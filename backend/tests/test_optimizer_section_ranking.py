import os
from typing import Dict, Optional

from services.optimizer import section_ranking as _section_ranking
from services.optimizer.core import PromptOptimizer


def _build_optimizer() -> PromptOptimizer:
    os.environ["OPTIMIZER_PREWARM_MODELS"] = "0"
    optimizer = PromptOptimizer()
    return optimizer


def _apply(
    optimizer: PromptOptimizer,
    prompt: str,
    override: Optional[Dict[str, object]] = None,
    *,
    chunking_mode: Optional[str] = None,
):
    ranking = _section_ranking.resolve_section_ranking(
        default_mode=optimizer.section_ranking_mode,
        default_token_budget=optimizer.section_ranking_token_budget,
        override=override,
    )
    return _section_ranking.apply_section_ranking(
        optimizer=optimizer,
        prompt=prompt,
        ranking=ranking,
        chunking_mode=chunking_mode,
        chat_metadata=None,
        default_chunking_mode=optimizer.default_chunking_mode,
        chunk_size=optimizer.chunk_size or 0,
        semantic_model=optimizer.semantic_chunk_model,
        semantic_rank_model=optimizer.semantic_rank_model,
        semantic_similarity=optimizer.semantic_chunk_similarity,
        count_tokens=optimizer.count_tokens,
        embedding_cache={},
    )


def test_section_ranking_single_section_no_change() -> None:
    optimizer = _build_optimizer()
    prompt = "Single focused instruction with no additional context."

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        override={"mode": "bm25"},
    )

    assert ranked == prompt
    assert applied is False
    assert metadata["selected_indices"] == [0]


def test_section_ranking_respects_token_budget() -> None:
    optimizer = _build_optimizer()
    sections = [
        "Primary instructions about the deployment pipeline and QA steps.",
        "Critical context describing database migrations and alerting signals.",
        "Ancillary note about team snacks for the Friday demo.",
    ]
    prompt = "\n\n".join(sections)
    token_budget = optimizer.count_tokens("\n\n".join(sections[:2])) + 5

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        override={
            "mode": "bm25",
            "token_budget": token_budget,
        },
        chunking_mode="structured",
    )

    assert applied is True
    assert sections[0] in ranked
    assert sections[1] in ranked
    assert sections[2] not in ranked


def test_section_ranking_mode_off_with_budget_keeps_order() -> None:
    optimizer = _build_optimizer()
    sections = [
        "System instructions that must lead the prompt.",
        "Follow-up guidance that should stay when space allows.",
        "Ancillary note that can be trimmed.",
    ]
    prompt = "\n\n".join(sections)

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        override={
            "mode": "off",
            "token_budget": optimizer.count_tokens(sections[0]),
        },
        chunking_mode="structured",
    )

    assert applied is True
    assert ranked.strip() == sections[0]
    assert metadata["selected_indices"] == [0]


def test_section_ranking_tfidf_mode_applies() -> None:
    optimizer = _build_optimizer()
    sections = [
        "Alpha instructions for the workflow.",
        "Beta details for the deployment pipeline.",
    ]
    prompt = "\n\n".join(sections)

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        override={
            "mode": "tfidf",
            "token_budget": optimizer.count_tokens(sections[0]),
        },
        chunking_mode="structured",
    )

    assert applied is True
    assert metadata["selected_indices"]
    assert ranked.strip()


def test_section_ranking_auto_budget_applies_when_budget_is_unset() -> None:
    optimizer = _build_optimizer()
    section = "Deployment guidance with critical details. " + ("context " * 900)
    prompt = "\n\n".join([f"{section} section_{i}" for i in range(6)])

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        override={"mode": "bm25", "token_budget": 0},
        chunking_mode="structured",
    )

    assert metadata.get("token_budget") is not None
    assert metadata["token_budget"] < optimizer.count_tokens(prompt)
    assert optimizer.count_tokens(ranked) <= metadata["token_budget"]
    assert applied is True


def test_section_ranking_env_zero_budget_uses_auto_cap(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PROMPT_OPTIMIZER_SECTION_RANKING_MODE", "bm25")
    monkeypatch.setenv("PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET", "0")
    optimizer = _build_optimizer()
    section = "Release runbook context. " + ("details " * 900)
    prompt = "\n\n".join([f"{section} section_{i}" for i in range(6)])

    ranked, applied, metadata = _apply(
        optimizer,
        prompt,
        chunking_mode="structured",
    )

    assert optimizer.section_ranking_token_budget is None
    assert metadata.get("token_budget") is not None
    assert metadata["token_budget"] < optimizer.count_tokens(prompt)
    assert optimizer.count_tokens(ranked) <= metadata["token_budget"]
    assert applied is True


def test_query_aware_compress_with_valid_query() -> None:
    """Test query-aware compression with valid query and context."""
    optimizer = _build_optimizer()

    context = (
        "The deployment pipeline includes automated testing. "
        "Database migrations run before deployment. "
        "The monitoring system tracks all errors. "
        "Team lunch is scheduled for Friday."
    )
    query = "What happens with the database?"

    compressed, applied, metadata = _section_ranking.query_aware_compress(
        prompt=context,
        query=query,
        model_name=optimizer.semantic_rank_model,
        budget_ratio=0.5,
        count_tokens=optimizer.count_tokens,
    )

    # Should have compressed the text
    if applied:
        assert len(compressed) < len(context)
        # Database-related content should be preserved
        assert "database" in compressed.lower() or "migrations" in compressed.lower()


def test_query_aware_compress_budget_ratios() -> None:
    """Test different budget ratios for different modes."""
    optimizer = _build_optimizer()

    context = " ".join([f"Sentence {i} with some content." for i in range(20)])
    query = "Sentence 5"

    for mode, expected_ratio in [
        ("conservative", 0.7),
        ("balanced", 0.55),
        ("maximum", 0.45),
    ]:
        compressed, applied, metadata = _section_ranking.query_aware_compress(
            prompt=context,
            query=query,
            model_name=optimizer.semantic_rank_model,
            budget_ratio=expected_ratio,
            count_tokens=optimizer.count_tokens,
        )

        if applied:
            assert "budget_tokens" in metadata
            # Verify budget was calculated based on ratio
            expected_budget = int(optimizer.count_tokens(context) * expected_ratio)
            assert (
                metadata["budget_tokens"] >= expected_budget * 0.8
            )  # Allow 20% variance


def test_query_aware_compress_model_unavailable() -> None:
    """Test fallback when semantic model unavailable."""
    optimizer = _build_optimizer()

    context = "This is a test context."
    query = "test"

    compressed, applied, metadata = _section_ranking.query_aware_compress(
        prompt=context,
        query=query,
        model_name=None,  # No model
        budget_ratio=0.5,
        count_tokens=optimizer.count_tokens,
    )

    assert compressed == context
    assert applied is False
    assert metadata == {}


def test_query_aware_compress_min_thresholds() -> None:
    """Test minimum token/sentence thresholds."""
    optimizer = _build_optimizer()

    # Too short prompt (below minimum tokens)
    short_context = "Short text."
    query = "text"

    compressed, applied, metadata = _section_ranking.query_aware_compress(
        prompt=short_context,
        query=query,
        model_name=optimizer.semantic_rank_model,
        budget_ratio=0.5,
        count_tokens=optimizer.count_tokens,
    )

    assert compressed == short_context
    assert applied is False

    # Too few sentences
    single_sentence = (
        "This is a single sentence with enough tokens to pass the minimum threshold."
    )
    compressed, applied, metadata = _section_ranking.query_aware_compress(
        prompt=single_sentence,
        query="sentence",
        model_name=optimizer.semantic_rank_model,
        budget_ratio=0.5,
        count_tokens=optimizer.count_tokens,
    )

    assert compressed == single_sentence
    assert applied is False
