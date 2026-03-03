import os
import tempfile
from pathlib import Path


DEFAULT_TEST_DB = Path(tempfile.gettempdir()) / "tokemizer_test_max_prepass.db"


def _configure_test_environment() -> None:
    os.environ["OPTIMIZER_PREWARM_MODELS"] = "0"
    if DEFAULT_TEST_DB.exists():
        DEFAULT_TEST_DB.unlink()
    os.environ["DB_PATH"] = str(DEFAULT_TEST_DB)


_configure_test_environment()


from services.optimizer import max_prepass as _max_prepass
from services.optimizer.core import PromptOptimizer


def _count_tokens(text: str) -> int:
    return len(text.split())


def test_budgeted_prepass_preserves_constraints_and_placeholders() -> None:
    prompt = (
        "System instructions: must never expose API keys. "
        "Use placeholder __PROTECT_0__ exactly as-is. "
        "General context about release planning and sprint updates. "
        "General context about release planning and sprint updates. "
        "General context about release planning and sprint updates."
    )
    config = _max_prepass.BudgetedPrepassConfig(
        enabled=True,
        minimum_tokens=20,
        budget_ratio=0.6,
        max_sentences=6,
    )

    compressed, applied, metadata = _max_prepass.budgeted_sentence_span_prepass(
        prompt=prompt,
        query="release planning",
        count_tokens=_count_tokens,
        config=config,
    )

    assert applied is True
    assert "must never expose api keys" in compressed.lower()
    assert "__PROTECT_0__" in compressed
    assert metadata["protected_indices"]


def test_budgeted_prepass_hard_preserves_explicit_protected_ranges() -> None:
    sentence_a = "Background sentence about onboarding."
    sentence_b = "Preserve this billing constraint sentence exactly."
    sentence_c = "Another background sentence that can be removed."
    prompt = " ".join([sentence_a, sentence_b, sentence_c, sentence_c])
    start = prompt.index(sentence_b)
    end = start + len(sentence_b)
    config = _max_prepass.BudgetedPrepassConfig(
        enabled=True,
        minimum_tokens=10,
        budget_ratio=0.45,
        max_sentences=3,
    )

    compressed, applied, _ = _max_prepass.budgeted_sentence_span_prepass(
        prompt=prompt,
        query="onboarding",
        count_tokens=_count_tokens,
        protected_ranges=[(start, end)],
        config=config,
    )

    assert applied is True
    assert sentence_b in compressed


def test_budgeted_prepass_prefers_non_redundant_sentence_when_budget_tight() -> None:
    # Two identical sentences + one unique sentence. With a tight budget, the
    # prepass should prefer keeping the unique content (higher redundancy_score).
    repeated = "Alpha bravo charlie delta echo foxtrot."
    unique = "One two three four five six."
    prompt = f"{repeated} {repeated} {unique}"
    config = _max_prepass.BudgetedPrepassConfig(
        enabled=True,
        minimum_tokens=1,
        budget_ratio=0.34,  # ~one sentence worth of budget
        max_sentences=1,
    )

    compressed, applied, _ = _max_prepass.budgeted_sentence_span_prepass(
        prompt=prompt,
        query=None,
        count_tokens=_count_tokens,
        config=config,
    )

    assert applied is True
    assert "one two three four five six" in compressed.lower()
    assert "alpha bravo charlie" not in compressed.lower()


def test_maximum_mode_prepass_feature_flag_integration(monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", "1")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS", "80")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO", "0.55")

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "auto"
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_a, **_k: 0.99,
    )
    monkeypatch.setattr(
        optimizer,
        "_maybe_prune_low_entropy",
        lambda text: text,
    )
    monkeypatch.setattr(
        optimizer,
        "_optimize_with_token_classifier",
        lambda text, **_k: (text, False, {}),
    )
    repeated = " ".join(
        [
            f"Sentence {index}: deployment checklist includes QA validation and monitoring handoff."
            for index in range(120)
        ]
    )
    prompt = (
        "System: must never remove compliance constraints. "
        + repeated
        + " Use __PROTECT_0__ verbatim."
    )

    result = optimizer.optimize(prompt, optimization_mode="maximum")

    techniques = result.get("techniques_applied") or []
    assert "Maximum Mode Budgeted Pre-Pass" in techniques

    if "Maximum Mode Budgeted Pre-Pass" in techniques:
        assert result["stats"].get("maximum_prepass_selected_sentences") is not None
        assert result["stats"].get("maximum_prepass_policy_source") == "forced"
        assert result["stats"].get("maximum_prepass_policy_mode") == "auto"
        assert result["stats"].get("maximum_prepass_policy_enabled_override") is True
        assert (
            result["stats"].get("maximum_prepass_policy_adaptive_budget_ratio")
            is not None
        )
        assert (
            result["stats"].get("maximum_prepass_policy_adaptive_redundancy_ratio")
            is not None
        )
        assert (
            result["stats"].get("maximum_prepass_policy_adaptive_constraint_density")
            is not None
        )
    # Always preserve placeholders when present
    assert "__PROTECT_0__" in result["optimized_output"]


def test_maximum_mode_prepass_policy_off_disables_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", raising=False)

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "off"
    policy = optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=9000,
        content_profile=type("Profile", (), {"name": "general_prose"})(),
        query_hint="compliance",
        redundancy_estimate=0.35,
    )

    assert policy["policy_source"] == "off"
    assert policy["policy_mode"] == "off"
    assert policy["enabled"] is False


def test_maximum_mode_prepass_policy_auto_enables_for_large_risk_aware_profiles(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES", raising=False)

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "auto"
    policy = optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=5000,
        content_profile=type("Profile", (), {"name": "heavy_document"})(),
        query_hint="incident timeline",
        redundancy_estimate=0.3,
    )

    assert policy["policy_source"] == "auto"
    assert policy["enabled"] is True


def test_maximum_mode_prepass_policy_auto_disables_for_non_risk_aware_profiles(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", raising=False)

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "auto"
    policy = optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=8000,
        content_profile=type("Profile", (), {"name": "code"})(),
        query_hint=None,
        redundancy_estimate=0.4,
    )

    assert policy["enabled"] is False


def test_maximum_mode_prepass_policy_conservative_vs_aggressive(monkeypatch) -> None:
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", raising=False)

    conservative_optimizer = PromptOptimizer()
    conservative_optimizer.maximum_prepass_policy = "conservative"
    conservative = conservative_optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=2000,
        content_profile=type("Profile", (), {"name": "general_prose"})(),
        query_hint=None,
        redundancy_estimate=0.1,
    )

    aggressive_optimizer = PromptOptimizer()
    aggressive_optimizer.maximum_prepass_policy = "aggressive"
    aggressive = aggressive_optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=2000,
        content_profile=type("Profile", (), {"name": "general_prose"})(),
        query_hint=None,
        redundancy_estimate=0.1,
    )

    assert conservative["enabled"] is True
    assert aggressive["enabled"] is True
    assert conservative["minimum_tokens"] > aggressive["minimum_tokens"]
    assert conservative["budget_ratio"] > aggressive["budget_ratio"]


def test_maximum_mode_prepass_policy_env_overrides_take_precedence(monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", "1")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS", "2000")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO", "0.7")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES", "100")

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "off"
    policy = optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=500,
        content_profile=type("Profile", (), {"name": "code"})(),
        query_hint=None,
        redundancy_estimate=0.0,
    )

    assert policy["enabled"] is False
    assert policy["policy_mode"] == "off"
    assert policy["minimum_tokens"] == 2000
    assert policy["budget_ratio"] == 0.7
    assert policy["max_sentences"] == 100
    assert policy["policy_source"] == "forced_small_prompt"
    assert policy["enabled_override"] is True


def test_budgeted_prepass_adapts_ratio_up_for_dense_constraints() -> None:
    prompt = " ".join(
        [
            "Must include PCI controls in every response.",
            "Shall not remove auditability requirements.",
            "Required: preserve compliance tags.",
            "General planning context for rollout and deployment.",
            "General planning context for rollout and deployment.",
        ]
    )
    config = _max_prepass.BudgetedPrepassConfig(
        enabled=True,
        minimum_tokens=1,
        budget_ratio=0.5,
        max_sentences=8,
    )

    _compressed, applied, metadata = _max_prepass.budgeted_sentence_span_prepass(
        prompt=prompt,
        query="",
        count_tokens=_count_tokens,
        config=config,
    )

    assert applied is True
    assert metadata["constraint_density"] > 0.1
    assert metadata["resolved_budget_ratio"] >= 0.5
    assert metadata["auto_safety_floor_ratio"] is not None


def test_budgeted_prepass_adapts_ratio_down_for_redundant_sparse_constraints() -> None:
    repeated = "General release planning status update and owner tracking."
    prompt = " ".join([repeated] * 7 + ["Distinct risk mitigation sentence."])
    config = _max_prepass.BudgetedPrepassConfig(
        enabled=True,
        minimum_tokens=1,
        budget_ratio=0.7,
        max_sentences=10,
    )

    _compressed, applied, metadata = _max_prepass.budgeted_sentence_span_prepass(
        prompt=prompt,
        query="risk",
        count_tokens=_count_tokens,
        config=config,
    )

    assert applied is True
    assert metadata["constraint_density"] < 0.1
    assert metadata["redundancy_ratio"] > 0.3
    assert metadata["resolved_budget_ratio"] < 0.7


def test_maximum_mode_prepass_policy_auto_records_adaptive_signals(monkeypatch) -> None:
    monkeypatch.delenv("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", raising=False)

    optimizer = PromptOptimizer()
    optimizer.maximum_prepass_policy = "auto"
    policy = optimizer._resolve_maximum_prepass_policy(
        prompt_tokens=5000,
        content_profile=type("Profile", (), {"name": "heavy_document"})(),
        query_hint=None,
        redundancy_estimate=0.4,
        constraint_density=0.02,
    )

    assert policy["policy_source"] == "auto"
    assert policy["adaptive_applied"] is True
    assert policy["adaptive_redundancy_ratio"] == 0.4
    assert policy["adaptive_constraint_density"] == 0.02
