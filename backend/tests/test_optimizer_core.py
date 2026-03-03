from __future__ import annotations

import os
import ast
import json
import logging
import math
import re
from types import SimpleNamespace
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple
from uuid import uuid4

import pytest


def _new_test_db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"tokemizer_test_core_{os.getpid()}_{uuid4().hex}.db"


DEFAULT_TEST_DB = _new_test_db_path()


def _configure_test_environment() -> None:
    global DEFAULT_TEST_DB
    DEFAULT_TEST_DB = _new_test_db_path()
    os.environ["OPTIMIZER_PREWARM_MODELS"] = "0"
    os.environ["DB_PATH"] = str(DEFAULT_TEST_DB)


_configure_test_environment()

import database


from services.optimizer import preservation
from services.optimizer import chunking as chunking_helpers
from services.optimizer.core import PromptOptimizer, _segment_allows_contextual_canon
from services.optimizer.lexical import canonicalize_entities
from services.optimizer.pipeline_config import OPTIMIZATION_MODES
from services.optimizer.router import SmartContext, classify_content, get_profile


@pytest.fixture(autouse=True)
def _disable_optimizer_prewarm(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.optimizer.metrics as optimizer_metrics
    import services.optimizer.token_classifier as token_classifier

    monkeypatch.setenv("OPTIMIZER_PREWARM_MODELS", "0")
    original_encode = optimizer_metrics.encode_texts_with_plan

    def _safe_encode_texts_with_plan(*args, **kwargs):
        embeddings = original_encode(*args, **kwargs)
        if embeddings is not None:
            return embeddings

        texts = args[0] if args else kwargs.get("texts", [])
        if optimizer_metrics.np is not None:
            return [
                optimizer_metrics.np.array([float(index + 1)], dtype=float)
                for index, _ in enumerate(texts)
            ]
        return [[float(index + 1)] for index, _ in enumerate(texts)]

    def _no_op_token_classifier(
        text: str,
        **_kwargs,
    ):
        return text, False, {"keep_ratio": 1.0, "decisions": 0, "removals": 0}

    monkeypatch.setattr(
        optimizer_metrics, "encode_texts_with_plan", _safe_encode_texts_with_plan
    )
    monkeypatch.setattr(
        token_classifier, "compress_with_token_classifier", _no_op_token_classifier
    )
    monkeypatch.setattr(
        token_classifier, "evaluate_shadow_classifier", lambda *_a, **_k: {}
    )


def _placeholder_ranges(text: str, preserved: Dict) -> List[Tuple[int, int]]:
    optimizer = PromptOptimizer()
    return optimizer._get_placeholder_ranges(text, preserved)


def test_get_placeholder_ranges_filters_non_preserved_tokens() -> None:
    preserved = {
        "code_blocks": ["print('hello')", "print('bye')"],
        "urls": ["https://example.com"],
    }
    text = (
        "Start __CODE_0__ middle __URL_0__ __QUOTE_0__ "
        "end __CODE_1__ and another __CODE_0__"
    )

    ranges = _placeholder_ranges(text, preserved)

    expected = [
        (6, 16),
        (24, 33),
        (50, 60),
        (73, 83),
    ]

    assert ranges == expected


def test_get_placeholder_ranges_handles_repeated_tokens_within_text() -> None:
    preserved = {"code_blocks": ["print('hello')"]}
    text = "__CODE_0__ start(__CODE_0__) middle __CODE_0__"

    ranges = _placeholder_ranges(text, preserved)

    expected = [
        (0, 10),
        (17, 27),
        (36, 46),
    ]

    assert ranges == expected


def test_canonicalize_entities_caches_patterns() -> None:
    """Test that canonicalization replacers are cached for repeated use."""
    built_in = {"javascript": "JS", "typescript": "TS"}
    config_overrides = {"application": "app"}

    text1 = "This javascript application uses typescript"
    result1, map1 = canonicalize_entities(text1, built_in, config_overrides)

    from services.optimizer.trie_replacer import get_canonical_replacer

    replacer1 = get_canonical_replacer(map1)
    replacer2 = get_canonical_replacer(dict(map1))
    assert replacer1 is replacer2

    text2 = "Another javascript application with typescript"
    result2, map2 = canonicalize_entities(text2, built_in, config_overrides)

    # Verify correct replacements
    assert "JS" in result1
    assert "TS" in result1
    assert "app" in result1
    assert map1 == map2
    assert result2


def test_canonicalize_entities_preserves_behavior() -> None:
    """Test that cached version produces same results as non-cached."""
    built_in = {"maximum": "max", "minimum": "min"}
    config_overrides = {"performance": "perf"}

    text = "The maximum performance and minimum values"
    result, mapping = canonicalize_entities(text, built_in, config_overrides)

    # Verify all replacements occurred
    assert "max" in result.lower()
    assert "perf" in result.lower()
    assert "min" in result.lower()
    assert mapping == {"maximum": "max", "minimum": "min", "performance": "perf"}


def test_classify_content_profiles() -> None:
    code_prompt = "def foo():\n    return 1\n" * 10
    json_prompt = (
        '{"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}], '
        '"meta": {"count": 2, "version": "1.0.0"}, '
        '"notes": "keep structure intact"}'
    )
    dialogue_prompt = (
        "User: Can you summarize the report findings?\n"
        "Assistant: Sure, here is a quick summary.\n"
        "User: Include the risks and next steps.\n"
        "Assistant: I will include both sections."
    )
    technical_prompt = (
        "API Reference\n"
        "GET /v1/metrics returns telemetry for the last 24 hours.\n"
        "Response includes latency_ms and tokens_saved fields."
    )
    prose_prompt = (
        "The project aims to improve prompt optimization throughput while keeping "
        "semantic fidelity intact across varied workloads and client traffic."
    )

    assert classify_content(code_prompt) == "code"
    assert classify_content(json_prompt) == "json"
    assert classify_content(dialogue_prompt) == "dialogue"
    assert classify_content(technical_prompt) == "technical_doc"
    assert classify_content(prose_prompt) == "general_prose"


def test_estimate_sentence_redundancy_ratio_preserves_digits_for_structured_lines() -> None:
    optimizer = PromptOptimizer()
    prompt = "id: 123\nid: 456\nid: 456\n"
    ratio = optimizer._estimate_sentence_redundancy_ratio(prompt)
    assert ratio == pytest.approx(1 / 3)


def test_estimate_sentence_redundancy_ratio_does_not_flag_distinct_structured_ids() -> None:
    optimizer = PromptOptimizer()
    prompt = "\n".join([f"id: {i}" for i in range(25)])
    ratio = optimizer._estimate_sentence_redundancy_ratio(prompt)
    assert ratio == 0.0


def test_prechunk_sentence_dedup_preserves_repeated_constraints() -> None:
    optimizer = PromptOptimizer()
    prompt = (
        "You must never share secrets. You must never share secrets.\n\n"
        "Provide a concise answer."
    )

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == prompt
    assert changed is False


def test_prechunk_sentence_dedup_removes_generic_filler_inside_section() -> None:
    optimizer = PromptOptimizer()
    prompt = "This is additional context for background only. This is additional context for background only."

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == "This is additional context for background only."
    assert changed is True


def test_prechunk_sentence_dedup_keeps_one_duplicate_across_section_boundary() -> None:
    optimizer = PromptOptimizer()
    prompt = (
        "This is additional context for background only.\n\n"
        "This is additional context for background only.\n\n"
        "This is additional context for background only."
    )

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == (
        "This is additional context for background only.\n\n"
        "This is additional context for background only."
    )
    assert changed is True


def test_prechunk_sentence_dedup_balanced_removes_lower_confidence_safe_duplicate() -> None:
    optimizer = PromptOptimizer()
    optimizer._get_state().optimization_mode = "balanced"
    prompt = (
        "Background context details help align quality and tone for this request! "
        "Background context details help align quality and tone for this request!"
    )

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == "Background context details help align quality and tone for this request!"
    assert changed is True


def test_prechunk_sentence_dedup_conservative_keeps_lower_confidence_safe_duplicate() -> None:
    optimizer = PromptOptimizer()
    optimizer._get_state().optimization_mode = "conservative"
    prompt = (
        "Background context details help align quality and tone for this request! "
        "Background context details help align quality and tone for this request!"
    )

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == prompt
    assert changed is False


def test_prechunk_sentence_dedup_maximum_removes_cross_section_duplicate() -> None:
    optimizer = PromptOptimizer()
    optimizer._get_state().optimization_mode = "maximum"
    prompt = (
        "This is additional context for background only.\n\n"
        "This is additional context for background only.\n\n"
        "This is additional context for background only."
    )

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == "This is additional context for background only."
    assert changed is True


def test_prechunk_sentence_dedup_preserves_short_imperative_repetition() -> None:
    optimizer = PromptOptimizer()
    prompt = "Answer briefly! Answer briefly!"

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == prompt
    assert changed is False


def test_prechunk_sentence_dedup_preserves_extra_section_spacing_when_unchanged() -> None:
    optimizer = PromptOptimizer()
    prompt = "Ensure accuracy first.\n\n\nEnsure accuracy first."

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == prompt
    assert changed is False


def test_prechunk_sentence_dedup_preserves_trailing_blank_lines_when_unchanged() -> None:
    optimizer = PromptOptimizer()
    prompt = "Ensure accuracy first.\n\n"

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped == prompt
    assert changed is False


def test_prechunk_sentence_dedup_reports_change_when_duplicate_removed() -> None:
    optimizer = PromptOptimizer()
    prompt = "This is additional context for background only. This is additional context for background only."

    deduped, changed = optimizer._deduplicate_normalized_sentences(prompt)

    assert deduped != prompt
    assert changed is True


def _reload_database(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    import importlib

    import database

    importlib.reload(database)
    database.init_db()
    return database


def test_custom_canonicals_apply_in_fastpath(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _reload_database(tmp_path, monkeypatch)
    database.create_user_canonical_mapping("cust_a", "application", "APPX")
    database.toggle_ootb_mapping("cust_a", "application", enabled=False)

    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 1000

    prompt = " ".join(["This application is nice."] * 10)
    result = optimizer.optimize(prompt, customer_id="cust_a")

    assert "APPX" in result["optimized_output"]


def test_custom_canonicals_apply_in_chunking(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _reload_database(tmp_path, monkeypatch)
    database.create_user_canonical_mapping("cust_a", "application", "APPX")
    database.toggle_ootb_mapping("cust_a", "application", enabled=False)

    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.chunk_size = 1
    optimizer.chunk_threshold = 1
    optimizer.default_chunking_mode = "fixed"
    optimizer.semantic_guard_enabled = False
    monkeypatch.setattr(
        "services.optimizer.core.resolve_smart_context",
        lambda *_a, **_k: SmartContext(
            enable_frequency_learning=False,
            use_discourse_weighting=True,
            chunking_mode="fixed",
            section_ranking_enabled=False,
            preserve_digits=False,
            description="forced fixed chunking for test",
        ),
    )

    prompt = " ".join(["This application is nice."] * 10)
    result = optimizer.optimize(prompt, customer_id="cust_a")

    assert "APPX" in result["optimized_output"]


def test_profile_threshold_modifiers() -> None:
    code_profile = get_profile("code")
    json_profile = get_profile("json")
    prose_profile = get_profile("general_prose")

    assert math.isclose(code_profile.get_threshold_modifier("entropy_budget", 1.0), 0.4)
    assert math.isclose(
        code_profile.get_threshold_modifier("summarize_threshold", 1.0), 1.1
    )
    assert math.isclose(
        json_profile.get_threshold_modifier("summarize_threshold", 1.0), 1.2
    )
    assert math.isclose(
        prose_profile.get_threshold_modifier("entropy_budget", 1.0), 1.15
    )


def test_fastpath_disabled_passes_are_real_pipeline_passes() -> None:
    import services.optimizer.core as optimizer_core

    source = Path(optimizer_core.__file__).read_text(encoding="utf-8")
    pass_names = set(
        re.findall(
            r"should_skip_pass\(\s*[\"']([^\"']+)[\"']\s*\)",
            source,
        )
    )

    # If a pass name is in _FASTPATH_DISABLED_PASSES but isn't consulted by the pipeline,
    # it is dead configuration and should be removed.
    virtual_references = {"compress_examples", "summarize_history", "prune_low_entropy"}
    missing = sorted(
        (set(optimizer_core._FASTPATH_DISABLED_PASSES) - pass_names) - virtual_references
    )
    assert missing == []


def _extract_should_skip_pass_names() -> set[str]:
    import services.optimizer.core as optimizer_core

    source = Path(optimizer_core.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    pass_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "should_skip_pass":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            pass_names.add(first_arg.value)
    return pass_names


def test_pipeline_config_pass_governance() -> None:
    import services.optimizer.core as optimizer_core

    pass_names = _extract_should_skip_pass_names()
    virtual_toggle_whitelist: set[str] = {
        "compress_examples",
        "summarize_history",
        "prune_low_entropy",
    }

    unknown_disabled_by_mode: dict[str, list[str]] = {}
    unknown_toggles_by_mode: dict[str, list[str]] = {}

    for mode, mode_config in OPTIMIZATION_MODES.items():
        disabled_passes = set(mode_config.get("disabled_passes", []))
        unknown_disabled = sorted(disabled_passes - pass_names - virtual_toggle_whitelist)
        if unknown_disabled:
            unknown_disabled_by_mode[mode] = unknown_disabled

        toggle_passes = set(mode_config.get("pass_toggles", {}).keys())
        unknown_toggles = sorted(toggle_passes - pass_names - virtual_toggle_whitelist)
        if unknown_toggles:
            unknown_toggles_by_mode[mode] = unknown_toggles

    unknown_fastpath = sorted(
        (set(optimizer_core._FASTPATH_DISABLED_PASSES) - pass_names)
        - virtual_toggle_whitelist
    )

    assert not (
        unknown_disabled_by_mode or unknown_toggles_by_mode or unknown_fastpath
    ), (
        "Unknown optimizer pass references detected. "
        f"disabled_passes={unknown_disabled_by_mode}; "
        f"pass_toggles={unknown_toggles_by_mode}; "
        f"_FASTPATH_DISABLED_PASSES={unknown_fastpath}; "
        f"known_should_skip_passes={sorted(pass_names)}"
    )


def test_token_classifier_prepass_does_not_short_circuit_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.default_chunking_mode = "off"
    optimizer.chunk_threshold = 10**9
    optimizer.semantic_guard_enabled = False

    seen: Dict[str, str] = {}

    def _fake_classifier(*args, **kwargs):
        text = kwargs.get("text") or (args[1] if len(args) > 1 else "")
        return f"{text} CLASSIFIED", True, {}

    def _fake_pipeline(self, prompt: str, *args, **kwargs):
        seen["pipeline_prompt"] = prompt
        return f"{prompt} PIPELINED"

    monkeypatch.setattr(
        PromptOptimizer,
        "_optimize_with_token_classifier",
        _fake_classifier,
    )
    monkeypatch.setattr(PromptOptimizer, "_optimize_pipeline", _fake_pipeline)

    result = optimizer.optimize("hello", optimization_mode="maximum")
    assert seen["pipeline_prompt"].endswith("CLASSIFIED")
    assert result["optimized_output"].endswith("PIPELINED")


def test_maximum_short_prompt_caps_multi_candidate_settings() -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 1000
    state = optimizer._get_state()
    state.original_tokens = 97

    settings = optimizer._resolve_multi_candidate_settings(
        "token_classifier", "maximum"
    )

    assert settings["max_candidates"] == 1


def test_entropy_prune_estimation_failure_is_fail_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer._get_state().optimization_mode = "maximum"

    monkeypatch.setattr(
        optimizer, "_entropy_prune_budget", lambda *_a, **_k: 64
    )
    monkeypatch.setattr(
        optimizer,
        "_estimate_avg_nll",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("entropy failed")),
    )

    source = "Keep this text exactly."
    result = optimizer._maybe_prune_low_entropy(source)
    assert result == source


def test_semantic_chunking_without_model_fails_fast() -> None:
    optimizer = PromptOptimizer()

    with pytest.raises(RuntimeError) as exc_info:
        chunking_helpers.chunk_prompt(
            optimizer,
            "First paragraph.\n\nSecond paragraph.",
            chunk_size=128,
            strategy="semantic",
            semantic_model=None,
        )

    assert "strict mode forbids fallback to structured chunking" in str(exc_info.value)


def test_semantic_guard_chunk_retry_disables_entity_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.default_chunking_mode = "fixed"
    optimizer.chunk_size = 1
    optimizer.chunk_threshold = 1
    optimizer.semantic_guard_enabled = True
    optimizer.semantic_guard_per_pass_enabled = False

    calls: List[Dict[str, object]] = []

    def _fake_classifier(*args, **kwargs):
        text = kwargs.get("text") or (args[1] if len(args) > 1 else "")
        return f"{text} CLASSIFIED", True, {}

    def _fake_chunking(self, prompt: str, mode: str, optimization_mode: str, **kwargs):
        calls.append(
            {
                "optimization_mode": optimization_mode,
                "force_disabled_passes": set(kwargs.get("force_disabled_passes") or []),
            }
        )
        return "x", [SimpleNamespace(metadata={"strategy": "fixed"})]

    monkeypatch.setattr(PromptOptimizer, "_optimize_with_token_classifier", _fake_classifier)
    monkeypatch.setattr(PromptOptimizer, "_optimize_with_chunking", _fake_chunking)
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_a, **_k: 0.99,
    )
    monkeypatch.setattr(optimizer, "_resolve_semantic_guard_threshold", lambda: 0.95)
    monkeypatch.setattr(optimizer, "_lexical_similarity", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(
        "services.optimizer.core.resolve_smart_context",
        lambda *_a, **_k: SmartContext(
            enable_frequency_learning=False,
            use_discourse_weighting=True,
            chunking_mode="fixed",
            section_ranking_enabled=False,
            preserve_digits=False,
            description="forced for test",
        ),
    )

    optimizer.optimize(
        "This prompt is intentionally long enough to trigger chunking in the test.",
        optimization_mode="maximum",
    )

    assert len(calls) == 1
    assert calls[0]["optimization_mode"] == "maximum"
    assert "alias_named_entities" in calls[0]["force_disabled_passes"]


def test_conservative_mode_keeps_entropy_pruning_enabled() -> None:
    disabled_passes = set(OPTIMIZATION_MODES["conservative"]["disabled_passes"])
    assert "prune_low_entropy" not in disabled_passes


def test_query_aware_floor_skips_section_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.default_chunking_mode = "off"
    optimizer.chunk_size = 100
    optimizer.chunk_threshold = 1
    optimizer.semantic_guard_enabled = False
    optimizer.semantic_guard_per_pass_enabled = False

    query_aware_prompt = " ".join(["context"] * 150)
    source_prompt = " ".join(["original"] * 300)
    ranking_calls = {"count": 0}
    pipeline_prompt = {"value": ""}

    monkeypatch.setattr(
        "services.optimizer.core.resolve_smart_context",
        lambda *_a, **_k: SmartContext(
            enable_frequency_learning=False,
            use_discourse_weighting=True,
            chunking_mode="off",
            section_ranking_enabled=True,
            preserve_digits=False,
            description="forced for test",
        ),
    )
    monkeypatch.setattr(
        optimizer,
        "_resolve_maximum_prepass_policy",
        lambda **_k: {
            "enabled": False,
            "minimum_tokens": optimizer.maximum_prepass_min_tokens,
            "budget_ratio": optimizer.maximum_prepass_budget_ratio,
            "max_sentences": optimizer.maximum_prepass_max_sentences,
            "policy_source": "test",
            "policy_mode": "off",
            "enabled_override": False,
        },
    )
    monkeypatch.setattr(
        "services.optimizer.core._section_ranking.query_aware_compress",
        lambda **_k: (query_aware_prompt, True, {}),
    )

    def _unexpected_section_ranking(*_args, **_kwargs):
        ranking_calls["count"] += 1
        return query_aware_prompt, False, {"selected_indices": []}

    def _fake_pipeline(self, prompt: str, *args, **kwargs):
        pipeline_prompt["value"] = prompt
        return prompt

    monkeypatch.setattr(
        "services.optimizer.core._section_ranking.apply_section_ranking",
        _unexpected_section_ranking,
    )
    monkeypatch.setattr(
        PromptOptimizer,
        "_optimize_with_token_classifier",
        lambda self, prompt, **_kwargs: (prompt, False, {}),
    )
    monkeypatch.setattr(PromptOptimizer, "_optimize_pipeline", _fake_pipeline)

    result = optimizer.optimize(source_prompt, optimization_mode="maximum", query="focus")

    assert ranking_calls["count"] == 0
    assert pipeline_prompt["value"] == query_aware_prompt
    assert "Query-Aware Compression" in result["techniques_applied"]
    assert not any(
        technique.startswith("Context Section Ranking")
        for technique in result["techniques_applied"]
    )


def test_section_ranking_rolls_back_below_hard_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.default_chunking_mode = "off"
    optimizer.chunk_size = 100
    optimizer.chunk_threshold = 1
    optimizer.semantic_guard_enabled = False
    optimizer.semantic_guard_per_pass_enabled = False

    query_aware_prompt = " ".join(["context"] * 180)
    ranked_prompt = " ".join(["trimmed"] * 80)
    source_prompt = " ".join(["original"] * 300)
    ranking_calls = {"count": 0}
    pipeline_prompt = {"value": ""}

    monkeypatch.setattr(
        "services.optimizer.core.resolve_smart_context",
        lambda *_a, **_k: SmartContext(
            enable_frequency_learning=False,
            use_discourse_weighting=True,
            chunking_mode="off",
            section_ranking_enabled=True,
            preserve_digits=False,
            description="forced for test",
        ),
    )
    monkeypatch.setattr(
        optimizer,
        "_resolve_maximum_prepass_policy",
        lambda **_k: {
            "enabled": False,
            "minimum_tokens": optimizer.maximum_prepass_min_tokens,
            "budget_ratio": optimizer.maximum_prepass_budget_ratio,
            "max_sentences": optimizer.maximum_prepass_max_sentences,
            "policy_source": "test",
            "policy_mode": "off",
            "enabled_override": False,
        },
    )
    monkeypatch.setattr(
        "services.optimizer.core._section_ranking.query_aware_compress",
        lambda **_k: (query_aware_prompt, True, {}),
    )

    def _fake_section_ranking(*_args, **_kwargs):
        ranking_calls["count"] += 1
        return ranked_prompt, True, {"selected_indices": [0]}

    def _fake_pipeline(self, prompt: str, *args, **kwargs):
        pipeline_prompt["value"] = prompt
        return prompt

    monkeypatch.setattr(
        "services.optimizer.core._section_ranking.apply_section_ranking",
        _fake_section_ranking,
    )
    monkeypatch.setattr(
        PromptOptimizer,
        "_optimize_with_token_classifier",
        lambda self, prompt, **_kwargs: (prompt, False, {}),
    )
    monkeypatch.setattr(PromptOptimizer, "_optimize_pipeline", _fake_pipeline)

    result = optimizer.optimize(source_prompt, optimization_mode="maximum", query="focus")

    assert ranking_calls["count"] == 1
    assert pipeline_prompt["value"] == query_aware_prompt
    assert "Query-Aware Compression" in result["techniques_applied"]
    assert not any(
        technique.startswith("Context Section Ranking")
        for technique in result["techniques_applied"]
    )


def test_contextual_canonicalizations_skip_code_and_json() -> None:
    optimizer = PromptOptimizer()

    code_prompt = (
        "def foo():\n"
        "    return 'please note that this value is required'\n"
        "def bar():\n"
        "    return 'keep in mind that the output matters'\n"
    )
    code_prompt = code_prompt + "def baz():\n    return 1\n" * 5
    code_profile = get_profile("code")
    code_output = optimizer._optimize_pipeline(
        code_prompt,
        mode="basic",
        optimization_mode="balanced",
        content_type="code",
        content_profile=code_profile,
    )
    assert code_output == code_prompt

    json_prompt = (
        '{"note": "please note that this payload is immutable", '
        '"context": "keep in mind that changes are rejected", '
        '"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]}'
    )
    json_profile = get_profile("json")
    json_output = optimizer._optimize_pipeline(
        json_prompt,
        mode="basic",
        optimization_mode="balanced",
        content_type="json",
        content_profile=json_profile,
        json_policy={"minify": False},
    )
    assert json_output == json_prompt


def test_contextual_canonicalizations_skip_negation_and_questions() -> None:
    assert (
        _segment_allows_contextual_canon("Please note that this is required.") is True
    )
    assert (
        _segment_allows_contextual_canon("Please note that this is not required.")
        is False
    )
    assert (
        _segment_allows_contextual_canon("Please note that this is required?") is False
    )


def test_disabled_ootb_canonical_not_applied_with_contextual_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()

    def _fake_combined(_customer_id: str | None) -> Dict[str, str]:
        return {"as soon as possible": "ASAP"}

    def _fake_disabled(_customer_id: str | None) -> List[str]:
        return ["for example"]

    monkeypatch.setattr(database, "get_combined_canonical_mappings", _fake_combined)
    monkeypatch.setattr(database, "list_disabled_ootb_mappings", _fake_disabled)

    prompt = "Schedule as soon as possible, for example, after kickoff."
    profile = get_profile("general_prose")
    output = optimizer._optimize_pipeline(
        prompt,
        mode="basic",
        optimization_mode="balanced",
        content_type="general_prose",
        content_profile=profile,
        customer_id="customer-123",
    )

    assert "ASAP" in output
    assert "for example" in output
    assert "e.g." not in output


def test_custom_canonical_override_applies_over_smart_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()

    def _fake_combined(_customer_id: str | None) -> Dict[str, str]:
        return {"baseline": "base"}

    def _fake_disabled(_customer_id: str | None) -> List[str]:
        return []

    monkeypatch.setattr(database, "get_combined_canonical_mappings", _fake_combined)
    monkeypatch.setattr(database, "list_disabled_ootb_mappings", _fake_disabled)

    prompt = "We should, for example, verify overrides."
    profile = get_profile("general_prose")
    output = optimizer._optimize_pipeline(
        prompt,
        mode="basic",
        optimization_mode="balanced",
        content_type="general_prose",
        content_profile=profile,
        customer_id="customer-123",
        custom_canonicals={"for example": "for-eg"},
    )

    assert "for-eg" in output
    assert "e.g." not in output


def test_disabled_ootb_keeps_custom_override_in_contextual_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()

    def _fake_combined(_customer_id: str | None) -> Dict[str, str]:
        return {"for example": "for-eg"}

    def _fake_disabled(_customer_id: str | None) -> List[str]:
        return ["for example"]

    monkeypatch.setattr(database, "get_combined_canonical_mappings", _fake_combined)
    monkeypatch.setattr(database, "list_disabled_ootb_mappings", _fake_disabled)

    prompt = "We should, for example, verify overrides."
    profile = get_profile("general_prose")
    output = optimizer._optimize_pipeline(
        prompt,
        mode="basic",
        optimization_mode="balanced",
        content_type="general_prose",
        content_profile=profile,
        customer_id="customer-123",
    )

    assert "for-eg" in output
    assert "e.g." not in output


def test_json_payload_round_trip_with_default_policy() -> None:
    optimizer = PromptOptimizer()
    prompt = '{"title": "Hello", "summary": "Preserved    value   with  spaces"}'

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    assert json.loads(output) == json.loads(prompt)


def test_force_preserve_respects_protect_regions() -> None:
    optimizer = PromptOptimizer()
    text = "<protect>secret 007</protect> codeX2"

    result, preserved = preservation.extract_and_preserve(
        optimizer,
        text,
        force_digits=True,
    )

    assert preserved["protected"] == ["secret 007"]
    assert preserved["forced"] == ["codeX2"]
    assert result.startswith("__PROTECT_0__")


def test_extract_and_preserve_keeps_multitick_inline_code() -> None:
    optimizer = PromptOptimizer()
    text = "Inline ``code `span` here`` and ```inline `code` span``` plus `single`."

    _, preserved = preservation.extract_and_preserve(optimizer, text)

    assert preserved["code_blocks"] == [
        "``code `span` here``",
        "```inline `code` span```",
        "`single`",
    ]


def test_extract_and_preserve_ignores_shell_escape_backticks() -> None:
    optimizer = PromptOptimizer()
    command = (
        "docker compose -f docker-compose.yml -f docker-compose.windows-test.yml "
        "exec -T backend ` python scripts/seed_admin.py usman@tokem.com "
        "'Usman111!' 'Usman Saleem'"
    )

    _, preserved = preservation.extract_and_preserve(optimizer, command * 3)

    assert preserved["code_blocks"] == []


def test_fastpath_applies_lexical_cleanup() -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 128
    prompt = "Please provide a concise summary of the findings."

    result = optimizer.optimize(prompt, mode="basic")

    assert result["stats"]["fast_path"] is True
    assert "fast path lexical cleanup" in " ".join(
        technique.lower() for technique in result["techniques_applied"]
    )
    assert "please" not in result["optimized_output"].lower()


def test_fastpath_deduplicates_repeated_sentences() -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 200
    prompt = "A TM Forum-focused architecture advisor. " * 3

    result = optimizer.optimize(prompt, mode="basic")
    output = result["optimized_output"].strip()

    sentences = [
        sentence for sentence in re.split(r"(?<=[.!?])\s+", output) if sentence
    ]

    assert result["stats"]["fast_path"] is True
    assert len(sentences) == 1
    assert "Content Deduplication" in result["techniques_applied"]


def test_deduplicates_repeated_shell_command_block() -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0

    command = (
        "docker compose -f docker-compose.yml -f docker-compose.windows-test.yml "
        "exec -T backend ` python scripts/seed_admin.py usman@tokem.com "
        "'Usman111!' 'Usman Saleem'"
    )
    prompt = command * 3

    result = optimizer.optimize(prompt, mode="basic")
    output = result["optimized_output"]

    assert output.count("scripts/seed_admin.py") == 1
    assert output.count("docker compose -f docker-compose.yml") == 1
    assert "Content Deduplication" in result["techniques_applied"]


def test_deduplicate_near_duplicate_sentences() -> None:
    """Test that near-duplicate sentences (>92% similar) are deduplicated."""
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    original_guard = optimizer.semantic_guard_enabled
    optimizer.semantic_guard_enabled = False

    # Create near-duplicate sentences with very high overlap (>92%)
    # Only one word differs out of ~15 words = ~93% overlap
    sentence1 = (
        "Please execute the longitudinal analysis process on the given topic "
        "and provide comprehensive data review for the final detailed report."
    )
    sentence2 = (
        "Please execute the longitudinal analysis process on the specified topic "
        "and provide comprehensive data review for the final detailed report."
    )

    prompt = f"{sentence1} {sentence2}"

    try:
        result = optimizer.optimize(prompt, mode="basic")
    finally:
        optimizer.semantic_guard_enabled = original_guard
    output = result["optimized_output"]

    # Near-duplicates should be deduplicated, leaving only one
    phrase_count = output.lower().count("longitudinal analysis")
    assert (
        phrase_count == 1
    ), f"Expected 1 occurrence of near-duplicate, found {phrase_count}"
    assert "Content Deduplication" in result["techniques_applied"]


def test_fastpath_reduces_latency() -> None:
    """Test that the fast path is triggered for small prompts."""
    repeated_fragment = "repeat this instruction for analysis."
    prompt = "Please " + " ".join([repeated_fragment] * 25)

    fast_optimizer = PromptOptimizer()
    fast_optimizer.fastpath_token_threshold = 500

    # Run the fast optimizer
    fast_result = fast_optimizer.optimize(prompt, mode="basic")

    # Verify fast path was triggered
    assert fast_result["stats"]["fast_path"] is True

    slow_optimizer = PromptOptimizer()
    slow_optimizer.fastpath_token_threshold = 0

    # Run the slow optimizer
    slow_result = slow_optimizer.optimize(prompt, mode="basic")

    # Verify fast path was NOT triggered for the slow optimizer
    assert slow_result["stats"]["fast_path"] is False

    # Note: We don't assert on timing because it can be flaky
    # The important thing is that fast_path flag is correctly set


def test_coref_compression_skips_when_tokens_exceed_limit(monkeypatch, caplog) -> None:
    optimizer = PromptOptimizer()

    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text: str) -> List[int]:
            self.calls += 1
            tokens = text.strip().split()
            return list(range(len(tokens)))

    tokenizer = RecordingTokenizer()
    optimizer.tokenizer = tokenizer

    def _fail_get_coref_model():
        raise AssertionError("coref model should not be loaded for oversized prompts")

    monkeypatch.setattr(optimizer, "_get_coref_model", _fail_get_coref_model)
    long_prompt = "token " * 601

    with caplog.at_level(logging.WARNING):
        result = optimizer._compress_coreferences(long_prompt, preserved={})

    assert result == long_prompt
    assert tokenizer.calls == 1
    assert "Skipping coreference compression" in caplog.text


def test_coref_compression_skip_handles_missing_tokenizer(monkeypatch, caplog) -> None:
    optimizer = PromptOptimizer()
    optimizer.tokenizer = None

    def _fail_get_coref_model():
        raise AssertionError("coref model should be skipped when prompt too large")

    monkeypatch.setattr(optimizer, "_get_coref_model", _fail_get_coref_model)
    long_prompt = ("a " * 700).strip()

    with caplog.at_level(logging.WARNING):
        result = optimizer._compress_coreferences(long_prompt, preserved={})

    assert result == long_prompt
    assert "Skipping coreference compression" in caplog.text


def test_coref_compression_skips_for_arabic_input(monkeypatch, caplog) -> None:
    optimizer = PromptOptimizer()

    def _fail_get_coref_model():
        raise AssertionError("coref model should be skipped for Arabic input")

    monkeypatch.setattr(optimizer, "_get_coref_model", _fail_get_coref_model)
    mixed_prompt = "Please summarize this paragraph. هذا اختبار بسيط."

    with caplog.at_level(logging.WARNING):
        result = optimizer._compress_coreferences(mixed_prompt, preserved={})

    assert result == mixed_prompt
    assert "Arabic detected in input" in caplog.text


def test_count_tokens_fails_when_tokenizer_missing() -> None:
    """Strict mode must fail when tokenizer support is unavailable."""
    optimizer = PromptOptimizer()
    optimizer.tokenizer = None
    text = "a " * 600  # 1200 chars

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.count_tokens(text)
    assert "Exact token counting is unavailable" in str(exc_info.value)


def test_frequency_learning_still_runs_sentence_deduplication() -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    original_guard = optimizer.semantic_guard_enabled
    optimizer.semantic_guard_enabled = False
    repeated_sentence = (
        "Document the enterprise release management plan details for the program. "
    )
    prompt = repeated_sentence * 60

    try:
        optimized = optimizer.optimize(prompt, mode="basic")
    finally:
        optimizer.semantic_guard_enabled = original_guard
    output = optimized["optimized_output"]

    sentences = [
        sentence for sentence in re.split(r"(?<=[.!?])\s+", output.strip()) if sentence
    ]

    assert len(sentences) == 2
    assert "Adaptive Abbreviation Learning" in optimized["techniques_applied"]
    assert "Content Deduplication" in optimized["techniques_applied"]


def test_verbatim_paragraph_duplication_is_removed() -> None:
    """Test that complete duplicate paragraphs are removed early in the pipeline."""
    optimizer = PromptOptimizer()

    # Paragraph must be >= 100 chars to trigger verbatim block deduplication
    paragraph = (
        "Execute longitudinal analysis on [TOPIC]. First, establish baseline parameters: "
        "define the standard refresh interval for this domain based on market dynamics. "
        "For example, AI refresh cycle may be two weeks, clothing may be 3 months, "
        "construction may be 2 years. Calculate n=3 data points spanning 2 full cycles."
    )

    # Verify paragraph is long enough
    assert len(paragraph) >= 100, f"Test paragraph too short: {len(paragraph)} chars"

    # Simulate a prompt with the exact same paragraph duplicated
    prompt = f"{paragraph}\n\n{paragraph}"

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    # The duplicate paragraph should be removed
    # Count occurrences of a unique phrase from the paragraph
    phrase_count = output.lower().count("longitudinal analysis")
    assert phrase_count == 1, f"Expected 1 occurrence, found {phrase_count}"
    assert "Verbatim Block Deduplication" in optimized["techniques_applied"]


def test_inline_duplicate_block_is_removed() -> None:
    """Test that inline duplicate blocks (same line, no paragraph breaks) are removed."""
    optimizer = PromptOptimizer()

    # Simulate the user's exact problem: same content repeated on same line
    block = (
        "Execute longitudinal analysis on [TOPIC]. First, establish baseline parameters: "
        "define the standard refresh interval for this domain based on market dynamics "
        "(enterprise adoption cycles, regulatory changes, technology maturity curves). "
        "For example, AI refresh cycle may be two weeks, clothing may be 3 months, "
        "construction may be 2 years. Calculate n=3 data points spanning 2 full cycles."
    )

    prompt = f"{block} {block}"

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    # The duplicate block should be removed - only one "longitudinal analysis"
    phrase_count = output.lower().count("longitudinal analysis")
    assert phrase_count == 1, f"Expected 1 occurrence, found {phrase_count}"


def test_multi_sentence_sequence_duplicates_removed() -> None:
    """Test that sequences of 2+ consecutive duplicate sentences are removed."""
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0

    # Create a prompt with a sequence of 2 sentences repeated
    sentence1 = "First, analyze the data thoroughly."
    sentence2 = "Second, create a comprehensive report."
    unique = "Finally, present your findings."

    prompt = f"{sentence1} {sentence2} {unique} {sentence1} {sentence2}"

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    # The duplicate sequence should be removed
    first_count = output.lower().count("first")
    second_count = output.lower().count("second")
    assert first_count == 1, f"Expected 1 'first', found {first_count}"
    assert second_count == 1, f"Expected 1 'second', found {second_count}"


def test_sentence_deduplication_preserves_semantic_guarded_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_a, **_k: 0.99,
    )

    prompt = (
        "You are an expert assistant.\n"
        "Please answer clearly and concisely.\n"
        "Please answer clearly and concisely.\n"
        "Focus on actionable steps.\n"
        "Focus on actionable steps.\n"
        "Do not hallucinate.\n"
    )

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = result["optimized_output"]

    lowered_output = output.lower()
    assert lowered_output.count("answer clearly and concisely") == 1
    assert lowered_output.count("focus on actionable steps") == 1
    assert result["stats"]["semantic_similarity_source"] in {
        "deduplication_collapse",
        "exact_repetition_collapse",
        "model",
    }
    assert "Semantic Guard Rollback" not in result["techniques_applied"]


def test_verbatim_deduplication_preserves_unique_content() -> None:
    """Test that verbatim deduplication doesn't remove unique paragraphs."""
    optimizer = PromptOptimizer()

    paragraph1 = (
        "The first section describes the initial requirements. "
        "These are essential for project success."
    )
    paragraph2 = (
        "The second section covers implementation details. "
        "Follow these steps carefully."
    )
    paragraph3 = (
        "The third section explains testing procedures. "
        "Quality assurance is critical."
    )

    prompt = f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    # All unique paragraphs should be preserved (words may be shortened)
    # Check for key concepts that should remain
    assert "first" in output.lower() or "1." in output
    assert "second" in output.lower() or "2." in output or "impl" in output.lower()
    assert "third" in output.lower() or "3." in output or "quality" in output.lower()


def test_repeated_constraint_prefix_factoring() -> None:
    optimizer = PromptOptimizer()
    prompt = "Constraint: keep output short.\nConstraint: avoid filler.\nConstraint: be direct.\n"
    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="conservative")
    output = optimized["optimized_output"]

    assert "Constraint:" in output
    assert "keep output short" in output
    assert ";" in output


def test_repeated_must_prefix_factoring() -> None:
    optimizer = PromptOptimizer()
    prompt = "Must: keep schema stable.\nMust: avoid breaking changes.\nMust: document updates.\n"
    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="conservative")
    output = optimized["optimized_output"]

    assert "Must:" in output or "must:" in output
    assert "schema stable" in output
    assert ";" in output


def test_sliding_window_catches_partial_duplicates() -> None:
    """Test that the sliding window approach catches duplicates that aren't full paragraphs."""
    optimizer = PromptOptimizer()

    # Create a prompt with a significant repeated phrase (>20 tokens)
    repeated_phrase = (
        "The system shall process user requests in real-time with sub-second latency "
        "while maintaining full data consistency across all distributed nodes"
    )
    unique_content = "Introduction to the system requirements document."

    prompt = f"{unique_content} {repeated_phrase} Some middle content here. {repeated_phrase}"

    optimized = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = optimized["optimized_output"]

    # The repeated phrase should only appear once
    phrase_count = output.lower().count("sub-second latency")
    assert (
        phrase_count == 1
    ), f"Expected 1 occurrence of repeated phrase, found {phrase_count}"


def test_remove_verbatim_duplicate_blocks_method_directly() -> None:
    """Test the _remove_verbatim_duplicate_blocks method directly."""
    optimizer = PromptOptimizer()

    # Paragraph must be >= 100 chars to trigger deduplication
    paragraph = (
        "This is a test paragraph with enough content to meet the minimum length threshold. "
        "We need at least one hundred characters to trigger the verbatim block deduplication logic."
    )
    assert len(paragraph) >= 100, f"Test paragraph too short: {len(paragraph)} chars"
    text = f"{paragraph}\n\n{paragraph}"

    result, removed = optimizer._remove_verbatim_duplicate_blocks(text, preserved=None)

    assert removed is True
    # Result should have roughly half the length (minus separator)
    assert len(result) < len(text) * 0.7


def test_remove_multi_sentence_sequence_duplicates_method_directly() -> None:
    """Test the _remove_multi_sentence_sequence_duplicates method directly."""
    optimizer = PromptOptimizer()

    sentences = [
        "First sentence here.",
        "Second sentence follows.",
        "Third unique sentence.",
        "First sentence here.",
        "Second sentence follows.",
    ]
    directives = [False, False, False, False, False]

    (
        result_sentences,
        result_directives,
    ) = optimizer._remove_multi_sentence_sequence_duplicates(
        sentences, directives, sequence_length=2
    )

    # Should remove the duplicate sequence (indices 3 and 4)
    assert len(result_sentences) == 3
    assert "Third unique" in result_sentences[2]


def test_legend_consolidation_merges_entries_and_saves_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_a, **_k: 0.99,
    )
    url = "https://example.com/resources/long/path/to/architecture/overview"
    prompt = (
        "Title of the Section: Measure latency\n"
        "Description of the Section: Collect metrics\n"
        "Title of the Section: Reduce latency\n"
        "Description of the Section: Report findings\n"
        "Acme Corporation filed a report. Acme Corporation released updates. "
        "Acme Corporation announced earnings.\n"
        f"See {url} for details. Use {url} when referencing docs.\n"
        "Configure the dashboard (see section 4 for architecture details) "
        "and share it (see section 4 for architecture details).\n"
    )

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = result["optimized_output"]

    assert optimizer.count_tokens(output) <= optimizer.count_tokens(prompt)
    if "Legend:" in output:
        assert "Refs:" in output
        assert "Glossary:" in output
        assert "Labels: -" not in output
        assert "Aliases: -" not in output


def test_semantic_threshold_override_applies_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0

    prompt = (
        "Provide a detailed summary of the onboarding process, including timelines, "
        "owners, and required documentation."
    )
    optimized_text = (
        "Provide a concise summary of onboarding, timelines, owners, and documentation."
    )

    def _fake_pipeline(*_args, **_kwargs) -> str:
        return optimized_text

    monkeypatch.setattr(PromptOptimizer, "_optimize_pipeline", _fake_pipeline)
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_args, **_kwargs: 0.9,
    )
    monkeypatch.setattr(
        PromptOptimizer,
        "_lexical_similarity",
        lambda *_args, **_kwargs: 0.9,
    )

    result_override = optimizer.optimize(prompt, mode="basic", semantic_threshold=0.95)
    assert result_override["optimized_output"] == prompt
    assert result_override["techniques_applied"] == ["Semantic Guard Rollback"]

    result_default = optimizer.optimize(prompt, mode="basic")
    assert result_default["optimized_output"] == optimized_text
    assert "Semantic Guard Rollback" not in result_default["techniques_applied"]


def test_optimizer_collapses_repeated_huggingface_model_ids() -> None:
    optimizer = PromptOptimizer()
    optimizer.semantic_guard_enabled = False
    optimizer.fastpath_token_threshold = 1000

    prompt = "sentence-transformers/all-MiniLM-L6-v2" * 6
    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    assert result["optimized_output"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert result["stats"]["compression_percentage"] > 80


def test_extract_constraint_fingerprint_captures_key_signal_types() -> None:
    optimizer = PromptOptimizer()
    fingerprint = optimizer._extract_constraint_fingerprint(
        'You must include exactly 3 bullet points. Do not mention "beta". '
        "You should keep each bullet under 20 words and never use passive voice."
    )

    assert fingerprint["must_directives"]
    assert fingerprint["should_directives"]
    assert fingerprint["do_not_rules"]
    assert fingerprint["negations"]
    assert fingerprint["quoted_literals"] == ["beta"]
    assert any(
        "3" in constraint.get("numbers", [])
        for constraint in fingerprint["numeric_constraints"]
    )


def test_select_semantic_candidate_rejects_mandatory_constraint_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    monkeypatch.setattr(
        "services.optimizer.core._metrics.score_similarity",
        lambda *_a, **_k: 0.2,
    )
    state = optimizer._get_state()
    state.constraint_fingerprint = optimizer._extract_constraint_fingerprint(
        'You must include exactly 3 bullets and do not mention "beta".'
    )

    chosen, _meta = optimizer._select_semantic_candidate(
        "You must include exactly 3 bullets and do not mention beta.",
        [
            (
                "original",
                "You must include exactly 3 bullets and do not mention beta.",
                {},
            ),
            (
                "aggressive",
                "Mention beta and include 2 bullets.",
                {},
            ),
        ],
        pass_name="compress_examples",
        guard_threshold=0.0,
    )

    assert chosen == "You must include exactly 3 bullets and do not mention beta."


def test_verify_constraint_fingerprint_accepts_equivalent_constraint_candidate() -> None:
    optimizer = PromptOptimizer()
    fingerprint = optimizer._extract_constraint_fingerprint(
        "You must provide exactly 2 examples for the final answer and do not use emojis."
    )

    is_valid, failures = optimizer._verify_constraint_fingerprint(
        fingerprint,
        "Must provide exactly 2 examples for the final answer. Do not use emojis.",
    )

    assert is_valid
    assert not any(failure.get("severity") == "mandatory" for failure in failures)


def test_verify_constraint_fingerprint_rejects_negation_polarity_flip() -> None:
    optimizer = PromptOptimizer()
    fingerprint = optimizer._extract_constraint_fingerprint(
        "Do not expose API keys in the response."
    )

    is_valid, failures = optimizer._verify_constraint_fingerprint(
        fingerprint,
        "Expose API keys in the response.",
    )

    assert not is_valid
    assert any(
        failure.get("category") in {"negation", "do_not_rule"}
        and failure.get("severity") == "mandatory"
        for failure in failures
    )


def test_heavy_pass_precheck_skips_without_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.semantic_guard_enabled = False
    optimizer.semantic_guard_per_pass_enabled = False

    calls = {"compress_examples": 0, "summarize_history": 0}

    monkeypatch.setattr(
        optimizer,
        "_compress_examples",
        lambda text, *_a, **_k: calls.__setitem__("compress_examples", calls["compress_examples"] + 1) or text,
    )
    monkeypatch.setattr(
        "services.optimizer.core._history.summarize_history",
        lambda *_a, **_k: calls.__setitem__("summarize_history", calls["summarize_history"] + 1) or _a[1],
    )
    monkeypatch.setattr(
        optimizer,
        "_optimize_with_token_classifier",
        lambda text, **_k: (text, False, {}),
    )

    prompt = " ".join(["general context line"] * 1200)
    optimizer.optimize(prompt, optimization_mode="maximum")

    assert calls["compress_examples"] == 0
    assert calls["summarize_history"] == 0


def test_paragraph_semantic_dedup_prefers_constraint_overlap() -> None:
    optimizer = PromptOptimizer()
    paragraph_a = (
        "Policy paragraph version A. must keep SOC2 controls and auditing enabled. "
        "Additional explanation text repeated for policy guidance."
    )
    paragraph_b = (
        "Policy paragraph version B. keep SOC2 controls and auditing enabled. "
        "Additional explanation text repeated for policy guidance."
    )
    text = "\n\n".join([paragraph_a, paragraph_b, "Independent unique paragraph."])

    deduped, applied, removed = optimizer._apply_paragraph_semantic_dedup(
        text,
        query_hint="SOC2 auditing",
        preserved={
            "code_blocks": [],
            "quotes": [],
            "numbers": [],
            "urls": [],
            "citations": [],
            "protected": [],
            "forced": [],
            "json_tokens": [],
            "json_literals": [],
            "json_strings": [],
            "toon_blocks": [],
        },
    )

    assert applied is True
    assert removed >= 1
    assert "SOC2" in deduped


def test_paragraph_semantic_dedup_uses_query_hint_from_optimize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PromptOptimizer()
    optimizer.fastpath_token_threshold = 0
    optimizer.semantic_guard_enabled = False
    optimizer.semantic_guard_per_pass_enabled = False

    captured = {"query_hint": None}

    def _fake_paragraph_dedup(self, text: str, *, query_hint, preserved):
        captured["query_hint"] = query_hint
        return text, False, 0

    monkeypatch.setattr(
        PromptOptimizer,
        "_apply_paragraph_semantic_dedup",
        _fake_paragraph_dedup,
    )

    monkeypatch.setattr(optimizer, "count_tokens", lambda _text: 3001)
    monkeypatch.setattr(optimizer, "_tokens_for_stage_decision", lambda _text: 3001)

    long_text = "\n\n".join(["Long prose paragraph with compliance policy language. " * 120] * 8)
    optimizer._optimize_pipeline(
        long_text,
        "basic",
        "maximum",
        query_hint="SOC2 controls",
    )

    assert captured["query_hint"] == "SOC2 controls"
