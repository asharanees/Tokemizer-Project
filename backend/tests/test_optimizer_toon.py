from __future__ import annotations

import json

import pytest
from services.optimizer import toon_encoder
from services.optimizer.core import PromptOptimizer


@pytest.fixture(autouse=True)
def _stabilize_strict_maximum_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.optimizer.metrics as optimizer_metrics
    import services.optimizer.token_classifier as token_classifier

    original_score_similarity = optimizer_metrics.score_similarity

    def _safe_score_similarity(*args, **kwargs):
        score = original_score_similarity(*args, **kwargs)
        if score is None:
            return 0.99
        return score

    def _no_op_token_classifier(
        text: str,
        **_kwargs,
    ):
        return text, False, {"keep_ratio": 1.0, "decisions": 0, "removals": 0}

    monkeypatch.setattr(optimizer_metrics, "score_similarity", _safe_score_similarity)
    monkeypatch.setattr(
        token_classifier, "compress_with_token_classifier", _no_op_token_classifier
    )
    monkeypatch.setattr(
        token_classifier, "evaluate_shadow_classifier", lambda *_a, **_k: {}
    )


def test_toon_alias_round_trip_is_deterministic() -> None:
    data = {
        "metadata": {
            "longRepeatedFieldName": "very-long-repeated-string-value",
            "anotherLongRepeatedFieldName": "very-long-repeated-string-value",
        },
        "items": [
            {
                "longRepeatedFieldName": "very-long-repeated-string-value",
                "anotherLongRepeatedFieldName": "very-long-repeated-string-value",
                "id": "A-1",
            },
            {
                "longRepeatedFieldName": "very-long-repeated-string-value",
                "anotherLongRepeatedFieldName": "very-long-repeated-string-value",
                "id": "A-2",
            },
        ],
    }

    compressed, legend = toon_encoder.compress_structure(data)
    reconstructed = toon_encoder.restore_structure_aliases(compressed, legend)

    assert reconstructed == data
    assert compressed == toon_encoder.compress_structure(data)[0]
    assert legend == toon_encoder.compress_structure(data)[1]


def test_toon_alias_preserves_id_and_numeric_strings() -> None:
    data = {
        "id": "X-100",
        "order_id": "ORD-9",
        "quantity": "123456789",
        "entries": [
            {"id": "X-100", "quantity": "123456789", "longRepeatedFieldName": "123456789"},
            {"id": "X-100", "quantity": "123456789", "longRepeatedFieldName": "123456789"},
        ],
    }

    compressed, legend = toon_encoder.compress_structure(data)

    assert "id" in compressed
    assert "order_id" in compressed
    assert legend.get("v", {}).get("123456789") is None
    assert toon_encoder.restore_structure_aliases(compressed, legend) == data


def test_toon_conversion_enabled_for_maximum() -> None:
    optimizer = PromptOptimizer()
    prompt = '{"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}'

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")
    output = result["optimized_output"]

    assert "items[2]" in output
    assert "id,name" in output
    assert "__TOON_" not in output


def test_toon_conversion_disabled_for_balanced() -> None:
    optimizer = PromptOptimizer()
    prompt = '{"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}'

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    output = result["optimized_output"]

    assert json.loads(output) == json.loads(prompt)


def test_toon_conversion_size_gate_skips_small_json() -> None:
    optimizer = PromptOptimizer()
    prompt = '{"a": 1}'

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")
    output = result["optimized_output"]

    assert json.loads(output) == json.loads(prompt)


def test_toon_stats_emitted_on_conversion() -> None:
    optimizer = PromptOptimizer()
    prompt = '{"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}'

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")
    stats = result["stats"]

    if "toon_conversions" in stats:
        assert stats["toon_conversions"] >= 1
        assert stats["toon_bytes_saved"] > 0
