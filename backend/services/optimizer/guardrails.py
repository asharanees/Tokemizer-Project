"""Guardrail helpers for semantic and latency thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass(frozen=True)
class GuardrailResult:
    """Serialized result for a single guardrail evaluation."""

    name: str
    passed: bool
    threshold: float
    value: float
    detail: str


def evaluate_latency_guard(latency_ms: float) -> GuardrailResult:
    """Check that cumulative optimization latency stays below the configured guard."""
    threshold = config.SEMANTIC_GUARD_LATENCY_GUARD_MS
    return GuardrailResult(
        name="max_latency_ms",
        passed=latency_ms <= threshold,
        threshold=threshold,
        value=latency_ms,
        detail="Cumulative per-request latency should remain under this bound.",
    )


def evaluate_similarity_guard(similarity: Optional[float]) -> GuardrailResult:
    """Check that the worst-case semantic similarity does not violate the guard."""
    threshold = config.SEMANTIC_GUARD_THRESHOLD
    if similarity is None:
        return GuardrailResult(
            name="semantic_similarity",
            passed=False,
            threshold=threshold,
            value=-1.0,
            detail="No semantic similarity samples available to evaluate this guard.",
        )
    return GuardrailResult(
        name="semantic_similarity",
        passed=similarity >= threshold,
        threshold=threshold,
        value=similarity,
        detail="Minimum observed similarity across the sampled runs.",
    )


def evaluate_token_savings_guard(tokens_saved: float) -> GuardrailResult:
    """Check that average token savings meets the configured baseline."""
    threshold = config.SEMANTIC_GUARD_TOKEN_SAVINGS_BASELINE
    return GuardrailResult(
        name="tokens_saved",
        passed=tokens_saved >= threshold,
        threshold=threshold,
        value=tokens_saved,
        detail="Average tokens saved per run should stay above this baseline.",
    )
