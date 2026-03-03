import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import pytest
from services.optimizer.core import PromptOptimizer

_DATASET_DIR = Path(__file__).resolve().parents[2] / "tests" / "benchmarks" / "datasets"


@pytest.fixture(autouse=True)
def _stabilize_strict_maximum_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.optimizer.entropy as optimizer_entropy
    import services.optimizer.metrics as optimizer_metrics
    import services.optimizer.token_classifier as token_classifier

    original_score_similarity = optimizer_metrics.score_similarity

    class _NoOpEntropyScorer:
        @property
        def available(self) -> bool:
            return True

        def score_tokens(self, _text: str, skip_ranges=None):
            _ = skip_ranges
            return []

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
    monkeypatch.setattr(optimizer_entropy, "_get_scorer", lambda: _NoOpEntropyScorer())
    monkeypatch.setattr(
        optimizer_entropy, "_get_fast_scorer", lambda: _NoOpEntropyScorer()
    )
    monkeypatch.setattr(
        token_classifier, "compress_with_token_classifier", _no_op_token_classifier
    )
    monkeypatch.setattr(
        token_classifier, "evaluate_shadow_classifier", lambda *_a, **_k: {}
    )


def _load_dataset_samples(limit_per_file: int = 2) -> List[str]:
    samples: List[str] = []
    for dataset_file in sorted(_DATASET_DIR.glob("*.json")):
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
        for item in payload.get("samples", [])[:limit_per_file]:
            text = (item.get("text") or "").strip()
            if text:
                samples.append(text)
    return samples


def _lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _run_batch(optimizer: PromptOptimizer, prompts: List[str]) -> Dict[str, float]:
    reductions: List[float] = []
    latencies_ms: List[float] = []
    fidelities: List[float] = []

    for prompt in prompts:
        start = time.perf_counter()
        result = optimizer.optimize(prompt, optimization_mode="maximum")
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        stats = result["stats"]
        original_tokens = max(stats.get("original_tokens", 0), 1)
        optimized_tokens = max(stats.get("optimized_tokens", 0), 0)
        reductions.append((original_tokens - optimized_tokens) / original_tokens)

        fidelities.append(_lexical_overlap(prompt, result["optimized_output"]))

    return {
        "avg_token_reduction": mean(reductions) if reductions else 0.0,
        "avg_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
        "avg_semantic_fidelity": mean(fidelities) if fidelities else 0.0,
    }


def test_maximum_prepass_rollout_benchmark_compare(monkeypatch) -> None:
    baseline_env = {
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED": "0",
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS": "350",
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO": "0.70",
    }
    candidate_env = {
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED": "1",
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS": "350",
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO": "0.70",
    }
    for key, value in baseline_env.items():
        monkeypatch.setenv(key, value)

    seed_samples = _load_dataset_samples(limit_per_file=2)
    assert seed_samples

    # Expand samples so they cross large-prompt threshold in this test environment.
    prompts = ["\n".join([sample] * 25) for sample in seed_samples]

    baseline_optimizer = PromptOptimizer()
    baseline = _run_batch(baseline_optimizer, prompts)

    for key, value in candidate_env.items():
        monkeypatch.setenv(key, value)
    candidate_optimizer = PromptOptimizer()
    candidate = _run_batch(candidate_optimizer, prompts)

    # Rollout guard: candidate should maintain strong fidelity and not regress hard.
    assert candidate["avg_semantic_fidelity"] >= 0.30
    assert (
        candidate["avg_semantic_fidelity"] >= baseline["avg_semantic_fidelity"] - 0.2
    )

    # Comparison is computed for token reduction and latency across benchmark datasets.
    assert candidate["avg_token_reduction"] >= baseline["avg_token_reduction"] - 0.03
    assert candidate["avg_latency_ms"] <= baseline["avg_latency_ms"] * 1.25
