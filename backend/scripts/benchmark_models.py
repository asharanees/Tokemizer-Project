import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from services.model_cache_manager import resolve_cached_model_path
from services.optimizer import metrics as _metrics
from services.optimizer import section_ranking as _section_ranking

logger = logging.getLogger(__name__)


DEFAULT_SEMANTIC_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

BASELINE_SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OUTPUT_PATH = (
    APP_ROOT / "scripts" / "benchmark_outputs" / "embedding_model_benchmark.json"
)


@dataclass(frozen=True)
class _GuardPair:
    left: str
    right: str
    accepted: bool


@dataclass(frozen=True)
class _RankingSample:
    prompt: str
    sections: List[str]
    relevant_indices: List[int]


@dataclass(frozen=True)
class _CompressionSample:
    query: str
    sections: List[str]
    retention_tokens: List[str]


DEFAULT_ENTROPY_MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "distilgpt2",
]

DEFAULT_ENTROPY_FAST_MODELS = [
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
]


def _resolve_cached_models(model_names: Sequence[str], model_type: str) -> List[str]:
    available: List[str] = []
    for name in model_names:
        if resolve_cached_model_path(model_type, name):
            available.append(name)
        else:
            logger.warning("Model not cached: %s (%s)", name, model_type)
    return available


def _resolve_cached_semantic_models(model_names: Sequence[str]) -> List[str]:
    """Return semantic models cached for both guard and ranking workloads."""
    available: List[str] = []
    for name in model_names:
        guard_cached = bool(resolve_cached_model_path("semantic_guard", name))
        rank_cached = bool(resolve_cached_model_path("semantic_rank", name))
        if guard_cached and rank_cached:
            available.append(name)
            continue
        missing = []
        if not guard_cached:
            missing.append("semantic_guard")
        if not rank_cached:
            missing.append("semantic_rank")
        logger.warning(
            "Model not fully cached for semantic benchmark: %s (missing=%s)",
            name,
            ",".join(missing),
        )
    return available


def _semantic_guard_dataset() -> Dict[str, List[_GuardPair]]:
    common = [
        _GuardPair(
            "Summarize the project status update.",
            "Summarize the status update.",
            True,
        ),
        _GuardPair(
            "Please generate a concise checklist for deployment readiness.",
            "Generate a concise deployment readiness checklist.",
            True,
        ),
        _GuardPair(
            "Explain customer onboarding prerequisites and documents.",
            "List documents required for customer onboarding.",
            True,
        ),
    ]
    return {
        "default": common
        + [
            _GuardPair(
                "As a reminder, ensure security reviews are complete before launch.",
                "Ensure security reviews are complete before launch.",
                True,
            ),
            _GuardPair(
                "Use the ACME Analytics Platform for reporting.",
                "Use ACME Analytics Platform for reporting.",
                True,
            ),
            _GuardPair(
                "Rotate database credentials before production rollout.",
                "What is the weather this weekend?",
                False,
            ),
        ],
        "code": common
        + [
            _GuardPair(
                "Refactor this Python function to avoid mutable defaults.",
                "How do mutable default arguments break Python functions?",
                True,
            ),
            _GuardPair(
                "Convert this SQL query to use a CTE.",
                "Summarize yesterday's support tickets.",
                False,
            ),
        ],
        "dialogue": common
        + [
            _GuardPair(
                "Rewrite the assistant response with empathetic tone.",
                "Make this support response sound more empathetic.",
                True,
            ),
            _GuardPair(
                "Escalate billing complaint with an apology.",
                "Generate Kubernetes deployment YAML.",
                False,
            ),
        ],
    }


def _section_ranking_dataset() -> Dict[str, List[_RankingSample]]:
    return {
        "default": [
            _RankingSample(
                prompt="Prioritize sections relevant to customer onboarding and SLA updates.",
                sections=[
                    "Release notes: new dashboard widgets and UI tweaks.",
                    "Customer onboarding checklist with required documents and approvals.",
                    "SLA update proposal and escalation procedures.",
                    "Backend refactor notes and internal API cleanup.",
                    "Support coverage schedule for weekend rotations.",
                ],
                relevant_indices=[1, 2],
            )
        ],
        "code": [
            _RankingSample(
                prompt="Find sections covering API authentication and error handling in Python services.",
                sections=[
                    "Frontend color palette and typography updates.",
                    "Python API authentication middleware with token validation.",
                    "Error handling patterns and retry backoff for service calls.",
                    "Recruiting pipeline notes and interview schedule.",
                ],
                relevant_indices=[1, 2],
            )
        ],
        "dialogue": [
            _RankingSample(
                prompt="Rank sections about response tone and de-escalation for support chats.",
                sections=[
                    "Database index maintenance strategy for high-cardinality fields.",
                    "Support chat playbook: warm tone, acknowledgement, de-escalation.",
                    "Style guide for short assistant responses and follow-up questions.",
                    "Quarterly budget variance spreadsheet notes.",
                ],
                relevant_indices=[1, 2],
            )
        ],
    }


def _compression_dataset() -> Dict[str, List[_CompressionSample]]:
    return {
        "default": [
            _CompressionSample(
                query="Keep onboarding SLA commitments and escalation contacts.",
                sections=[
                    "SLA commitments include 99.9% uptime and 4-hour critical response.",
                    "Escalation contacts: support-leads@acme.example and pager channel.",
                    "Office lunch menu and parking updates.",
                ],
                retention_tokens=["99.9% uptime", "4-hour", "support-leads", "pager"],
            )
        ],
        "code": [
            _CompressionSample(
                query="Preserve authentication flow details for Python API requests.",
                sections=[
                    "Use bearer token auth with HMAC request signatures and nonce checks.",
                    "Retry policy includes exponential backoff with jitter.",
                    "Team offsite schedule and travel notes.",
                ],
                retention_tokens=["bearer token", "HMAC", "nonce"],
            )
        ],
        "dialogue": [
            _CompressionSample(
                query="Retain empathetic acknowledgement and clear next-step messaging.",
                sections=[
                    "Acknowledge frustration and apologize before proposing a next step.",
                    "Offer timeline transparency and a follow-up check-in message.",
                    "Kernel tuning notes for compute nodes.",
                ],
                retention_tokens=["apologize", "next step", "follow-up"],
            )
        ],
    }


def _cosine_pairs_from_embeddings(embeddings: Any, count: int) -> List[float]:
    return [
        float((embeddings[idx] * embeddings[idx + 1]).sum())
        for idx in range(0, count * 2, 2)
    ]


def _benchmark_semantic_models(
    models: Sequence[str],
    output_path: Path,
    baseline: str,
) -> Dict[str, Any]:
    guard_dataset = _semantic_guard_dataset()
    ranking_dataset = _section_ranking_dataset()
    compression_dataset = _compression_dataset()
    profile_names = sorted(
        set(guard_dataset) | set(ranking_dataset) | set(compression_dataset)
    )
    per_model: Dict[str, Dict[str, Any]] = {}

    logger.info("Semantic model benchmark (baseline=%s)", baseline)
    for model in models:
        guard_correct = 0
        guard_total = 0
        recall_scores: Dict[int, List[float]] = {1: [], 2: [], 3: []}
        compression_scores: List[float] = []
        per_profile: Dict[str, Dict[str, float]] = {}
        total_tokens = 0

        semantic_start = time.perf_counter()
        for profile in profile_names:
            profile_guard = guard_dataset.get(profile, [])
            profile_guard_correct = 0
            profile_guard_total = 0
            profile_recall_scores: Dict[int, List[float]] = {1: [], 2: [], 3: []}
            profile_compression_scores: List[float] = []

            if profile_guard:
                texts = [
                    text for pair in profile_guard for text in (pair.left, pair.right)
                ]
                embeddings = _metrics.encode_texts(texts, model)
                total_tokens += sum(max(len(text.split()), 1) for text in texts)
                if embeddings is not None:
                    scores = _cosine_pairs_from_embeddings(
                        embeddings, len(profile_guard)
                    )
                    for pair, score in zip(profile_guard, scores):
                        predicted = score >= 0.75
                        profile_guard_total += 1
                        guard_total += 1
                        if predicted == pair.accepted:
                            profile_guard_correct += 1
                            guard_correct += 1

            for sample in ranking_dataset.get(profile, []):
                specs = [type("Spec", (), {"text": text})() for text in sample.sections]
                section_scores, _ = _section_ranking._score_sections_semantic(
                    specs,
                    sample.prompt,
                    model,
                )
                total_tokens += sum(
                    max(len(text.split()), 1)
                    for text in [sample.prompt, *sample.sections]
                )
                if section_scores is None:
                    continue
                ranked = [
                    idx
                    for idx, _ in sorted(
                        section_scores.items(), key=lambda item: item[1], reverse=True
                    )
                ]
                relevant = set(sample.relevant_indices)
                for k in (1, 2, 3):
                    top_k = set(ranked[:k])
                    recall = len(top_k.intersection(relevant)) / max(len(relevant), 1)
                    profile_recall_scores[k].append(recall)
                    recall_scores[k].append(recall)

            for sample in compression_dataset.get(profile, []):
                specs = [type("Spec", (), {"text": text})() for text in sample.sections]
                section_scores, _ = _section_ranking._score_sections_semantic(
                    specs,
                    sample.query,
                    model,
                )
                total_tokens += sum(
                    max(len(text.split()), 1)
                    for text in [sample.query, *sample.sections]
                )
                if section_scores is None:
                    continue
                ranked = [
                    idx
                    for idx, _ in sorted(
                        section_scores.items(), key=lambda item: item[1], reverse=True
                    )
                ]
                retained_text = "\n".join(
                    sample.sections[idx] for idx in ranked[:2]
                ).lower()
                retention = sum(
                    1
                    for token in sample.retention_tokens
                    if token.lower() in retained_text
                ) / max(len(sample.retention_tokens), 1)
                profile_compression_scores.append(retention)
                compression_scores.append(retention)

            per_profile[profile] = {
                "semantic_guard_acceptance_fidelity": profile_guard_correct
                / max(profile_guard_total, 1),
                "section_ranking_recall_at_2": sum(profile_recall_scores[2])
                / max(len(profile_recall_scores[2]), 1),
                "query_aware_compression_retention_quality": sum(
                    profile_compression_scores
                )
                / max(len(profile_compression_scores), 1),
            }

        elapsed_ms = (time.perf_counter() - semantic_start) * 1000.0
        latency_ms_per_1k = elapsed_ms / max(total_tokens / 1000.0, 1e-6)
        per_model[model] = {
            "semantic_guard_acceptance_fidelity": guard_correct / max(guard_total, 1),
            "section_ranking_recall_at_k": {
                str(k): sum(values) / max(len(values), 1)
                for k, values in recall_scores.items()
            },
            "query_aware_compression_retention_quality": (
                sum(compression_scores) / max(len(compression_scores), 1)
            ),
            "latency_ms_per_1k_tokens": latency_ms_per_1k,
            "profiles": per_profile,
        }
        logger.info(
            "Model=%s guard_fidelity=%.3f recall@2=%.3f retention=%.3f latency_ms_per_1k=%.2f",
            model,
            per_model[model]["semantic_guard_acceptance_fidelity"],
            per_model[model]["section_ranking_recall_at_k"]["2"],
            per_model[model]["query_aware_compression_retention_quality"],
            per_model[model]["latency_ms_per_1k_tokens"],
        )

    baseline_metrics = per_model.get(baseline)
    if not baseline_metrics:
        logger.warning("Baseline metrics missing; recommendations not generated.")
        return {"baseline": baseline, "models": per_model}

    def dominates(candidate: Dict[str, Any], base_metrics: Dict[str, Any]) -> bool:
        quality_ok = (
            candidate["semantic_guard_acceptance_fidelity"]
            >= base_metrics["semantic_guard_acceptance_fidelity"]
            and candidate["section_ranking_recall_at_k"]["2"]
            >= base_metrics["section_ranking_recall_at_k"]["2"]
            and candidate["query_aware_compression_retention_quality"]
            >= base_metrics["query_aware_compression_retention_quality"]
        )
        latency_ok = (
            candidate["latency_ms_per_1k_tokens"]
            <= base_metrics["latency_ms_per_1k_tokens"]
        )
        return quality_ok and latency_ok

    default_recommendation = baseline
    for model, metrics in per_model.items():
        if model == baseline:
            continue
        if dominates(metrics, baseline_metrics):
            default_recommendation = model
            break

    profile_overrides: Dict[str, str] = {}
    for profile in profile_names:
        baseline_profile = baseline_metrics.get("profiles", {}).get(profile, {})
        for model, metrics in per_model.items():
            if model == baseline:
                continue
            candidate_profile = metrics.get("profiles", {}).get(profile, {})
            if not candidate_profile or not baseline_profile:
                continue
            quality_ok = (
                candidate_profile.get("semantic_guard_acceptance_fidelity", 0.0)
                >= baseline_profile.get("semantic_guard_acceptance_fidelity", 0.0)
                and candidate_profile.get("section_ranking_recall_at_2", 0.0)
                >= baseline_profile.get("section_ranking_recall_at_2", 0.0)
                and candidate_profile.get(
                    "query_aware_compression_retention_quality", 0.0
                )
                >= baseline_profile.get(
                    "query_aware_compression_retention_quality", 0.0
                )
            )
            latency_ok = (
                metrics["latency_ms_per_1k_tokens"]
                <= baseline_metrics["latency_ms_per_1k_tokens"]
            )
            if quality_ok and latency_ok:
                profile_overrides[profile] = model
                break

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline,
        "models": per_model,
        "recommendations": {
            "default": default_recommendation,
            "profile_overrides": profile_overrides,
            "dominance_rule": "quality_non_decreasing_and_latency_not_worse",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Wrote semantic benchmark results to %s", output_path)
    return output


def _benchmark_entropy_teacher_models(models: Sequence[str]) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers/torch unavailable; skipping entropy teacher benchmarks"
        )
        return

    samples = [
        "Summarize the weekly metrics and highlight anomalies.",
        "Provide a concise rollout plan for the API deprecation notice.",
    ]

    for model in models:
        try:
            model_path = resolve_cached_model_path("entropy", model)
            if not model_path:
                logger.warning("Entropy teacher model not cached: %s", model)
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            lm = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            lm.eval()
        except Exception as exc:
            logger.warning("Failed to load entropy teacher model %s: %s", model, exc)
            continue

        start = time.perf_counter()
        with torch.no_grad():
            losses = []
            for sample in samples:
                encoded = tokenizer(sample, return_tensors="pt")
                outputs = lm(**encoded, labels=encoded["input_ids"])
                loss = float(outputs.loss)
                losses.append(loss)
        latency_ms = (time.perf_counter() - start) * 1000.0
        avg_loss = sum(losses) / max(len(losses), 1)
        logger.info(
            "Entropy teacher model=%s latency_ms=%.2f avg_loss=%.4f",
            model,
            latency_ms,
            avg_loss,
        )


def _benchmark_entropy_fast_models(models: Sequence[str]) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning(
            "onnxruntime/numpy/transformers unavailable; skipping fast entropy benchmarks"
        )
        return

    samples = [
        "Summarize the weekly metrics and highlight anomalies.",
        "Provide a concise rollout plan for the API deprecation notice.",
    ]

    for model in models:
        try:
            model_path = resolve_cached_model_path("entropy_fast", model)
            if not model_path:
                logger.warning("Fast entropy model not cached: %s", model)
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            int8_path = Path(model_path) / "model.int8.onnx"
            onnx_path = (
                int8_path if int8_path.exists() else (Path(model_path) / "model.onnx")
            )
            if not onnx_path.exists():
                logger.warning("Fast entropy ONNX artifact missing for %s", model)
                continue
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            input_names = [item.name for item in session.get_inputs()]
        except Exception as exc:
            logger.warning("Failed to load fast entropy model %s: %s", model, exc)
            continue

        start = time.perf_counter()
        drop_means = []
        for sample in samples:
            encoded = tokenizer(sample, return_tensors="np", truncation=True)
            inputs = {}
            for name in input_names:
                value = encoded.get(name)
                if value is None and name == "token_type_ids":
                    value = np.zeros_like(encoded["input_ids"])
                if value is not None:
                    inputs[name] = value.astype("int64")
            if not inputs:
                continue
            outputs = session.run(None, inputs)
            if not outputs:
                continue
            logits = np.asarray(outputs[0])
            if logits.ndim == 3 and logits.shape[-1] >= 2:
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                probs = np.exp(logits)
                probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-8)
                drop_probs = probs[..., 1]
            else:
                drop_probs = 1.0 / (1.0 + np.exp(-logits))
            drop_means.append(float(np.mean(drop_probs)))

        latency_ms = (time.perf_counter() - start) * 1000.0
        avg_drop = sum(drop_means) / max(len(drop_means), 1)
        logger.info(
            "Entropy fast model=%s latency_ms=%.2f avg_drop_prob=%.4f",
            model,
            latency_ms,
            avg_drop,
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Benchmark semantic and entropy models."
    )
    parser.add_argument("--semantic-models", nargs="*", default=None)
    parser.add_argument("--entropy-models", nargs="*", default=None)
    parser.add_argument("--entropy-fast-models", nargs="*", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--baseline-semantic-model", default=BASELINE_SEMANTIC_MODEL)
    args = parser.parse_args()

    semantic_models = args.semantic_models or DEFAULT_SEMANTIC_MODELS
    entropy_models = args.entropy_models or DEFAULT_ENTROPY_MODELS
    entropy_fast_models = args.entropy_fast_models or DEFAULT_ENTROPY_FAST_MODELS

    semantic_models = _resolve_cached_semantic_models(semantic_models)
    baseline_semantic_model = args.baseline_semantic_model
    output_path = Path(args.output).resolve()
    if baseline_semantic_model not in semantic_models:
        logger.warning(
            "Skipping semantic benchmark: baseline model unavailable in both semantic caches: %s",
            baseline_semantic_model,
        )
    elif semantic_models:
        _benchmark_semantic_models(
            semantic_models,
            output_path=output_path,
            baseline=baseline_semantic_model,
        )
    else:
        logger.warning("No cached semantic models available for benchmarking.")

    entropy_models = _resolve_cached_models(entropy_models, "entropy")
    if entropy_models:
        _benchmark_entropy_teacher_models(entropy_models)
    else:
        logger.warning("No cached entropy teacher models available for benchmarking.")

    entropy_fast_models = _resolve_cached_models(entropy_fast_models, "entropy_fast")
    if entropy_fast_models:
        _benchmark_entropy_fast_models(entropy_fast_models)
    else:
        logger.warning("No cached fast entropy models available for benchmarking.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
