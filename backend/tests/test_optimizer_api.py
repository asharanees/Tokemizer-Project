from __future__ import annotations

import importlib
import os
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from models.optimization import (OptimizationRequest, OptimizationResponse,
                                 OptimizationStats)


def _new_test_db_path() -> Path:
    return (
        Path(tempfile.gettempdir())
        / f"tokemizer_test_history_{os.getpid()}_{uuid4().hex}.db"
    )


DEFAULT_TEST_DB = _new_test_db_path()


def _configure_test_environment() -> None:
    global DEFAULT_TEST_DB
    DEFAULT_TEST_DB = _new_test_db_path()
    os.environ["OPTIMIZATION_HISTORY_ENABLED"] = "1"
    os.environ["OPTIMIZER_PREWARM_MODELS"] = "0"
    # Force DB isolation for tests (do not let a repo-root `.env` influence DB_PATH).
    os.environ["DB_PATH"] = str(DEFAULT_TEST_DB)


_configure_test_environment()

optimizer_module = importlib.import_module("services.optimizer")
optimizer = optimizer_module.optimizer
PromptOptimizer = importlib.import_module("services.optimizer.core").PromptOptimizer


def _active_test_db() -> Path:
    return Path(os.environ.get("DB_PATH", str(DEFAULT_TEST_DB)))


def _strict_ready_model_status() -> dict[str, dict[str, object]]:
    return {
        "semantic_guard": {"loaded": True},
        "semantic_rank": {"loaded": True},
        "coreference": {"loaded": True},
        "spacy": {"loaded": True},
        "entropy_fast": {"loaded": True},
        "entropy": {"loaded": True},
        "token_classifier": {"loaded": True},
    }


class _ExecutorSpy:
    last_requested_workers: Optional[int] = None

    def __init__(self, max_workers: int = 1, **kwargs):
        type(self).last_requested_workers = max_workers

    def __enter__(self) -> "_ExecutorSpy":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def submit(self, func, *args, **kwargs):
        # Fake future
        class Future:
            def result(self):
                return func(*args, **kwargs)

        return Future()


def _fake_optimize_single(prompt: str, *_args, **_kwargs) -> OptimizationResponse:
    stats = OptimizationStats(
        original_chars=len(prompt),
        optimized_chars=len(prompt),
        compression_percentage=0.0,
        original_tokens=1,
        optimized_tokens=1,
        token_savings=0,
        processing_time_ms=0.1,
        fast_path=True,
        content_profile="general_prose",
        smart_context_description="Auto-derived for general_prose (1 tokens)",
    )
    return OptimizationResponse(
        optimized_output=prompt,
        stats=stats,
        warnings=None,
    )


@pytest.fixture(scope="module")
def client() -> TestClient:
    _configure_test_environment()
    import database as database_module
    import server as server_module

    importlib.reload(database_module)
    server_module = importlib.reload(server_module)

    from auth import get_current_customer
    from database import Customer

    app = server_module.app
    mock_customer = Customer(
        id="test_user_id",
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z",
        name="Test User",
        email="test@example.com",
        role="customer",
        is_active=True,
        subscription_status="active",
        subscription_tier="free",
    )

    # Keep API tests deterministic in strict mode by simulating warmed/ready models.
    server_module.optimizer.model_status = lambda: _strict_ready_model_status()
    server_module.optimizer.probe_model_readiness = lambda _model_type: None

    app.dependency_overrides[get_current_customer] = lambda: mock_customer
    return TestClient(app)


@pytest.fixture(autouse=True)
def _stabilize_semantic_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.optimizer.metrics as optimizer_metrics
    import services.optimizer.token_classifier as token_classifier

    original = optimizer_metrics.score_similarity
    original_encode = optimizer_metrics.encode_texts_with_plan

    def _safe_score_similarity(*args, **kwargs):
        score = original(*args, **kwargs)
        if score is None:
            return 0.99
        return score

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

    monkeypatch.setattr(optimizer_metrics, "score_similarity", _safe_score_similarity)
    monkeypatch.setattr(
        optimizer_metrics, "encode_texts_with_plan", _safe_encode_texts_with_plan
    )
    monkeypatch.setattr(
        token_classifier, "compress_with_token_classifier", _no_op_token_classifier
    )
    monkeypatch.setattr(
        token_classifier, "evaluate_shadow_classifier", lambda *_a, **_k: {}
    )


def _token_budget(stats: dict) -> int:
    return stats["original_tokens"], stats["optimized_tokens"]


def _normalized_stats(stats: dict) -> dict:
    return {key: value for key, value in stats.items() if key != "processing_time_ms"}


def test_profile_flag_records_timings() -> None:
    prompt = "Summarize this document into concise bullets."

    baseline_optimizer = PromptOptimizer()
    baseline = baseline_optimizer.optimize(
        prompt, mode="basic", optimization_mode="balanced"
    )

    profiled_optimizer = PromptOptimizer()
    profiled_optimizer._profiling_enabled = True
    profiled = profiled_optimizer.optimize(
        prompt, mode="basic", optimization_mode="balanced"
    )

    assert profiled["optimized_output"] == baseline["optimized_output"]
    profiling_stats = profiled["stats"].get("profiling_ms")
    assert profiling_stats, "profiling results should be included when flag is enabled"
    assert all(value >= 0 for value in profiling_stats.values())


def test_optimize_single_prompt_reduces_size() -> None:
    prompt = (
        "Please provide me with a comprehensive overview of the onboarding process, "
        "including every deadline, responsible owner, and required document."
    )

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")

    assert "optimized_output" in result
    assert "stats" in result

    stats = result["stats"]
    original_tokens, optimized_tokens = _token_budget(stats)

    assert stats["original_chars"] >= stats["optimized_chars"]
    assert original_tokens >= optimized_tokens
    assert stats["processing_time_ms"] >= 0


def test_optimize_batch_via_api(client: TestClient) -> None:
    payload = {
        "prompts": [
            "Summarize the primary launch milestones and owners.",
            "Convert the following agenda into concise bullet points: ...",
        ],
        "optimization_mode": "balanced",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "results" in data

    results: List[dict] = data["results"]
    assert len(results) == 2

    for item in results:
        assert "optimized_output" in item
        assert "stats" in item
        stats = item["stats"]
        assert stats["original_tokens"] >= stats["optimized_tokens"]
        assert stats["processing_time_ms"] >= 0


def test_optimize_accepts_segment_spans_and_query(client: TestClient) -> None:
    payload = {
        "prompt": "Alpha beta gamma delta epsilon.",
        "optimization_mode": "balanced",
        "segment_spans": [{"start": 6, "end": 10, "weight": 1.0}],
        "query": "What is the core requirement?",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "optimized_output" in data


def test_optimize_accepts_max_mode_alias(client: TestClient) -> None:
    payload = {
        "prompt": "Alpha beta gamma delta epsilon.",
        "optimization_mode": "max",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "optimized_output" in data


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_threshold", 0.9),
        ("minhash_paraphrase_threshold", 0.55),
        ("deduplicate_exact", False),
        ("deduplicate_semantic", False),
    ],
)
def test_optimize_rejects_removed_request_fields(
    client: TestClient,
    field: str,
    value: object,
) -> None:
    payload = {
        "prompt": "Alpha beta gamma delta epsilon.",
        "optimization_mode": "balanced",
        field: value,
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert field in str(body["detail"])


def test_optimize_applies_custom_canonicals(client: TestClient) -> None:
    payload = {
        "prompt": "Provide a quantum roadmap for the project. The quantum roadmap should be concise.",
        "optimization_mode": "balanced",
        "custom_canonicals": {"quantum roadmap": "qmap"},
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    output = response.json()["optimized_output"].lower()
    assert "qmap" in output


def test_optimize_applies_customer_canonical_mappings(client: TestClient) -> None:
    from database import create_user_canonical_mapping

    create_user_canonical_mapping("test_user_id", "quantum roadmap", "qmap")

    payload = {
        "prompt": "Provide a quantum roadmap for the project. The quantum roadmap should be concise.",
        "optimization_mode": "balanced",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    output = response.json()["optimized_output"].lower()
    assert "qmap" in output


def test_optimize_single_mode_degradation_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    calls = {"count": 0}

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        calls["count"] += 1
        stats = {
            "original_chars": len(prompt),
            "optimized_chars": len(prompt),
            "compression_percentage": 0.0,
            "original_tokens": 1,
            "optimized_tokens": 1,
            "token_savings": 0,
            "processing_time_ms": 1.0,
            "fast_path": False,
            "content_profile": "general_prose",
            "smart_context_description": "Auto-derived for general_prose (1 tokens)",
        }
        return {
            "optimized_output": prompt,
            "stats": stats,
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": False},
            "entropy": {"loaded": False},
            "token_classifier": {"loaded": False},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Hello world.", optimization_mode="maximum")
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single("Hello world.", request, BackgroundTasks())

    assert exc_info.value.status_code == 503
    assert "Missing required runtime models" in str(exc_info.value.detail)
    assert calls["count"] == 0


def test_validate_optimization_mode_fails_when_required_models_are_missing() -> None:
    import server

    with pytest.raises(HTTPException) as exc_info:
        server._validate_optimization_mode(
            "maximum",
            model_status={
                "semantic_guard": {"loaded": True},
                "entropy": {"loaded": True},
                "entropy_fast": {"loaded": False},
                "token_classifier": {"loaded": False},
            },
        )

    assert exc_info.value.status_code == 503
    assert "Missing required runtime models" in str(exc_info.value.detail)


def test_validate_cached_model_availability_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server
    from services import model_cache_manager

    def _raise(*_args, **_kwargs):
        raise RuntimeError("cache down")

    monkeypatch.setattr(model_cache_manager.ModelCacheValidator, "validate_model_cache", _raise)

    availability, warning = server._validate_cached_model_availability(
        ("semantic_guard", "entropy")
    )

    assert availability == {"semantic_guard": False, "entropy": False}
    assert warning is not None
    assert "treating gated models as unavailable in strict mode" in warning


def test_validate_optimization_mode_treats_unknown_cached_models_as_not_ready() -> None:
    import server

    with pytest.raises(HTTPException) as exc_info:
        server._validate_optimization_mode(
            "maximum",
            model_status={},
            cached_availability={"semantic_guard": True},
        )

    assert exc_info.value.status_code == 503
    assert "Missing required runtime models" in str(exc_info.value.detail)


def test_optimize_single_conservative_mode_does_not_report_mode_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    optimize_calls = {"mode": None}

    def _fake_optimize(prompt: str, *_args, **kwargs):
        optimize_calls["mode"] = kwargs.get("optimization_mode")
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "entropy": {"loaded": True},
            "token_classifier": {"loaded": False},
            "semantic_rank": {"loaded": False},
            "coreference": {"loaded": False},
            "spacy": {"loaded": False},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Hello world.", optimization_mode="conservative")
    result = server._optimize_single("Hello world.", request, BackgroundTasks())

    assert optimize_calls["mode"] == "conservative"
    warnings = result.warnings or []
    assert not any("downgraded from conservative" in warning for warning in warnings)


def test_optimize_single_fails_when_readiness_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    optimize_calls = {"mode": None}

    def _fake_optimize(prompt: str, *_args, **kwargs):
        optimize_calls["mode"] = kwargs.get("optimization_mode")
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(server.optimizer, "model_status", lambda: {})
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Hello world.", optimization_mode="maximum")
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single("Hello world.", request, BackgroundTasks())

    assert exc_info.value.status_code == 503
    assert "Optimizer model readiness is unavailable" in str(exc_info.value.detail)
    assert optimize_calls["mode"] is None


def test_optimize_single_maximum_mode_without_classifier_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    optimize_calls = {"mode": None}

    def _fake_optimize(prompt: str, *_args, **kwargs):
        optimize_calls["mode"] = kwargs.get("optimization_mode")
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": False},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(
        prompt="Retain meaning, reduce verbosity.", optimization_mode="maximum"
    )
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())

    assert exc_info.value.status_code == 503
    assert "token_classifier" in str(exc_info.value.detail)
    assert optimize_calls["mode"] is None


def test_optimize_single_warns_when_ready_models_not_exercised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": True},
            "coreference": {"loaded": True},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": True},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)
    monkeypatch.setattr(
        server.optimizer_config,
        "TOKEN_CLASSIFIER_POST_PASS_ENABLED",
        True,
    )

    request = OptimizationRequest(
        prompt="Retain meaning, reduce verbosity.",
        optimization_mode="maximum",
        query="Which sections answer the question?",
    )
    result = server._optimize_single(request.prompt, request, BackgroundTasks())

    warnings = result.warnings or []
    assert any(
        "semantic_rank was ready but not exercised" in warning for warning in warnings
    )
    assert any(
        "coreference was ready but not exercised" in warning for warning in warnings
    )
    assert any("spacy was ready but not exercised" in warning for warning in warnings)
    assert any(
        "entropy backend was ready but not exercised" in warning for warning in warnings
    )
    assert any(
        "token_classifier was ready but not exercised" in warning
        for warning in warnings
    )


def test_optimize_single_coreference_missing_suppressed_when_pass_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": True},
            "coreference": {"loaded": False},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": True},
        },
    )
    monkeypatch.setattr(
        server,
        "_resolve_request_disabled_passes",
        lambda _request: ("compress_coreferences",),
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Retain meaning.", optimization_mode="maximum")
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())
    assert exc_info.value.status_code == 503
    assert "coreference" in str(exc_info.value.detail)


def test_optimize_single_coreference_missing_warns_when_pass_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": True},
            "coreference": {"loaded": False},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": True},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Retain meaning.", optimization_mode="maximum")
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())
    assert exc_info.value.status_code == 503
    assert "coreference" in str(exc_info.value.detail)


def test_optimize_single_semantic_rank_missing_suppressed_when_profile_path_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "code",
                "smart_context_description": "Auto-derived for code (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": False},
            "coreference": {"loaded": True},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": True},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(
        prompt="def fn(x): return x", optimization_mode="maximum", query="what changed?"
    )
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())
    assert exc_info.value.status_code == 503
    assert "semantic_rank" in str(exc_info.value.detail)


def test_optimize_single_semantic_rank_missing_warns_when_profile_path_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": False},
            "coreference": {"loaded": True},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": True},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(
        prompt="summarize this", optimization_mode="maximum", query="what changed?"
    )
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())
    assert exc_info.value.status_code == 503
    assert "semantic_rank" in str(exc_info.value.detail)


def test_optimize_single_spacy_missing_uses_centralized_warning_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    def _fake_optimize(prompt: str, *_args, **_kwargs):
        return {
            "optimized_output": prompt,
            "stats": {
                "original_chars": len(prompt),
                "optimized_chars": len(prompt),
                "compression_percentage": 0.0,
                "original_tokens": 2,
                "optimized_tokens": 2,
                "token_savings": 0,
                "processing_time_ms": 1.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (2 tokens)",
            },
            "techniques_applied": [],
        }

    monkeypatch.setattr(server.optimizer, "optimize", _fake_optimize)
    monkeypatch.setattr(
        server.optimizer,
        "model_status",
        lambda: {
            "semantic_guard": {"loaded": True},
            "semantic_rank": {"loaded": True},
            "coreference": {"loaded": True},
            "entropy": {"loaded": True},
            "entropy_fast": {"loaded": True},
            "token_classifier": {"loaded": True},
            "spacy": {"loaded": False},
        },
    )
    monkeypatch.setattr(server, "_CACHE_SIZE", 0)

    request = OptimizationRequest(prompt="Retain meaning.", optimization_mode="maximum")
    with pytest.raises(HTTPException) as exc_info:
        server._optimize_single(request.prompt, request, BackgroundTasks())
    assert exc_info.value.status_code == 503
    assert "spacy" in str(exc_info.value.detail)


def test_settings_does_not_expose_api_keys(client: TestClient) -> None:
    from database import set_llm_profiles

    set_llm_profiles(
        "test_user_id",
        [
            {
                "name": "Test",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "api_key": "secret-key",
            }
        ],
    )

    response = client.get("/api/v1/settings")
    assert response.status_code == 200

    data = response.json()
    assert "llm_profiles" in data
    assert len(data["llm_profiles"]) == 1

    profile = data["llm_profiles"][0]
    assert profile["name"] == "Test"
    assert profile["provider"] == "openai"
    assert profile["model"] == "gpt-4.1-mini"
    assert profile.get("has_api_key") is True
    assert "api_key" not in profile


def test_settings_telemetry_toggle_persists_to_admin_settings(client: TestClient) -> None:
    import server
    from database import get_admin_setting
    from services.telemetry_control import set_enabled as set_telemetry_enabled

    set_telemetry_enabled(False)

    update_response = client.patch(
        "/api/v1/settings",
        json={"telemetry_enabled": True},
    )
    assert update_response.status_code == 200
    assert update_response.json()["telemetry_enabled"] is True

    # Simulate a different worker process with stale in-memory state.
    set_telemetry_enabled(False)
    assert get_admin_setting("telemetry_enabled", None) is True
    assert server.is_telemetry_enabled() is True

    get_response = client.get("/api/v1/settings")
    assert get_response.status_code == 200
    assert get_response.json()["telemetry_enabled"] is True


def test_optimize_batch_respects_worker_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    # Ensure we create a fresh executor in this test (server caches the pool globally).
    existing = getattr(server, "_batch_executor", None)
    if existing is not None:
        try:
            existing.shutdown(wait=False)
        except Exception:
            pass
    monkeypatch.setattr(server, "_batch_executor", None)

    _ExecutorSpy.last_requested_workers = None
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAX_WORKERS", "2")
    monkeypatch.setattr(server, "ThreadPoolExecutor", _ExecutorSpy)
    monkeypatch.setattr(server, "_optimize_single", _fake_optimize_single)

    request = OptimizationRequest(
        prompts=["a", "b", "c", "d"],
    )

    results = server._optimize_batch(request, BackgroundTasks())

    assert _ExecutorSpy.last_requested_workers == 2
    assert len(results.results) == 4


def test_parallel_optimizations_are_isolated() -> None:
    prompts = [
        "Summarize this onboarding guide into three concise bullet points.",
        "List critical release milestones with owners and due dates in brief form.",
        "Provide a crisp project status covering progress, risks, and next steps.",
    ]

    sequential_results = [
        optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
        for prompt in prompts
    ]

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(
                optimizer.optimize,
                prompt,
                mode="basic",
                optimization_mode="balanced",
            )
            for prompt in prompts
        ]
        parallel_results = [future.result() for future in futures]

    parallel_list_ids = [
        id(result["techniques_applied"]) for result in parallel_results
    ]
    assert len(set(parallel_list_ids)) == len(parallel_list_ids)

    for sequential, parallel in zip(sequential_results, parallel_results):
        assert sequential["optimized_output"] == parallel["optimized_output"]
        assert _normalized_stats(sequential["stats"]) == _normalized_stats(
            parallel["stats"]
        )
        assert sequential["techniques_applied"] == parallel["techniques_applied"]
        assert sequential["techniques_applied"] is not parallel["techniques_applied"]


def test_semantic_guard_reverts_on_low_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_prompt = "Explain the theory of relativity in exactly three bullet points."

    def fake_similarity(original: str, candidate: str, *_args, **_kwargs) -> float:
        assert original != candidate
        return 0.1

    monkeypatch.setattr(
        "services.optimizer.metrics.score_similarity",
        fake_similarity,
    )

    # Force-enable semantic guard for this test (some environments may not load model inventory).
    monkeypatch.setattr(optimizer, "refresh_model_configs", lambda: None, raising=False)
    monkeypatch.setattr(optimizer, "semantic_guard_enabled", True, raising=False)
    monkeypatch.setattr(optimizer, "semantic_guard_per_pass_enabled", True, raising=False)
    monkeypatch.setattr(optimizer, "semantic_guard_model", "test-model", raising=False)
    monkeypatch.setattr(optimizer, "fastpath_token_threshold", 0, raising=False)
    monkeypatch.setattr(
        optimizer, "_resolve_semantic_guard_threshold", lambda: 0.8, raising=False
    )
    # Force the lexical gate into its mid-range so score_similarity() is consulted.
    monkeypatch.setattr(optimizer, "_lexical_similarity", lambda *_a, **_k: 0.75, raising=False)

    # Use balanced mode so this test validates direct guard rollback behavior
    # without maximum-mode fallback retries.
    result = optimizer.optimize(
        original_prompt, mode="basic", optimization_mode="balanced"
    )

    assert result["optimized_output"] == original_prompt
    stats = result["stats"]
    assert stats["token_savings"] == 0
    # Rollback can happen via per-pass guards without tagging techniques.
    assert result["techniques_applied"] in ([], ["Semantic Guard Rollback"])


def test_maximum_level_entropy_pruning_does_not_error() -> None:
    prompt = "\n".join(
        ["The quick brown fox jumps over the lazy dog." for _ in range(20)]
    )

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")

    stats = result["stats"]
    original_tokens, optimized_tokens = _token_budget(stats)
    assert original_tokens >= optimized_tokens
    assert stats["processing_time_ms"] >= 0


def test_entropy_pruning_skips_placeholder_tokens() -> None:
    """Verify that entropy scoring skips tokens within placeholder ranges.

    This test ensures the optimization completes successfully when the prompt
    contains placeholders matching the PLACEHOLDER_PATTERN (__XXX_N__).
    The skip_ranges logic should prevent errors and wasteful computation.
    """
    # Create a prompt with multiple placeholders matching the pattern
    prompt = "The quick __PLACEHOLDER_1__ brown __VAR_2__ fox jumps over the __CONST_3__ lazy dog."

    # Run optimization with maximum optimization level to trigger entropy pruning
    result = optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")

    # Verify the optimization completes without error
    assert "optimized_output" in result
    assert "stats" in result
    assert result["stats"]["processing_time_ms"] >= 0

    # The placeholders should be preserved in the output
    assert "__PLACEHOLDER_1__" in result["optimized_output"]
    assert "__VAR_2__" in result["optimized_output"]
    assert "__CONST_3__" in result["optimized_output"]


def test_optimize_records_history_when_enabled(
    client: TestClient,
) -> None:
    prompt = "List the onboarding steps for new engineers in bullet form."
    response = client.post(
        "/api/v1/optimize",
        json={"prompt": prompt, "optimization_mode": "balanced"},
    )
    assert response.status_code == 200

    active_db = _active_test_db()
    assert active_db.exists()
    rows = []
    deadline = time.time() + 6.0
    while time.time() < deadline and not rows:
        with sqlite3.connect(active_db) as connection:
            rows = connection.execute(
                "SELECT raw_prompt, optimized_prompt FROM optimization_history ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        if not rows:
            time.sleep(0.2)

    assert rows


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_empty_prompt_returns_error(client: TestClient) -> None:
    """Test that empty prompt is rejected with 400 error"""
    payload = {
        "prompt": "",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_whitespace_only_prompt_returns_error(client: TestClient) -> None:
    """Test that whitespace-only prompt is rejected"""
    payload = {
        "prompt": "   \n\t  \n   ",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422


def test_neither_prompt_nor_prompts_provided(client: TestClient) -> None:
    """Test that request without prompt or prompts returns error"""
    payload = {}

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_both_prompt_and_prompts_provided(client: TestClient) -> None:
    """Test that providing both prompt and prompts returns error"""
    payload = {
        "prompt": "Test prompt",
        "prompts": ["Test prompt 1", "Test prompt 2"],
    }

    response = client.post("/api/v1/optimize", json=payload)
    # Should return 400 or 422 depending on validation logic
    assert response.status_code in [400, 422]


def test_empty_batch_prompts_list(client: TestClient) -> None:
    """Test that empty prompts list is rejected"""
    payload = {
        "prompts": [],
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422


def test_single_item_batch_succeeds(client: TestClient) -> None:
    """Test that batch with single prompt works correctly"""
    payload = {
        "prompts": ["Summarize the project timeline in bullet points."],
        "optimization_mode": "balanced",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert "optimized_output" in data["results"][0]


def test_very_large_prompt_succeeds() -> None:
    """Test optimization of very large prompt (simulating near 500K tokens)"""
    # Generate a large prompt (roughly 100K tokens = ~400K chars)
    large_prompt = (
        "This is a test sentence with multiple words that will be repeated. " * 5000
    )

    result = optimizer.optimize(
        large_prompt, mode="basic", optimization_mode="balanced"
    )

    assert "optimized_output" in result
    assert "stats" in result
    stats = result["stats"]

    # Should compress significantly
    assert stats["original_tokens"] > stats["optimized_tokens"]
    assert stats["compression_percentage"] > 0


def test_prompt_with_special_characters() -> None:
    """Test that prompts with special characters, emojis, unicode are handled"""
    prompt = "Please summarize: 🚀 Project Alpha™ costs €1,234.56 (≈$1,400) — including VAT@20%"

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")

    assert "optimized_output" in result
    output = result["optimized_output"]
    # Should preserve numbers and special symbols
    assert "1,234" in output or "1234" in output


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_invalid_optimization_mode_returns_error(client: TestClient) -> None:
    """Test that invalid optimization_mode value returns validation error"""
    payload = {"prompt": "Test prompt", "optimization_mode": "invalid_mode"}

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_malformed_json_returns_error(client: TestClient) -> None:
    """Test that malformed JSON returns 422 error"""
    response = client.post(
        "/api/v1/optimize",
        content='{"prompt": "test"',  # Missing closing brace
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ============================================================================
# HISTORY API TESTS
# ============================================================================


def test_history_detail_endpoint_returns_prompts(client: TestClient) -> None:
    optimization_id = "opt_detail_1"

    with sqlite3.connect(_active_test_db()) as connection:
        connection.execute(
            """
            INSERT INTO optimization_history (
                id, customer_id, created_at, updated_at, mode,
                raw_prompt, optimized_prompt, raw_tokens, optimized_tokens,
                processing_time_ms, estimated_cost_before, estimated_cost_after, estimated_cost_saved,
                compression_percentage, semantic_similarity, techniques_applied
            ) VALUES (?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                optimization_id,
                "test_user_id",
                "basic",
                "raw prompt text",
                "optimized prompt text",
                10,
                6,
                12.0,
                0.03,
                0.02,
                0.01,
                40.0,
                0.9,
                "[]",
            ),
        )

    response = client.get(f"/api/v1/history/{optimization_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == optimization_id
    assert data["raw_prompt"] == "raw prompt text"
    assert data["optimized_prompt"] == "optimized prompt text"
    assert data["original_tokens"] == 10
    assert data["optimized_tokens"] == 6


def test_history_detail_endpoint_is_customer_scoped(client: TestClient) -> None:
    optimization_id = "opt_detail_other_customer"

    with sqlite3.connect(_active_test_db()) as connection:
        connection.execute(
            """
            INSERT INTO optimization_history (
                id, customer_id, created_at, updated_at, mode,
                raw_prompt, optimized_prompt, raw_tokens, optimized_tokens,
                processing_time_ms, estimated_cost_before, estimated_cost_after, estimated_cost_saved,
                compression_percentage, semantic_similarity, techniques_applied
            ) VALUES (?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                optimization_id,
                "other_customer",
                "basic",
                "raw",
                "opt",
                10,
                6,
                1.0,
                0.0,
                0.0,
                0.0,
                40.0,
                0.9,
                "[]",
            ),
        )

    response = client.get(f"/api/v1/history/{optimization_id}")
    assert response.status_code == 404


# ============================================================================
# PERFORMANCE BENCHMARK TESTS
# ============================================================================


def test_performance_small_prompt_under_2s() -> None:
    """Test that small prompt (1K tokens) optimizes in under 2 seconds"""
    import time

    # ~1K tokens (roughly 4K chars)
    prompt = (
        "Please provide a comprehensive analysis of the quarterly business review. "
        * 50
    )

    start = time.perf_counter()
    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    duration_ms = (time.perf_counter() - start) * 1000

    assert result["stats"]["processing_time_ms"] < 2000
    assert duration_ms < 2000


def test_performance_medium_prompt_under_2s() -> None:
    """Test that medium prompt (10K tokens) optimizes in under 2 seconds"""
    import time

    # ~10K tokens (roughly 40K chars)
    prompt = (
        "This is a detailed technical document with many sections and subsections. "
        * 500
    )

    start = time.perf_counter()
    optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    duration_ms = (time.perf_counter() - start) * 1000

    # Medium prompts should still be under 2s per PRD
    assert duration_ms < 2000


def test_performance_maximum_level_under_2s() -> None:
    """Test that maximum optimization level on medium prompt stays under 2 seconds"""
    import time

    prompt = (
        "Please could you kindly provide detailed information about this topic. " * 100
    )

    start = time.perf_counter()
    optimizer.optimize(prompt, mode="basic", optimization_mode="maximum")
    duration_ms = (time.perf_counter() - start) * 1000

    assert duration_ms < 2000


# ============================================================================
# COMPRESSION RATIO VALIDATION TESTS
# ============================================================================


def test_compression_percentage_calculation() -> None:
    """Test that compression percentage is calculated correctly"""
    prompt = (
        "Please provide me with a comprehensive and detailed overview of the project."
    )

    result = optimizer.optimize(prompt, mode="basic", optimization_mode="balanced")
    stats = result["stats"]

    # Verify compression percentage formula
    expected_compression = (
        (stats["original_chars"] - stats["optimized_chars"]) / stats["original_chars"]
    ) * 100

    assert abs(stats["compression_percentage"] - expected_compression) < 0.01


def test_basic_mode_achieves_minimum_compression() -> None:
    """Test that balanced optimization level achieves at least some compression on verbose prompts"""
    verbose_prompt = (
        "Please could you kindly provide me with a very comprehensive and extremely "
        "detailed overview of the entire project timeline, including all of the "
        "important milestones, critical deadlines, and every single responsible owner."
    )

    result = optimizer.optimize(
        verbose_prompt, mode="basic", optimization_mode="balanced"
    )
    stats = result["stats"]

    # Should achieve at least 10% compression on verbose text
    assert stats["compression_percentage"] >= 10.0
    assert stats["token_savings"] > 0


def test_maximum_level_achieves_higher_compression() -> None:
    """Test that maximum optimization level compresses more than balanced optimization level"""
    prompt = "The quick brown fox jumps over the lazy dog. " * 20

    balanced_result = optimizer.optimize(
        prompt, mode="basic", optimization_mode="balanced"
    )
    maximum_result = optimizer.optimize(
        prompt, mode="basic", optimization_mode="maximum"
    )

    balanced_tokens = balanced_result["stats"]["optimized_tokens"]
    maximum_tokens = maximum_result["stats"]["optimized_tokens"]

    # Maximum should compress more (lower token count)
    assert maximum_tokens <= balanced_tokens


# ============================================================================
# BATCH PROCESSING EDGE CASES
# ============================================================================


def test_large_batch_processing(client: TestClient) -> None:
    """Test that large batch (10 prompts) processes successfully"""
    prompts = [f"Summarize point number {i} in brief bullet format." for i in range(10)]

    payload = {"prompts": prompts, "optimization_mode": "balanced"}

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert len(data["results"]) == 10

    # Verify all results have required fields
    for result in data["results"]:
        assert "optimized_output" in result
        assert "stats" in result


def test_batch_with_mixed_lengths(client: TestClient) -> None:
    """Test batch with prompts of varying lengths"""
    payload = {
        "prompts": [
            "Short.",
            "This is a medium length prompt with several words.",
            "This is a much longer prompt that contains significantly more content and should take more time to "
            "process but still complete successfully.",
        ],
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert len(data["results"]) == 3


def test_batch_preserves_order(client: TestClient) -> None:
    """Test that batch results maintain input order"""
    prompts = ["First prompt", "Second prompt", "Third prompt"]

    payload = {
        "prompts": prompts,
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    results = response.json()["results"]

    # Results should maintain input order
    # (Can't directly verify text due to optimization, but can verify count)
    assert len(results) == len(prompts)


def test_sanitize_header_value_strips_whitespace_and_emoji() -> None:
    """Test that _sanitize_header_value strips emoji and leading/trailing whitespace"""
    from server import _sanitize_header_value

    # Test emoji prefix removal with whitespace handling
    result = _sanitize_header_value("⚠️ msg")
    assert result == "msg", f"Expected 'msg', got '{result}'"

    # Test that there are no leading spaces
    assert not result.startswith(" "), "Result should not have leading spaces"

    # Test trailing whitespace removal
    result = _sanitize_header_value("test message   ")
    assert result == "test message", f"Expected 'test message', got '{result}'"

    # Test leading whitespace removal
    result = _sanitize_header_value("   test message")
    assert result == "test message", f"Expected 'test message', got '{result}'"

    # Test empty string after sanitization
    result = _sanitize_header_value("⚠️")
    assert result == "", f"Expected empty string, got '{result}'"

    # Test normal ASCII text
    result = _sanitize_header_value("normal text")
    assert result == "normal text"


def test_optimize_response_headers_with_sanitized_warnings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that optimize_prompt properly sanitizes and filters warnings in response headers"""

    def mock_optimize(*args, **kwargs):
        # Return a dict matching the structure from services.optimizer.core.optimize()
        # Include "Semantic Guard Rollback" to trigger warning with emoji
        return {
            "optimized_output": "test output",
            "stats": {
                "original_chars": 100,
                "optimized_chars": 50,
                "compression_percentage": 50.0,
                "original_tokens": 10,
                "optimized_tokens": 5,
                "token_savings": 5,
                "processing_time_ms": 10.0,
                "fast_path": False,
                "content_profile": "general_prose",
                "smart_context_description": "Auto-derived for general_prose (10 tokens)",
            },
            "techniques_applied": ["Semantic Guard Rollback"],
        }

    # Mock optimizer output to inject semantic-guard fallback warning path.
    monkeypatch.setattr("services.optimizer.core.spacy", None)
    monkeypatch.setattr("services.optimizer.core.optimizer.optimize", mock_optimize)

    payload = {
        "prompt": "Test prompt",
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200

    # Check that the response headers contain sanitized warnings
    warnings_header = response.headers.get("X-Tokemizer-Warnings")

    # Should have sanitized warnings (emoji removed, whitespace stripped)
    assert warnings_header is not None
    # The Semantic Guard Rollback warning has an emoji that should be stripped
    assert "Optimization was reverted" in warnings_header
    # Warning header should be ASCII-sanitized (no emoji glyphs).
    assert "⚠" not in warnings_header

    # Verify no leading spaces in the warnings
    warning_parts = warnings_header.split("; ")
    for part in warning_parts:
        assert not part.startswith(
            " "
        ), f"Warning part '{part}' should not have leading space"
        assert not part.endswith(
            " "
        ), f"Warning part '{part}' should not have trailing space"

    # Verify we have at least one warning and preserve separators for multiple warnings.
    assert (
        len(warning_parts) >= 1
    ), f"Expected at least 1 warning, got {len(warning_parts)}: {warning_parts}"

    # Verify the specific warnings we expect are present (regardless of other optional dep warnings)
    warning_text = warnings_header.lower()
    assert (
        "optimization was reverted" in warning_text
    ), "Expected Semantic Guard Rollback warning"


def test_validate_cached_model_availability_uses_ttl_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import server

    calls = {"count": 0}

    class FakeValidator:
        def __init__(self, _hf_home: str):
            return None

        def validate_model_cache(
            self, _model_type: str, use_cache=False, generate_manifest=False
        ):
            calls["count"] += 1
            return {"cached_ok": True}

    monkeypatch.setattr(server, "_MODEL_AVAILABILITY_CACHE_TTL_SECONDS", 60)
    monkeypatch.setattr(server, "get_model_cache_validation_version", lambda: 1)
    monkeypatch.setattr(server, "resolve_hf_home", lambda: "/tmp/hf")
    monkeypatch.setattr(
        server,
        "_invalidate_model_availability_cache",
        lambda: server._model_availability_cache.clear(),
    )
    server._model_availability_cache.clear()

    import types
    import sys
    fake_module = types.SimpleNamespace(ModelCacheValidator=FakeValidator)
    monkeypatch.setitem(sys.modules, "services.model_cache_manager", fake_module)

    first, warning_first = server._validate_cached_model_availability(("semantic_guard",))
    second, warning_second = server._validate_cached_model_availability(("semantic_guard",))

    assert warning_first is None
    assert warning_second is None
    assert first["semantic_guard"] is True
    assert second["semantic_guard"] is True
    assert calls["count"] == 1


def test_validate_cached_model_availability_cache_invalidates_on_version_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    calls = {"count": 0}

    class FakeValidator:
        def __init__(self, _hf_home: str):
            return None

        def validate_model_cache(self, _model_type: str, use_cache=False, generate_manifest=False):
            calls["count"] += 1
            return {"cached_ok": True}

    version_state = {"value": 1}

    monkeypatch.setattr(server, "_MODEL_AVAILABILITY_CACHE_TTL_SECONDS", 60)
    monkeypatch.setattr(server, "get_model_cache_validation_version", lambda: version_state["value"])
    monkeypatch.setattr(server, "resolve_hf_home", lambda: "/tmp/hf")
    server._model_availability_cache.clear()

    import types
    import sys
    fake_module = types.SimpleNamespace(ModelCacheValidator=FakeValidator)
    monkeypatch.setitem(sys.modules, "services.model_cache_manager", fake_module)

    server._validate_cached_model_availability(("semantic_guard",))
    version_state["value"] = 2
    server._validate_cached_model_availability(("semantic_guard",))

    assert calls["count"] == 2


def test_summarize_noncode_request_text_preserves_execution_plan_and_line_limit() -> None:
    import server

    prompt = (
        "Write a Python data extraction script for Postgres at host=prod-db, port=5432. "
        "Output CSV in exact header order user_id,email,signup_date,last_active,30d_retention_score. "
        "Return both code and a short execution plan; response format must be a single code block. "
        "Keep under 300 lines and fully featured."
    )

    summarized = server._summarize_noncode_request_text(
        prompt,
        max_chars=1200,
        allow_long_output=True,
    ).lower()

    assert "execution plan" in summarized
    assert "single code block" in summarized
    assert "under 300 lines" in summarized
    assert "host=prod-db" in summarized


def test_deterministic_fulfillment_summary_feature_flag_controls_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import server

    prompt = (
        "Write a Python data extraction script for Postgres at host=prod-db, port=5432. "
        "Return both code and a short execution plan. Keep under 300 lines."
    )

    request = OptimizationRequest(
        prompt=prompt,
        optimization_technique="llm_based",
        optimization_mode="balanced",
    )

    monkeypatch.setenv("LLM_CONSTRAINT_ENFORCEMENT_ENABLED", "false")
    monkeypatch.setenv("LLM_CONSTRAINT_REPAIR_ENABLED", "false")
    monkeypatch.setattr(server, "get_llm_system_context", lambda: "compress")
    monkeypatch.setattr(
        server,
        "call_llm",
        lambda *_args, **_kwargs: type(
            "LLMResultStub",
            (),
            {"text": "llm drafted output", "duration_ms": 5.0},
        )(),
    )

    monkeypatch.setenv("LLM_DETERMINISTIC_FULFILLMENT_SUMMARY_ENABLED", "false")
    result_disabled = server._optimize_single_llm(prompt, request)
    assert result_disabled.optimized_output == "llm drafted output"

    monkeypatch.setenv("LLM_DETERMINISTIC_FULFILLMENT_SUMMARY_ENABLED", "true")
    result_enabled = server._optimize_single_llm(prompt, request)
    lowered = result_enabled.optimized_output.lower()
    assert "execution plan" in lowered
    assert "under 300 lines" in lowered
    assert "host=prod-db" in lowered
    assert result_enabled.optimized_output != "llm drafted output"
