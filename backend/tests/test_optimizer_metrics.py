from typing import List, Sequence

import pytest
from services.optimizer import metrics


class _DummyTokenizer:
    def __init__(self, limit: int) -> None:
        self.model_max_length = limit
        self.init_kwargs = {"model_max_length": limit}

    def encode(
        self, text: str, add_special_tokens: bool = True, truncation: bool = False
    ):
        return list(range(len(text.split())))


class _DummyEncoder:
    def __init__(self, limit: int) -> None:
        self.max_seq_length = limit
        self.tokenizer = _DummyTokenizer(limit)
        self.calls: List[Sequence[str]] = []

    def encode(self, texts, *, batch_size: int):
        self.calls.append(list(texts))
        return [[float(len(text.split()))] for text in texts]


def test_encode_texts_skips_overlong_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = _DummyEncoder(limit=5)
    monkeypatch.setattr(metrics, "_load_encoder", lambda _model, _type: encoder)

    # Create a segment exceeding the synthetic 5 token limit
    result = metrics.encode_texts(["one two three four five six"], model_name="dummy")

    assert result is None
    assert encoder.calls == []


def test_encode_texts_returns_embeddings_when_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if metrics.np is None:
        pytest.skip("NumPy unavailable in test environment.")
    encoder = _DummyEncoder(limit=6)
    monkeypatch.setattr(metrics, "_load_encoder", lambda _model, _type: encoder)

    texts = ["short text", "even shorter"]
    result = metrics.encode_texts(texts, model_name="dummy")

    assert result is not None
    assert len(result) == len(texts)
    assert encoder.calls == [texts]


def test_score_similarity_returns_none_for_overlong_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if metrics.np is None:
        pytest.skip("NumPy unavailable in test environment.")
    encoder = _DummyEncoder(limit=4)
    monkeypatch.setattr(metrics, "_load_encoder", lambda _model, _type: encoder)

    similarity = metrics.score_similarity(
        "one two three four five", "ok", model_name="dummy"
    )

    assert similarity is None
    assert encoder.calls == []


def test_encoder_cache_shared_across_semantic_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class _DummyOnnxEncoder:
        def __init__(self, model_type: str, model_name: str, use_int8: bool) -> None:
            self.model_type = model_type
            self.model_name = model_name
            self.use_int8 = use_int8
            self.available = True
            created.append((model_type, model_name, use_int8))

    monkeypatch.setattr(metrics, "_OnnxEncoder", _DummyOnnxEncoder)
    metrics.reset_encoder_cache()

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_guard = metrics._get_cached_encoder("semantic_guard", model_name, False)
    encoder_rank = metrics._get_cached_encoder("semantic_rank", model_name, False)

    assert encoder_guard is encoder_rank
    assert len(created) == 1


def test_encode_texts_with_plan_reuses_embeddings_without_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_plan = {
        "embedding_vectors": {
            ("semantic_shared", "dummy", "query"): [1.0],
            ("semantic_shared", "dummy", "context"): [2.0],
        },
        "metrics": {
            "embedding_reuse_count": 0.0,
            "embedding_calls_saved": 0.0,
            "embedding_wall_clock_savings_ms": 0.0,
            "avg_encode_call_ms": 5.0,
        },
    }

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("encode_texts should not run when all vectors are planned")

    monkeypatch.setattr(metrics, "encode_texts", _fail_if_called)

    resolved = metrics.encode_texts_with_plan(
        ["query", "context"],
        "dummy",
        model_type="semantic_rank",
        semantic_plan=semantic_plan,
        shared_embeddings=True,
    )

    assert resolved == [[1.0], [2.0]]
    assert semantic_plan["metrics"]["embedding_reuse_count"] == 2.0
    assert semantic_plan["metrics"]["embedding_calls_saved"] == 1.0
    assert semantic_plan["metrics"]["embedding_wall_clock_savings_ms"] == 5.0


def test_encode_texts_with_plan_populates_missing_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_plan = {
        "embedding_vectors": {
            ("semantic_shared", "dummy", "query"): [1.0],
        },
        "metrics": {
            "embedding_reuse_count": 0.0,
            "embedding_calls_saved": 0.0,
            "embedding_wall_clock_savings_ms": 0.0,
        },
    }

    def _encode_missing(texts, model_name, **kwargs):
        assert list(texts) == ["context"]
        assert model_name == "dummy"
        return [[3.0]]

    monkeypatch.setattr(metrics, "encode_texts", _encode_missing)

    resolved = metrics.encode_texts_with_plan(
        ["query", "context"],
        "dummy",
        model_type="semantic_rank",
        semantic_plan=semantic_plan,
        shared_embeddings=True,
    )

    assert resolved == [[1.0], [3.0]]
    assert semantic_plan["embedding_vectors"][
        ("semantic_shared", "dummy", "context")
    ] == [3.0]
    assert semantic_plan["metrics"]["embedding_reuse_count"] == 1.0
    assert semantic_plan["metrics"]["observed_encode_call_count"] == 1.0


def test_select_embeddings_mean_pools_token_embeddings() -> None:
    if metrics.np is None:
        pytest.skip("NumPy unavailable in test environment.")

    encoder = metrics._OnnxEncoder.__new__(metrics._OnnxEncoder)
    token_embeddings = metrics.np.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]],
            [[2.0, 2.0], [8.0, 8.0], [10.0, 10.0], [12.0, 12.0]],
        ],
        dtype="float32",
    )
    attention_mask = metrics.np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype="int64")

    selected = encoder._select_embeddings(
        [token_embeddings],
        attention_mask=attention_mask,
        expected_batch_size=2,
    )

    assert selected is not None
    assert selected.shape == (2, 2)
    assert metrics.np.allclose(selected[0], [3.0, 0.0])
    assert metrics.np.allclose(selected[1], [2.0, 2.0])


def test_select_embeddings_prefers_direct_sentence_embeddings() -> None:
    if metrics.np is None:
        pytest.skip("NumPy unavailable in test environment.")

    encoder = metrics._OnnxEncoder.__new__(metrics._OnnxEncoder)
    token_embeddings = metrics.np.ones((2, 3, 2), dtype="float32")
    sentence_embeddings = metrics.np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")

    selected = encoder._select_embeddings(
        [token_embeddings, sentence_embeddings],
        attention_mask=metrics.np.ones((2, 3), dtype="int64"),
        expected_batch_size=2,
    )

    assert selected is not None
    assert metrics.np.allclose(selected, sentence_embeddings)
