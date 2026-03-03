"""Semantic similarity metrics helpers for the optimizer."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - environment dependent
    import numpy as np
except ImportError:  # pragma: no cover - dependency handled gracefully
    np = None  # type: ignore

try:  # pragma: no cover - environment dependent
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - dependency handled gracefully
    ort = None
    AutoTokenizer = None

from services.model_cache_manager import (
    resolve_cached_model_artifact,
    resolve_tokenizer_root_from_artifact,
)

from . import config

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_DEFAULT_ENCODER_MAX_SEQ_LENGTH = 512
_TOKENIZER_SENTINEL_LIMIT = 1_000_000
_INT8_SIMILARITY_THRESHOLD = 4000
_INT8_SCALE = 127.0
_ENCODER_CACHE_SIZE = 2


def _resolve_embedding_cache_size() -> int:
    raw_value = os.environ.get("EMBEDDING_CACHE_SIZE", "512")
    try:
        value = int(raw_value)
        return max(value, 0)
    except (TypeError, ValueError):
        return 512


_EMBEDDING_CACHE_SIZE = _resolve_embedding_cache_size()
_EMBEDDING_CACHE: "OrderedDict[Tuple[str, str, str], Any]" = OrderedDict()
_EMBEDDING_CACHE_LOCK = threading.Lock()
_SHARED_EMBEDDING_TYPES = {"semantic_guard", "semantic_rank", "semantic_chunk"}
_SHARED_EMBEDDING_TELEMETRY_KEY = ("__shared__", "__telemetry__", "__reuse__")
_ENCODER_CACHE: "OrderedDict[Tuple[str, str, bool], Optional[_OnnxEncoder]]" = (
    OrderedDict()
)


class _OnnxEncoder:
    def __init__(self, model_type: str, model_name: str, use_int8: bool) -> None:
        self._model_type = model_type
        self._model_name = model_name
        self._use_int8 = use_int8
        self._session = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self.max_seq_length = _DEFAULT_ENCODER_MAX_SEQ_LENGTH
        self._available = False
        self._load()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def available(self) -> bool:
        return (
            self._available
            and self._session is not None
            and self._tokenizer is not None
        )

    def _load(self) -> None:
        if ort is None or AutoTokenizer is None:
            logger.warning(
                "onnxruntime or transformers not available; semantic similarity metrics disabled"
            )
            return

        preferred = "model.int8.onnx" if self._use_int8 else "model.onnx"
        onnx_path = resolve_cached_model_artifact(
            self._model_type, self._model_name, preferred
        )
        if onnx_path is None and self._use_int8:
            onnx_path = resolve_cached_model_artifact(
                self._model_type, self._model_name, "model.onnx"
            )
        if onnx_path is None:
            logger.warning(
                "ONNX model not found for %s: %s",
                self._model_type,
                self._model_name,
            )
            return

        model_path = resolve_tokenizer_root_from_artifact(onnx_path)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
        except Exception as exc:
            logger.warning(
                "Failed to load tokenizer for %s %s: %s",
                self._model_type,
                self._model_name,
                exc,
            )
            self._tokenizer = None
            return

        try:
            self._session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._input_names = [item.name for item in self._session.get_inputs()]
        except Exception as exc:
            logger.warning(
                "Failed to load ONNX session for %s %s: %s",
                self._model_type,
                self._model_name,
                exc,
            )
            self._session = None
            return

        _configure_encoder_limits(self)
        self._available = True

    @staticmethod
    def _mean_pool_token_embeddings(
        token_embeddings: Any, attention_mask: Optional[Any]
    ):
        if np is None:
            return None
        array = np.asarray(token_embeddings)
        if array.ndim != 3:
            return None
        if attention_mask is None:
            return np.mean(array, axis=1)

        mask = np.asarray(attention_mask)
        if mask.ndim != 2:
            return np.mean(array, axis=1)
        if mask.shape[0] != array.shape[0] or mask.shape[1] != array.shape[1]:
            return np.mean(array, axis=1)

        expanded = mask.astype("float32")[..., None]
        masked_sum = np.sum(array * expanded, axis=1)
        token_count = np.sum(expanded, axis=1)
        token_count = np.clip(token_count, 1e-12, None)
        return masked_sum / token_count

    def _select_embeddings(
        self,
        outputs: Sequence[Any],
        *,
        attention_mask: Optional[Any],
        expected_batch_size: int,
    ) -> Optional[Any]:
        if np is None:
            return None
        if not outputs:
            return None

        for output in outputs:
            array = np.asarray(output)
            if array.ndim == 2 and array.shape[0] == expected_batch_size:
                return array

        for output in outputs:
            array = np.asarray(output)
            if array.ndim == 1 and expected_batch_size == 1:
                return array.reshape(1, -1)

        for output in outputs:
            pooled = self._mean_pool_token_embeddings(output, attention_mask)
            if (
                pooled is not None
                and pooled.ndim == 2
                and pooled.shape[0] == expected_batch_size
            ):
                return pooled

        fallback = np.asarray(outputs[0])
        if fallback.ndim == 2 and fallback.shape[0] == expected_batch_size:
            return fallback
        return None

    def encode(self, texts: Sequence[str], *, batch_size: int) -> Optional[Any]:
        if not texts or not self.available or np is None:
            return None

        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:
            return None

        max_length = _resolve_max_sequence_length(self)
        outputs: List[Any] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            inputs: Dict[str, Any] = {}
            for name in self._input_names:
                value = encoded.get(name)
                if value is None and name == "token_type_ids":
                    value = np.zeros_like(encoded["input_ids"])
                if value is not None:
                    inputs[name] = value.astype("int64")
            if not inputs:
                return None
            result = session.run(None, inputs)
            embeddings = self._select_embeddings(
                result,
                attention_mask=inputs.get("attention_mask"),
                expected_batch_size=len(batch),
            )
            if embeddings is None:
                return None
            embeddings = embeddings.astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            outputs.append(embeddings / norms)
        if not outputs:
            return None
        return np.vstack(outputs)


def _load_encoder(model_name: str, model_type: str) -> Optional[_OnnxEncoder]:
    """Load and cache the sentence encoder instance."""
    return _get_cached_encoder(model_type, model_name, config.ONNX_USE_INT8)


def reset_encoder_cache() -> None:
    """Clear the cached sentence-transformer encoder instance."""
    with _MODEL_LOCK:
        _ENCODER_CACHE.clear()


def _resolve_embedding_cache_key(
    model_type: str,
    model_name: str,
    text: str,
    *,
    shared_embeddings: bool,
) -> Tuple[str, str, str]:
    if shared_embeddings and model_type in _SHARED_EMBEDDING_TYPES:
        return ("semantic_shared", model_name, text)
    return (model_type, model_name, text)


def _resolve_encoder_cache_key(
    model_type: str, model_name: str, use_int8: bool
) -> Tuple[str, str, bool]:
    if model_type in _SHARED_EMBEDDING_TYPES:
        return ("semantic_shared", model_name, use_int8)
    return (model_type, model_name, use_int8)


def _record_shared_embedding_hit(
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]],
) -> None:
    if embedding_cache is None:
        return
    current = embedding_cache.get(_SHARED_EMBEDDING_TELEMETRY_KEY)
    try:
        embedding_cache[_SHARED_EMBEDDING_TELEMETRY_KEY] = int(current or 0) + 1
    except Exception:
        embedding_cache[_SHARED_EMBEDDING_TELEMETRY_KEY] = 1


def get_shared_embedding_reuse_count(
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]],
) -> int:
    if embedding_cache is None:
        return 0
    try:
        return int(embedding_cache.get(_SHARED_EMBEDDING_TELEMETRY_KEY) or 0)
    except Exception:
        return 0


def _get_cached_embedding(
    model_type: str,
    model_name: str,
    text: str,
    *,
    shared_embeddings: bool,
) -> Optional[Any]:
    key = _resolve_embedding_cache_key(
        model_type,
        model_name,
        text,
        shared_embeddings=shared_embeddings,
    )
    with _EMBEDDING_CACHE_LOCK:
        embedding = _EMBEDDING_CACHE.get(key)
        if embedding is not None:
            _EMBEDDING_CACHE.move_to_end(key)
        return embedding


def _set_cached_embedding(
    model_type: str,
    model_name: str,
    text: str,
    embedding: Any,
    *,
    shared_embeddings: bool,
) -> None:
    key = _resolve_embedding_cache_key(
        model_type,
        model_name,
        text,
        shared_embeddings=shared_embeddings,
    )
    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE[key] = embedding
        _EMBEDDING_CACHE.move_to_end(key)
        if len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_SIZE:
            _EMBEDDING_CACHE.popitem(last=False)


def _valid_length(value: Optional[int]) -> Optional[int]:
    if isinstance(value, int) and 0 < value < _TOKENIZER_SENTINEL_LIMIT:
        return value
    return None


def _resolve_max_sequence_length(encoder) -> int:
    candidates: List[int] = []
    direct_limit = _valid_length(getattr(encoder, "max_seq_length", None))
    if direct_limit is not None:
        candidates.append(direct_limit)

    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is not None:
        for attr in ("model_max_length", "max_len_single_sentence"):
            token_limit = _valid_length(getattr(tokenizer, attr, None))
            if token_limit is not None:
                candidates.append(token_limit)

        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            init_limit = _valid_length(init_kwargs.get("model_max_length"))
            if init_limit is not None:
                candidates.append(init_limit)

    if not candidates:
        return _DEFAULT_ENCODER_MAX_SEQ_LENGTH

    resolved = min(candidates)
    return max(1, min(resolved, _DEFAULT_ENCODER_MAX_SEQ_LENGTH))


def _configure_encoder_limits(encoder) -> None:
    max_length = _resolve_max_sequence_length(encoder)
    try:
        encoder.max_seq_length = max_length
    except Exception:  # pragma: no cover - attribute missing/readonly
        pass

    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        return

    if hasattr(tokenizer, "model_max_length"):
        try:
            tokenizer.model_max_length = max_length
        except Exception:  # pragma: no cover - attribute may be property
            pass

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["model_max_length"] = max_length


def _get_cached_encoder(
    model_type: str, model_name: str, use_int8: bool
) -> Optional[_OnnxEncoder]:
    """Cache wrapper for the encoder to avoid repeated downloads/initialization."""

    cache_key = _resolve_encoder_cache_key(model_type, model_name, use_int8)
    with _MODEL_LOCK:
        encoder = _ENCODER_CACHE.get(cache_key)
        if encoder is not None or cache_key in _ENCODER_CACHE:
            _ENCODER_CACHE.move_to_end(cache_key)
            return encoder

        encoder = _OnnxEncoder(model_type, model_name, use_int8)
        if encoder.available:
            resolved = encoder
        else:
            resolved = None

        _ENCODER_CACHE[cache_key] = resolved
        _ENCODER_CACHE.move_to_end(cache_key)
        if len(_ENCODER_CACHE) > _ENCODER_CACHE_SIZE:
            _ENCODER_CACHE.popitem(last=False)
    return resolved


def _estimate_token_length(text: str, tokenizer, *, max_fallback: int) -> int:
    if not text:
        return 0

    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return min(len(text.split()), max_fallback)

    try:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(token_ids)
    except Exception as exc:  # pragma: no cover - tokenizer edge cases
        logger.debug("Failed to estimate token length for semantic model: %s", exc)
        return min(len(text.split()), max_fallback)


def _texts_fit_encoder(encoder, texts: Sequence[str]) -> bool:
    max_length = _resolve_max_sequence_length(encoder)
    tokenizer = getattr(encoder, "tokenizer", None)

    for text in texts:
        length = _estimate_token_length(text, tokenizer, max_fallback=max_length * 2)
        if length > max_length:
            logger.info(
                "Skipping semantic embeddings: text length %s tokens exceeds encoder limit %s",
                length,
                max_length,
            )
            return False

    return True


def score_similarity(
    original: str,
    candidate: str,
    model_name: str,
    *,
    model_type: str = "semantic_guard",
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
) -> Optional[float]:
    """Return cosine similarity between two prompts using a cached encoder."""

    if not original or not candidate:
        return 0.0

    if np is None:
        logger.warning("NumPy not available; semantic similarity metrics disabled")
        return None

    embeddings = encode_texts(
        [original, candidate],
        model_name,
        model_type=model_type,
        embedding_cache=embedding_cache,
    )
    if not embeddings or len(embeddings) != 2:
        return None

    original_vector, candidate_vector = embeddings
    if len(original) + len(candidate) >= _INT8_SIMILARITY_THRESHOLD:
        quantized = np.vstack([original_vector, candidate_vector]) * _INT8_SCALE
        quantized = np.clip(quantized, -127, 127).astype(np.int8)
        similarity = float(np.dot(quantized[0], quantized[1])) / (_INT8_SCALE**2)
    else:
        similarity = float(np.dot(original_vector, candidate_vector))
    return max(min(similarity, 1.0), -1.0)


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    *,
    model_type: str = "semantic_guard",
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    shared_embeddings: Optional[bool] = None,
):
    """Return normalized embeddings for a batch of texts.

    When dependencies are unavailable the function returns ``None`` so callers
    can gracefully fall back to non-semantic strategies without attempting to
    load heavyweight models inside hot code paths.
    """

    if not texts:
        return []

    if np is None:
        logger.warning("NumPy not available; semantic chunking disabled")
        return None

    encoder = _load_encoder(model_name, model_type)
    if encoder is None:
        return None

    if not _texts_fit_encoder(encoder, texts):
        return None

    if shared_embeddings is None:
        shared_embeddings = config.SHARED_SEMANTIC_EMBEDDINGS

    missing_texts: List[str] = []
    missing_seen: Dict[str, None] = {}
    cache_hits: Dict[int, Any] = {}
    for index, text in enumerate(texts):
        key = _resolve_embedding_cache_key(
            model_type,
            model_name,
            text,
            shared_embeddings=bool(shared_embeddings),
        )
        cached = embedding_cache.get(key) if embedding_cache is not None else None
        if cached is None:
            cached = _get_cached_embedding(
                model_type,
                model_name,
                text,
                shared_embeddings=bool(shared_embeddings),
            )
        if cached is not None:
            cache_hits[index] = cached
            if shared_embeddings and model_type in _SHARED_EMBEDDING_TYPES:
                _record_shared_embedding_hit(embedding_cache)
            continue
        if text not in missing_seen:
            missing_texts.append(text)
            missing_seen[text] = None

    try:
        if missing_texts:
            batch_size = max(8, len(missing_texts))
            encoded_missing = encoder.encode(missing_texts, batch_size=batch_size)
            if encoded_missing is None:
                return None
            for text, embedding in zip(missing_texts, encoded_missing):
                key = _resolve_embedding_cache_key(
                    model_type,
                    model_name,
                    text,
                    shared_embeddings=bool(shared_embeddings),
                )
                _set_cached_embedding(
                    model_type,
                    model_name,
                    text,
                    embedding,
                    shared_embeddings=bool(shared_embeddings),
                )
                if embedding_cache is not None:
                    embedding_cache[key] = embedding
    except Exception as exc:  # pragma: no cover - runtime encoding failures
        logger.error("Failed to generate embeddings for semantic chunking: %s", exc)
        return None

    embeddings: List[Any] = []
    for index, text in enumerate(texts):
        cached = cache_hits.get(index)
        if cached is None:
            key = _resolve_embedding_cache_key(
                model_type,
                model_name,
                text,
                shared_embeddings=bool(shared_embeddings),
            )
            cached = embedding_cache.get(key) if embedding_cache is not None else None
            if cached is None:
                cached = _get_cached_embedding(
                    model_type,
                    model_name,
                    text,
                    shared_embeddings=bool(shared_embeddings),
                )
            if cached is None:
                return None
            if embedding_cache is not None:
                embedding_cache[key] = cached
            if shared_embeddings and model_type in _SHARED_EMBEDDING_TYPES:
                _record_shared_embedding_hit(embedding_cache)
        embeddings.append(cached)

    return embeddings


def encode_texts_with_plan(
    texts: Sequence[str],
    model_name: str,
    *,
    model_type: str = "semantic_guard",
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    shared_embeddings: Optional[bool] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
):
    """Resolve embeddings via a request-scoped semantic plan before encoding misses."""

    if not texts:
        return []

    plan_embeddings = None
    plan_metrics = None
    if isinstance(semantic_plan, dict):
        plan_embeddings = semantic_plan.setdefault("embedding_vectors", {})
        plan_metrics = semantic_plan.setdefault("metrics", {})

    if not isinstance(plan_embeddings, dict):
        return encode_texts(
            texts,
            model_name,
            model_type=model_type,
            embedding_cache=embedding_cache,
            shared_embeddings=shared_embeddings,
        )

    if shared_embeddings is None:
        shared_embeddings = config.SHARED_SEMANTIC_EMBEDDINGS

    resolved: List[Any] = [None] * len(texts)
    missing: List[str] = []
    missing_seen: Dict[str, None] = {}
    reuse_hits = 0

    for index, text in enumerate(texts):
        key = _resolve_embedding_cache_key(
            model_type,
            model_name,
            text,
            shared_embeddings=bool(shared_embeddings),
        )
        cached = plan_embeddings.get(key)
        if cached is not None:
            resolved[index] = cached
            reuse_hits += 1
            continue
        if text not in missing_seen:
            missing.append(text)
            missing_seen[text] = None

    if plan_metrics is not None and reuse_hits:
        plan_metrics["embedding_reuse_count"] = float(
            plan_metrics.get("embedding_reuse_count", 0.0)
        ) + float(reuse_hits)

    if not missing:
        if plan_metrics is not None:
            plan_metrics["embedding_calls_saved"] = (
                float(plan_metrics.get("embedding_calls_saved", 0.0)) + 1.0
            )
            avg_ms = float(plan_metrics.get("avg_encode_call_ms", 0.0) or 0.0)
            if avg_ms > 0:
                plan_metrics["embedding_wall_clock_savings_ms"] = (
                    float(plan_metrics.get("embedding_wall_clock_savings_ms", 0.0))
                    + avg_ms
                )
        return resolved

    start = time.perf_counter()
    encoded_missing = encode_texts(
        missing,
        model_name,
        model_type=model_type,
        embedding_cache=embedding_cache,
        shared_embeddings=shared_embeddings,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if encoded_missing is None:
        return None

    for text, embedding in zip(missing, encoded_missing):
        key = _resolve_embedding_cache_key(
            model_type,
            model_name,
            text,
            shared_embeddings=bool(shared_embeddings),
        )
        plan_embeddings[key] = embedding

    if plan_metrics is not None:
        total_ms = (
            float(plan_metrics.get("observed_encode_call_ms_total", 0.0)) + elapsed_ms
        )
        total_calls = float(plan_metrics.get("observed_encode_call_count", 0.0)) + 1.0
        plan_metrics["observed_encode_call_ms_total"] = total_ms
        plan_metrics["observed_encode_call_count"] = total_calls
        plan_metrics["avg_encode_call_ms"] = total_ms / max(total_calls, 1.0)

    for index, text in enumerate(texts):
        if resolved[index] is not None:
            continue
        key = _resolve_embedding_cache_key(
            model_type,
            model_name,
            text,
            shared_embeddings=bool(shared_embeddings),
        )
        resolved[index] = plan_embeddings.get(key)
        if resolved[index] is None:
            return None

    return resolved


def warm_up(
    model_name: Optional[str] = None, *, model_type: str = "semantic_guard"
) -> None:
    """Pre-load the sentence-transformer model to avoid cold-start latency.

    This helper triggers model download and initialization during application
    startup rather than on first API request. Should be called from the main
    optimizer warm_up routine.

    Args:
        model_name: Name of the sentence-transformer model to pre-load.
            Defaults to the model_inventory entry for ``model_type``.
        model_type: Model inventory type to resolve cached artifacts.
    """
    try:
        if model_name is None:
            from services.model_cache_manager import get_model_configs

            configs = get_model_configs()
            model_entry = configs.get(model_type)
            if not model_entry or not model_entry.get("model_name"):
                logger.error(
                    "Semantic warm-up skipped: missing '%s' entry in Model Inventory. "
                    "Add the entry before enabling this feature.",
                    model_type,
                )
                return
            model_name = model_entry["model_name"]

        encoder = _load_encoder(model_name, model_type)
        if encoder is not None:
            logger.info("Semantic ONNX model '%s' loaded during warm-up", model_name)
        else:
            logger.warning(
                "Failed to load semantic ONNX model '%s' during warm-up",
                model_name,
            )
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.warning(
            "Semantic ONNX warm-up failed for model '%s': %s", model_name, exc
        )
