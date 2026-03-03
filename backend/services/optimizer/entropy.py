"""Entropy-driven pruning helpers for the prompt optimizer pipeline."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.model_cache_manager import (
    get_model_configs,
    resolve_cached_model_artifact,
    resolve_cached_model_path,
    resolve_tokenizer_root_from_artifact,
)

logger = logging.getLogger(__name__)

try:
    _ENTROPY_TRANSFORMER_MIN_TOKENS = max(
        0, int(os.environ.get("PROMPT_OPTIMIZER_ENTROPY_TRANSFORMER_MIN_TOKENS", "0"))
    )
except (TypeError, ValueError):  # pragma: no cover - environment dependent
    _ENTROPY_TRANSFORMER_MIN_TOKENS = 512


try:  # pragma: no cover - optional dependency
    import numpy as np
    import onnxruntime as ort
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore
    ort = None  # type: ignore


try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError:  # pragma: no cover - environment dependent
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore
    F = None  # type: ignore


PLACEHOLDER_PATTERN = re.compile(r"__[^_\s]+?_\d+__")
_ENTROPY_MODEL_NAME: Optional[str] = None
_ENTROPY_FAST_MODEL_NAME: Optional[str] = None


@dataclass
class TokenEntropy:
    """Container describing surprisal measurements for a token span."""

    start: int
    end: int
    entropy: float
    confidence: Optional[float] = None


def character_entropy(text: str) -> float:
    """Compute Shannon entropy over character bigrams for the given text."""

    if not text:
        return 0.0

    cleaned = text.strip()
    if len(cleaned) < 2:
        return 0.0

    bigrams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    if not bigrams:
        return 0.0

    counts = {}
    for item in bigrams:
        counts[item] = counts.get(item, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


class _EntropyModel:
    """Lazy loader for causal language models used for surprisal scoring."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._available = False
        self._model_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if (
            AutoModelForCausalLM is None
            or AutoTokenizer is None
            or torch is None
            or F is None
        ):
            logger.debug("transformers not available; entropy pruning unavailable")
            return

        configs = get_model_configs()
        entropy_config = configs.get("entropy")
        if not entropy_config or not entropy_config.get("model_name"):
            logger.error(
                "Entropy pruning disabled: model inventory missing "
                "'entropy' entry with a valid model_name."
            )
            return

        model_name = entropy_config["model_name"]
        self._model_name = model_name

        global _ENTROPY_MODEL_NAME
        _ENTROPY_MODEL_NAME = model_name

        model_path = resolve_cached_model_path("entropy", model_name)
        if model_path is None:
            logger.warning(
                "Entropy model cache missing for %s; entropy pruning disabled",
                model_name,
            )
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True
            )
            self._model.eval()
            self._available = True
            logger.debug("Loaded entropy model %s", model_name)
        except Exception as exc:  # pragma: no cover - download/runtime failures
            logger.warning("Failed to load entropy model %s: %s", model_name, exc)
            self._model = None
            self._tokenizer = None
            self._available = False

    @property
    def available(self) -> bool:
        return (
            self._available and self._model is not None and self._tokenizer is not None
        )

    def _max_length(self) -> Optional[int]:
        if self._tokenizer is None:
            return None

        max_length = getattr(self._tokenizer, "model_max_length", None)
        return (
            int(max_length) if isinstance(max_length, int) and max_length > 0 else None
        )

    def score_tokens(
        self, text: str, skip_ranges: Optional[Sequence[Tuple[int, int]]] = None
    ) -> List[TokenEntropy]:
        """Return entropy scores for each lexical token in text.

        Args:
            text: Input text to score
            skip_ranges: Optional sequence of (start, end) character offset ranges to skip scoring

        Returns:
            List of TokenEntropy objects for scored tokens (excluding skipped ranges)
        """

        if not text.strip():
            return []

        if not self.available:
            raise RuntimeError("Entropy scorer unavailable")

        assert self._model is not None  # for mypy
        assert self._tokenizer is not None
        max_length = self._max_length()

        try:
            encoded = self._tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                return_overflowing_tokens=True,
            )
        except Exception as exc:  # pragma: no cover - tokenizer specific failure
            raise RuntimeError(f"Entropy tokenization failed: {exc}") from exc

        num_truncated_tokens = int(encoded.get("num_truncated_tokens", 0) or 0)
        overflowing = encoded.get("overflowing_tokens") or []
        if num_truncated_tokens > 0 or len(overflowing) > 0:
            raise RuntimeError("Entropy scoring text exceeds tokenizer max length")

        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise RuntimeError("Entropy tokenizer does not provide offset mapping")

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        offsets = offsets[0]

        length = int(input_ids.size(0))
        if length <= 1:
            return []

        if (
            _ENTROPY_TRANSFORMER_MIN_TOKENS > 0
            and length < _ENTROPY_TRANSFORMER_MIN_TOKENS
        ):
            return []

        try:
            with torch.no_grad():
                outputs = self._model(
                    input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
        except Exception as exc:  # pragma: no cover - runtime/device issues
            raise RuntimeError(f"Entropy model inference failed: {exc}") from exc

        logits = outputs.logits[0]
        shift_logits = logits[:-1]
        shift_labels = input_ids[1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll_values = -log_probs[torch.arange(shift_labels.size(0)), shift_labels]
        probs = log_probs.exp()
        default_nll = float(nll_values.max().item()) if nll_values.numel() else 0.0

        scores: List[TokenEntropy] = []
        for index in range(length):
            if attention_mask[index].item() == 0:
                continue

            start, end = offsets[index].tolist()
            if end <= start:
                continue

            if skip_ranges and _range_overlaps((start, end), skip_ranges):
                continue

            token_id = input_ids[index].item()
            if token_id in self._tokenizer.all_special_ids:
                continue

            if index == 0:
                nll = default_nll
                confidence = None
            else:
                nll = float(nll_values[index - 1].item())
                confidence = (
                    float(probs[index - 1, token_id].item())
                    if 0 <= token_id < probs.size(-1)
                    else None
                )

            scores.append(
                TokenEntropy(start=start, end=end, entropy=nll, confidence=confidence)
            )

        return scores


class _EntropyFastModel:
    """Fast ONNX token-level scorer that predicts drop probabilities."""

    def __init__(self) -> None:
        self._session = None
        self._torch_model = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._runtime: Optional[str] = None
        self._available = False
        self._model_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if AutoTokenizer is None:
            logger.debug("transformers unavailable; fast entropy disabled")
            return

        configs = get_model_configs()
        fast_config = configs.get("entropy_fast")
        if not fast_config or not fast_config.get("model_name"):
            logger.debug(
                "Model inventory missing 'entropy_fast'; fast entropy disabled"
            )
            return

        model_name = fast_config["model_name"]
        self._model_name = model_name
        global _ENTROPY_FAST_MODEL_NAME
        _ENTROPY_FAST_MODEL_NAME = model_name

        preferred = "model.int8.onnx"
        onnx_path = resolve_cached_model_artifact("entropy_fast", model_name, preferred)
        if onnx_path is None:
            onnx_path = resolve_cached_model_artifact(
                "entropy_fast", model_name, "model.onnx"
            )

        if onnx_path is not None and ort is not None and np is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    resolve_tokenizer_root_from_artifact(onnx_path),
                    local_files_only=True,
                )
                self._session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
                self._input_names = [item.name for item in self._session.get_inputs()]
                self._runtime = "onnx"
                self._available = True
                logger.debug("Loaded fast entropy ONNX model %s", model_name)
                return
            except Exception as exc:  # pragma: no cover - runtime failure
                logger.warning(
                    "Failed to load fast entropy ONNX model %s: %s", model_name, exc
                )

        if AutoModelForTokenClassification is None or torch is None:
            logger.warning(
                "Fast entropy model %s requires either ONNX artifacts or transformers+torch token classification runtime",
                model_name,
            )
            self._session = None
            self._torch_model = None
            self._tokenizer = None
            self._available = False
            return

        model_path = resolve_cached_model_path("entropy_fast", model_name)
        if model_path is None:
            logger.warning("Fast entropy model cache missing for %s", model_name)
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self._torch_model = AutoModelForTokenClassification.from_pretrained(
                model_path, local_files_only=True
            )
            self._torch_model.eval()
            self._runtime = "transformers"
            self._available = True
            logger.debug("Loaded fast entropy transformers model %s", model_name)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("Failed to load fast entropy model %s: %s", model_name, exc)
            self._session = None
            self._torch_model = None
            self._tokenizer = None
            self._available = False

    @property
    def available(self) -> bool:
        if not self._available or self._tokenizer is None:
            return False
        if self._runtime == "onnx":
            return self._session is not None
        if self._runtime == "transformers":
            return self._torch_model is not None
        return False

    def _max_length(self) -> int:
        if self._tokenizer is None:
            return 512
        max_length = getattr(self._tokenizer, "model_max_length", 512)
        if isinstance(max_length, int) and max_length > 0:
            return max_length
        return 512

    def _extract_drop_probs(self, output: Any) -> Optional[Any]:
        if np is None:
            return None
        array = np.asarray(output)
        if array.size == 0:
            return None
        if array.ndim == 3:
            channels = int(array.shape[-1])
            if channels >= 2:
                logits = array
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                exp_scores = np.exp(logits)
                probs = exp_scores / np.maximum(
                    np.sum(exp_scores, axis=-1, keepdims=True), 1e-8
                )
                return probs[..., 1]
            return array[..., 0]
        if array.ndim == 2:
            if np.max(array) > 1.0 or np.min(array) < 0.0:
                return 1.0 / (1.0 + np.exp(-array))
            return array
        if array.ndim == 1:
            return array.reshape(1, -1)
        return None

    def score_tokens(
        self, text: str, skip_ranges: Optional[Sequence[Tuple[int, int]]] = None
    ) -> List[TokenEntropy]:
        if not text.strip():
            return []
        if not self.available:
            raise RuntimeError("Fast entropy scorer unavailable")

        assert self._tokenizer is not None
        max_length = self._max_length()
        drop_probs: Any
        input_ids: Any
        attention_mask: Any
        offsets: Any
        if self._runtime == "onnx":
            assert self._session is not None
            assert np is not None
            try:
                encoded = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="np",
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_overflowing_tokens=True,
                )
            except Exception as exc:  # pragma: no cover - tokenizer dependent
                raise RuntimeError(f"Fast entropy tokenization failed: {exc}") from exc

            num_truncated_tokens = int(encoded.get("num_truncated_tokens", 0) or 0)
            overflowing = encoded.get("overflowing_tokens") or []
            if num_truncated_tokens > 0 or len(overflowing) > 0:
                raise RuntimeError(
                    "Fast entropy scoring text exceeds tokenizer max length"
                )

            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise RuntimeError(
                    "Fast entropy tokenizer does not provide offset mapping"
                )

            try:
                inputs: Dict[str, Any] = {}
                for name in self._input_names:
                    value = encoded.get(name)
                    if value is None and name == "token_type_ids":
                        value = np.zeros_like(encoded["input_ids"])
                    if value is not None:
                        inputs[name] = value.astype("int64")
                if not inputs:
                    raise RuntimeError("Fast entropy model inputs are empty")
                outputs = self._session.run(None, inputs)
            except Exception as exc:  # pragma: no cover - runtime issues
                raise RuntimeError(f"Fast entropy inference failed: {exc}") from exc

            if not outputs:
                raise RuntimeError("Fast entropy inference returned no outputs")
            drop_probs = self._extract_drop_probs(outputs[0])
            if drop_probs is None:
                raise RuntimeError("Fast entropy output parsing failed")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if drop_probs.shape[:2] != input_ids.shape[:2]:
                seq = min(drop_probs.shape[1], input_ids.shape[1])
                drop_probs = drop_probs[:, :seq]
                input_ids = input_ids[:, :seq]
                attention_mask = attention_mask[:, :seq]
                offsets = offsets[:, :seq]
        elif self._runtime == "transformers":
            assert self._torch_model is not None
            assert torch is not None
            assert F is not None
            try:
                encoded = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    stride=max(0, min(64, max_length // 4)),
                    return_overflowing_tokens=True,
                )
            except Exception as exc:  # pragma: no cover - tokenizer dependent
                raise RuntimeError(f"Fast entropy tokenization failed: {exc}") from exc

            num_truncated_tokens = int(encoded.get("num_truncated_tokens", 0) or 0)
            overflowing = encoded.get("overflowing_tokens") or []
            if num_truncated_tokens > 0 or len(overflowing) > 0:
                raise RuntimeError(
                    "Fast entropy scoring text exceeds tokenizer max length"
                )

            offsets = encoded.get("offset_mapping")
            if offsets is None:
                raise RuntimeError(
                    "Fast entropy tokenizer does not provide offset mapping"
                )

            model_inputs = {
                key: value
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            try:
                with torch.no_grad():
                    outputs = self._torch_model(**model_inputs)
            except Exception as exc:
                raise RuntimeError(f"Fast entropy inference failed: {exc}") from exc

            logits = outputs.logits
            if logits.ndim == 3 and logits.shape[-1] >= 2:
                probs = F.softmax(logits, dim=-1)
                drop_probs = probs[..., 1].detach().cpu().numpy()
            else:
                drop_probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            input_ids = encoded["input_ids"].detach().cpu().numpy()
            attention_mask = encoded["attention_mask"].detach().cpu().numpy()
            offsets = offsets.detach().cpu().numpy()
        else:
            raise RuntimeError("Fast entropy scorer runtime is unavailable")

        scored: Dict[Tuple[int, int], TokenEntropy] = {}
        for batch_index in range(input_ids.shape[0]):
            for token_index in range(input_ids.shape[1]):
                if int(attention_mask[batch_index, token_index]) == 0:
                    continue
                offset = offsets[batch_index, token_index]
                start, end = offset.tolist() if hasattr(offset, "tolist") else offset
                start = int(start)
                end = int(end)
                if end <= start:
                    continue
                if skip_ranges and _range_overlaps((start, end), skip_ranges):
                    continue
                token_id = int(input_ids[batch_index, token_index])
                if token_id in self._tokenizer.all_special_ids:
                    continue

                drop_probability = float(drop_probs[batch_index, token_index])
                drop_probability = max(1e-6, min(1.0 - 1e-6, drop_probability))
                keep_probability = 1.0 - drop_probability
                entropy_value = -math.log(max(keep_probability, 1e-6))
                score = TokenEntropy(
                    start=start,
                    end=end,
                    entropy=entropy_value,
                    confidence=max(drop_probability, keep_probability),
                )
                key = (start, end)
                previous = scored.get(key)
                if previous is None or score.entropy < previous.entropy:
                    scored[key] = score

        return list(scored.values())


_SCORER: Optional[_EntropyModel] = None
_FAST_SCORER: Optional[_EntropyFastModel] = None


def _get_scorer() -> _EntropyModel:
    global _SCORER
    if _SCORER is None:
        _SCORER = _EntropyModel()
    return _SCORER


def _get_fast_scorer() -> _EntropyFastModel:
    global _FAST_SCORER
    if _FAST_SCORER is None:
        _FAST_SCORER = _EntropyFastModel()
    return _FAST_SCORER


def reset_entropy_model() -> None:
    global _SCORER, _FAST_SCORER
    _SCORER = None
    _FAST_SCORER = None
    global _ENTROPY_MODEL_NAME
    _ENTROPY_MODEL_NAME = None
    global _ENTROPY_FAST_MODEL_NAME
    _ENTROPY_FAST_MODEL_NAME = None


def _placeholder_ranges(text: str) -> List[Tuple[int, int]]:
    return [
        (match.start(), match.end()) for match in PLACEHOLDER_PATTERN.finditer(text)
    ]


def _range_overlaps(span: Tuple[int, int], ranges: Sequence[Tuple[int, int]]) -> bool:
    span_start, span_end = span
    for start, end in ranges:
        if span_start < end and span_end > start:
            return True
    return False


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _expand_span(text: str, start: int, end: int) -> Tuple[int, int]:
    length = len(text)
    span_start = start
    span_end = end

    while span_start > 0 and text[span_start - 1].isspace():
        span_start -= 1

    while span_start > 0 and text[span_start - 1] not in ".!?\n":
        if text[span_start - 1].isspace():
            break
        span_start -= 1

    while span_start > 0 and text[span_start - 1].isspace():
        span_start -= 1

    while span_end < length and text[span_end : span_end + 1].isspace():
        span_end += 1

    while span_end < length and text[span_end : span_end + 1] not in ".!?\n":
        if text[span_end].isspace():
            break
        span_end += 1

    while span_end < length and text[span_end : span_end + 1] in ".!?":
        span_end += 1

    while span_end < length and text[span_end : span_end + 1].isspace():
        span_end += 1

    return max(0, span_start), min(length, span_end)


def _merge_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted(ranges)
    if not ordered:
        return []

    merged: List[Tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def prune_low_entropy(
    text: str,
    budget: int,
    *,
    protected_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    min_confidence: Optional[float] = None,
    backend_preference: str = "entropy_fast",
) -> Tuple[str, int, List[Tuple[int, int]]]:
    """Remove the lowest-entropy regions within the provided character budget."""

    if not text or budget <= 0:
        return text, 0, []

    fast_scorer = _get_fast_scorer()
    teacher_scorer = _get_scorer()
    # Compute placeholder ranges first to skip scoring them
    placeholder_ranges = _placeholder_ranges(text)
    protected = list(placeholder_ranges)
    if protected_ranges:
        protected.extend(protected_ranges)
    protected = _merge_ranges(protected)
    selected = backend_preference.strip().lower()
    token_scores: List[TokenEntropy]
    if selected in {"teacher", "entropy"}:
        if not teacher_scorer.available:
            raise RuntimeError("Required entropy backend 'entropy' is unavailable")
        token_scores = teacher_scorer.score_tokens(text, skip_ranges=protected)
    elif selected in {"fast", "entropy_fast"}:
        if not fast_scorer.available:
            raise RuntimeError("Required entropy backend 'entropy_fast' is unavailable")
        token_scores = fast_scorer.score_tokens(text, skip_ranges=protected)
    else:
        raise ValueError(
            "Unknown entropy backend preference: "
            f"{backend_preference}. Expected one of: entropy_fast, fast, entropy, teacher."
        )

    if not token_scores:
        return text, 0, []

    candidates = []
    for token in token_scores:
        if _range_overlaps((token.start, token.end), protected):
            continue
        if (
            min_confidence is not None
            and token.confidence is not None
            and token.confidence < min_confidence
        ):
            continue
        candidates.append(token)

    if not candidates:
        return text, 0, []

    candidates.sort(key=lambda item: (item.entropy, item.end - item.start))

    removals: List[Tuple[int, int]] = []
    removed = 0

    for token in candidates:
        if removed >= budget:
            break

        span = _expand_span(text, token.start, token.end)
        if _range_overlaps(span, protected):
            continue

        span_length = span[1] - span[0]
        if span_length <= 0:
            continue

        if span_length > budget:
            tight_start, tight_end = token.start, token.end
            while tight_start > 0 and text[tight_start - 1].isspace():
                tight_start -= 1
            while tight_end < len(text) and text[tight_end : tight_end + 1].isspace():
                tight_end += 1
            span = (tight_start, tight_end)
            if _range_overlaps(span, protected):
                continue
            span_length = span[1] - span[0]
            if span_length <= 0 or span_length > budget:
                continue

        if span_length + removed > budget:
            continue

        if any(start < span[1] and end > span[0] for start, end in removals):
            continue

        removals.append(span)
        removed += span_length

    if not removals:
        return text, 0, []

    merged = _merge_ranges(removals)

    rebuilt: List[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            rebuilt.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        rebuilt.append(text[cursor:])

    result = _normalize_spacing("".join(rebuilt))
    removed_chars = sum(end - start for start, end in merged)

    return result, removed_chars, merged


def warm_up() -> None:
    """Pre-load the entropy model to avoid cold-start latency.

    This helper triggers model download and initialization during application
    startup rather than on first API request. Should be called from the main
    optimizer warm_up routine.

    The model name is fixed to the configured default (see `_ENTROPY_MODEL_NAME`).
    """
    try:
        scorer = _get_scorer()
        if scorer.available:
            logger.info("Entropy model loaded during warm-up")
        else:
            logger.warning(
                "Entropy model not available during warm-up (dependencies may be missing)"
            )
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.warning("Entropy model warm-up failed: %s", exc)
