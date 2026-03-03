"""Token classification compression helpers."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.model_cache_manager import (
    get_model_configs,
    resolve_cached_model_artifact,
    resolve_cached_model_path,
    resolve_tokenizer_root_from_artifact,
)

from . import config
from . import entropy as _entropy

logger = logging.getLogger(__name__)

_TOKEN_CLASSIFIER_STRIDE = 64
_TOKEN_CLASSIFIER_MODEL_NAME: Optional[str] = None
_TOKEN_CLASSIFIER_SHADOW_MODEL_ENV = "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_SHADOW_MODEL"
_STAGE1_ENTROPY_LITE_KEEP_THRESHOLD = 0.62

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - environment dependent
    ort = None
    AutoTokenizer = None

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModelForTokenClassification
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    AutoModelForTokenClassification = None


@dataclass(frozen=True)
class TokenDecision:
    start: int
    end: int
    keep: bool


@dataclass(frozen=True)
class CombinedDecisionMetadata:
    decisions: int
    combined_keep_ratio: float
    entropy_high_ratio: float
    entropy_high_threshold: float
    keep_threshold: float
    classifier_weight: float
    entropy_weight: float


def _summarize_decisions(
    decisions: Sequence[TokenDecision],
) -> Dict[str, float | int]:
    removals = sum(1 for decision in decisions if not decision.keep)
    keep_ratio = 1.0 - (removals / max(len(decisions), 1))
    return {
        "keep_ratio": keep_ratio,
        "decisions": len(decisions),
        "removals": removals,
    }


def _compare_decisions(
    active: Sequence[TokenDecision], shadow: Sequence[TokenDecision]
) -> Optional[Dict[str, float | int]]:
    if not active or not shadow:
        return None
    active_map = {(decision.start, decision.end): decision.keep for decision in active}
    shadow_map = {(decision.start, decision.end): decision.keep for decision in shadow}
    shared = set(active_map).intersection(shadow_map)
    if not shared:
        return None
    agreements = sum(1 for key in shared if active_map.get(key) == shadow_map.get(key))
    disagreements = len(shared) - agreements
    agreement_ratio = agreements / max(len(shared), 1)
    return {
        "agreement_ratio": agreement_ratio,
        "disagreements": disagreements,
        "shared_decisions": len(shared),
    }


def _resolve_token_classifier_model_name(model_name: Optional[str]) -> Optional[str]:
    """Resolve the model name from inventory unless overridden explicitly."""
    if model_name:
        return model_name

    configs = get_model_configs()
    entry = configs.get("token_classifier")
    if not entry or not entry.get("model_name"):
        logger.error(
            "Token classifier disabled: missing 'token_classifier' entry in Model Inventory."
            " Add the entry before enabling this feature."
        )
        return None

    return entry["model_name"]


class _TokenClassifierModel:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._session = None
        self._torch_model = None
        self._runtime: Optional[str] = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._available = False
        self._load()

    def _load(self) -> None:
        if AutoTokenizer is None:
            logger.debug("transformers unavailable; token classifier disabled")
            return

        preferred = "model.int8.onnx" if config.ONNX_USE_INT8 else "model.onnx"
        onnx_path = resolve_cached_model_artifact(
            "token_classifier", self._model_name, preferred
        )
        if onnx_path is not None and ort is not None and np is not None:
            model_path = resolve_tokenizer_root_from_artifact(onnx_path)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True
                )
                self._session = ort.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
                self._input_names = [item.name for item in self._session.get_inputs()]
                self._runtime = "onnx"
                self._available = True
                logger.debug("Loaded token classifier ONNX model %s", self._model_name)
                return
            except Exception as exc:  # pragma: no cover - runtime failures
                logger.warning(
                    "Failed to load token classifier ONNX model %s: %s",
                    self._model_name,
                    exc,
                )

        if AutoModelForTokenClassification is None or torch is None:
            logger.warning(
                "Token classifier model %s requires either ONNX artifacts or transformers+torch token classification runtime",
                self._model_name,
            )
            self._session = None
            self._torch_model = None
            self._available = False
            return

        model_path = resolve_cached_model_path("token_classifier", self._model_name)
        if model_path is None:
            logger.warning(
                "Token classifier model cache missing for %s",
                self._model_name,
            )
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
            logger.debug(
                "Loaded token classifier transformers model %s", self._model_name
            )
        except Exception as exc:
            logger.warning(
                "Failed to load token classifier model %s: %s",
                self._model_name,
                exc,
            )
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

    def _max_length(self) -> Optional[int]:
        if self._tokenizer is None:
            return None
        max_length = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(max_length, int) and max_length > 0:
            return max_length
        return None

    def _resolve_stride(
        self,
        text: str,
        *,
        max_length: int,
        content_profile_name: Optional[str],
    ) -> int:
        base_stride = min(_TOKEN_CLASSIFIER_STRIDE, max(0, max_length // 4))
        if base_stride <= 0:
            return 0

        words = re.findall(r"\b\w+\b", text.lower())
        token_count = len(words)
        unique_ratio = len(set(words)) / max(token_count, 1)
        repetition_ratio = 1.0 - unique_ratio
        code_like_hits = len(
            re.findall(
                r"[`{}\[\]()=;:#<>]|\b(def|class|import|return|if|for|while)\b", text
            )
        )
        directive_hits = len(
            re.findall(
                r"\b(must|should|ensure|required|do not|never|always|only)\b",
                text.lower(),
            )
        )
        density = (code_like_hits + directive_hits) / max(token_count, 1)

        profile_name = (content_profile_name or "").lower()
        prose_profiles = {
            "general_prose",
            "heavy_document",
            "technical_doc",
            "markdown",
        }
        dense_profiles = {"code", "json", "dialogue"}
        if (
            profile_name in prose_profiles
            and repetition_ratio >= 0.35
            and density < 0.12
        ) or (repetition_ratio >= 0.45 and density < 0.1):
            return max(
                base_stride,
                min(max(1, max_length // 2), max(base_stride + 1, base_stride * 2)),
            )
        if profile_name in dense_profiles or density >= 0.18:
            return max(16, base_stride // 2)
        return base_stride

    @staticmethod
    def _segment_entropy_lite(token: str) -> float:
        if not token:
            return 0.0
        token_len = len(token)
        unique_chars = len(set(token)) / token_len
        symbol_ratio = len(re.findall(r"[^A-Za-z0-9_]", token)) / token_len
        digit_ratio = len(re.findall(r"\d", token)) / token_len
        uppercase_ratio = len(re.findall(r"[A-Z]", token)) / token_len
        repeated_char_ratio = max(0.0, 1.0 - unique_chars)
        raw_score = (
            0.35 * unique_chars
            + 0.3 * symbol_ratio
            + 0.2 * digit_ratio
            + 0.15 * uppercase_ratio
            - 0.2 * repeated_char_ratio
        )
        return max(0.0, min(1.0, raw_score))

    def _stage1_prefilter(
        self,
        text: str,
        *,
        protected_ranges: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float]:
        guaranteed_keep: List[Tuple[int, int]] = []
        candidate_drop: List[Tuple[int, int]] = []
        segments = list(re.finditer(r"\S+", text))

        for segment in segments:
            span = (segment.start(), segment.end())
            token = segment.group(0)
            entropy_lite = self._segment_entropy_lite(token)
            protected = bool(protected_ranges) and _entropy._range_overlaps(
                span, protected_ranges
            )
            has_code_symbols = bool(re.search(r"[`{}\[\]()=;:#<>]|::|->|==", token))
            has_directive = bool(
                re.search(
                    r"\b(must|should|ensure|required|never|always|only|import|return|def|class)\b",
                    token.lower(),
                )
            )
            keep = (
                protected
                or has_code_symbols
                or has_directive
                or entropy_lite >= _STAGE1_ENTROPY_LITE_KEEP_THRESHOLD
            )
            if keep:
                guaranteed_keep.append(span)
            else:
                candidate_drop.append(span)

        skip_ratio = len(guaranteed_keep) / max(len(segments), 1)
        return guaranteed_keep, candidate_drop, skip_ratio

    @staticmethod
    def _candidate_windows(
        candidate_drop: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not candidate_drop:
            return []
        windows: List[Tuple[int, int]] = []
        start, end = candidate_drop[0]
        for next_start, next_end in candidate_drop[1:]:
            if next_start - end <= 2:
                end = next_end
                continue
            windows.append((start, end))
            start, end = next_start, next_end
        windows.append((start, end))
        return windows

    def _predict_keep_offsets(
        self,
        text: str,
        *,
        min_confidence: Optional[float],
        stride: int,
    ) -> Dict[Tuple[int, int], bool]:
        assert self._tokenizer is not None
        max_length = self._max_length() or 512
        if self._runtime == "onnx":
            assert self._session is not None
            assert np is not None
            try:
                encoded = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="np",
                    truncation=True,
                    max_length=max_length,
                    stride=stride,
                    return_overflowing_tokens=True,
                )
            except Exception as exc:  # pragma: no cover - tokenizer failures
                logger.debug("Token classification tokenization failed: %s", exc)
                return {}

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            offsets = encoded.get("offset_mapping")
            if offsets is None:
                return {}

            try:
                inputs: Dict[str, Any] = {}
                for name in self._input_names:
                    value = encoded.get(name)
                    if value is None and name == "token_type_ids":
                        value = np.zeros_like(encoded["input_ids"])
                    if value is not None:
                        inputs[name] = value.astype("int64")
                if not inputs:
                    return {}
                outputs = self._session.run(None, inputs)
            except Exception as exc:  # pragma: no cover - runtime issues
                logger.debug("Token classification inference failed: %s", exc)
                return {}

            if not outputs:
                return {}
            logits = outputs[0]
            predictions = np.argmax(logits, axis=-1)
            probs = None
            if min_confidence is not None:
                logits_max = np.max(logits, axis=-1, keepdims=True)
                exp_scores = np.exp(logits - logits_max)
                probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        elif self._runtime == "transformers":
            assert self._torch_model is not None
            assert torch is not None
            try:
                encoded = self._tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    stride=stride,
                    return_overflowing_tokens=True,
                )
            except Exception as exc:
                logger.debug("Token classification tokenization failed: %s", exc)
                return {}

            offsets = encoded.get("offset_mapping")
            if offsets is None:
                return {}
            model_inputs = {
                key: value
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            try:
                with torch.no_grad():
                    outputs = self._torch_model(**model_inputs)
            except Exception as exc:
                logger.debug("Token classification inference failed: %s", exc)
                return {}
            logits_t = outputs.logits.detach().cpu()
            predictions_t = torch.argmax(logits_t, dim=-1)
            probs_t = (
                torch.softmax(logits_t, dim=-1) if min_confidence is not None else None
            )
            predictions = predictions_t.numpy()
            probs = probs_t.numpy() if probs_t is not None else None
            input_ids = encoded["input_ids"].detach().cpu().numpy()
            attention_mask = encoded["attention_mask"].detach().cpu().numpy()
            offsets = offsets.detach().cpu().numpy()
        else:
            return {}

        decisions: Dict[Tuple[int, int], bool] = {}
        batch_size, seq_len = predictions.shape
        for batch_index in range(batch_size):
            for token_index in range(seq_len):
                if attention_mask[batch_index, token_index] == 0:
                    continue
                offset = offsets[batch_index, token_index]
                if hasattr(offset, "tolist"):
                    start, end = offset.tolist()
                else:
                    start, end = offset
                if end <= start:
                    continue
                token_id = int(input_ids[batch_index, token_index])
                if token_id in self._tokenizer.all_special_ids:
                    continue

                keep = bool(predictions[batch_index, token_index] == 1)
                if probs is not None and keep:
                    confidence = float(probs[batch_index, token_index, 1])
                    if confidence < min_confidence:
                        keep = False
                key = (start, end)
                previous = decisions.get(key)
                decisions[key] = keep if previous is None else previous or keep
        return decisions

    def classify(
        self,
        text: str,
        *,
        protected_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        min_confidence: Optional[float] = None,
        content_profile_name: Optional[str] = None,
    ) -> Tuple[List[TokenDecision], Dict[str, float]]:
        if not text.strip() or not self.available:
            return [], {
                "stage1_skip_ratio": 0.0,
                "stage1_candidate_ratio": 0.0,
                "adaptive_stride": 0.0,
            }

        assert self._tokenizer is not None
        protected = _entropy._merge_ranges(protected_ranges or [])
        guaranteed_keep, candidate_drop, skip_ratio = self._stage1_prefilter(
            text,
            protected_ranges=protected,
        )
        max_length = self._max_length() or 512
        stride = self._resolve_stride(
            text,
            max_length=max_length,
            content_profile_name=content_profile_name,
        )

        classified_offsets: Dict[Tuple[int, int], bool] = {}
        for start, end in self._candidate_windows(candidate_drop):
            span_text = text[start:end]
            decisions = self._predict_keep_offsets(
                span_text,
                min_confidence=min_confidence,
                stride=stride,
            )
            for (local_start, local_end), keep in decisions.items():
                global_key = (start + local_start, start + local_end)
                previous = classified_offsets.get(global_key)
                classified_offsets[global_key] = (
                    keep if previous is None else previous or keep
                )

        segments = list(re.finditer(r"\S+", text))
        results: List[TokenDecision] = []
        guaranteed_set = set(guaranteed_keep)
        for segment in segments:
            span = (segment.start(), segment.end())
            if span in guaranteed_set or (
                protected and _entropy._range_overlaps(span, protected)
            ):
                results.append(TokenDecision(start=span[0], end=span[1], keep=True))
                continue

            token_kept = False
            for (start, end), keep in classified_offsets.items():
                if start < span[1] and end > span[0] and keep:
                    token_kept = True
                    break
            results.append(TokenDecision(start=span[0], end=span[1], keep=token_kept))
        metadata: Dict[str, float] = {
            "stage1_skip_ratio": skip_ratio,
            "stage1_candidate_ratio": len(candidate_drop) / max(len(segments), 1),
            "adaptive_stride": float(stride),
        }
        return results, metadata


_CLASSIFIER: Optional[_TokenClassifierModel] = None
_SHADOW_CLASSIFIER: Optional[_TokenClassifierModel] = None
_SHADOW_CLASSIFIER_MODEL_NAME: Optional[str] = None


def _get_classifier(
    model_name: Optional[str] = None,
) -> Optional[_TokenClassifierModel]:
    global _CLASSIFIER, _TOKEN_CLASSIFIER_MODEL_NAME

    resolved_name = _resolve_token_classifier_model_name(model_name)
    if not resolved_name:
        return None

    if _CLASSIFIER is None or resolved_name != _CLASSIFIER._model_name:
        _CLASSIFIER = _TokenClassifierModel(resolved_name)

    _TOKEN_CLASSIFIER_MODEL_NAME = resolved_name
    return _CLASSIFIER


def _resolve_shadow_model_name() -> Optional[str]:
    shadow = os.environ.get(_TOKEN_CLASSIFIER_SHADOW_MODEL_ENV)
    if shadow:
        return shadow.strip() or None
    return None


def _get_shadow_classifier(
    model_name: Optional[str] = None,
) -> Optional[_TokenClassifierModel]:
    global _SHADOW_CLASSIFIER, _SHADOW_CLASSIFIER_MODEL_NAME

    resolved_name = model_name or _resolve_shadow_model_name()
    if not resolved_name:
        return None

    if _SHADOW_CLASSIFIER is None or resolved_name != _SHADOW_CLASSIFIER_MODEL_NAME:
        _SHADOW_CLASSIFIER = _TokenClassifierModel(resolved_name)
        _SHADOW_CLASSIFIER_MODEL_NAME = resolved_name
    return _SHADOW_CLASSIFIER


def reset_token_classifier() -> None:
    global _CLASSIFIER, _SHADOW_CLASSIFIER, _SHADOW_CLASSIFIER_MODEL_NAME
    _CLASSIFIER = None
    _SHADOW_CLASSIFIER = None
    _SHADOW_CLASSIFIER_MODEL_NAME = None
    global _TOKEN_CLASSIFIER_MODEL_NAME
    _TOKEN_CLASSIFIER_MODEL_NAME = None


def _resolve_combined_weights() -> Tuple[float, float, float, float]:
    classifier_weight = max(
        0.0, float(config.TOKEN_CLASSIFIER_COMBINED_CLASSIFIER_WEIGHT)
    )
    entropy_weight = max(0.0, float(config.TOKEN_CLASSIFIER_COMBINED_ENTROPY_WEIGHT))
    keep_threshold = max(
        0.0, min(1.0, float(config.TOKEN_CLASSIFIER_COMBINED_KEEP_THRESHOLD))
    )
    entropy_threshold = max(
        0.0, float(config.TOKEN_CLASSIFIER_COMBINED_ENTROPY_HIGH_THRESHOLD)
    )
    return classifier_weight, entropy_weight, keep_threshold, entropy_threshold


def _entropy_high_ranges(
    text: str,
    *,
    protected_ranges: Optional[Sequence[Tuple[int, int]]],
    threshold: float,
) -> List[Tuple[int, int]]:
    protected = _entropy._merge_ranges(protected_ranges or [])
    scorer = _entropy._get_scorer()
    if not scorer.available:
        raise RuntimeError(
            "Entropy scorer unavailable for token-classifier combined decisioning."
        )
    scores = scorer.score_tokens(text, skip_ranges=protected)

    high_ranges = [
        (score.start, score.end)
        for score in scores
        if score.entropy >= threshold
        and not _entropy._range_overlaps((score.start, score.end), protected)
    ]
    return _entropy._merge_ranges(high_ranges)


def _combine_token_decisions(
    text: str,
    decisions: Sequence[TokenDecision],
    *,
    protected_ranges: Optional[Sequence[Tuple[int, int]]],
) -> Tuple[List[TokenDecision], CombinedDecisionMetadata]:
    (
        classifier_weight,
        entropy_weight,
        keep_threshold,
        entropy_threshold,
    ) = _resolve_combined_weights()
    protected = _entropy._merge_ranges(protected_ranges or [])
    entropy_ranges = _entropy_high_ranges(
        text, protected_ranges=protected, threshold=entropy_threshold
    )
    combined: List[TokenDecision] = []
    entropy_high_count = 0
    keep_count = 0

    total_weight = classifier_weight + entropy_weight
    for decision in decisions:
        span = (decision.start, decision.end)
        if protected and _entropy._range_overlaps(span, protected):
            combined_keep = True
        else:
            entropy_high = bool(entropy_ranges) and _entropy._range_overlaps(
                span, entropy_ranges
            )
            if entropy_high:
                entropy_high_count += 1
            if total_weight <= 0.0:
                combined_keep = decision.keep
            else:
                classifier_score = 1.0 if decision.keep else 0.0
                entropy_score = 1.0 if entropy_high else 0.0
                combined_score = (
                    classifier_weight * classifier_score
                    + entropy_weight * entropy_score
                ) / total_weight
                combined_keep = combined_score >= keep_threshold

        if combined_keep:
            keep_count += 1
        combined.append(
            TokenDecision(start=decision.start, end=decision.end, keep=combined_keep)
        )

    decisions_count = len(decisions)
    combined_keep_ratio = keep_count / decisions_count if decisions_count else 1.0
    entropy_high_ratio = (
        entropy_high_count / decisions_count if decisions_count else 0.0
    )
    metadata = CombinedDecisionMetadata(
        decisions=decisions_count,
        combined_keep_ratio=combined_keep_ratio,
        entropy_high_ratio=entropy_high_ratio,
        entropy_high_threshold=entropy_threshold,
        keep_threshold=keep_threshold,
        classifier_weight=classifier_weight,
        entropy_weight=entropy_weight,
    )
    return combined, metadata


def compress_with_token_classifier(
    text: str,
    *,
    protected_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    model_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_keep_ratio: Optional[float] = None,
    content_profile_name: Optional[str] = None,
) -> Tuple[str, bool, Dict[str, float]]:
    classifier = _get_classifier(model_name)
    if classifier is None or not classifier.available:
        raise RuntimeError(
            "Token classifier is unavailable; strict mode requires token_classifier."
        )

    decisions, stage1_metadata = classifier.classify(
        text,
        protected_ranges=protected_ranges,
        min_confidence=min_confidence,
        content_profile_name=content_profile_name,
    )
    if not decisions:
        return (
            text,
            False,
            {
                "keep_ratio": 1.0,
                "decisions": 0,
                "removals": 0,
                **stage1_metadata,
                "compression_delta": 0.0,
            },
        )

    combined_decisions, combined_metadata = _combine_token_decisions(
        text, decisions, protected_ranges=protected_ranges
    )
    removals = [
        (decision.start, decision.end)
        for decision in combined_decisions
        if not decision.keep
    ]
    if not removals:
        return (
            text,
            False,
            {
                "keep_ratio": combined_metadata.combined_keep_ratio,
                "decisions": len(decisions),
                "removals": 0,
                "combined_keep_ratio": combined_metadata.combined_keep_ratio,
                "entropy_high_ratio": combined_metadata.entropy_high_ratio,
                "combined_entropy_high_threshold": combined_metadata.entropy_high_threshold,
                "combined_keep_threshold": combined_metadata.keep_threshold,
                "combined_classifier_weight": combined_metadata.classifier_weight,
                "combined_entropy_weight": combined_metadata.entropy_weight,
                **stage1_metadata,
                "compression_delta": 0.0,
            },
        )

    keep_ratio = combined_metadata.combined_keep_ratio
    if min_keep_ratio is not None and keep_ratio < min_keep_ratio:
        return (
            text,
            False,
            {
                "keep_ratio": keep_ratio,
                "decisions": len(decisions),
                "removals": len(removals),
                "combined_keep_ratio": combined_metadata.combined_keep_ratio,
                "entropy_high_ratio": combined_metadata.entropy_high_ratio,
                "combined_entropy_high_threshold": combined_metadata.entropy_high_threshold,
                "combined_keep_threshold": combined_metadata.keep_threshold,
                "combined_classifier_weight": combined_metadata.classifier_weight,
                "combined_entropy_weight": combined_metadata.entropy_weight,
                **stage1_metadata,
                "compression_delta": 0.0,
            },
        )

    merged = _entropy._merge_ranges(removals)
    rebuilt: List[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            rebuilt.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        rebuilt.append(text[cursor:])

    result = _entropy._normalize_spacing("".join(rebuilt))
    compression_delta = (len(text) - len(result)) / max(len(text), 1)
    return (
        result,
        True,
        {
            "keep_ratio": keep_ratio,
            "decisions": len(decisions),
            "removals": len(removals),
            "combined_keep_ratio": combined_metadata.combined_keep_ratio,
            "entropy_high_ratio": combined_metadata.entropy_high_ratio,
            "combined_entropy_high_threshold": combined_metadata.entropy_high_threshold,
            "combined_keep_threshold": combined_metadata.keep_threshold,
            "combined_classifier_weight": combined_metadata.classifier_weight,
            "combined_entropy_weight": combined_metadata.entropy_weight,
            **stage1_metadata,
            "compression_delta": compression_delta,
        },
    )


def evaluate_shadow_classifier(
    text: str,
    *,
    protected_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    min_confidence: Optional[float] = None,
    min_keep_ratio: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    shadow_name = _resolve_shadow_model_name()
    if not shadow_name:
        return None

    classifier = _get_shadow_classifier(shadow_name)
    if classifier is None or not classifier.available:
        return {"model_name": shadow_name, "available": False}

    decisions, stage1_metadata = classifier.classify(
        text,
        protected_ranges=protected_ranges,
        min_confidence=min_confidence,
    )
    if not decisions:
        return {
            "model_name": shadow_name,
            "available": True,
            "keep_ratio": 1.0,
            "decisions": 0,
            "removals": 0,
            "applied": False,
            **stage1_metadata,
            "compression_delta": 0.0,
        }

    summary = _summarize_decisions(decisions)
    combined_decisions, combined_metadata = _combine_token_decisions(
        text, decisions, protected_ranges=protected_ranges
    )
    removals = sum(1 for decision in combined_decisions if not decision.keep)
    keep_ratio = combined_metadata.combined_keep_ratio
    applied = removals > 0 and (min_keep_ratio is None or keep_ratio >= min_keep_ratio)
    payload: Dict[str, Any] = {
        "model_name": shadow_name,
        "available": True,
        "keep_ratio": summary["keep_ratio"],
        "decisions": int(summary["decisions"]),
        "removals": int(summary["removals"]),
        "applied": applied,
        "combined_keep_ratio": combined_metadata.combined_keep_ratio,
        "entropy_high_ratio": combined_metadata.entropy_high_ratio,
        "combined_entropy_high_threshold": combined_metadata.entropy_high_threshold,
        "combined_keep_threshold": combined_metadata.keep_threshold,
        "combined_classifier_weight": combined_metadata.classifier_weight,
        "combined_entropy_weight": combined_metadata.entropy_weight,
        **stage1_metadata,
        "compression_delta": 0.0,
    }

    active_classifier = _get_classifier()
    if (
        active_classifier is not None
        and active_classifier.available
        and _TOKEN_CLASSIFIER_MODEL_NAME
        and _TOKEN_CLASSIFIER_MODEL_NAME != shadow_name
    ):
        active_decisions, _ = active_classifier.classify(
            text,
            protected_ranges=protected_ranges,
            min_confidence=min_confidence,
        )
        active_summary = _summarize_decisions(active_decisions)
        comparison = _compare_decisions(active_decisions, decisions)
        payload["comparison"] = {
            "active_model_name": _TOKEN_CLASSIFIER_MODEL_NAME,
            "active_keep_ratio": active_summary["keep_ratio"],
            "active_decisions": int(active_summary["decisions"]),
            "active_removals": int(active_summary["removals"]),
            "shadow_keep_ratio_delta": summary["keep_ratio"]
            - active_summary["keep_ratio"],
            "shadow_removals_delta": summary["removals"] - active_summary["removals"],
            "agreement_ratio": comparison["agreement_ratio"] if comparison else None,
            "disagreements": int(comparison["disagreements"]) if comparison else None,
            "shared_decisions": (
                int(comparison["shared_decisions"]) if comparison else None
            ),
        }
    return payload


def warm_up(model_name: Optional[str] = None) -> None:
    try:
        classifier = _get_classifier(model_name)
        if classifier is None:
            logger.warning(
                "Token classifier warm-up skipped: no model configured in inventory."
            )
            return

        if classifier.available:
            logger.info("Token classifier model loaded during warm-up")
        else:
            logger.warning("Token classifier model unavailable during warm-up")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Token classifier warm-up failed: %s", exc)
