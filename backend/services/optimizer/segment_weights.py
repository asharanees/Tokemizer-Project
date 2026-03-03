from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config


def resolve_label_weight(label: Optional[str]) -> float:
    if not label:
        return config.DISCOURSE_DEFAULT_WEIGHT

    weight = config.DISCOURSE_LABEL_WEIGHTS.get(
        label.lower(), config.DISCOURSE_DEFAULT_WEIGHT
    )
    return max(0.0, min(float(weight), 1.0))


def normalize_span_weight(span: Dict[str, Any]) -> float:
    weight_value = span.get("weight")
    if weight_value is not None:
        try:
            weight = float(weight_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive casting
            weight = resolve_label_weight(span.get("label"))
        else:
            return max(0.0, min(weight, 1.0))

    return resolve_label_weight(span.get("label"))


def clip_segment_spans(
    spans: Sequence[Dict[str, Any]], *, text_length: int
) -> List[Dict[str, Any]]:
    clipped: List[Dict[str, Any]] = []
    for span in spans:
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except (TypeError, ValueError):  # pragma: no cover - defensive casting
            continue

        if end <= start or start >= text_length:
            continue

        clipped.append(
            {
                "start": max(start, 0),
                "end": min(end, text_length),
                "label": span.get("label"),
                "weight": normalize_span_weight(span),
            }
        )

    return clipped


def weight_for_range(start: int, end: int, spans: Sequence[Dict[str, Any]]) -> float:
    if start >= end:
        return 1.0

    for span in spans:
        span_start = span.get("start", 0)
        span_end = span.get("end", 0)
        if start < span_end and end > span_start:
            try:
                weight = float(span.get("weight", 1.0))
            except (TypeError, ValueError):  # pragma: no cover - defensive casting
                return 1.0
            return max(0.0, min(weight, 1.0))

    return 1.0


def placeholder_weights_from_spans(
    text: str, spans: Sequence[Dict[str, Any]]
) -> List[float]:
    segments = config.PLACEHOLDER_PATTERN.split(text)
    segment_count = len(segments)
    if segment_count == 0:
        return []

    weights: List[float] = []
    cursor = 0
    for match in config.PLACEHOLDER_PATTERN.finditer(text):
        weights.append(weight_for_range(cursor, match.start(), spans))
        cursor = match.end()

    weights.append(weight_for_range(cursor, len(text), spans))

    if len(weights) < segment_count:
        weights.extend([1.0] * (segment_count - len(weights)))

    return weights[:segment_count]


def analyze_segment_spans(
    text: str,
    discourse_analyzer: Any,
    segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    if not text.strip():
        segments = config.PLACEHOLDER_PATTERN.split(text)
        return [], [1.0] * len(segments)

    try:
        discourse_segments = discourse_analyzer.segment(text)
    except Exception:  # pragma: no cover - discourse analysis is best effort
        discourse_segments = []

    spans: List[Dict[str, Any]] = []
    if segment_spans:
        spans.extend(segment_spans)
    cursor = 0
    for segment in discourse_segments:
        block = segment.text.strip()
        if not block:
            continue

        start = text.find(block, cursor)
        if start < 0:
            start = cursor
        end = start + len(block)
        cursor = max(end, cursor)

        spans.append(
            {
                "start": start,
                "end": end,
                "label": segment.label,
                "weight": resolve_label_weight(segment.label),
            }
        )

    spans = clip_segment_spans(spans, text_length=len(text))
    weights = placeholder_weights_from_spans(text, spans)
    return spans, weights


__all__ = [
    "analyze_segment_spans",
    "clip_segment_spans",
    "normalize_span_weight",
    "placeholder_weights_from_spans",
    "resolve_label_weight",
    "weight_for_range",
]
