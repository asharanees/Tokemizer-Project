"""Named entity aliasing helpers."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .placeholders import span_overlaps_placeholder

_TITLE_CASE_PATTERN = re.compile(
    r"\b(?:[A-Z][\w&+.\-]*\s+){1,4}[A-Z][\w&+.\-]*\b"
)


def _normalize_entity(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _collect_spacy_entities(doc, *, allowed_labels: Sequence[str]) -> List[Tuple[str, int, int]]:
    entities: List[Tuple[str, int, int]] = []
    if not doc or not getattr(doc, "ents", None):
        return entities
    for ent in doc.ents:
        if allowed_labels and ent.label_ not in allowed_labels:
            continue
        if ent.text:
            entities.append((ent.text.strip(), ent.start_char, ent.end_char))
    return entities


def _collect_regex_entities(text: str) -> List[Tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in _TITLE_CASE_PATTERN.finditer(text)]


def alias_named_entities(
    text: str,
    *,
    nlp_model,
    placeholder_ranges: Sequence[Tuple[int, int]],
    token_counter: Callable[[str], int],
    min_occurrences: int,
    min_chars: int,
    max_aliases: int,
    alias_prefix: str,
    allowed_labels: Sequence[str],
    reserved_tokens: Optional[Sequence[str]] = None,
) -> Tuple[str, bool, Optional[List[Tuple[str, str]]], int]:
    if not text:
        return text, False, None, 0

    entities: List[Tuple[str, int, int]] = []
    if nlp_model is not None and "ner" in getattr(nlp_model, "pipe_names", []):
        try:
            doc = nlp_model(text)
            entities = _collect_spacy_entities(doc, allowed_labels=allowed_labels)
        except Exception:
            entities = []
    if not entities:
        entities = _collect_regex_entities(text)

    if not entities:
        return text, False, None, 0

    groups: Dict[str, List[Tuple[int, int, str]]] = {}
    for value, start, end in entities:
        if len(value) < min_chars:
            continue
        if "__" in value or "TOON" in value:
            continue
        if span_overlaps_placeholder(
            type("Span", (), {"start_char": start, "end_char": end})(),
            list(placeholder_ranges),
        ):
            continue
        key = _normalize_entity(value)
        groups.setdefault(key, []).append((start, end, value))

    candidates: List[Tuple[str, List[Tuple[int, int, str]]]] = []
    for key, spans in groups.items():
        if len(spans) >= min_occurrences:
            candidates.append((key, spans))

    if not candidates:
        return text, False, None, 0

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    reserved = {token.lower() for token in (reserved_tokens or [])}
    legend_entries: List[Tuple[str, str]] = []
    total_saved = 0
    selected: List[Tuple[str, List[Tuple[int, int, str]], str, str]] = []
    current_net = 0

    for key, spans in candidates:
        if len(selected) >= max_aliases:
            break
        canonical = spans[0][2]
        original_tokens = token_counter(canonical)
        alias_tag = f"{alias_prefix}{len(selected) + 1}"
        alias_lower = alias_tag.lower()
        if alias_lower in reserved:
            continue
        if re.search(rf"\b{re.escape(alias_tag)}\b", text, flags=re.IGNORECASE):
            continue
        alias_tokens = token_counter(alias_tag)
        saved = (len(spans) - 1) * (original_tokens - alias_tokens)
        if saved <= 0:
            continue
        tentative_entries = legend_entries + [(alias_tag, canonical)]
        legend = "Aliases: " + ", ".join(
            f"{alias}={value}" for alias, value in tentative_entries
        )
        legend_cost = token_counter(legend)
        tentative_total = total_saved + saved
        tentative_net = tentative_total - legend_cost
        if tentative_net <= current_net:
            continue
        selected.append((key, spans, canonical, alias_tag))
        legend_entries.append((alias_tag, canonical))
        total_saved = tentative_total
        current_net = tentative_net

    if not selected:
        return text, False, None, 0

    replacements: List[Tuple[int, int, str]] = []
    for _key, spans, _canonical, alias_tag in selected:
        spans_sorted = sorted(spans, key=lambda item: item[0])
        for start, end, _value in spans_sorted[1:]:
            replacements.append((start, end, alias_tag))

    if not replacements:
        return text, False, None, 0

    replacements.sort(key=lambda item: item[0], reverse=True)
    filtered: List[Tuple[int, int, str]] = []
    last_start = len(text) + 1
    for start, end, alias_tag in replacements:
        if end > last_start:
            continue
        filtered.append((start, end, alias_tag))
        last_start = start

    updated = text
    for start, end, alias_tag in filtered:
        updated = updated[:start] + alias_tag + updated[end:]

    return updated, True, legend_entries, current_net
