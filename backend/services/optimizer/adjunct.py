"""Adjunct clause trimming using spaCy parses."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from . import entropy as _entropy


_LEADING_PUNCT = re.compile(r"^[\s,;:\-–—]+")


def _range_overlaps(span: Tuple[int, int], ranges: Sequence[Tuple[int, int]]) -> bool:
    if not ranges:
        return False
    start, end = span
    for r_start, r_end in ranges:
        if start < r_end and end > r_start:
            return True
    return False


def _in_parenthetical(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    left = text.rfind("(", 0, start)
    right = text.find(")", end)
    if left == -1 or right == -1:
        return None
    if left < start < right and left < end <= right:
        return (left, right + 1)
    return None


def _leading_span(sent_text: str, span_start: int, sent_start: int) -> bool:
    prefix = sent_text[: span_start - sent_start]
    return not _LEADING_PUNCT.sub("", prefix).strip()


def _find_clause_end(tokens, start_index: int) -> Optional[int]:
    for idx in range(start_index, len(tokens)):
        token = tokens[idx]
        if token.text in {",", ";", "—", "–", ":"}:
            return token.idx + len(token.text)
    return None


def trim_adjunct_clauses(
    text: str,
    *,
    nlp_model,
    placeholder_ranges: Sequence[Tuple[int, int]],
    allowlist_phrases: Sequence[Sequence[str]],
    allowed_deps: Sequence[str],
    negation_tokens: Sequence[str],
    condition_tokens: Sequence[str],
    modal_tokens: Sequence[str],
    token_counter: Optional[Callable[[str], int]] = None,
) -> Tuple[str, int]:
    if not text or nlp_model is None:
        return text, 0

    doc = nlp_model(text)
    if not doc or not getattr(doc, "sents", None):
        return text, 0

    removals: List[Tuple[int, int]] = []

    for sent in doc.sents:
        tokens = list(sent)
        if not tokens:
            continue
        sent_text = text[sent.start_char : sent.end_char]
        token_texts = [token.text.lower() for token in tokens]

        for phrase in allowlist_phrases:
            if not phrase:
                continue
            phrase_len = len(phrase)
            for idx in range(len(tokens) - phrase_len + 1):
                if token_texts[idx : idx + phrase_len] != list(phrase):
                    continue

                span = sent[idx : idx + phrase_len]
                span_start = span.start_char
                span_end = span.end_char
                parenthetical = _in_parenthetical(text, span_start, span_end)

                if parenthetical:
                    removal_start, removal_end = parenthetical
                else:
                    if not _leading_span(sent_text, span_start, sent.start_char):
                        continue
                    clause_end = _find_clause_end(tokens, idx + phrase_len)
                    if clause_end is None:
                        continue
                    removal_start = span_start
                    removal_end = clause_end

                if _range_overlaps((removal_start, removal_end), placeholder_ranges):
                    continue

                root = span.root
                if root is None or root.dep_ not in allowed_deps:
                    continue

                span_tokens = [
                    token.text.lower()
                    for token in doc
                    if token.idx >= removal_start and token.idx < removal_end
                ]
                if any(token in negation_tokens for token in span_tokens):
                    continue
                if any(token in condition_tokens for token in span_tokens):
                    continue
                if any(token in modal_tokens for token in span_tokens):
                    continue

                removals.append((removal_start, removal_end))
                break

    if not removals:
        return text, 0

    removals = sorted(set(removals), key=lambda item: item[0])
    merged: List[Tuple[int, int]] = []
    for start, end in removals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    updated = text
    for start, end in reversed(merged):
        updated = updated[:start] + updated[end:]
    updated = _entropy._normalize_spacing(updated)

    if token_counter is None:
        return updated, 0
    removed_tokens = max(token_counter(text) - token_counter(updated), 0)
    return updated, removed_tokens
