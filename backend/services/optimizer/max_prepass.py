"""Budgeted sentence/span pre-pass for maximum-mode large prompts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

_PLACEHOLDER_PATTERN = re.compile(r"__\w+_\d+__")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_CONSTRAINT_KEYWORD_PATTERN = re.compile(
    r"\b(?:must|shall|required|never|do not|don't|cannot|only|exactly)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class BudgetedPrepassConfig:
    enabled: bool
    minimum_tokens: int
    budget_ratio: float
    max_sentences: int
    budget_floor_ratio: float = 0.2
    budget_cap_ratio: float = 0.95
    adaptive_budgeting: bool = True


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _split_sentence_spans(text: str) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        start, end = match.span()
        chunk = text[start:end].strip()
        if not chunk:
            continue
        leading_ws = len(text[start:end]) - len(text[start:end].lstrip())
        trailing_ws = len(text[start:end]) - len(text[start:end].rstrip())
        normalized_start = start + leading_ws
        normalized_end = end - trailing_ws
        if normalized_end <= normalized_start:
            continue
        spans.append(
            SentenceSpan(
                start=normalized_start,
                end=normalized_end,
                text=text[normalized_start:normalized_end],
            )
        )
    if spans:
        return spans

    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    return [SentenceSpan(start=start, end=start + len(stripped), text=stripped)]


def _entropy_lite(sentence: str) -> float:
    tokens = _tokenize(sentence)
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = float(len(tokens))
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(max(probability, 1e-12))
    max_entropy = math.log(max(len(counts), 1)) if counts else 1.0
    if max_entropy <= 0:
        return 0.0
    return min(entropy / max_entropy, 1.0)


def _span_overlaps(span: Tuple[int, int], protected: Sequence[Tuple[int, int]]) -> bool:
    for start, end in protected:
        if span[0] < end and span[1] > start:
            return True
    return False


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: List[Tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        m_start, m_end = merged[-1]
        if start <= m_end:
            merged[-1] = (m_start, max(m_end, end))
            continue
        merged.append((start, end))
    return merged


def _constraint_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for span in _split_sentence_spans(text):
        if _CONSTRAINT_KEYWORD_PATTERN.search(span.text):
            ranges.append((span.start, span.end))
    return ranges


def _constraint_hit_count(text: str) -> int:
    return len(_constraint_ranges(text))


def _query_relevance(sentence_tokens: Set[str], query_tokens: Set[str]) -> float:
    if not sentence_tokens or not query_tokens:
        return 0.0
    overlap = sentence_tokens.intersection(query_tokens)
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_tokens), 1)


def budgeted_sentence_span_prepass(
    *,
    prompt: str,
    query: Optional[str],
    count_tokens: Callable[[str], int],
    protected_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    config: Optional[BudgetedPrepassConfig] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    resolved = config or BudgetedPrepassConfig(
        enabled=False,
        minimum_tokens=3500,
        budget_ratio=0.7,
        max_sentences=120,
        budget_floor_ratio=0.2,
        budget_cap_ratio=0.95,
        adaptive_budgeting=True,
    )
    if not resolved.enabled or not prompt:
        return prompt, False, {}

    total_tokens = max(count_tokens(prompt), 0)
    if total_tokens < max(resolved.minimum_tokens, 1):
        return prompt, False, {}

    sentence_spans = _split_sentence_spans(prompt)
    if len(sentence_spans) < 2:
        return prompt, False, {}

    query_tokens = set(_tokenize(query or ""))

    min_ratio = min(0.95, max(0.2, resolved.budget_floor_ratio))
    max_ratio = min(0.95, max(min_ratio, resolved.budget_cap_ratio))
    budget_ratio = min(max_ratio, max(min_ratio, resolved.budget_ratio))

    redundancy_counts: Dict[str, int] = {}
    sentence_tokens_cache: List[Set[str]] = []
    for span in sentence_spans:
        token_set = set(_tokenize(span.text))
        sentence_tokens_cache.append(token_set)
        signature = " ".join(sorted(token_set))
        redundancy_counts[signature] = redundancy_counts.get(signature, 0) + 1

    total_sentences = max(len(sentence_spans), 1)
    duplicated_sentences = sum(max(count - 1, 0) for count in redundancy_counts.values())
    redundancy_ratio = duplicated_sentences / total_sentences
    constraint_hits = _constraint_hit_count(prompt)
    constraint_density = constraint_hits / total_sentences

    adaptive_budget_ratio = budget_ratio
    if resolved.adaptive_budgeting:
        if constraint_density >= 0.2:
            adaptive_budget_ratio += 0.1
        elif constraint_density >= 0.1:
            adaptive_budget_ratio += 0.05

        if redundancy_ratio >= 0.35 and constraint_density <= 0.18:
            adaptive_budget_ratio -= 0.12
        elif redundancy_ratio >= 0.22 and constraint_density <= 0.12:
            adaptive_budget_ratio -= 0.07

    adaptive_budget_ratio = min(max_ratio, max(min_ratio, adaptive_budget_ratio))

    auto_safety_floor_ratio: Optional[float] = None
    if not query_tokens:
        auto_safety_floor_ratio = 0.5
        if constraint_density >= 0.15:
            auto_safety_floor_ratio = 0.6
        elif redundancy_ratio < 0.2:
            auto_safety_floor_ratio = 0.58
        auto_safety_floor_ratio = min(max_ratio, max(min_ratio, auto_safety_floor_ratio))
        adaptive_budget_ratio = max(adaptive_budget_ratio, auto_safety_floor_ratio)

    budget_ratio = adaptive_budget_ratio
    target_tokens = max(int(total_tokens * budget_ratio), 1)
    if target_tokens >= total_tokens:
        return prompt, False, {}

    placeholder_ranges = [
        match.span() for match in _PLACEHOLDER_PATTERN.finditer(prompt)
    ]
    hard_protected = list(protected_ranges or [])
    hard_protected.extend(placeholder_ranges)
    hard_protected.extend(_constraint_ranges(prompt))
    hard_protected = _merge_ranges(hard_protected)

    scored: List[Tuple[int, float, int, bool]] = []
    for index, span in enumerate(sentence_spans):
        token_set = sentence_tokens_cache[index]
        signature = " ".join(sorted(token_set))
        query_score = _query_relevance(token_set, query_tokens)
        redundancy_score = 1.0 / max(redundancy_counts.get(signature, 1), 1)
        entropy_score = _entropy_lite(span.text)
        span_tuple = (span.start, span.end)
        protected = _span_overlaps(span_tuple, hard_protected)
        protection_score = 1.0 if protected else 0.0

        aggregate = (
            (0.45 * query_score)
            + (0.3 * protection_score)
            + (0.15 * redundancy_score)
            + (0.1 * entropy_score)
        )
        scored.append((index, aggregate, count_tokens(span.text), protected))

    protected_indices = [item[0] for item in scored if item[3]]
    selected: List[int] = sorted(set(protected_indices))
    token_total = sum(scored[i][2] for i in selected)

    for index, _, tokens, _ in sorted(scored, key=lambda item: item[1], reverse=True):
        if index in selected:
            continue
        if len(selected) >= resolved.max_sentences:
            break
        if selected and token_total + tokens > target_tokens:
            continue
        selected.append(index)
        token_total += tokens
        if token_total >= target_tokens:
            break

    if not selected:
        best = max(scored, key=lambda item: item[1])[0]
        selected = [best]

    selected = sorted(set(selected))
    savings_ratio = (total_tokens - token_total) / max(total_tokens, 1)
    if savings_ratio < 0.08:
        return (
            prompt,
            False,
            {
                "selected_indices": selected,
                "target_tokens": target_tokens,
                "protected_indices": protected_indices,
                "resolved_budget_ratio": budget_ratio,
                "redundancy_ratio": redundancy_ratio,
                "constraint_density": constraint_density,
                "constraint_hits": constraint_hits,
                "auto_safety_floor_ratio": auto_safety_floor_ratio,
                "savings_ratio": savings_ratio,
                "skipped_reason": "low_expected_savings",
            },
        )

    rebuilt = "\n".join(
        sentence_spans[index].text.strip() for index in selected
    ).strip()
    if not rebuilt:
        return prompt, False, {}

    if rebuilt == prompt.strip():
        return (
            prompt,
            False,
            {
                "selected_indices": selected,
                "target_tokens": target_tokens,
                "protected_indices": protected_indices,
                "resolved_budget_ratio": budget_ratio,
                "redundancy_ratio": redundancy_ratio,
                "constraint_density": constraint_density,
                "constraint_hits": constraint_hits,
                "auto_safety_floor_ratio": auto_safety_floor_ratio,
            },
        )

    return (
        rebuilt,
        True,
        {
            "selected_indices": selected,
            "target_tokens": target_tokens,
            "protected_indices": protected_indices,
            "original_tokens": total_tokens,
            "selected_tokens": token_total,
            "resolved_budget_ratio": budget_ratio,
            "redundancy_ratio": redundancy_ratio,
            "constraint_density": constraint_density,
            "constraint_hits": constraint_hits,
            "auto_safety_floor_ratio": auto_safety_floor_ratio,
        },
    )
