"""
Lexical simplification helpers for prompt optimization.

This module contains all text transformation functions that perform
pattern-based lexical simplifications, including:
- Politeness phrase removal
- Verbose pattern replacement
- Redundant phrase elimination
- Synonym shortenings
- Canonicalization (long form -> short form)
- Filler word removal
- Number and unit normalization
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import (Callable, Collection, Dict, List, Literal, Match, Optional,
                    Sequence, Set, Tuple)

try:
    from quantulum3 import parser as quantulum_parser

    QUANTULUM_AVAILABLE = True
except ImportError:
    QUANTULUM_AVAILABLE = False


from ..repetition import RepetitionDetector
from . import config

_FAST_UNIT_PATTERNS = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.FAST_UNIT_PATTERNS
)

_AGGRESSIVE_PUNCTUATION_PATTERNS = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in config.AGGRESSIVE_PUNCTUATION_PATTERNS
)

WORD_PATTERN = re.compile(r"\b[\w']+\b")

# Module-level regex to detect numeric cues before invoking Quantulum parsing
_NUMERIC_CUE_PATTERN = re.compile(
    r"\d|"  # Any digit
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)

# Module-level compiled pattern for thousand separators to avoid recompilation
_THOUSAND_SEPARATOR_PATTERN = re.compile(r"\b\d{1,3}(?:[ ,_]\d{3})+(?:\.\d+)?\b")

_SPELLED_MAGNITUDE_PATTERN = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
_SPELLED_MAGNITUDE_PREFIXES = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_SPELLED_MAGNITUDE_SUFFIXES = {
    "thousand": "000",
    "million": "000000",
    "billion": "000000000",
    "trillion": "000000000000",
}


InstructionCategory = Literal["politeness", "verbose", "redundant", "format", "filler"]
InstructionRule = Tuple[re.Pattern, str]
InstructionRuleTable = Dict[InstructionCategory, Tuple[InstructionRule, ...]]

_INSTRUCTION_CATEGORY_ORDER: Tuple[InstructionCategory, ...] = (
    "politeness",
    "format",
    "verbose",
    "redundant",
    "filler",
)

# Batch politeness patterns into a single combined regex for better performance.
# Sort longest-first to reduce premature matches that would shadow longer phrases.
_POLITENESS_PATTERNS_SORTED = sorted(config.POLITENESS_PATTERNS, key=len, reverse=True)
_COMBINED_POLITENESS_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _POLITENESS_PATTERNS_SORTED),
    re.IGNORECASE,
)

_INSTRUCTION_RULES: InstructionRuleTable = {
    "politeness": ((_COMBINED_POLITENESS_PATTERN, ""),),
    "verbose": tuple(
        (re.compile(pattern, re.IGNORECASE), replacement)
        for pattern, replacement in config.VERBOSE_PATTERNS
    ),
    "redundant": tuple(
        (re.compile(pattern, re.IGNORECASE), replacement)
        for pattern, replacement in config.REDUNDANT_PHRASES
    ),
}

format_rules = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.INSTRUCTION_FORMAT_PATTERNS
)

if format_rules:
    _INSTRUCTION_RULES["format"] = format_rules
else:
    _INSTRUCTION_RULES["format"] = tuple()

if config.FILLER_WORDS:
    filler_alternatives = "|".join(
        sorted((re.escape(word) for word in config.FILLER_WORDS), key=len, reverse=True)
    )
    filler_pattern = re.compile(rf"\b(?:{filler_alternatives})\b\s*", re.IGNORECASE)
    filler_rules: List[InstructionRule] = []
    if getattr(config, "MULTIWORD_FILLER_PHRASES", None):
        phrase_alternatives = "|".join(
            sorted(
                (re.escape(phrase) for phrase in config.MULTIWORD_FILLER_PHRASES),
                key=len,
                reverse=True,
            )
        )
        filler_rules.append(
            (
                re.compile(rf"\b(?:{phrase_alternatives})\b[,\s]*", re.IGNORECASE),
                "",
            )
        )
    filler_rules.append((filler_pattern, ""))
    _INSTRUCTION_RULES["filler"] = tuple(filler_rules)
else:
    if getattr(config, "MULTIWORD_FILLER_PHRASES", None):
        phrase_alternatives = "|".join(
            sorted(
                (re.escape(phrase) for phrase in config.MULTIWORD_FILLER_PHRASES),
                key=len,
                reverse=True,
            )
        )
        multiword_pattern = re.compile(
            rf"\b(?:{phrase_alternatives})\b[,\s]*", re.IGNORECASE
        )
        _INSTRUCTION_RULES["filler"] = ((multiword_pattern, ""),)
    else:
        _INSTRUCTION_RULES["filler"] = tuple()


_CONTRACTION_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.CONTRACTION_MAP.items()
)

_BOILERPLATE_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.BOILERPLATE_PATTERNS
)

# Consecutive duplicate word/phrase removal patterns
_CONSECUTIVE_DUPLICATE_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.CONSECUTIVE_DUPLICATE_PATTERNS
)

# Paradoxical phrase collapse patterns
_PARADOXICAL_PHRASE_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.PARADOXICAL_PHRASE_PATTERNS
)

# Repeated phrase consolidation patterns
_REPEATED_PHRASE_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in config.REPEATED_PHRASE_CONSOLIDATION
)

# Word token detector used by frequency abbreviation learning.
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]+")

# Cheap detectors to skip phrase optimizations when patterns cannot match.
_CONSECUTIVE_DUP_DETECTOR = re.compile(
    r"\b(\w+(?:\s+\w+){0,3})(?:[,\s]+\1)+\b", re.IGNORECASE
)
_PARADOXICAL_PHRASE_DETECTOR = re.compile(r"\b(?:but|and)\s+also\b", re.IGNORECASE)
_REPEATED_PHRASE_DETECTOR = re.compile(
    r"(?:\band\s+and\b|\bthank you\b|\bkindly do the needful\b|"
    r"\bsummary of the summary\b|\bplan for the plan\b|"
    r"\bexplanation of the explanation\b|"
    r"\bfull (?:plan|explanation|summary|outline)\b|\bappreciate it\b)",
    re.IGNORECASE,
)

_PAREN_CLAUSE_PATTERN = re.compile(r"\(([^)]{1,50})\)")
_CLAUSE_HEDGE_PATTERN = re.compile(
    r"\b(?:essentially|basically|kind of|sort of|more or less)\b\s*",
    re.IGNORECASE,
)
_WHICH_CLAUSE_PATTERN = re.compile(
    r",\s*which (?:is|are|was|were) (?:a |an |the )?([^,.;]{1,40})",
    re.IGNORECASE,
)

_ORDINAL_LIST_PATTERN = re.compile(
    r"\b(First|Firstly),?\s+(.*?)\s+"
    r"(Second|Secondly),?\s+(.*?)\s+"
    r"(Third|Thirdly),?\s+(.*?)(?:\s+(Fourth|Fourthly),?\s+(.*?))?(?=\n|$|\.)",
    re.IGNORECASE | re.DOTALL,
)

_FOLLOWING_LIST_PATTERN = re.compile(
    r"(?:The following|These)\s+(\w+):?\s*([^\n.]{60,}?)\s*(?=\n|$|\.)",
    re.IGNORECASE,
)

_DECIMAL_REDUCTION_PATTERN = re.compile(r"\b-?\d+\.\d{3,}\b(?!\.\d)")

_MACRO_TOKEN_PATTERN = re.compile(r"__\S+?__|[A-Za-z0-9][A-Za-z0-9'_-]*")
_MACRO_PLACEHOLDER_PATTERN = re.compile(r"^__\w+__")
_MACRO_LEGEND_PREFIX = "Legend:"
_MACRO_ALIAS_OPEN = "⟦"
_MACRO_ALIAS_CLOSE = "⟧"

FIELD_LABEL_PATTERN = re.compile(r"(?m)^(?P<label>[A-Za-z][A-Za-z0-9 _-]{1,30}):")
_BULLET_ITEM_PATTERN = re.compile(
    r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s+(?P<item>.+)$",
    re.MULTILINE,
)
_PARENTHETICAL_PATTERN = re.compile(r"\([^()]{1,60}\)")
PARENTHETICAL_ALIAS_OPEN = "⟮"
PARENTHETICAL_ALIAS_CLOSE = "⟯"
_MAX_PARENTHETICAL_LEGEND = 6


def _get_segment_weight(
    segment_weights: Optional[Sequence[float]], index: int
) -> Optional[float]:
    if not segment_weights or index >= len(segment_weights):
        return None

    try:
        weight = float(segment_weights[index])
    except (TypeError, ValueError):  # pragma: no cover - defensive casting
        return None

    if weight < 0.0:
        return 0.0
    if weight > 1.0:
        return 1.0
    return weight


def _filter_categories_for_weight(
    active_categories: Collection[str], weight: Optional[float]
) -> Set[InstructionCategory]:
    allowed = {
        category
        for category in active_categories
        if category in _INSTRUCTION_CATEGORY_ORDER
    }

    if weight is None:
        return {
            category for category in _INSTRUCTION_CATEGORY_ORDER if category in allowed
        }

    if weight >= config.SEGMENT_WEIGHT_HIGH:
        allowed.difference_update({"filler"})
    elif weight >= config.SEGMENT_WEIGHT_MODERATE:
        allowed.discard("filler")

    return {category for category in _INSTRUCTION_CATEGORY_ORDER if category in allowed}


def _apply_instruction_rules(
    segment: str,
    *,
    active_categories: Collection[InstructionCategory],
    weight: Optional[float],
) -> Tuple[str, Set[InstructionCategory]]:
    if not segment or not active_categories:
        return segment, set()

    categories_for_segment = _filter_categories_for_weight(active_categories, weight)
    if not categories_for_segment:
        return segment, set()

    updated_segment = segment
    triggered: Set[InstructionCategory] = set()
    for category in _INSTRUCTION_CATEGORY_ORDER:
        if category not in categories_for_segment:
            continue

        rules = _INSTRUCTION_RULES.get(category, tuple())
        if not rules:
            continue

        for pattern, replacement in rules:
            updated_segment, count = pattern.subn(replacement, updated_segment)
            if count:
                triggered.add(category)

    return updated_segment, triggered


def clean_instruction_noise(
    text: str,
    placeholder_pattern: re.Pattern,
    *,
    enabled_categories: Optional[Collection[str]] = None,
    segment_weights: Optional[Sequence[float]] = None,
) -> Tuple[str, Set[InstructionCategory]]:
    """Apply instruction-cleanup rules in a single pass while preserving placeholders."""

    if not text:
        return text, set()

    if enabled_categories is None:
        active_categories: Set[InstructionCategory] = set(_INSTRUCTION_CATEGORY_ORDER)
    else:
        active_categories = {
            category
            for category in enabled_categories
            if category in _INSTRUCTION_CATEGORY_ORDER
        }

    if not active_categories:
        return text, set()

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    triggered_categories: Set[InstructionCategory] = set()
    processed_segments: List[str] = []

    for index, segment in enumerate(segments):
        if not segment:
            processed_segments.append(segment)
            continue

        weight = _get_segment_weight(segment_weights, index)
        updated_segment, segment_triggered = _apply_instruction_rules(
            segment,
            active_categories=active_categories,
            weight=weight,
        )
        if segment_triggered:
            triggered_categories.update(segment_triggered)
        processed_segments.append(updated_segment)

    rebuilt: List[str] = []
    for index, segment in enumerate(processed_segments):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt), triggered_categories


def _match_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _apply_contractions_segment(segment: str) -> str:
    if not segment or not _CONTRACTION_RULES:
        return segment

    result = segment
    for pattern, contraction in _CONTRACTION_RULES:
        result = pattern.sub(lambda m: _match_case(contraction, m.group(0)), result)
    return result


def apply_contractions(text: str, placeholder_pattern: re.Pattern) -> str:
    """Convert expanded forms to contractions while preserving placeholders."""
    if not text or not _CONTRACTION_RULES:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = [_apply_contractions_segment(segment) for segment in segments]

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def compress_boilerplate(
    text: str, *, placeholder_ranges: Sequence[Tuple[int, int]]
) -> str:
    """Compress common boilerplate sections while preserving placeholders."""
    if not text or not _BOILERPLATE_RULES:
        return text

    replacements: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []

    def overlaps_existing(start: int, end: int) -> bool:
        return any(
            start < existing_end and end > existing_start
            for existing_start, existing_end in occupied
        )

    for pattern, replacement in _BOILERPLATE_RULES:
        for match in pattern.finditer(text):
            start, end = match.span()
            if span_overlaps_placeholder(start, end, placeholder_ranges):
                continue
            if overlaps_existing(start, end):
                continue
            snippet = match.group(0).lower()
            if any(keyword.lower() in snippet for keyword in config.DIRECTIVE_KEYWORDS):
                continue
            if replacement and len(replacement) >= len(match.group(0)):
                continue

            replacements.append((start, end, replacement))
            occupied.append((start, end))

    if not replacements:
        return text

    replacements.sort(key=lambda item: item[0], reverse=True)
    updated = text
    for start, end, replacement in replacements:
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _choose_label_alias(label: str, reserved: Set[str]) -> Optional[str]:
    cleaned = "".join(word[0] for word in label.split() if word).upper()
    candidates: List[str] = []
    if cleaned:
        candidates.append(cleaned[:1])
        candidates.append(cleaned[:2])
    else:
        candidates.append(label[:1].upper())
    for candidate in candidates:
        candidate = candidate.strip()
        if 0 < len(candidate) <= 2 and candidate not in reserved:
            return candidate
    prefix = cleaned[:1] if cleaned else label[:1].upper()
    if prefix:
        for digit in range(1, 10):
            candidate = f"{prefix}{digit}"
            if len(candidate) <= 2 and candidate not in reserved:
                return candidate
    return None


def compress_field_labels(
    text: str,
    *,
    placeholder_ranges: Sequence[Tuple[int, int]],
    token_counter: Optional[Callable[[str], int]] = None,
) -> Tuple[str, bool, Optional[List[Tuple[str, str]]], int]:
    """Alias repeated leading labels with short tokens and insert a legend."""
    if not text:
        return text, False, None, 0

    matches: Dict[str, List[Tuple[int, int, str]]] = {}
    for match in FIELD_LABEL_PATTERN.finditer(text):
        start, end = match.span()
        if span_overlaps_placeholder(start, end, placeholder_ranges):
            continue
        label = match.group("label").strip()
        if len(label) <= 2:
            continue
        snippet = match.group(0)
        matches.setdefault(label, []).append((start, end, snippet))

    candidates = {
        label: entries for label, entries in matches.items() if len(entries) >= 2
    }
    if not candidates:
        return text, False, None, 0

    counter = token_counter or (
        lambda value: _estimate_token_count(value, token_counter=None)
    )
    reserved: Set[str] = set()
    replacements: List[Tuple[int, int, str]] = []
    legend_entries: List[Tuple[str, str]] = []
    original_tokens = 0
    alias_tokens = 0

    for label, entries in candidates.items():
        alias = _choose_label_alias(label, reserved)
        if not alias:
            continue
        alias_token = counter(f"{alias}:")
        entry_token = counter(entries[0][2])
        total_entry_tokens = entry_token * len(entries)
        total_alias_tokens = alias_token * len(entries)
        if total_entry_tokens <= total_alias_tokens:
            continue
        reserved.add(alias)
        legend_entries.append((alias, label))
        original_tokens += total_entry_tokens
        alias_tokens += total_alias_tokens
        for start, end, _ in entries:
            replacements.append((start, end, f"{alias}:"))

    if not replacements:
        return text, False, None, 0

    legend = (
        f"Labels: {', '.join(f'{alias}={label}' for alias, label in legend_entries)}"
    )
    legend_cost = counter(legend)
    net_savings = original_tokens - alias_tokens - legend_cost
    if net_savings <= 0:
        return text, False, None, 0

    replacements.sort(key=lambda item: item[0], reverse=True)
    updated = text
    for start, end, alias_text in replacements:
        updated = updated[:start] + alias_text + updated[end:]

    return updated, True, legend_entries, net_savings


def _parenthetical_replacer(match: Match[str]) -> str:
    content = match.group(1)
    lowered = content.lower()
    if "__" in content or any(char.isdigit() for char in content):
        return match.group(0)
    if any(keyword in lowered for keyword in config.DIRECTIVE_KEYWORDS):
        return match.group(0)
    if re.search(r"\b(?:not|never|no)\b", lowered):
        return match.group(0)
    if ":" in content or ";" in content or "=" in content:
        return match.group(0)
    if re.search(r"[A-Z]{2,}", content):
        return match.group(0)
    return " "


def _which_clause_replacer(match: Match[str]) -> str:
    content = match.group(1).strip()
    lowered = content.lower()
    if "__" in content or any(char.isdigit() for char in content):
        return match.group(0)
    if any(keyword in lowered for keyword in config.DIRECTIVE_KEYWORDS):
        return match.group(0)
    if re.search(r"\b(?:not|never|no)\b", lowered):
        return match.group(0)
    replacement = f" - {content}"
    if len(replacement) >= len(match.group(0)):
        return match.group(0)
    return replacement


def _compress_clause_segment(segment: str, *, weight: Optional[float]) -> str:
    if not segment:
        return segment

    if weight is not None and weight >= config.SEGMENT_WEIGHT_HIGH:
        return segment

    result = _PAREN_CLAUSE_PATTERN.sub(_parenthetical_replacer, segment)
    result = _CLAUSE_HEDGE_PATTERN.sub("", result)
    result = _WHICH_CLAUSE_PATTERN.sub(_which_clause_replacer, result)
    return result


def compress_clauses(
    text: str,
    placeholder_pattern: re.Pattern,
    *,
    segment_weights: Optional[Sequence[float]] = None,
) -> str:
    """Remove low-signal parenthetical clauses and hedging language."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed: List[str] = []
    for index, segment in enumerate(segments):
        if not segment:
            processed.append(segment)
            continue

        weight = _get_segment_weight(segment_weights, index)
        processed.append(_compress_clause_segment(segment, weight=weight))

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def _ordinal_list_replacer(match: Match[str]) -> str:
    items = [match.group(2), match.group(4), match.group(6)]
    if match.group(8):
        items.append(match.group(8))
    cleaned = [item.strip(" ,;") for item in items if item]
    if any(len(item) > 120 for item in cleaned):
        return match.group(0)
    replacement = "\n".join(
        f"{index + 1}. {item.strip()}" for index, item in enumerate(cleaned)
    )
    if len(replacement) >= len(match.group(0)):
        return match.group(0)
    return replacement


def _following_list_replacer(match: Match[str]) -> str:
    category = match.group(1)
    items_text = match.group(2)
    if "," not in items_text and " and " not in items_text:
        return match.group(0)
    parts = re.split(r",\s*(?:and\s*)?|\s+and\s+", items_text)
    items = [item.strip() for item in parts if item.strip()]
    if len(items) < 3:
        return match.group(0)
    if any(len(item) > 120 for item in items):
        return match.group(0)
    replacement = f"{category}:\n" + "\n".join(f"- {item}" for item in items)
    if len(replacement) >= len(match.group(0)):
        return match.group(0)
    return replacement


def _compress_list_segment(
    segment: str,
    *,
    weight: Optional[float],
    token_counter: Optional[Callable[[str], int]],
    enable_prefix_suffix_factoring: bool,
) -> str:
    if not segment:
        return segment

    if weight is not None and weight >= config.SEGMENT_WEIGHT_HIGH:
        return segment

    result = _ORDINAL_LIST_PATTERN.sub(_ordinal_list_replacer, segment)
    result = _FOLLOWING_LIST_PATTERN.sub(_following_list_replacer, result)
    result, factored = _apply_list_prefix_suffix_factoring(
        result,
        token_counter=token_counter,
        enable_factoring=enable_prefix_suffix_factoring,
    )
    return result


def _detect_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def _strip_prefix_text(
    item: str, word_matches: Sequence[Match[str]], length: int
) -> str:
    if len(word_matches) < length:
        return ""
    end = word_matches[length - 1].end()
    return item[end:].lstrip()


def _strip_suffix_text(
    item: str, word_matches: Sequence[Match[str]], length: int
) -> str:
    if len(word_matches) < length:
        return item.rstrip()
    start = word_matches[-length].start()
    return item[:start].rstrip()


def _evaluate_common_prefix(
    items: Sequence[str],
    spans: Sequence[Sequence[Match[str]]],
    *,
    original_tokens: int,
    token_counter: Callable[[str], int],
) -> Optional[Tuple[str, int]]:
    max_words = 6
    min_words = 2
    for length in range(max_words, min_words - 1, -1):
        if any(len(matches) < length for matches in spans):
            continue
        base_sequence = tuple(match.group(0).lower() for match in spans[0][:length])
        if any(
            tuple(match.group(0).lower() for match in matches[:length]) != base_sequence
            for matches in spans[1:]
        ):
            continue
        prefix_end = spans[0][length - 1].end()
        prefix_text = items[0][:prefix_end].strip()
        remainders: List[str] = []
        for idx, item in enumerate(items):
            remainder = _strip_prefix_text(item, spans[idx], length)
            if not remainder:
                break
            remainders.append(remainder)
        else:
            candidate = f"{prefix_text}: " + "; ".join(remainders)
            candidate_tokens = token_counter(candidate)
            return candidate, original_tokens - candidate_tokens
    return None


def _evaluate_common_suffix(
    items: Sequence[str],
    spans: Sequence[Sequence[Match[str]]],
    *,
    original_tokens: int,
    token_counter: Callable[[str], int],
) -> Optional[Tuple[str, int]]:
    max_words = 6
    min_words = 2
    for length in range(max_words, min_words - 1, -1):
        if any(len(matches) < length for matches in spans):
            continue
        base_sequence = tuple(match.group(0).lower() for match in spans[0][-length:])
        if any(
            tuple(match.group(0).lower() for match in matches[-length:])
            != base_sequence
            for matches in spans[1:]
        ):
            continue
        suffix_start = spans[0][-length].start()
        suffix_text = items[0][suffix_start:].strip()
        if not suffix_text:
            continue
        remainders: List[str] = []
        for idx, item in enumerate(items):
            remainder = _strip_suffix_text(item, spans[idx], length)
            if not remainder:
                break
            remainders.append(remainder)
        else:
            candidate = "; ".join(remainders)
            candidate = f"{candidate} ({suffix_text})"
            candidate_tokens = token_counter(candidate)
            return candidate, original_tokens - candidate_tokens
    return None


def _factor_list_block(
    block_lines: Sequence[str], *, token_counter: Callable[[str], int]
) -> Optional[str]:
    block_text = "".join(block_lines)
    if "__" in block_text or "TOON" in block_text:
        return None
    items: List[str] = []
    spans: List[List[Match[str]]] = []
    for line in block_lines:
        match = _BULLET_ITEM_PATTERN.match(line)
        if not match:
            return None
        item = match.group("item").strip()
        items.append(item)
        spans.append(list(WORD_PATTERN.finditer(item)))
    if len(items) < 2:
        return None
    original_tokens = token_counter(block_text)
    prefix_candidate = _evaluate_common_prefix(
        items,
        spans,
        original_tokens=original_tokens,
        token_counter=token_counter,
    )
    suffix_candidate = _evaluate_common_suffix(
        items,
        spans,
        original_tokens=original_tokens,
        token_counter=token_counter,
    )
    best_candidate: Optional[Tuple[str, int]] = None
    if prefix_candidate and prefix_candidate[1] > 0:
        best_candidate = prefix_candidate
    if suffix_candidate and suffix_candidate[1] > 0:
        if not best_candidate or suffix_candidate[1] > best_candidate[1]:
            best_candidate = suffix_candidate
    if not best_candidate:
        return None
    line_ending = _detect_line_ending(block_lines[-1])
    return best_candidate[0] + line_ending


def _apply_list_prefix_suffix_factoring(
    segment: str,
    *,
    token_counter: Optional[Callable[[str], int]],
    enable_factoring: bool,
) -> Tuple[str, bool]:
    if not segment or not enable_factoring:
        return segment, False
    counter = token_counter or (
        lambda value: _estimate_token_count(value, token_counter=None)
    )
    lines = segment.splitlines(keepends=True)
    rebuilt: List[str] = []
    index = 0
    changed = False
    while index < len(lines):
        line = lines[index]
        if not _BULLET_ITEM_PATTERN.match(line):
            rebuilt.append(line)
            index += 1
            continue
        block_lines: List[str] = []
        while index < len(lines) and _BULLET_ITEM_PATTERN.match(lines[index]):
            block_lines.append(lines[index])
            index += 1
        factored = _factor_list_block(block_lines, token_counter=counter)
        if factored:
            rebuilt.append(factored)
            changed = True
        else:
            rebuilt.extend(block_lines)
    return "".join(rebuilt), changed


def extract_parenthetical_glossary(
    text: str,
    *,
    placeholder_ranges: Sequence[Tuple[int, int]],
    token_counter: Optional[Callable[[str], int]] = None,
    min_token_savings: int = 5,
) -> Tuple[str, bool, Optional[List[Tuple[str, str]]], int]:
    if not text:
        return text, False, None, 0

    counter = token_counter or (
        lambda value: _estimate_token_count(value, token_counter=None)
    )
    candidates: Dict[str, List[Tuple[int, int]]] = {}

    for match in _PARENTHETICAL_PATTERN.finditer(text):
        start, end = match.span()
        if span_overlaps_placeholder(start, end, placeholder_ranges):
            continue
        phrase = match.group(0).strip()
        if len(phrase) <= 3:
            continue
        if "__" in phrase or "TOON" in phrase:
            continue
        candidates.setdefault(phrase, []).append((start, end))

    phrase_entries = [
        (phrase, spans, len(spans) * counter(phrase))
        for phrase, spans in candidates.items()
        if len(spans) >= 2
    ]
    if not phrase_entries:
        return text, False, None, 0

    phrase_entries.sort(key=lambda item: item[2], reverse=True)
    selected: List[Tuple[str, List[Tuple[int, int]], str, int]] = []
    for phrase, spans, original_tokens in phrase_entries:
        alias_tag = (
            f"{PARENTHETICAL_ALIAS_OPEN}P{len(selected) + 1}{PARENTHETICAL_ALIAS_CLOSE}"
        )
        alias_tokens = len(spans) * counter(alias_tag)
        saved = original_tokens - alias_tokens
        if saved <= 0:
            continue
        selected.append((phrase, spans, alias_tag, saved))
        if len(selected) >= _MAX_PARENTHETICAL_LEGEND:
            break

    if not selected:
        return text, False, None, 0

    total_saved = sum(item[3] for item in selected)
    legend_items = [f"{alias}={phrase}" for phrase, _, alias, _ in selected]
    legend = f"Glossary: {', '.join(legend_items)}"
    legend_cost = counter(legend)
    net_savings = total_saved - legend_cost
    if net_savings <= min_token_savings:
        return text, False, None, 0

    replacements: List[Tuple[int, int, str]] = []
    for phrase, spans, alias, _ in selected:
        for start, end in spans:
            replacements.append((start, end, alias))

    replacements.sort(key=lambda item: item[0], reverse=True)
    updated = text
    for start, end, alias in replacements:
        updated = updated[:start] + alias + updated[end:]

    legend_entries = [(alias, phrase) for phrase, _, alias, _ in selected]
    return updated, True, legend_entries, net_savings


def compress_lists(
    text: str,
    placeholder_pattern: re.Pattern,
    *,
    segment_weights: Optional[Sequence[float]] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    enable_prefix_suffix_factoring: bool = True,
) -> str:
    """Compress verbose list phrasing into denser formats."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed: List[str] = []
    for index, segment in enumerate(segments):
        if not segment:
            processed.append(segment)
            continue

        weight = _get_segment_weight(segment_weights, index)
        processed.append(
            _compress_list_segment(
                segment,
                weight=weight,
                token_counter=token_counter,
                enable_prefix_suffix_factoring=enable_prefix_suffix_factoring,
            )
        )

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def _reduce_decimal_match(match: Match[str]) -> str:
    number = match.group(0)
    try:
        value = float(number)
    except ValueError:
        return number

    if value.is_integer():
        return str(int(value))

    if abs(value) < 1000:
        rounded = round(value, 2)
        if rounded == 0 and value != 0:
            return number
        return f"{rounded:g}"

    return str(int(round(value)))


def _reduce_numeric_precision_segment(segment: str) -> str:
    if not segment:
        return segment

    return _DECIMAL_REDUCTION_PATTERN.sub(_reduce_decimal_match, segment)


def reduce_numeric_precision(text: str, placeholder_pattern: re.Pattern) -> str:
    """Reduce unnecessary numeric precision in decimal values."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = [_reduce_numeric_precision_segment(segment) for segment in segments]

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def shorten_synonyms(
    text: str,
    placeholder_pattern: re.Pattern,
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    segment_weights: Optional[Sequence[float]] = None,
) -> str:
    """Replace verbose synonyms with shorter equivalents while preserving placeholders."""
    if not text or not config.SYNONYM_SHORTENINGS:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    token_cache: Dict[str, int] = {}
    processed_segments = [
        _shorten_synonyms_segment(
            segment,
            weight=_get_segment_weight(segment_weights, index),
            token_counter=token_counter,
            token_cache=token_cache,
        )
        for index, segment in enumerate(segments)
    ]

    result_parts: List[str] = []
    for index, segment in enumerate(processed_segments):
        result_parts.append(segment)
        if index < len(placeholders):
            result_parts.append(placeholders[index])

    return "".join(result_parts)


def _normalized_token_count(
    value: str,
    *,
    token_counter: Optional[Callable[[str], int]],
    token_cache: Dict[str, int],
) -> Optional[int]:
    if token_counter is None:
        return None

    cached = token_cache.get(value)
    if cached is not None:
        return cached

    try:
        count = token_counter(value)
    except Exception:  # pragma: no cover - defensive against tokenizer errors
        return None

    if not value:
        normalized = 0
    else:
        normalized = max(int(count), 1)

    token_cache[value] = normalized
    return normalized


def _shorten_synonyms_segment(
    segment: str,
    *,
    weight: Optional[float],
    token_counter: Optional[Callable[[str], int]],
    token_cache: Dict[str, int],
) -> str:
    if not segment:
        return segment

    high_priority = weight is not None and weight >= config.SEGMENT_WEIGHT_HIGH
    moderate_priority = weight is not None and weight >= config.SEGMENT_WEIGHT_MODERATE

    def substitute(match: Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        replacement = config.SYNONYM_SHORTENINGS.get(lower)
        if not replacement:
            return word

        if high_priority:
            return word

        if token_counter is not None:
            original_tokens = _normalized_token_count(
                word,
                token_counter=token_counter,
                token_cache=token_cache,
            )
            if original_tokens is None:
                return word

            replacement_variant = replacement
            if word.isupper():
                replacement_variant = replacement.upper()
            elif word[0].isupper():
                replacement_variant = replacement.capitalize()

            replacement_tokens = _normalized_token_count(
                replacement_variant,
                token_counter=token_counter,
                token_cache=token_cache,
            )
            if replacement_tokens is None or replacement_tokens >= original_tokens:
                return word

            token_delta = original_tokens - replacement_tokens
            if moderate_priority and token_delta < 2:
                return word

            return replacement_variant

        if moderate_priority:
            return word

        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement

    return WORD_PATTERN.sub(substitute, segment)


def canonicalize_entities(
    text: str,
    built_in_canonicalizations: Dict[str, str],
    config_canonicalizations: Dict[str, str],
    custom_map: Optional[Dict[str, str]] = None,
    disable_defaults: bool = False,
) -> Tuple[str, Dict[str, str]]:
    canonical_map = merge_canonicalizations(
        built_in_canonicalizations,
        config_canonicalizations,
        custom_map,
        disable_defaults,
    )

    if not text or not canonical_map:
        return text, canonical_map

    from .trie_replacer import trie_canonicalize

    return trie_canonicalize(text, canonical_map), canonical_map


def merge_canonicalizations(
    built_in: Dict[str, str],
    config_overrides: Dict[str, str],
    custom_map: Optional[Dict[str, str]],
    disable_defaults: bool = False,
) -> Dict[str, str]:
    """
    Merge built-in canonicalizations with configuration and request custom mappings.

    Args:
        built_in: Built-in default canonicalizations from config
        config_overrides: Configuration overrides (merged with built-in)
        custom_map: Per-request custom canonicalization mappings
        disable_defaults: If True, ignore built_in and config_overrides, use only custom_map

    Returns:
        Merged canonicalization dictionary

    Behavior:
        - If disable_defaults=True: Return only custom_map (empty dict if None)
        - If disable_defaults=False: Merge built_in + config_overrides + custom_map
          Custom mappings OVERRIDE defaults on conflict (case-insensitive matching)
    """
    if disable_defaults:
        return dict(custom_map) if custom_map else {}

    # Start with built-in defaults
    merged = dict(built_in)

    # Merge config overrides
    if config_overrides:
        merged.update(config_overrides)

    # Add custom mappings - custom mappings OVERRIDE defaults when there's a conflict
    # Per user requirement: custom mappings MUST have precedence over OOTB mappings
    if custom_map:
        for key, value in custom_map.items():
            # Check if this key already exists in defaults (case-insensitive)
            existing_key = next((k for k in merged if k.lower() == key.lower()), None)
            if existing_key:
                # OVERRIDE: Custom mappings take precedence over defaults
                # Remove the existing default and add the custom mapping
                logging.debug(
                    f"Custom mapping '{key}' -> '{value}' overrides "
                    f"default mapping '{existing_key}' -> '{merged[existing_key]}'"
                )
                del merged[existing_key]
            # Add custom mapping (overriding any previous default)
            merged[key] = value

    return merged


def _normalize_numbers_and_units_segment(segment: str) -> str:
    if not segment:
        return segment

    normalized = segment
    if _SPELLED_MAGNITUDE_PATTERN.search(normalized):
        normalized = _SPELLED_MAGNITUDE_PATTERN.sub(
            lambda match: (
                _SPELLED_MAGNITUDE_PREFIXES.get(match.group(1).lower(), match.group(1))
                + _SPELLED_MAGNITUDE_SUFFIXES.get(match.group(2).lower(), "")
            ),
            normalized,
        )

    # Apply fast regex-based unit normalization (Technique 1)
    for pattern, replacement in _FAST_UNIT_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    # Only invoke Quantulum parsing if segment contains numeric cues
    if QUANTULUM_AVAILABLE and _NUMERIC_CUE_PATTERN.search(segment):
        try:
            quantities = quantulum_parser.parse(normalized)
        except (
            Exception
        ) as exc:  # pragma: no cover - depends on optional dependency state
            logging.debug("Quantulum parsing failed: %s", exc)
            quantities = []
        if quantities:
            rebuilt: List[str] = []
            last_index = 0
            for quantity in quantities:
                start, end = quantity.span
                rebuilt.append(normalized[last_index:start])
                rebuilt.append(format_quantity(quantity))
                last_index = end
            rebuilt.append(normalized[last_index:])
            normalized = "".join(rebuilt)

    return standardize_thousand_separators(normalized)


def normalize_numbers_and_units(
    text: str,
    placeholder_pattern: re.Pattern,
) -> str:
    """Normalize numbers and units while preserving placeholders."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = [_normalize_numbers_and_units_segment(segment) for segment in segments]

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


_SYMBOLIC_REPLACEMENT_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), symbol)
    for word, symbol in config.SYMBOLIC_REPLACEMENTS.items()
)


def _apply_symbolic_replacements_segment(segment: str) -> str:
    if not segment:
        return segment

    result = segment
    for pattern, symbol in _SYMBOLIC_REPLACEMENT_PATTERNS:
        result = pattern.sub(symbol, result)
    return result


def apply_symbolic_replacements(
    text: str,
    placeholder_pattern: re.Pattern,
) -> str:
    """Replace common words with shorter symbolic equivalents."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = [_apply_symbolic_replacements_segment(segment) for segment in segments]

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


_ARTICLE_PATTERN = re.compile(r"\b(?:the|a|an)\b\s+", re.IGNORECASE)


def _remove_articles_segment(segment: str) -> str:
    if not segment:
        return segment
    return _ARTICLE_PATTERN.sub("", segment)


def remove_articles(
    text: str,
    placeholder_pattern: re.Pattern,
) -> str:
    """Remove definite and indefinite articles."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = [_remove_articles_segment(segment) for segment in segments]

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def format_quantity(quantity) -> str:
    """Format a quantulum3 quantity with normalized number and unit."""
    surface = quantity.surface
    if "__" in surface:
        return surface

    number_str = format_number_value(quantity.value)
    unit = getattr(quantity, "unit", None)

    if not unit or unit.name.lower() == "dimensionless":
        return number_str

    unit_str = get_unit_abbreviation(unit)
    if not unit_str:
        unit_name = getattr(unit, "name", "")
        combined = f"{number_str} {unit_name}".strip()
        return combined

    # Prefix symbols that should touch the number (currency, percent)
    if unit_str == "%":
        return f"{number_str}{unit_str}"
    if unit_str and unit_str[0] in ("$", "€", "£", "¥", "₹"):
        return f"{unit_str}{number_str}"

    if unit_str in config.COMPACT_UNIT_ABBREVIATIONS:
        return f"{number_str}{unit_str}"

    return f"{number_str} {unit_str}"


def format_number_value(value) -> str:
    """Convert numeric value or range to string with standard thousands separators."""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        parts = [format_number_value(part) for part in value]
        return " - ".join(parts)

    numeric = value
    if isinstance(numeric, Fraction):
        numeric = Decimal(numeric.numerator) / Decimal(numeric.denominator)

    try:
        decimal_value = Decimal(str(numeric))
    except (InvalidOperation, ValueError, TypeError):
        return str(numeric)

    if decimal_value == decimal_value.to_integral():
        return str(int(decimal_value))

    normalized = decimal_value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def get_unit_abbreviation(unit) -> str:
    """Retrieve a concise unit abbreviation."""
    name = unit.name.lower() if getattr(unit, "name", None) else ""
    entity = getattr(unit, "entity", None)
    if entity and getattr(entity, "name", None) == "currency":
        code = getattr(unit, "code", None)
        symbol = None
        if code:
            symbol = config.CURRENCY_SYMBOLS.get(str(code).lower())
        if not symbol and name:
            symbol = config.CURRENCY_SYMBOLS.get(name)
        if symbol:
            return symbol

    if name in config.UNIT_ABBREVIATIONS:
        return config.UNIT_ABBREVIATIONS[name]

    # Attempt prefix match for compound unit names (e.g., degree Celsius)
    for candidate, abbr in config.UNIT_ABBREVIATIONS.items():
        if name.startswith(candidate):
            return abbr

    symbols = getattr(unit, "symbols", None)
    if symbols:
        for symbol in symbols:
            if symbol:
                return symbol

    symbol = getattr(unit, "symbol", None)
    if symbol:
        return symbol

    return name


def standardize_thousand_separators(text: str) -> str:
    """Normalize thousands separators in numbers for token efficiency."""

    def replacer(match: Match[str]) -> str:
        raw = match.group(0)
        cleaned = re.sub(r"[ ,_]", "", raw)
        if not cleaned:
            return raw
        if "." in cleaned:
            integer, decimal = cleaned.split(".", 1)
            return f"{int(integer)}.{decimal}"
        return str(int(cleaned))

    return _THOUSAND_SEPARATOR_PATTERN.sub(replacer, text)


def _collapse_consecutive_duplicates_segment(segment: str) -> str:
    """Remove consecutive duplicate words/phrases from a segment."""
    if not segment or not _CONSECUTIVE_DUPLICATE_RULES:
        return segment

    if "/" in segment and len(segment) >= 20:
        max_unit = len(segment) // 2
        for unit_length in range(1, max_unit + 1):
            if len(segment) % unit_length != 0:
                continue
            unit = segment[:unit_length]
            if "/" not in unit:
                continue
            if segment == unit * (len(segment) // unit_length):
                return unit

    result = segment
    for pattern, replacement in _CONSECUTIVE_DUPLICATE_RULES:
        result = pattern.sub(replacement, result)
    return result


def _collapse_paradoxical_phrases_segment(segment: str) -> str:
    """Collapse paradoxical/contradictory statements in a segment."""
    if not segment or not _PARADOXICAL_PHRASE_RULES:
        return segment

    result = segment
    for pattern, replacement in _PARADOXICAL_PHRASE_RULES:
        result = pattern.sub(replacement, result)
    return result


def _consolidate_repeated_phrases_segment(segment: str) -> str:
    """Consolidate repeated phrases in a segment."""
    if not segment or not _REPEATED_PHRASE_RULES:
        return segment

    result = segment
    for pattern, replacement in _REPEATED_PHRASE_RULES:
        result = pattern.sub(replacement, result)
    return result


# Final cleanup patterns to remove artifacts after all transformations
_FINAL_CLEANUP_COMMON_PATTERNS = [
    # "and and and" -> "and"
    (re.compile(r"\b(and)\s+\1(?:\s+\1)+\b", re.IGNORECASE), r"\1"),
    # "I really I really" -> "I really"
    (re.compile(r"\b(I\s+\w+)\s+\1(?:\s+\1)*\b", re.IGNORECASE), r"\1"),
    # Trailing "and and and" at end
    (re.compile(r"\s+and(?:\s+and)+\s*$", re.IGNORECASE), " and"),
    # Empty parentheses
    (re.compile(r"\(\s*\)"), ""),
    # Orphan quotes
    (re.compile(r'"\s*"'), ""),
    # Fix artifact: "kthe" -> "k the" (corrupted "know the")
    (re.compile(r"\bkthe\b", re.IGNORECASE), "k the"),
    # Fix artifact: "nthe" -> "n the"
    (re.compile(r"\bnthe\b", re.IGNORECASE), "n the"),
]
_FINAL_CLEANUP_WHITESPACE_PATTERNS = [
    # Multiple spaces
    (re.compile(r"  +"), " "),
    # Space before punctuation
    (re.compile(r"\s+([.,!?;:])"), r"\1"),
]
_FINAL_CLEANUP_PUNCTUATION_PATTERNS = [
    # Multiple punctuation
    (re.compile(r"([.,!?])\1+"), r"\1"),
    # Trailing comma before period
    (re.compile(r",\s*\."), "."),
]


_ABRUPT_LEADING_CONNECTOR_PATTERN = re.compile(
    r"^\s*[,;:—–-]*\s*"
    r"(?P<connector>"
    r"and|but|or|so|then|also|plus|however|moreover|additionally|furthermore|"
    r"nevertheless|nonetheless|still|anyway|anyways"
    r")\b(?!/)"
    r"(?:\s+|\s*[,;:—–-]\s*)",
    re.IGNORECASE,
)


def _first_alpha_char(value: str) -> Optional[str]:
    for ch in value:
        if ch.isalpha():
            return ch
    return None


def _trim_abrupt_leading_connectors(text: str) -> str:
    if not text:
        return text

    result = text
    removed = False
    for _ in range(2):
        match = _ABRUPT_LEADING_CONNECTOR_PATTERN.match(result)
        if not match:
            break

        connector = match.group("connector") or ""
        remaining = result[match.end() :]
        first_alpha = _first_alpha_char(remaining.lstrip(" \t\r\n\"'“”‘’([{<"))
        if connector.islower() or (first_alpha is not None and first_alpha.islower()):
            result = remaining.lstrip(" \t\r\n,;:—–-")
            removed = True
            continue

        break

    if removed and result:
        stripped = result.lstrip()
        if stripped and stripped[0].isalpha() and stripped[0].islower():
            result = stripped[0].upper() + stripped[1:]
        else:
            result = stripped

    return result



def final_text_cleanup(
    text: str,
    *,
    normalize_whitespace: bool,
    compress_punctuation: bool,
) -> str:
    """Apply final cleanup to remove artifacts from optimization passes."""
    if not text:
        return text

    result = text
    for pattern, replacement in _FINAL_CLEANUP_COMMON_PATTERNS:
        result = pattern.sub(replacement, result)

    if normalize_whitespace:
        for pattern, replacement in _FINAL_CLEANUP_WHITESPACE_PATTERNS:
            result = pattern.sub(replacement, result)

    if compress_punctuation:
        for pattern, replacement in _FINAL_CLEANUP_PUNCTUATION_PATTERNS:
            result = pattern.sub(replacement, result)

    result = _trim_abrupt_leading_connectors(result)
    return result.strip() if normalize_whitespace else result


def compress_punctuation(
    text: str,
    placeholder_pattern: re.Pattern,
) -> str:
    """Apply aggressive punctuation compression (Technique 2)."""
    if not text:
        return text

    segments = placeholder_pattern.split(text)
    placeholders = placeholder_pattern.findall(text)

    processed = []
    for segment in segments:
        if not segment:
            processed.append(segment)
            continue

        result = segment
        for pattern, replacement in _AGGRESSIVE_PUNCTUATION_PATTERNS:
            result = pattern.sub(replacement, result)
        processed.append(result)

    rebuilt: List[str] = []
    for index, segment in enumerate(processed):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])

    return "".join(rebuilt)


def apply_frequency_abbreviations(
    text: str,
    canonical_map: Dict[str, str],
    preserved: Dict,
    placeholder_tokens: Set[str],
    placeholder_pattern: re.Pattern,
    token_counter: Optional[Callable[[str], int]] = None,
    min_occurrences: int = 3,
    max_new: int = 5,
    min_phrase_chars: int = 12,
) -> Tuple[str, Dict[str, str], Optional[List[Tuple[str, str]]], int]:
    """Learn abbreviations for frequently repeated multi-word phrases."""
    if not text:
        return text, {}, None, 0

    reserved_tokens: Set[str] = {value.lower() for value in canonical_map.values()}
    reserved_tokens.update(token.lower() for token in placeholder_tokens)

    word_matches = list(_WORD_PATTERN.finditer(text))

    occurrences: Dict[str, int] = defaultdict(int)
    exemplars: Dict[str, str] = {}

    for length in range(2, 5):
        if len(word_matches) < length:
            break

        for index in range(len(word_matches) - length + 1):
            start = word_matches[index].start()
            end = word_matches[index + length - 1].end()
            snippet = text[start:end]

            if any(punct in snippet for punct in ".:,;!?"):
                continue

            words = [word_matches[index + offset].group(0) for offset in range(length)]
            phrase = " ".join(words)
            normalized = phrase.lower()

            if len(phrase.replace(" ", "")) < min_phrase_chars:
                continue

            if normalized in canonical_map:
                continue

            occurrences[normalized] += 1
            if normalized not in exemplars:
                exemplars[normalized] = phrase

    candidates = [
        (normalized, count, exemplars[normalized])
        for normalized, count in occurrences.items()
        if count >= min_occurrences
    ]

    if not candidates:
        return text, {}, None, 0

    leading_stopwords = {"the", "a", "an"}

    processed_candidates: List[Tuple[str, int, int, str, List[str], str, bool]] = []
    for normalized, count, exemplar in candidates:
        words = exemplar.split()
        if not words:
            continue

        core_words = list(words)
        while core_words and core_words[0].lower() in leading_stopwords:
            core_words.pop(0)

        if not core_words:
            continue

        core_phrase = " ".join(core_words).lower()
        starts_with_stopword = len(core_words) != len(words)
        estimated_savings = max(count - 1, 0) * len(exemplar)
        processed_candidates.append(
            (
                normalized,
                count,
                estimated_savings,
                exemplar,
                core_words,
                core_phrase,
                starts_with_stopword,
            )
        )

    if not processed_candidates:
        return text, {}, None, 0

    processed_candidates.sort(
        key=lambda item: (-item[2], -item[1], item[6], -len(item[4]), -len(item[3]))
    )

    learned: Dict[str, str] = {}
    seen_cores: List[Tuple[str, Tuple[str, ...]]] = []

    for (
        normalized,
        count,
        _estimated_savings,
        exemplar,
        core_words,
        core_phrase,
        _,
    ) in processed_candidates:
        if len(learned) >= max_new:
            break

        core_tokens = tuple(word.lower() for word in core_words)

        if any(existing_core == core_phrase for existing_core, _ in seen_cores):
            continue

        if any(
            phrases_overlap(existing_tokens, core_tokens)
            for _, existing_tokens in seen_cores
        ):
            continue

        abbreviation = "".join(word[0] for word in core_words).upper()
        if not abbreviation:
            continue

        if abbreviation.lower() in reserved_tokens:
            continue

        learned[exemplar] = abbreviation
        reserved_tokens.add(abbreviation.lower())
        seen_cores.append((core_phrase, core_tokens))

    if not learned:
        return text, {}, None, 0

    replacements: List[Tuple[int, int, str]] = []
    placeholder_spans = [match.span() for match in placeholder_pattern.finditer(text)]

    for phrase, abbreviation in learned.items():
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            if span_overlaps_placeholder(start, end, placeholder_spans):
                continue

            replacements.append((start, end, abbreviation))

    if not replacements:
        return text, {}, None, 0

    replacements.sort(key=lambda item: item[0], reverse=True)
    updated_text = text
    for start_char, end_char, replacement_text in replacements:
        updated_text = (
            updated_text[:start_char] + replacement_text + updated_text[end_char:]
        )

    counter = token_counter or (
        lambda value: _estimate_token_count(value, token_counter=None)
    )
    tokens_before = counter(text)
    tokens_after = counter(updated_text)
    legend_entries = [(abbr, phrase) for phrase, abbr in learned.items()]
    legend = "Abbrev: " + ", ".join(
        f"{alias}={value}" for alias, value in legend_entries
    )
    legend_cost = counter(legend)
    net_savings = tokens_before - tokens_after - legend_cost
    if net_savings <= 0:
        return text, {}, None, 0

    return updated_text, learned, legend_entries, net_savings


@dataclass(frozen=True)
class MacroCandidate:
    phrase: str
    spans: Tuple[Tuple[int, int], ...]
    tokens: Tuple[str, ...]

    @property
    def occurrences(self) -> int:
        return len(self.spans)


def apply_macro_dictionary(
    text: str,
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    placeholder_tokens: Optional[Set[str]] = None,
    min_savings_ratio: float = 0.015,
    min_length: int = 3,
    min_occurrences: int = 2,
    max_macros: int = 4,
) -> Tuple[str, Dict[str, str]]:
    """Build a macro legend and replace repeated phrases with aliases."""
    if not text:
        return text, {}

    placeholder_tokens = placeholder_tokens or set()
    if _MACRO_LEGEND_PREFIX in text:
        return text, {}

    tokens = _tokenize_macro_candidates(text, placeholder_tokens)
    if len(tokens) < min_length * min_occurrences:
        return text, {}

    detector = RepetitionDetector(
        min_length=min_length, min_occurrences=min_occurrences
    )
    fragments = detector.find_repetitions([token.value for token in tokens])
    if not fragments:
        return text, {}

    candidates = _build_macro_candidates(text, tokens, fragments)
    if not candidates:
        return text, {}

    selected = _select_macro_candidates(candidates, max_macros=max_macros)
    if not selected:
        return text, {}

    alias_map = _assign_macro_aliases(text, selected)
    if not alias_map:
        return text, {}

    replacements = _build_macro_replacements(selected, alias_map)
    if not replacements:
        return text, {}

    updated = _apply_macro_replacements(text, replacements)
    legend = _build_macro_legend(alias_map)
    updated_with_legend = _insert_macro_legend(updated, legend)

    tokens_before = _estimate_token_count(text, token_counter=token_counter)
    tokens_after = _estimate_token_count(
        updated_with_legend, token_counter=token_counter
    )
    if tokens_before <= 0 or tokens_after >= tokens_before:
        return text, {}

    savings_ratio = (tokens_before - tokens_after) / tokens_before
    if savings_ratio < min_savings_ratio:
        return text, {}

    return updated_with_legend, alias_map


@dataclass(frozen=True)
class _MacroToken:
    value: str
    start: int
    end: int
    is_placeholder: bool


def _tokenize_macro_candidates(
    text: str, placeholder_tokens: Set[str]
) -> List[_MacroToken]:
    tokens: List[_MacroToken] = []
    for match in _MACRO_TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        is_placeholder = (
            value in placeholder_tokens
            or _MACRO_PLACEHOLDER_PATTERN.match(value) is not None
        )
        tokens.append(
            _MacroToken(
                value=value,
                start=match.start(),
                end=match.end(),
                is_placeholder=is_placeholder,
            )
        )
    return tokens


def _build_macro_candidates(
    text: str, tokens: Sequence[_MacroToken], fragments: Sequence
) -> List[MacroCandidate]:
    candidates: List[MacroCandidate] = []
    for fragment in fragments:
        if fragment.length < 2:
            continue

        start_index = fragment.positions[0]
        end_index = start_index + fragment.length
        if end_index > len(tokens):
            continue

        if any(token.is_placeholder for token in tokens[start_index:end_index]):
            continue

        sample_span = (tokens[start_index].start, tokens[end_index - 1].end)
        phrase = text[sample_span[0] : sample_span[1]]
        if "\n" in phrase or " " not in phrase:
            continue

        spans: List[Tuple[int, int]] = []
        consistent = True
        for position in fragment.positions:
            if position + fragment.length > len(tokens):
                consistent = False
                break
            span = (
                tokens[position].start,
                tokens[position + fragment.length - 1].end,
            )
            if text[span[0] : span[1]] != phrase:
                consistent = False
                break
            spans.append(span)

        if not consistent or len(spans) < 2:
            continue

        candidates.append(
            MacroCandidate(
                phrase=phrase,
                spans=tuple(spans),
                tokens=tuple(token.value for token in tokens[start_index:end_index]),
            )
        )

    return candidates


def _select_macro_candidates(
    candidates: Sequence[MacroCandidate], *, max_macros: int
) -> List[MacroCandidate]:
    if not candidates:
        return []

    def _estimated_reduction(candidate: MacroCandidate) -> int:
        return max(candidate.occurrences - 1, 0) * len(candidate.phrase)

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -_estimated_reduction(candidate),
            -candidate.occurrences,
            -len(candidate.phrase),
        ),
    )

    selected: List[MacroCandidate] = []
    occupied_spans: List[Tuple[int, int]] = []

    for candidate in ordered:
        if len(selected) >= max_macros:
            break

        non_overlapping = []
        for span in sorted(candidate.spans, key=lambda item: item[0]):
            if any(
                not (span[1] <= occupied[0] or span[0] >= occupied[1])
                for occupied in occupied_spans
            ):
                continue
            if non_overlapping and span[0] < non_overlapping[-1][1]:
                continue
            non_overlapping.append(span)

        if len(non_overlapping) < 2:
            continue

        selected.append(
            MacroCandidate(
                phrase=candidate.phrase,
                spans=tuple(non_overlapping),
                tokens=candidate.tokens,
            )
        )
        occupied_spans.extend(non_overlapping)

    return selected


def _assign_macro_aliases(
    text: str, candidates: Sequence[MacroCandidate]
) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    used_aliases: Set[str] = set()

    for candidate in candidates:
        alias = None
        for index in range(702):
            proposed = _format_macro_alias(index)
            if proposed in used_aliases or proposed in text:
                continue
            alias = proposed
            used_aliases.add(proposed)
            break
        if alias is None:
            break
        alias_map[alias] = candidate.phrase

    return alias_map


def _format_macro_alias(index: int) -> str:
    base = 26
    letters: List[str] = []
    index += 1
    while index > 0:
        index -= 1
        letters.append(chr(ord("A") + (index % base)))
        index //= base
    label = "".join(reversed(letters))
    return f"{_MACRO_ALIAS_OPEN}{label}{_MACRO_ALIAS_CLOSE}"


def _build_macro_replacements(
    candidates: Sequence[MacroCandidate],
    alias_map: Dict[str, str],
) -> List[Tuple[int, int, str]]:
    replacements: List[Tuple[int, int, str]] = []
    phrase_to_alias = {phrase: alias for alias, phrase in alias_map.items()}

    for candidate in candidates:
        alias = phrase_to_alias.get(candidate.phrase)
        if not alias:
            continue
        for span in candidate.spans:
            replacements.append((span[0], span[1], alias))

    replacements.sort(key=lambda item: item[0], reverse=True)
    return replacements


def _apply_macro_replacements(
    text: str, replacements: Sequence[Tuple[int, int, str]]
) -> str:
    updated = text
    for start, end, alias in replacements:
        updated = updated[:start] + alias + updated[end:]
    return updated


def _build_macro_legend(alias_map: Dict[str, str]) -> str:
    parts = [f"{alias}={phrase}" for alias, phrase in alias_map.items()]
    return f"{_MACRO_LEGEND_PREFIX} " + "; ".join(parts)


def _insert_macro_legend(text: str, legend: str) -> str:
    leading = text[: len(text) - len(text.lstrip())]
    body = text[len(leading) :]
    if not body:
        return f"{leading}{legend}"
    return f"{leading}{legend}\n{body}"


def _prepend_legend(text: str, legend: str) -> str:
    leading = text[: len(text) - len(text.lstrip())]
    body = text[len(leading) :]
    legend_text = legend.rstrip()
    if legend_text and legend_text[-1] not in ".!?":
        legend_text = f"{legend_text}."
    if not body:
        return f"{leading}{legend_text}"
    return f"{leading}{legend_text}\n{body}"



def _estimate_token_count(
    text: str, *, token_counter: Optional[Callable[[str], int]]
) -> int:
    token_cache: Dict[str, int] = {}
    normalized = _normalized_token_count(
        text,
        token_counter=token_counter,
        token_cache=token_cache,
    )
    if normalized is not None:
        return normalized
    return max(len(WORD_PATTERN.findall(text)), 1)


def phrases_overlap(first: Sequence[str], second: Sequence[str]) -> bool:
    """Return True when two token sequences share any contiguous overlap."""
    if not first or not second:
        return False

    len_first = len(first)
    len_second = len(second)

    for offset in range(-len_second + 1, len_first):
        overlap_start = max(0, offset)
        overlap_end = min(len_first, offset + len_second)
        if overlap_start >= overlap_end:
            continue

        match = True
        for index in range(overlap_start, overlap_end):
            if first[index] != second[index - offset]:
                match = False
                break

        if match:
            return True

    return False


def span_overlaps_placeholder(
    start: int, end: int, placeholder_spans: Sequence[Tuple[int, int]]
) -> bool:
    """Check if a span overlaps with any placeholder."""
    for p_start, p_end in placeholder_spans:
        if not (end <= p_start or start >= p_end):
            return True
    return False
