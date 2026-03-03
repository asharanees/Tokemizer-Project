"""Structural compression helpers (line-level factoring, etc.)."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]+")
_TRAILING_PUNCT = re.compile(r"[,:;\-–]+$")
_LEADING_PUNCT = re.compile(r"^[\s,:;\-–]+")


def _range_overlaps(span: Tuple[int, int], ranges: Sequence[Tuple[int, int]]) -> bool:
    if not ranges:
        return False
    start, end = span
    for r_start, r_end in ranges:
        if start < r_end and end > r_start:
            return True
    return False


def _tokenize_with_spans(line: str) -> List[Tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in _TOKEN_PATTERN.finditer(line)]


def _normalize_token(token: str) -> str:
    return token.lower() if token.isalnum() else token


def _common_prefix_len(tokens: Sequence[Sequence[str]]) -> int:
    if not tokens:
        return 0
    min_len = min(len(seq) for seq in tokens)
    length = 0
    for idx in range(min_len):
        token = tokens[0][idx]
        if all(seq[idx] == token for seq in tokens[1:]):
            length += 1
        else:
            break
    return length


def _common_suffix_len(tokens: Sequence[Sequence[str]]) -> int:
    if not tokens:
        return 0
    min_len = min(len(seq) for seq in tokens)
    length = 0
    for idx in range(1, min_len + 1):
        token = tokens[0][-idx]
        if all(seq[-idx] == token for seq in tokens[1:]):
            length += 1
        else:
            break
    return length


def _prefix_text_from_tokens(tokens: Sequence[Tuple[str, int, int]], length: int, line: str) -> str:
    if length <= 0 or length > len(tokens):
        return ""
    _, _, end = tokens[length - 1]
    return line[:end].strip()


def _suffix_text_from_tokens(tokens: Sequence[Tuple[str, int, int]], length: int, line: str) -> str:
    if length <= 0 or length > len(tokens):
        return ""
    _, start, _ = tokens[-length]
    return line[start:].strip()


def _strip_prefix(line: str, prefix_end: int) -> str:
    remainder = line[prefix_end:]
    remainder = _LEADING_PUNCT.sub("", remainder)
    return remainder.strip()


def _strip_suffix(line: str, suffix_start: int) -> str:
    remainder = line[:suffix_start]
    remainder = _TRAILING_PUNCT.sub("", remainder)
    return remainder.strip()


def _build_prefix_template(
    lines: Sequence[str],
    token_spans: Sequence[Sequence[Tuple[str, int, int]]],
    prefix_len: int,
) -> Optional[str]:
    if prefix_len < 2:
        return None
    prefix_text = _prefix_text_from_tokens(token_spans[0], prefix_len, lines[0])
    if not prefix_text or len(prefix_text) < 4:
        return None
    prefix_end = token_spans[0][prefix_len - 1][2]
    items: List[str] = []
    for line, spans in zip(lines, token_spans):
        end = spans[prefix_len - 1][2]
        item = _strip_prefix(line, end)
        if not item:
            return None
        items.append(item)
    delimiter = "" if prefix_text.endswith((":","-","–")) else ":"
    return f"{prefix_text}{delimiter} " + "; ".join(items)


def _format_enumerated_prefix(label: str, items: Sequence[str]) -> Optional[str]:
    cleaned = _LEADING_PUNCT.sub("", label).strip()
    cleaned = _TRAILING_PUNCT.sub("", cleaned).strip()
    if len(cleaned) < 3:
        return None
    delimiter = ":" if cleaned and cleaned[-1].isalnum() else ""
    return f"{cleaned}{delimiter} " + "; ".join(items)


def _build_enumerated_prefix(
    lines: Sequence[str],
    token_spans: Sequence[Sequence[Tuple[str, int, int]]],
    prefix_len: int,
) -> Optional[str]:
    if prefix_len < 1:
        return None
    prefix_text = _prefix_text_from_tokens(token_spans[0], prefix_len, lines[0])
    if not prefix_text:
        return None
    items: List[str] = []
    for line, spans in zip(lines, token_spans):
        end = spans[prefix_len - 1][2]
        item = _strip_prefix(line, end)
        if not item:
            return None
        items.append(item)
    return _format_enumerated_prefix(prefix_text, items)


def _build_enumerated_suffix(
    lines: Sequence[str],
    token_spans: Sequence[Sequence[Tuple[str, int, int]]],
    suffix_len: int,
) -> Optional[str]:
    if suffix_len < 1:
        return None
    suffix_text = _suffix_text_from_tokens(token_spans[0], suffix_len, lines[0])
    if not suffix_text:
        return None
    items: List[str] = []
    for line, spans in zip(lines, token_spans):
        start = spans[-suffix_len][1]
        item = _strip_suffix(line, start)
        if not item:
            return None
        items.append(item)
    return _format_enumerated_prefix(suffix_text, items)


def _build_suffix_template(
    lines: Sequence[str],
    token_spans: Sequence[Sequence[Tuple[str, int, int]]],
    suffix_len: int,
) -> Optional[str]:
    if suffix_len < 2:
        return None
    suffix_text = _suffix_text_from_tokens(token_spans[0], suffix_len, lines[0])
    if not suffix_text or len(suffix_text) < 4:
        return None
    suffix_start = token_spans[0][-suffix_len][1]
    items: List[str] = []
    for line, spans in zip(lines, token_spans):
        start = spans[-suffix_len][1]
        item = _strip_suffix(line, start)
        if not item:
            return None
        items.append(item)
    return " / ".join(items) + f" + {suffix_text}"


def _strip_line_ending(line: str) -> Tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _eligible_line(line: str) -> bool:
    if not line.strip():
        return False
    if "__" in line or "TOON" in line:
        return False
    return True


def compress_repeated_prefix_suffix(
    text: str,
    *,
    placeholder_ranges: Sequence[Tuple[int, int]],
    token_counter: Callable[[str], int],
    min_group_size: int = 3,
) -> str:
    if not text:
        return text

    lines = text.splitlines(keepends=True)
    positions: List[Tuple[int, int]] = []
    cursor = 0
    for line in lines:
        start = cursor
        cursor += len(line)
        positions.append((start, cursor))

    rebuilt: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        line_start, line_end = positions[index]
        if (
            not _eligible_line(line)
            or _range_overlaps((line_start, line_end), placeholder_ranges)
        ):
            rebuilt.append(line)
            index += 1
            continue

        block_lines: List[str] = []
        block_positions: List[Tuple[int, int]] = []
        j = index
        while j < len(lines):
            candidate = lines[j]
            cand_start, cand_end = positions[j]
            if not _eligible_line(candidate):
                break
            if _range_overlaps((cand_start, cand_end), placeholder_ranges):
                break
            block_lines.append(candidate)
            block_positions.append((cand_start, cand_end))
            j += 1
        if len(block_lines) < min_group_size:
            rebuilt.append(line)
            index += 1
            continue

        best_rewrite = None
        best_span = (index + 1, index + 1)
        best_savings = 0

        for end in range(index + min_group_size, j + 1):
            candidate_lines = block_lines[: end - index]
            normalized: List[List[str]] = []
            spans: List[List[Tuple[str, int, int]]] = []
            stripped_lines: List[str] = []
            line_ending = ""
            for entry in candidate_lines:
                stripped, ending = _strip_line_ending(entry)
                if not stripped.strip():
                    break
                token_spans = _tokenize_with_spans(stripped)
                if len(token_spans) < 2:
                    break
                spans.append(token_spans)
                normalized.append([_normalize_token(tok) for tok, _, _ in token_spans])
                stripped_lines.append(stripped)
                line_ending = ending
            else:
                prefix_len = _common_prefix_len(normalized)
                suffix_len = _common_suffix_len(normalized)
                prefix_template = _build_prefix_template(
                    stripped_lines, spans, prefix_len
                )
                suffix_template = _build_suffix_template(
                    stripped_lines, spans, suffix_len
                )
                original = "".join(candidate_lines)
                original_tokens = token_counter(original)
                if prefix_template:
                    candidate = prefix_template + line_ending
                    savings = original_tokens - token_counter(candidate)
                    if savings > best_savings:
                        best_savings = savings
                        best_rewrite = candidate
                        best_span = (index, end)
                if suffix_template:
                    candidate = suffix_template + line_ending
                    savings = original_tokens - token_counter(candidate)
                    if savings > best_savings:
                        best_savings = savings
                        best_rewrite = candidate
                        best_span = (index, end)

        if best_rewrite and best_savings > 0:
            rebuilt.append(best_rewrite)
            index = best_span[1]
            continue

        rebuilt.append(line)
        index += 1

    return "".join(rebuilt)


def compress_enumerated_prefix_suffix(
    text: str,
    *,
    placeholder_ranges: Sequence[Tuple[int, int]],
    token_counter: Callable[[str], int],
    min_group_size: int = 3,
) -> str:
    if not text:
        return text

    lines = text.splitlines(keepends=True)
    positions: List[Tuple[int, int]] = []
    cursor = 0
    for line in lines:
        start = cursor
        cursor += len(line)
        positions.append((start, cursor))

    rebuilt: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        line_start, line_end = positions[index]
        if (
            not _eligible_line(line)
            or _range_overlaps((line_start, line_end), placeholder_ranges)
        ):
            rebuilt.append(line)
            index += 1
            continue

        block_lines: List[str] = []
        j = index
        while j < len(lines):
            candidate = lines[j]
            cand_start, cand_end = positions[j]
            if not _eligible_line(candidate):
                break
            if _range_overlaps((cand_start, cand_end), placeholder_ranges):
                break
            block_lines.append(candidate)
            j += 1
        if len(block_lines) < min_group_size:
            rebuilt.append(line)
            index += 1
            continue

        best_rewrite = None
        best_span = (index + 1, index + 1)
        best_savings = 0

        for end in range(index + min_group_size, j + 1):
            candidate_lines = block_lines[: end - index]
            normalized: List[List[str]] = []
            spans: List[List[Tuple[str, int, int]]] = []
            stripped_lines: List[str] = []
            line_ending = ""
            for entry in candidate_lines:
                stripped, ending = _strip_line_ending(entry)
                if not stripped.strip():
                    break
                token_spans = _tokenize_with_spans(stripped)
                if len(token_spans) < 2:
                    break
                spans.append(token_spans)
                normalized.append([_normalize_token(tok) for tok, _, _ in token_spans])
                stripped_lines.append(stripped)
                line_ending = ending
            else:
                prefix_len = _common_prefix_len(normalized)
                suffix_len = _common_suffix_len(normalized)
                prefix_template = _build_enumerated_prefix(
                    stripped_lines, spans, prefix_len
                )
                suffix_template = _build_enumerated_suffix(
                    stripped_lines, spans, suffix_len
                )
                original = "".join(candidate_lines)
                original_tokens = token_counter(original)
                if prefix_template:
                    candidate = prefix_template + line_ending
                    savings = original_tokens - token_counter(candidate)
                    if savings > best_savings:
                        best_savings = savings
                        best_rewrite = candidate
                        best_span = (index, end)
                if suffix_template:
                    candidate = suffix_template + line_ending
                    savings = original_tokens - token_counter(candidate)
                    if savings > best_savings:
                        best_savings = savings
                        best_rewrite = candidate
                        best_span = (index, end)

        if best_rewrite and best_savings > 0:
            rebuilt.append(best_rewrite)
            index = best_span[1]
            continue

        rebuilt.append(line)
        index += 1

    return "".join(rebuilt)
