from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple

from . import config


def get_placeholder_tokens(
    preserved: Dict,
    *,
    prefix_map: Optional[Dict[str, str]] = None,
    include_json_strings: bool = False,
) -> Set[str]:
    tokens: Set[str] = set()
    prefixes = prefix_map or config.PLACEHOLDER_PREFIXES
    for key, prefix in prefixes.items():
        values = preserved.get(key, []) if isinstance(preserved, dict) else []
        for index in range(len(values)):
            tokens.add(f"__{prefix}_{index}__")

    if include_json_strings and isinstance(preserved, dict):
        for entry in preserved.get("json_strings", []) or []:
            open_token = entry.get("open_token")
            close_token = entry.get("close_token")
            if open_token:
                tokens.add(open_token)
            if close_token:
                tokens.add(close_token)

    return tokens


def build_placeholder_normalization_map(
    preserved: Dict,
    *,
    prefix_map: Optional[Dict[str, str]] = None,
    include_json_strings: bool = False,
) -> Dict[str, str]:
    normalization: Dict[str, str] = {}
    prefixes = prefix_map or config.PLACEHOLDER_PREFIXES
    for key, prefix in prefixes.items():
        values = preserved.get(key, []) if isinstance(preserved, dict) else []
        for index, value in enumerate(values):
            digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
            normalization[f"__{prefix}_{index}__"] = f"__{prefix}_{digest}__"

    if include_json_strings and isinstance(preserved, dict):
        for index, entry in enumerate(preserved.get("json_strings", []) or []):
            open_token = entry.get("open_token")
            close_token = entry.get("close_token")
            if open_token:
                normalization[open_token] = f"__JSONSTROPEN_{index}__"
            if close_token:
                normalization[close_token] = f"__JSONSTRCLOSE_{index}__"

    return normalization


def get_placeholder_ranges(
    text: str,
    preserved: Optional[Dict],
    *,
    prefix_map: Optional[Dict[str, str]] = None,
    include_json_strings: bool = False,
) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if not text:
        return ranges

    preserved_tokens = get_placeholder_tokens(
        preserved or {},
        prefix_map=prefix_map,
        include_json_strings=include_json_strings,
    )
    if not preserved_tokens:
        return ranges

    for token in preserved_tokens:
        start = 0
        while True:
            index = text.find(token, start)
            if index == -1:
                break
            ranges.append((index, index + len(token)))
            start = index + len(token)

    return ranges


def span_overlaps_placeholder(
    span: object, placeholder_ranges: List[Tuple[int, int]]
) -> bool:
    if not placeholder_ranges or span is None:
        return False

    start = getattr(span, "start_char", None)
    end = getattr(span, "end_char", None)
    if start is None or end is None:
        return False

    for placeholder_start, placeholder_end in placeholder_ranges:
        if start < placeholder_end and end > placeholder_start:
            return True

    return False


__all__ = [
    "build_placeholder_normalization_map",
    "get_placeholder_ranges",
    "get_placeholder_tokens",
    "span_overlaps_placeholder",
]
