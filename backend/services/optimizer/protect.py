"""Utilities for parsing explicit <protect> tags in prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal


class ProtectTagError(ValueError):
    """Raised when <protect> tags are malformed."""


@dataclass(frozen=True)
class ProtectChunk:
    """Represents a parsed segment from a string containing <protect> tags."""

    type: Literal["text", "protected"]
    value: str


_TAG_PATTERN = re.compile(r"<(/)?protect\b([^>]*)>", re.IGNORECASE)


def _append_text(chunks: List[ProtectChunk], value: str) -> None:
    if not value:
        return
    if chunks and chunks[-1].type == "text":
        # Coalesce consecutive text chunks to avoid fragmentation.
        chunks[-1] = ProtectChunk("text", chunks[-1].value + value)
    else:
        chunks.append(ProtectChunk("text", value))


def parse_protect_tags(text: str) -> List[ProtectChunk]:
    """Parse <protect> tags into ordered text and protected chunks.

    Returns a list of :class:`ProtectChunk` instances describing the safe output
    order. Any malformed tag usage raises :class:`ProtectTagError` with a
    diagnostic message that includes the approximate offset of the problem.
    """

    if not text:
        return []

    chunks: List[ProtectChunk] = []
    stack: List[tuple[int, List[str]]] = []  # (start_index, buffer)
    last_index = 0

    for match in _TAG_PATTERN.finditer(text):
        start, end = match.span()
        preceding = text[last_index:start]
        if stack:
            stack[-1][1].append(preceding)
        else:
            _append_text(chunks, preceding)

        last_index = end

        is_closing = bool(match.group(1))
        suffix = match.group(2) or ""

        if is_closing:
            if suffix.strip():
                raise ProtectTagError(
                    f"Closing </protect> tag cannot contain attributes at index {start}."
                )
            if not stack:
                raise ProtectTagError(
                    f"Closing </protect> tag at index {start} does not match any opening tag."
                )

            open_start, buffer = stack.pop()
            protected_content = "".join(buffer)

            if stack:
                stack[-1][1].append(protected_content)
            else:
                chunks.append(ProtectChunk("protected", protected_content))
        else:
            stack.append((start, []))

    trailing = text[last_index:]
    if stack:
        stack[-1][1].append(trailing)
    else:
        _append_text(chunks, trailing)

    if stack:
        open_start, _ = stack[0]
        raise ProtectTagError(f"Unclosed <protect> tag starting at index {open_start}.")

    return chunks
