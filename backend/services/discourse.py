"""Discourse segmentation helpers for prioritizing prompt sections."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DiscourseSegment:
    """Container describing a discourse-aware section of text."""

    text: str
    label: str
    confidence: float = 0.0


class DiscourseAnalyzer:
    """Segment and classify prompt sections using fast lexical heuristics.

    This intentionally avoids heavyweight parsing dependencies to keep request
    latency predictable in production.
    """

    _HEADING_PATTERN = re.compile(r"(?<!\w)([A-Z][A-Za-z0-9 ]{0,40}):")
    _MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
    _BULLET_PATTERN = re.compile(
        r"^\s*(?:[-*•+]|(?:\(?\d+\)?[\.)]))\s+\S", re.MULTILINE
    )
    _CODE_FENCE_PATTERN = re.compile(r"```")
    _EXAMPLE_PATTERN = re.compile(r"\b(example|for example|e\.g\.|sample)\b", re.I)
    _BACKGROUND_PATTERN = re.compile(r"\b(background|context|overview|notes?)\b", re.I)
    _CONSTRAINT_PATTERN = re.compile(
        r"\b(must|must not|should|should not|do not|don't|never|required)\b", re.I
    )
    _INSTRUCTION_PATTERN = re.compile(
        r"\b(please|respond|return|output|write|generate|provide|follow)\b", re.I
    )

    def __init__(self, directive_keywords: Optional[Iterable[str]] = None):
        self._directive_keywords = {kw.lower() for kw in directive_keywords or []}

    def segment(self, text: str) -> List[DiscourseSegment]:
        """Return discourse-aware segments for the provided text."""

        blocks = self._basic_segment(text)
        segments: List[DiscourseSegment] = []

        for block in blocks:
            label, confidence = self._classify_block(block)
            segments.append(
                DiscourseSegment(text=block, label=label, confidence=confidence)
            )

        return segments

    def _basic_segment(self, text: str) -> List[str]:
        """Split text using whitespace and heading heuristics."""

        if not text or not text.strip():
            return []

        sections = [
            section.strip() for section in text.split("\n\n") if section.strip()
        ]

        if len(sections) > 1:
            return sections

        if self._MARKDOWN_HEADING_PATTERN.search(text):
            blocks = re.split(r"\n(?=#{1,6}\s+\S)", text)
            return [block.strip() for block in blocks if block.strip()]

        if self._BULLET_PATTERN.search(text):
            lines = [line.rstrip() for line in text.splitlines()]
            blocks: List[str] = []
            buffer: List[str] = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if buffer:
                        blocks.append("\n".join(buffer).strip())
                        buffer = []
                    continue
                if self._BULLET_PATTERN.match(line) and buffer:
                    blocks.append("\n".join(buffer).strip())
                    buffer = [line]
                else:
                    buffer.append(line)
            if buffer:
                blocks.append("\n".join(buffer).strip())
            return blocks or [text]

        matches = list(self._HEADING_PATTERN.finditer(text))

        if not matches:
            return [text]

        segments: List[str] = []
        last_index = 0

        for idx, match in enumerate(matches):
            start = match.start()

            if start > last_index:
                prefix = text[last_index:start].strip()
                if prefix:
                    segments.append(prefix)

            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                segments.append(section_text)

            last_index = end

        return segments or [text]

    def _classify_block(self, block: str) -> Tuple[str, float]:
        """Assign a discourse label to the provided block."""

        block_stripped = block.strip()
        if not block_stripped:
            return "background", 0.0
        return self._heuristic_label(block_stripped)

    def _heuristic_label(self, text: str) -> Tuple[str, float]:
        """Fallback heuristic labelling based on lexical cues."""

        lowered = text.lower()

        if any(
            lowered.startswith(prefix) for prefix in ("instruction:", "instructions:")
        ):
            return "instruction", 0.9

        keyword_hits = sum(
            1
            for keyword in self._directive_keywords
            if re.search(r"\\b" + re.escape(keyword) + r"\\b", lowered)
        )
        has_constraint = bool(self._CONSTRAINT_PATTERN.search(text)) or keyword_hits > 0
        has_instruction = bool(self._INSTRUCTION_PATTERN.search(text))
        has_example = bool(self._EXAMPLE_PATTERN.search(text)) or bool(
            self._CODE_FENCE_PATTERN.search(text)
        )
        has_background = bool(self._BACKGROUND_PATTERN.search(text))

        if has_constraint:
            confidence = 0.75 + min(0.2, keyword_hits * 0.05)
            return "constraint", min(confidence, 0.95)

        if has_example and not has_instruction:
            return "example", 0.65

        if has_instruction:
            return "instruction", 0.6 if not has_example else 0.55

        if has_background:
            return "background", 0.65

        return "background", 0.3
