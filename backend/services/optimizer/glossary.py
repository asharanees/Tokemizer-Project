"""Glossary aggregation utilities for aliasing/compression passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass
class GlossaryCollector:
    lines: List[str] = field(default_factory=list)
    net_token_savings: int = 0

    def add_entries(
        self,
        label: str,
        entries: Sequence[Tuple[str, str]],
        *,
        net_savings: int,
    ) -> None:
        if not entries or net_savings <= 0:
            return
        line = f"{label}: " + ", ".join(
            f"{alias}={value}" for alias, value in entries
        )
        if not line.strip():
            return
        self.lines.append(line)
        self.net_token_savings += net_savings

    def build_legend(
        self, token_counter: Callable[[str], int]
    ) -> Optional[str]:
        if not self.lines:
            return None
        categories = {
            line.split(":", 1)[0]: line for line in self.lines if ":" in line
        }
        ordered_labels = ("Labels", "Aliases", "Refs", "Glossary")
        normalized_lines: List[str] = []
        for label in ordered_labels:
            line = categories.get(label)
            if line:
                normalized_lines.append(line)
        for line in self.lines:
            if line not in normalized_lines:
                normalized_lines.append(line)
        if not normalized_lines:
            return None
        legend_body = "\n".join(normalized_lines)
        legend = f"Legend:\n{legend_body}"
        if self.net_token_savings <= token_counter(legend):
            return None
        return legend


    def has_entries(self) -> bool:
        return bool(self.lines)
