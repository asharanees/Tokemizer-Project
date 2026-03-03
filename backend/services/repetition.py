"""Utilities for detecting repeated token sequences beyond simple n-gram windows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RepeatedFragment:
    """Represents a repeated sequence of tokens identified in the text."""

    sequence: Tuple[str, ...]
    positions: Tuple[int, ...]

    @property
    def length(self) -> int:
        """Return the number of tokens in the repeated fragment."""

        return len(self.sequence)


class RepetitionDetector:
    """Detect repeated token sequences using a grammar-inspired approach.

    The detector incrementally grows repeated n-grams, similar to how the
    Sequitur algorithm promotes repeated digrams into grammar rules. By
    extending shared prefixes until a divergence occurs, we obtain maximal
    repeated fragments that can be compressed safely.
    """

    def __init__(self, min_length: int = 20, min_occurrences: int = 2) -> None:
        self.min_length = max(2, int(min_length))
        self.min_occurrences = max(2, int(min_occurrences))

    def find_repetitions(
        self,
        tokens: Sequence[str],
        *,
        min_length: int | None = None,
        min_occurrences: int | None = None,
    ) -> List[RepeatedFragment]:
        """Return repeated fragments longer than the configured threshold."""

        if not tokens:
            return []

        min_length = max(2, int(min_length or self.min_length))
        min_occurrences = max(2, int(min_occurrences or self.min_occurrences))
        token_count = len(tokens)

        if token_count < min_length * min_occurrences:
            return []

        # Stage 1: collect candidate windows that appear multiple times.
        window_positions = defaultdict(list)
        for start in range(0, token_count - min_length + 1):
            window = tuple(tokens[start : start + min_length])
            window_positions[window].append(start)

        fragments: Dict[Tuple[str, ...], List[int]] = {}

        for window, positions in window_positions.items():
            if len(positions) < min_occurrences:
                continue

            # Extend the shared sequence greedily to build a maximal fragment.
            extended_length = self._extend_sequence(tokens, positions, min_length)
            if extended_length < min_length:
                continue

            canonical_sequence = tuple(
                tokens[positions[0] : positions[0] + extended_length]
            )
            valid_positions = [
                idx
                for idx in positions
                if idx + extended_length <= token_count
                and tuple(tokens[idx : idx + extended_length]) == canonical_sequence
            ]

            if len(valid_positions) < min_occurrences:
                continue

            existing = fragments.get(canonical_sequence)
            if existing is None:
                fragments[canonical_sequence] = list(sorted(set(valid_positions)))
            else:
                merged = set(existing)
                merged.update(valid_positions)
                fragments[canonical_sequence] = list(sorted(merged))

        repeated_fragments = [
            RepeatedFragment(sequence=sequence, positions=tuple(sorted(pos_list)))
            for sequence, pos_list in fragments.items()
            if len(pos_list) >= min_occurrences
        ]

        # Sort largest fragments first so consumers can prioritise maximal spans.
        repeated_fragments.sort(
            key=lambda fragment: (-fragment.length, fragment.positions[0])
        )
        return repeated_fragments

    @staticmethod
    def _extend_sequence(
        tokens: Sequence[str], starts: Iterable[int], base_length: int
    ) -> int:
        """Extend a repeated sequence while all occurrences remain identical."""

        max_length = base_length
        starts = list(starts)
        token_count = len(tokens)

        while True:
            next_tokens = []
            for start in starts:
                next_index = start + max_length
                if next_index >= token_count:
                    next_tokens.append(None)
                else:
                    next_tokens.append(tokens[next_index])

            unique_next = set(next_tokens)
            if len(unique_next) != 1:
                break

            next_token = unique_next.pop()
            if next_token is None:
                break

            max_length += 1

        return max_length
