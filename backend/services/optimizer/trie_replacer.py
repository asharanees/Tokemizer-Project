"""Trie-based pattern matching engine using Aho-Corasick algorithm.

This module provides O(N) single-pass multi-pattern replacement,
replacing the O(N×M) approach of iterating through M regex patterns.

Task #7: Trie-based replacer (Aho-Corasick)

Performance:
- Current: O(N × M) where M = number of patterns (~200+)
- This module: O(N) single pass regardless of pattern count

Usage:
    replacer = TrieReplacer({"approximately": "about", "utilize": "use"})
    result = replacer.replace("We approximately utilize this method")
    # Result: "We about use this method"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config


@dataclass
class TrieNode:
    """Node in the Aho-Corasick automaton."""

    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    fail: Optional["TrieNode"] = None
    output: List[Tuple[str, str]] = field(
        default_factory=list
    )  # (pattern, replacement)
    depth: int = 0


class TrieReplacer:
    """Single-pass multi-pattern replacement using Aho-Corasick automaton.

    This class builds an Aho-Corasick automaton from a dictionary of
    pattern -> replacement mappings and performs all replacements in
    a single O(N) pass through the text.

    Features:
    - Case-insensitive matching with case preservation
    - Word boundary detection (only matches whole words)
    - Greedy longest-match-first semantics
    - Thread-safe after construction

    Example:
        replacer = TrieReplacer({
            "approximately": "about",
            "utilize": "use",
            "in order to": "to",
        })
        result = replacer.replace("We approximately utilize this in order to help")
        # Result: "We about use this to help"
    """

    def __init__(
        self,
        replacements: Dict[str, str],
        case_sensitive: bool = False,
        word_boundaries: bool = True,
    ):
        """Initialize the Trie replacer.

        Args:
            replacements: Dictionary mapping patterns to replacements
            case_sensitive: If False, match case-insensitively but preserve original case
            word_boundaries: If True, only match at word boundaries
        """
        self._replacements = replacements
        self._case_sensitive = case_sensitive
        self._word_boundaries = word_boundaries
        self._root = TrieNode()
        self._pattern_count = 0

        # Build the automaton
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self) -> None:
        """Build the trie from replacement patterns."""
        # Sort by length descending for greedy matching
        sorted_patterns = sorted(
            self._replacements.items(), key=lambda x: len(x[0]), reverse=True
        )

        for pattern, replacement in sorted_patterns:
            if not pattern:
                continue

            search_pattern = pattern if self._case_sensitive else pattern.lower()
            node = self._root

            for i, char in enumerate(search_pattern):
                if char not in node.children:
                    node.children[char] = TrieNode(depth=i + 1)
                node = node.children[char]

            # Store the original pattern for case preservation
            node.output.append((pattern, replacement))
            self._pattern_count += 1

    def _build_failure_links(self) -> None:
        """Build failure links using BFS (Aho-Corasick algorithm)."""
        from collections import deque

        queue: deque[TrieNode] = deque()

        # Initialize failure links for depth-1 nodes
        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        # BFS to build failure links
        while queue:
            current = queue.popleft()

            for char, child in current.children.items():
                queue.append(child)

                # Follow failure links to find longest proper suffix
                fail_node = current.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail

                child.fail = fail_node.children[char] if fail_node else self._root

                # Merge outputs from failure link
                if child.fail and child.fail.output:
                    # Don't add shorter patterns if we have longer ones
                    pass  # Keep greedy behavior

    def _is_word_boundary(self, text: str, pos: int) -> bool:
        """Check if position is at a word boundary."""
        if pos <= 0:
            return True
        if pos >= len(text):
            return True

        prev_char = text[pos - 1]
        return not (prev_char.isalnum() or prev_char == "_")

    def _is_word_end(self, text: str, pos: int) -> bool:
        """Check if position is at end of a word."""
        if pos >= len(text):
            return True

        next_char = text[pos]
        return not (next_char.isalnum() or next_char == "_")

    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserve the case pattern of the original in the replacement."""
        if self._case_sensitive:
            return replacement

        if not original or not replacement:
            return replacement

        if replacement != replacement.lower():
            return replacement

        if original.isupper():
            return replacement.upper()
        if original.islower():
            return replacement.lower()
        if original[0].isupper() and original[1:].islower():
            return replacement.capitalize()

        return replacement

    def find_matches(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Find all pattern matches in the text.

        Returns:
            List of (start, end, pattern, replacement) tuples, sorted by position.
            Overlapping matches are resolved by greedy longest-first.
        """
        if not text or not self._pattern_count:
            return []

        search_text = text if self._case_sensitive else text.lower()
        matches: List[Tuple[int, int, str, str]] = []

        node = self._root
        i = 0

        while i < len(search_text):
            char = search_text[i]

            # Follow failure links until we find a match or reach root
            while node is not self._root and char not in node.children:
                node = node.fail if node.fail else self._root

            if char in node.children:
                node = node.children[char]

                # Check for pattern match at this position
                if node.output:
                    for pattern, replacement in node.output:
                        pattern_len = len(pattern)
                        start = i - pattern_len + 1
                        end = i + 1

                        # Verify word boundaries if required
                        if self._word_boundaries:
                            if not self._is_word_boundary(text, start):
                                continue
                            if not self._is_word_end(text, end):
                                continue

                        # Get original text for case preservation
                        original = text[start:end]
                        preserved_replacement = self._preserve_case(
                            original, replacement
                        )

                        matches.append((start, end, pattern, preserved_replacement))
                        break  # Greedy: take first (longest) match

            i += 1

        # Remove overlapping matches (keep earlier ones)
        if not matches:
            return []

        matches.sort(
            key=lambda m: (m[0], -(m[1] - m[0]))
        )  # Sort by start, then by length desc

        non_overlapping: List[Tuple[int, int, str, str]] = []
        last_end = -1

        for start, end, pattern, replacement in matches:
            if start >= last_end:
                non_overlapping.append((start, end, pattern, replacement))
                last_end = end

        return non_overlapping

    def replace(self, text: str) -> str:
        """Replace all pattern occurrences in the text.

        Args:
            text: Input text to process

        Returns:
            Text with all patterns replaced
        """
        matches = self.find_matches(text)

        if not matches:
            return text

        # Build result by replacing matched segments
        result_parts: List[str] = []
        last_end = 0

        for start, end, pattern, replacement in matches:
            # Add text before this match
            result_parts.append(text[last_end:start])
            # Add replacement
            result_parts.append(replacement)
            last_end = end

        # Add remaining text
        result_parts.append(text[last_end:])

        return "".join(result_parts)

    @property
    def pattern_count(self) -> int:
        """Return the number of patterns in the automaton."""
        return self._pattern_count


class CachedTrieReplacer:
    """Singleton cache for TrieReplacer instances.

    Caches replacers by their configuration to avoid rebuilding
    the automaton on every request.
    """

    _cache: Dict[Tuple[Tuple[Tuple[str, str], ...], bool, bool], TrieReplacer] = {}
    _max_cache_size: int = 10
    _lock = threading.Lock()

    @classmethod
    def get_replacer(
        cls,
        replacements: Dict[str, str],
        case_sensitive: bool = False,
        word_boundaries: bool = True,
    ) -> TrieReplacer:
        """Get or create a cached TrieReplacer.

        Args:
            replacements: Pattern -> replacement dictionary
            case_sensitive: Case sensitivity flag
            word_boundaries: Word boundary flag

        Returns:
            Cached or new TrieReplacer instance
        """
        cache_key = (
            tuple(sorted(replacements.items())),
            case_sensitive,
            word_boundaries,
        )

        with cls._lock:
            cached = cls._cache.get(cache_key)
            if cached is not None:
                return cached

            if len(cls._cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._cache))
                del cls._cache[oldest_key]

            replacer = TrieReplacer(replacements, case_sensitive, word_boundaries)
            cls._cache[cache_key] = replacer
            return replacer

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the replacer cache."""
        with cls._lock:
            cls._cache.clear()


def get_canonical_replacer(canonical_map: Dict[str, str]) -> TrieReplacer:
    """Get a cached canonical entity replacer."""
    return CachedTrieReplacer.get_replacer(
        canonical_map,
        case_sensitive=False,
        word_boundaries=True,
    )


def trie_canonicalize(text: str, canonical_map: Dict[str, str]) -> str:
    """Canonicalize entities using Trie-based O(N) matching.

    Args:
        text: Input text
        canonical_map: Entity -> canonical form mappings

    Returns:
        Text with entities canonicalized
    """
    replacer = get_canonical_replacer(canonical_map)
    return replacer.replace(text)


def apply_phrase_dictionary(
    text: str,
    phrase_map: Dict[str, str],
    count_tokens: Callable[[str], int],
) -> Tuple[str, bool, Dict[str, Any]]:
    if not text or not phrase_map:
        return text, False, {}

    replacer = CachedTrieReplacer.get_replacer(
        phrase_map,
        case_sensitive=False,
        word_boundaries=True,
    )

    segments = config.PLACEHOLDER_PATTERN.split(text)
    placeholders = config.PLACEHOLDER_PATTERN.findall(text)
    used: Dict[str, str] = {}
    replaced_segments: List[str] = []

    for segment in segments:
        matches = replacer.find_matches(segment)
        if matches:
            for _start, _end, phrase, symbol in matches:
                used[phrase] = symbol
            replaced_segments.append(replacer.replace(segment))
        else:
            replaced_segments.append(segment)

    if not used:
        return text, False, {}

    rebuilt: List[str] = []
    for index, segment in enumerate(replaced_segments):
        rebuilt.append(segment)
        if index < len(placeholders):
            rebuilt.append(placeholders[index])
    replaced_text = "".join(rebuilt)

    dictionary_entries = [
        f"{symbol}={phrase}"
        for phrase, symbol in sorted(used.items(), key=lambda item: -len(item[0]))
    ]
    prefix = "[DICT] " + "; ".join(dictionary_entries) + "\n"
    candidate = prefix + replaced_text

    if count_tokens(candidate) >= count_tokens(text):
        return text, False, {}

    return (
        candidate,
        True,
        {
            "dictionary_size": len(dictionary_entries),
            "used_phrases": list(used.keys()),
        },
    )


# Benchmark utilities
def benchmark_trie_vs_regex(
    text: str,
    replacements: Dict[str, str],
    iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark Trie-based vs regex-based replacement.

    Args:
        text: Sample text to process
        replacements: Pattern -> replacement dictionary
        iterations: Number of iterations for timing

    Returns:
        Dictionary with timing results
    """
    import re as regex_module
    import time

    # Build Trie replacer
    trie_replacer = TrieReplacer(replacements)

    # Build regex patterns
    regex_patterns = [
        (
            regex_module.compile(
                r"\b" + regex_module.escape(k) + r"\b", regex_module.IGNORECASE
            ),
            v,
        )
        for k, v in sorted(replacements.items(), key=lambda x: -len(x[0]))
    ]

    # Benchmark Trie
    start = time.perf_counter()
    for _ in range(iterations):
        _ = trie_replacer.replace(text)
    trie_time = (time.perf_counter() - start) * 1000 / iterations

    # Benchmark Regex
    start = time.perf_counter()
    for _ in range(iterations):
        regex_result = text
        for pattern, replacement in regex_patterns:
            regex_result = pattern.sub(replacement, regex_result)
    regex_time = (time.perf_counter() - start) * 1000 / iterations

    return {
        "trie_ms": trie_time,
        "regex_ms": regex_time,
        "speedup": regex_time / trie_time if trie_time > 0 else 0,
        "pattern_count": len(replacements),
        "text_length": len(text),
    }


__all__ = [
    "TrieNode",
    "TrieReplacer",
    "CachedTrieReplacer",
    "get_canonical_replacer",
    "apply_phrase_dictionary",
    "trie_canonicalize",
    "benchmark_trie_vs_regex",
]
