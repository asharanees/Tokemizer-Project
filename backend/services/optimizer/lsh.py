"""
Zero-dependency MinHash LSH implementation for semantic deduplication.

This module provides a pure-Python implementation of Locality Sensitive Hashing (LSH)
using MinHash signatures, enabling fast O(N) semantic similarity detection
without external dependencies like datasketch or heavy ML models.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple


class MinHashSignature:
    """Implementation of MinHash signature for a set of shingles."""

    def __init__(self, num_permutations: int = 128, num_perm: Optional[int] = None):
        if num_perm is not None:
            num_permutations = num_perm
        self.num_permutations = num_permutations
        self.signature = [float("inf")] * num_permutations
        # Generate fixed seeds for deterministic hashing across runs/platforms
        self.seeds = [(i * 2654435761) & 0xFFFFFFFF for i in range(num_permutations)]

    def _hash(self, text: str, seed: int) -> int:
        """Deterministic bitwise hash function."""
        h = seed
        for char in text:
            h = ((h << 5) - h) + ord(char)
            h &= 0xFFFFFFFF  # Keep as 32-bit unsigned
        return abs(h)

    def update(self, content: bytes) -> None:
        """Compatibility method for datasketch.MinHash. Updates signature with a single element."""
        # Use string representation of bytes for hashing
        text = content.decode("utf-8", errors="ignore")
        for i in range(self.num_permutations):
            h = self._hash(text, self.seeds[i])
            if h < self.signature[i]:
                self.signature[i] = h

    def jaccard(self, other: MinHashSignature) -> float:
        """Alias for jaccard_similarity to match datasketch API."""
        return self.jaccard_similarity(other)

    def jaccard_similarity(self, other: MinHashSignature) -> float:
        """Calculate Jaccard similarity between two signatures."""
        if self.num_permutations != other.num_permutations:
            return 0.0

        matches = 0
        for i in range(self.num_permutations):
            if self.signature[i] == other.signature[i]:
                matches += 1
        return matches / self.num_permutations


class LSHIndex:
    """Locality Sensitive Hashing index for fast candidate retrieval."""

    def __init__(self, num_bands: int = 16, num_permutations: int = 128):
        if num_bands <= 0:
            num_bands = 1
        if num_bands > num_permutations:
            num_bands = num_permutations

        self.num_bands = num_bands
        self.num_permutations = num_permutations
        # rows_per_band is now calculated dynamically to handle remainders
        self.buckets: Dict[str, List[str]] = {}
        self.signatures: Dict[str, MinHashSignature] = {}

    def insert(self, identifier: str, signature: MinHashSignature) -> None:
        """Insert a signature into the LSH index."""
        self.signatures[identifier] = signature
        sig = signature.signature

        for b in range(self.num_bands):
            # Dynamic calculation to distribute remainder permutations
            start = (b * self.num_permutations) // self.num_bands
            end = ((b + 1) * self.num_permutations) // self.num_bands
            # Create a bucket key for this band's segment of the signature
            band_segment = ",".join(map(str, sig[start:end]))
            bucket_key = f"b{b}:{band_segment}"

            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            self.buckets[bucket_key].append(identifier)

    def query(self, signature: MinHashSignature) -> Set[str]:
        """Find candidate identifiers that might be similar."""
        candidates: Set[str] = set()
        sig = signature.signature

        for b in range(self.num_bands):
            start = (b * self.num_permutations) // self.num_bands
            end = ((b + 1) * self.num_permutations) // self.num_bands
            band_segment = ",".join(map(str, sig[start:end]))
            bucket_key = f"b{b}:{band_segment}"

            if bucket_key in self.buckets:
                candidates.update(self.buckets[bucket_key])

        return candidates

    def get_similar(
        self, identifier: str, threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Get identifiers of similar items above the threshold."""
        signature = self.signatures.get(identifier)
        if not signature:
            return []

        candidates = self.query(signature)
        results: List[Tuple[str, float]] = []

        for candidate_id in candidates:
            if candidate_id == identifier:
                continue

            candidate_sig = self.signatures.get(candidate_id)
            if not candidate_sig:
                continue

            similarity = signature.jaccard_similarity(candidate_sig)
            if similarity >= threshold:
                results.append((candidate_id, similarity))

        # Sort by similarity descending
        return sorted(results, key=lambda x: x[1], reverse=True)


class SentenceLSHIndex:
    """
    High-level wrapper for sentence-level LSH indexing and deduplication.
    Replaces the previous datasketch dependency.
    """

    def __init__(self, threshold: float = 0.75, num_perm: int = 128):
        self._threshold = threshold
        self._num_perm = num_perm
        # Clamp bands to be at least 1 and at most num_perm
        safe_bands = max(1, min(16, num_perm // 2))
        self._lsh = LSHIndex(num_bands=safe_bands, num_permutations=num_perm)

    @property
    def is_available(self) -> bool:
        """Always available in this zero-dependency implementation."""
        return True

    def create_signature(self, text: str, shingle_size: int = 2) -> MinHashSignature:
        """Create a MinHash signature for the given text using shingles."""
        shingles = self.generate_shingles(text, shingle_size)
        sig = MinHashSignature(self._num_perm)
        for shingle in shingles:
            sig.update(shingle.encode("utf-8"))
        return sig

    def generate_shingles(self, text: str, shingle_size: int = 2) -> List[str]:
        """Generate word-level shingles from text."""
        normalized = self._normalize(text)
        words = normalized.split()
        if not words:
            return []

        if len(words) < shingle_size:
            return [normalized]

        shingles: List[str] = []
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i : i + shingle_size])
            shingles.append(shingle)
        return shingles

    def _normalize(self, text: str) -> str:
        """Standard normalization for shingling."""
        # Lowercase, remove accents, keep only alphanumeric and spaces
        text = text.lower()
        text = "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def add_sentence(self, identifier: str, signature: MinHashSignature) -> bool:
        """Register a sentence MinHash signature."""
        self._lsh.insert(identifier, signature)
        return True

    def query_similar(self, signature: MinHashSignature) -> List[str]:
        """Return identifiers of sentences similar to the given signature."""
        candidates = self._lsh.query(signature)
        results: List[str] = []

        for candidate_id in candidates:
            candidate_sig = self._lsh.signatures.get(candidate_id)
            if candidate_sig:
                similarity = signature.jaccard_similarity(candidate_sig)
                if similarity >= self._threshold:
                    results.append(candidate_id)

        return results


__all__ = ["MinHashSignature", "LSHIndex", "SentenceLSHIndex"]
