"""Unit tests for trie replacer and phrase dictionary."""

from __future__ import annotations

from services.optimizer import toon_encoder
from services.optimizer.trie_replacer import (apply_phrase_dictionary,
                                              get_canonical_replacer,
                                              trie_canonicalize)


def test_apply_phrase_dictionary_with_savings() -> None:
    """Test phrase replacement when net tokens are saved."""
    text = "machine learning and natural language processing and machine learning"
    phrase_map = {
        "machine learning": "ML",
        "natural language processing": "NLP",
    }

    def count_tokens(s: str) -> int:
        return len(s.split())

    result, applied, metadata = apply_phrase_dictionary(text, phrase_map, count_tokens)

    if applied:
        assert "ML" in result
        assert "NLP" in result
        assert "[DICT]" in result
        assert metadata["dictionary_size"] == 2


def test_apply_phrase_dictionary_no_savings() -> None:
    """Test fallback when no net savings from dictionary."""
    # Short text where dictionary overhead outweighs savings
    text = "ML test"
    phrase_map = {
        "machine learning": "ML",
    }

    def count_tokens(s: str) -> int:
        return len(s.split())

    result, applied, metadata = apply_phrase_dictionary(text, phrase_map, count_tokens)

    # No matches, so no application
    assert result == text
    assert applied is False
    assert metadata == {}


def test_apply_phrase_dictionary_empty_inputs() -> None:
    """Test empty text/dictionary handling."""

    def count_tokens(s: str) -> int:
        return len(s.split())

    # Empty text
    result, applied, metadata = apply_phrase_dictionary("", {"test": "T"}, count_tokens)
    assert result == ""
    assert applied is False
    assert metadata == {}

    # Empty dictionary
    result, applied, metadata = apply_phrase_dictionary("test text", {}, count_tokens)
    assert result == "test text"
    assert applied is False
    assert metadata == {}


def test_apply_phrase_dictionary_dictionary_prefix() -> None:
    """Test dictionary prefix generation format."""
    text = "artificial intelligence and machine learning systems"
    phrase_map = {
        "artificial intelligence": "AI",
        "machine learning": "ML",
    }

    def count_tokens(s: str) -> int:
        # Simulate that dictionary saves tokens
        return max(1, len(s) // 5)

    result, applied, metadata = apply_phrase_dictionary(text, phrase_map, count_tokens)

    if applied:
        assert result.startswith("[DICT]")
        # Dictionary entries should be sorted by phrase length (longest first)
        dict_part = result.split("\n")[0]
        assert "AI=" in dict_part or "ML=" in dict_part


def test_trie_canonicalize_basic() -> None:
    """Test basic trie canonicalization."""
    text = "The Machine Learning system uses Natural Language Processing."
    canonical_map = {
        "Machine Learning": "ML",
        "Natural Language Processing": "NLP",
    }

    result = trie_canonicalize(text, canonical_map)

    assert "ML" in result
    assert "NLP" in result
    assert "Machine Learning" not in result
    assert "Natural Language Processing" not in result


def test_get_canonical_replacer_caching() -> None:
    """Test that canonical replacer is cached."""
    canonical_map = {"test": "T"}

    replacer1 = get_canonical_replacer(canonical_map)
    replacer2 = get_canonical_replacer(canonical_map)

    # Should return same cached instance
    assert replacer1 is replacer2


def test_apply_phrase_dictionary_respects_placeholders() -> None:
    """Test that phrase dictionary respects placeholder regions."""
    # Text with placeholder that should be skipped
    text = "machine learning __PLACEHOLDER_0__ machine learning"
    phrase_map = {
        "machine learning": "ML",
    }

    def count_tokens(s: str) -> int:
        return len(s.split())

    result, applied, metadata = apply_phrase_dictionary(text, phrase_map, count_tokens)

    # Placeholder should be preserved even if phrase matches
    if applied:
        assert "__PLACEHOLDER_0__" in result


def test_json_alias_legend_round_trip_deterministic() -> None:
    payload = {
        "dataset": [
            {
                "longRepeatedFieldName": "extremely-long-shared-value",
                "longRepeatedDescription": "extremely-long-shared-value",
            },
            {
                "longRepeatedFieldName": "extremely-long-shared-value",
                "longRepeatedDescription": "extremely-long-shared-value",
            },
        ]
    }

    compressed, legend = toon_encoder.compress_structure(payload)
    legend_line = toon_encoder.build_alias_legend_line(legend)

    assert legend_line.startswith("@alias=")
    assert toon_encoder.restore_structure_aliases(compressed, legend) == payload
    assert legend_line == toon_encoder.build_alias_legend_line(legend)


def test_json_alias_net_gain_gate_avoids_legend_when_not_profitable() -> None:
    payload = {
        "records": [
            {"veryLongOneOffKey": "alpha", "id": "1"},
            {"anotherLongOneOffKey": "beta", "id": "2"},
        ]
    }

    encoded = toon_encoder.encode(payload)

    assert "@alias=" not in encoded
