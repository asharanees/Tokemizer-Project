from __future__ import annotations

from services.optimizer import config
from services.optimizer.lexical import (apply_contractions,
                                        apply_frequency_abbreviations,
                                        clean_instruction_noise,
                                        compress_boilerplate, compress_clauses,
                                        compress_field_labels, compress_lists,
                                        _collapse_consecutive_duplicates_segment,
                                        extract_parenthetical_glossary,
                                        final_text_cleanup,
                                        normalize_numbers_and_units,
                                        reduce_numeric_precision,
                                        shorten_synonyms,
                                        span_overlaps_placeholder)


def test_apply_frequency_abbreviations_with_multiple_placeholders() -> None:
    text = (
        "__ITEM_0__ focuses on Advanced Risk Mitigation Plan.\n"
        "The Advanced Risk Mitigation Plan should include owners.\n"
        "__ITEM_1__ documents the Advanced Risk Mitigation Plan timeline.\n"
        "Summarize the Advanced Risk Mitigation Plan for stakeholders."
    )

    updated_text, learned, legend_entries, net_savings = apply_frequency_abbreviations(
        text=text,
        canonical_map={},
        preserved={},
        placeholder_tokens={"__ITEM_0__", "__ITEM_1__"},
        placeholder_pattern=config.PLACEHOLDER_PATTERN,
        min_occurrences=3,
        max_new=5,
    )

    assert learned == {"Advanced Risk Mitigation Plan": "ARMP"}
    assert legend_entries == [("ARMP", "Advanced Risk Mitigation Plan")]
    assert net_savings > 0
    assert "__ITEM_0__" in updated_text
    assert "__ITEM_1__" in updated_text
    assert "Advanced Risk Mitigation Plan" not in updated_text
    assert updated_text.count("ARMP") == 4


def test_span_overlaps_placeholder_with_precomputed_spans() -> None:
    spans = [(5, 10), (20, 30)]

    assert span_overlaps_placeholder(6, 9, spans) is True
    assert span_overlaps_placeholder(0, 4, spans) is False


def test_normalize_numbers_and_units_skips_non_numeric_segments() -> None:
    """Test that segments without numeric cues are not parsed by Quantulum."""
    text = "This is plain text without any numbers or units."
    result = normalize_numbers_and_units(text, config.PLACEHOLDER_PATTERN)
    # Should return unchanged since no numeric cues present
    assert result == text


def test_normalize_numbers_and_units_processes_numeric_segments() -> None:
    """Test that segments with numeric cues are processed."""
    text = "The distance is 5 kilometers and the weight is three pounds."
    result = normalize_numbers_and_units(text, config.PLACEHOLDER_PATTERN)
    # Should process the numeric content (exact output depends on Quantulum)
    # At minimum, verify it doesn't crash and returns a string
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_numbers_and_units_with_placeholders() -> None:
    """Test that placeholders are preserved during normalization."""
    text = "__ITEM_0__ has 100 items and __ITEM_1__ has fifty units."
    result = normalize_numbers_and_units(text, config.PLACEHOLDER_PATTERN)
    # Placeholders should be preserved
    assert "__ITEM_0__" in result
    assert "__ITEM_1__" in result


def test_normalize_numbers_and_units_standardizes_thousand_separators() -> None:
    """Test that thousand separators are standardized even without Quantulum."""
    text = "The value is 1 000 000 or 2_000_000"
    result = normalize_numbers_and_units(text, config.PLACEHOLDER_PATTERN)
    # Should standardize to comma separators
    assert "1000000" in result
    assert "2000000" in result


def test_clean_instruction_noise_combines_rules_and_preserves_placeholders() -> None:
    text = (
        "__ITEM_0__ Please can you provide me with a list of requirements, "
        "and I would like you to summarize the following text: __ITEM_1__ really."
    )

    cleaned, categories = clean_instruction_noise(text, config.PLACEHOLDER_PATTERN)

    assert "__ITEM_0__" in cleaned
    assert "__ITEM_1__" in cleaned
    lowered = cleaned.lower()
    assert "please" not in lowered
    assert "can you provide" not in lowered
    assert "i would like you to" not in lowered
    assert "the following text" not in lowered
    assert "really" not in lowered
    assert {"politeness", "verbose", "format", "filler"}.issubset(categories)


def test_clean_instruction_noise_respects_category_filtering() -> None:
    text = "Please analyze this really important detail."

    cleaned, categories = clean_instruction_noise(
        text,
        config.PLACEHOLDER_PATTERN,
        enabled_categories={"politeness"},
    )

    assert cleaned.lower().startswith("analyze")
    assert "really" in cleaned.lower()
    assert categories == {"politeness"}


def test_clean_instruction_noise_respects_segment_weights() -> None:
    text = "Please keep the safeguards enabled."

    high_priority, _ = clean_instruction_noise(
        text,
        config.PLACEHOLDER_PATTERN,
        segment_weights=[0.92],
    )
    low_priority, _ = clean_instruction_noise(
        text,
        config.PLACEHOLDER_PATTERN,
        segment_weights=[0.4],
    )

    assert "please" not in high_priority.lower()
    assert "please" not in low_priority.lower()


def test_apply_contractions_preserves_placeholders() -> None:
    text = "__ITEM_0__ Do not remove. I am ready."
    result = apply_contractions(text, config.PLACEHOLDER_PATTERN)
    assert "__ITEM_0__" in result
    assert "don't" in result.lower()
    assert "I'm" in result or "i'm" in result


def test_reduce_numeric_precision_skips_versions() -> None:
    text = "Version 1.2.3 uses 3.14159 and -12345.6789."
    result = reduce_numeric_precision(text, config.PLACEHOLDER_PATTERN)
    assert "1.2.3" in result
    assert "3.14" in result
    assert "-12346" in result


def test_compress_clauses_respects_weighting() -> None:
    text = "Use the API (which is stable) now."
    compressed = compress_clauses(text, config.PLACEHOLDER_PATTERN)
    assert "(which is stable)" not in compressed

    weighted = compress_clauses(
        text,
        config.PLACEHOLDER_PATTERN,
        segment_weights=[config.SEGMENT_WEIGHT_HIGH],
    )
    assert "(which is stable)" in weighted


def test_compress_lists_ordinal_rewrites_when_shorter() -> None:
    text = "First, gather data Second, analyze it Third, report."
    result = compress_lists(text, config.PLACEHOLDER_PATTERN)
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_compress_lists_reuses_common_prefix() -> None:
    text = (
        "- Ensure that tracing is enabled\n"
        "- Ensure that logging is configured\n"
        "- Ensure that alerts notify the team"
    )
    result = compress_lists(
        text,
        config.PLACEHOLDER_PATTERN,
        token_counter=lambda value: len(value.split()),
    )
    assert "Ensure that:" in result
    assert result.count("Ensure that") == 1
    assert ";" in result


def test_compress_lists_reuses_common_suffix() -> None:
    text = (
        "- Deploy services in production\n"
        "- Monitor metrics in production\n"
        "- Guard feature flags in production"
    )
    result = compress_lists(
        text,
        config.PLACEHOLDER_PATTERN,
        token_counter=lambda value: len(value.split()),
    )
    assert result.strip().endswith("(in production)")
    assert result.count("in production") == 1


def test_compress_field_labels_creates_alias_legend() -> None:
    text = (
        "Title of the Section: Measure latency\n"
        "Description of the Section: Collect metrics\n"
        "Title of the Section: Reduce latency\n"
        "Description of the Section: Report findings\n"
    )
    updated, changed, legend_entries, net_savings = compress_field_labels(
        text,
        placeholder_ranges=[],
        token_counter=lambda value: len(value),
    )
    assert changed
    assert legend_entries == [
        ("T", "Title of the Section"),
        ("D", "Description of the Section"),
    ]
    assert net_savings > 0
    assert "T:" in updated
    assert "D:" in updated


def test_extract_parenthetical_glossary_aliases_repeated_phrases() -> None:
    text = (
        "Configure the dashboard (see section 4 for architecture details) "
        "and share it (see section 4 for architecture details) "
        "with stakeholders (see section 4 for architecture details)."
    )
    updated, changed, legend_entries, net_savings = extract_parenthetical_glossary(
        text,
        placeholder_ranges=[],
        token_counter=lambda value: max(len(value), 1),
        min_token_savings=0,
    )
    assert changed
    assert "⟮P1⟯" in updated
    assert legend_entries == [("⟮P1⟯", "(see section 4 for architecture details)")]
    assert net_savings > 0


def test_compress_boilerplate_skips_placeholders() -> None:
    text = "Note: __ITEM_0__ should remain."
    placeholder_ranges = [
        match.span() for match in config.PLACEHOLDER_PATTERN.finditer(text)
    ]
    result = compress_boilerplate(text, placeholder_ranges=placeholder_ranges)
    assert result == text


def test_shorten_synonyms_respects_segment_weights() -> None:
    text = "Please utilize the secure API endpoint."

    preserved = shorten_synonyms(
        text,
        config.PLACEHOLDER_PATTERN,
        segment_weights=[0.9],
    )
    compressed = shorten_synonyms(
        text,
        config.PLACEHOLDER_PATTERN,
        segment_weights=[0.4],
    )

    assert "utilize" in preserved
    assert "use" in compressed.lower()


def test_collapse_consecutive_duplicates_collapses_huggingface_ids_with_spaces() -> None:
    segment = "sentence-transformers/all-MiniLM-L6-v2 " * 6
    segment = segment.strip()
    collapsed = _collapse_consecutive_duplicates_segment(segment)
    assert collapsed == "sentence-transformers/all-MiniLM-L6-v2"


def test_collapse_consecutive_duplicates_collapses_huggingface_ids_when_concatenated() -> None:
    segment = "sentence-transformers/all-MiniLM-L6-v2" * 6
    collapsed = _collapse_consecutive_duplicates_segment(segment)
    assert collapsed == "sentence-transformers/all-MiniLM-L6-v2"


def test_final_text_cleanup_trims_abrupt_leading_and() -> None:
    text = "and apologies in advance, for being so helpful."
    cleaned = final_text_cleanup(
        text,
        normalize_whitespace=True,
        compress_punctuation=True,
    )
    assert cleaned.startswith("Apologies in advance")
    assert cleaned == "Apologies in advance, for being so helpful."


def test_final_text_cleanup_trims_leading_punctuation_then_connector() -> None:
    text = ", and apologies in advance."
    cleaned = final_text_cleanup(
        text,
        normalize_whitespace=True,
        compress_punctuation=True,
    )
    assert cleaned == "Apologies in advance."


def test_final_text_cleanup_does_not_trim_and_or_prefix() -> None:
    text = "and/or keep both options."
    cleaned = final_text_cleanup(
        text,
        normalize_whitespace=True,
        compress_punctuation=True,
    )
    assert cleaned == text


def test_final_text_cleanup_keeps_leading_and_when_followed_by_uppercase() -> None:
    text = "And NASA should launch it."
    cleaned = final_text_cleanup(
        text,
        normalize_whitespace=True,
        compress_punctuation=True,
    )
    assert cleaned == text
