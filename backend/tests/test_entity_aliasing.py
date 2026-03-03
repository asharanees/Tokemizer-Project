from __future__ import annotations

from services.optimizer.entity_aliasing import alias_named_entities


def test_alias_named_entities_replaces_repeated_title_case() -> None:
    text = (
        "Acme Corporation filed a report. "
        "Acme Corporation released updates. "
        "Acme Corporation announced earnings. "
        "Acme Corporation hosted a call. "
        "Acme Corporation published guidance."
    )
    updated, applied, legend_entries, net_savings = alias_named_entities(
        text,
        nlp_model=None,
        placeholder_ranges=[],
        token_counter=lambda value: len(value.split()),
        min_occurrences=3,
        min_chars=12,
        max_aliases=3,
        alias_prefix="E",
        allowed_labels=(),
        reserved_tokens=[],
    )
    assert applied is True
    assert legend_entries == [("E1", "Acme Corporation")]
    assert net_savings > 0
    assert updated.count("Acme Corporation") == 1
    assert updated.count("E1") == 4


def test_alias_named_entities_skips_when_no_break_even_savings() -> None:
    text = (
        "Acme Corporation filed a report. "
        "Acme Corporation released updates. "
        "Acme Corporation announced earnings."
    )
    updated, applied, legend_entries, net_savings = alias_named_entities(
        text,
        nlp_model=None,
        placeholder_ranges=[],
        token_counter=len,
        min_occurrences=3,
        min_chars=12,
        max_aliases=3,
        alias_prefix="E",
        allowed_labels=(),
        reserved_tokens=[],
    )
    assert applied is False
    assert legend_entries is None
    assert net_savings == 0
    assert updated == text
