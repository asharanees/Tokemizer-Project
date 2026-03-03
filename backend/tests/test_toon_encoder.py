from __future__ import annotations

from services.optimizer import toon_encoder


def test_tabular_array_encoding() -> None:
    value = {"items": [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}]}

    output = toon_encoder.encode(value)

    assert output == "items[2] id,name\n  1,Alpha\n  2,Beta"


def test_delimiter_selection_prefers_low_quote_option() -> None:
    values = ["a,b", "c"]

    assert toon_encoder.select_delimiter(values) == "|"


def test_string_quoting_rules() -> None:
    value = {"note": "  spaced", "flag": "null", "dash": "-start"}

    output = toon_encoder.encode(value)

    assert output == 'note: "  spaced"\nflag: "null"\ndash: "-start"'


def test_key_folding_single_chain() -> None:
    value = {"a": {"b": {"c": 1}}}

    output = toon_encoder.encode(value)

    assert output == "a.b.c: 1"


def test_key_folding_collision_avoids_dotted_sibling() -> None:
    value = {"a": {"b": 1}, "a.b": 2}

    output = toon_encoder.encode(value)

    assert output == 'a\n  b: 1\n"a.b": 2'


def test_list_item_tabular_header_on_dash_line() -> None:
    value = [{"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], "tag": "x"}]

    output = toon_encoder.encode(value)

    assert output == "[1]\n  - items[2] id,name\n    1,A\n    2,B\n    tag: x"


def test_inline_primitive_array_encoding() -> None:
    value = {"tags": ["admin", "ops", "dev"]}

    output = toon_encoder.encode(value)

    assert output == "tags[3]: admin,ops,dev"
