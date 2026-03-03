from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_DELIMITER_CANDIDATES: Tuple[str, ...] = (",", "|", "\t")
_RESERVED_LITERALS = {"null", "true", "false"}
_NUMERIC_LIKE_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_DEFAULT_FLATTEN_DEPTH = 4
_MIN_BLOCK_CHARS = 200
_ALIAS_MIN_LEN = 8
_ALIAS_MIN_REPETITIONS = 2
_ALIAS_KEY_PREFIX = "~k"
_ALIAS_VALUE_PREFIX = "~v"
_ALIAS_LEGEND_PREFIX = "@alias="


@dataclass(frozen=True)
class ToonEncodeOptions:
    indent: int = 2
    document_delimiter: str = "\n"
    enable_key_folding: bool = False
    flatten_depth: int = _DEFAULT_FLATTEN_DEPTH


def encode(value: Any, *, options: Optional[ToonEncodeOptions] = None) -> str:
    settings = options or _auto_options(value)
    lines = _encode_value(value, 0, settings)
    baseline = "\n".join(lines)

    compressed_value, legend = compress_structure(value)
    if not legend:
        return baseline
    compressed_lines = _encode_value(compressed_value, 0, settings)
    candidate = "\n".join([*compressed_lines, build_alias_legend_line(legend)])
    if len(candidate) < len(baseline):
        return candidate
    return baseline


def compress_structure(value: Any) -> Tuple[Any, Dict[str, Dict[str, str]]]:
    key_counts: Dict[str, int] = {}
    value_counts: Dict[str, int] = {}
    _collect_alias_candidates(value, key_counts, value_counts)

    key_aliases = _build_alias_table(
        key_counts,
        prefix=_ALIAS_KEY_PREFIX,
        skip_id_like_keys=True,
    )
    value_aliases = _build_alias_table(
        value_counts,
        prefix=_ALIAS_VALUE_PREFIX,
        skip_id_like_keys=False,
    )
    if not key_aliases and not value_aliases:
        return value, {}

    alias_value = _apply_aliases(value, key_aliases, value_aliases)
    legend: Dict[str, Dict[str, str]] = {}
    if key_aliases:
        legend["k"] = {alias: key for key, alias in key_aliases.items()}
    if value_aliases:
        legend["v"] = {alias: original for original, alias in value_aliases.items()}
    return alias_value, legend


def restore_structure_aliases(value: Any, legend: Dict[str, Dict[str, str]]) -> Any:
    if not legend:
        return value
    key_restore = legend.get("k", {})
    value_restore = legend.get("v", {})

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            rebuilt: Dict[str, Any] = {}
            for key, child in node.items():
                restored_key = key_restore.get(str(key), str(key))
                rebuilt[restored_key] = _walk(child)
            return rebuilt
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, str):
            return value_restore.get(node, node)
        return node

    return _walk(value)


def build_alias_legend_line(legend: Dict[str, Dict[str, str]]) -> str:
    return f"{_ALIAS_LEGEND_PREFIX}{json.dumps(legend, ensure_ascii=False, separators=(',', ':'))}"


def should_convert_block(value: Any, source_length: int) -> bool:
    if source_length <= 0:
        return False
    if not isinstance(value, (dict, list)):
        return False
    if source_length >= _MIN_BLOCK_CHARS:
        return True
    return _contains_tabular_array(value)


def select_delimiter(values: Iterable[str]) -> str:
    best = _DELIMITER_CANDIDATES[0]
    best_quotes: Optional[int] = None
    for delimiter in _DELIMITER_CANDIDATES:
        quotes = sum(1 for value in values if _needs_quotes(value, delimiter))
        if best_quotes is None or quotes < best_quotes:
            best = delimiter
            best_quotes = quotes
    return best


def should_fold_keys(value: Any) -> bool:
    return _max_fold_chain_length(value) > 1


def choose_flatten_depth(value: Any) -> int:
    max_chain = _max_fold_chain_length(value)
    if max_chain <= 1:
        return 0
    return min(_DEFAULT_FLATTEN_DEPTH, max_chain)


def _auto_options(value: Any) -> ToonEncodeOptions:
    enable_folding = should_fold_keys(value)
    flatten_depth = choose_flatten_depth(value) if enable_folding else 0
    return ToonEncodeOptions(
        enable_key_folding=enable_folding, flatten_depth=flatten_depth
    )


def _encode_value(value: Any, indent: int, options: ToonEncodeOptions) -> List[str]:
    if isinstance(value, dict):
        return _encode_object(value, indent, options)
    if isinstance(value, list):
        return _encode_array(value, indent, options)
    return [_indent_line(_encode_primitive(value, options.document_delimiter), indent)]


def _encode_object(
    value: Dict[str, Any], indent: int, options: ToonEncodeOptions
) -> List[str]:
    lines: List[str] = []
    siblings = {str(key) for key in value.keys()}
    for key, child in value.items():
        allow_dots = False
        key_str = str(key)
        if options.enable_key_folding and options.flatten_depth > 1:
            folded_key, folded_value, folded = _fold_key_chain(
                key_str,
                child,
                siblings,
                options.flatten_depth,
            )
            if folded:
                key_str = folded_key
                child = folded_value
                allow_dots = True
        lines.extend(
            _encode_object_entry(
                key_str,
                child,
                indent,
                options,
                allow_dots=allow_dots,
            )
        )
    return lines


def _encode_object_entry(
    key: str,
    value: Any,
    indent: int,
    options: ToonEncodeOptions,
    *,
    allow_dots: bool = False,
) -> List[str]:
    key_token = _encode_key(key, options.document_delimiter, allow_dots=allow_dots)
    if isinstance(value, dict):
        if not value:
            return [_indent_line(f"{key_token}: {{}}", indent)]
        lines = [_indent_line(key_token, indent)]
        lines.extend(_encode_object(value, indent + options.indent, options))
        return lines
    if isinstance(value, list):
        return _encode_array(value, indent, options, key=key_token)
    encoded_value = _encode_primitive(value, options.document_delimiter)
    return [_indent_line(f"{key_token}: {encoded_value}", indent)]


def _encode_array(
    value: Sequence[Any],
    indent: int,
    options: ToonEncodeOptions,
    *,
    key: Optional[str] = None,
    header_prefix: Optional[str] = None,
) -> List[str]:
    tabular_fields = _tabular_fields(value)
    if tabular_fields is not None:
        delimiter = _select_array_delimiter(value, fields=tabular_fields)
        header_prefix = _build_array_header_prefix(
            value,
            delimiter,
            indent,
            key=key,
            header_prefix=header_prefix,
        )
        return _encode_tabular_array(
            value,
            header_prefix,
            indent + options.indent,
            delimiter,
            tabular_fields,
            options,
        )

    if value and all(_is_primitive(item) for item in value):
        delimiter = _select_array_delimiter(value)
        header_prefix = _build_array_header_prefix(
            value,
            delimiter,
            indent,
            key=key,
            header_prefix=header_prefix,
        )
        encoded_values = [_encode_primitive(item, delimiter) for item in value]
        return [f"{header_prefix}: {delimiter.join(encoded_values)}"]

    delimiter = _select_array_delimiter(value)
    header_prefix = _build_array_header_prefix(
        value,
        delimiter,
        indent,
        key=key,
        header_prefix=header_prefix,
    )

    lines = [header_prefix]
    item_indent = indent + options.indent
    for item in value:
        lines.extend(_encode_list_item(item, item_indent, options, delimiter))
    return lines


def _build_array_header_prefix(
    value: Sequence[Any],
    delimiter: str,
    indent: int,
    *,
    key: Optional[str],
    header_prefix: Optional[str],
) -> str:
    delimiter_indicator = _delimiter_indicator(delimiter)
    header = f"[{len(value)}{delimiter_indicator}]"
    if header_prefix is None:
        prefix = _indent_line("", indent)
        if key is not None:
            return f"{prefix}{key}{header}"
        return f"{prefix}{header}"
    return f"{header_prefix}{header}"


def _encode_tabular_array(
    value: Sequence[Dict[str, Any]],
    header_prefix: str,
    row_indent: int,
    delimiter: str,
    fields: Sequence[str],
    options: ToonEncodeOptions,
) -> List[str]:
    optimized_fields = _optimize_header_row(fields)
    header_fields = [
        _encode_key(field, delimiter, allow_dots=False) for field in optimized_fields
    ]
    header_line = header_prefix
    if header_fields:
        header_line = f"{header_prefix} {delimiter.join(header_fields)}"

    lines = [header_line]
    for row in value:
        row_values = []
        for field in fields:
            cell = row.get(field)
            row_values.append(_encode_primitive(cell, delimiter))
        lines.append(_indent_line(delimiter.join(row_values), row_indent))
    return lines


def _encode_list_item(
    value: Any,
    indent: int,
    options: ToonEncodeOptions,
    delimiter: str,
) -> List[str]:
    prefix = f"{_indent_line('', indent)}- "
    if isinstance(value, dict):
        items = list(value.items())
        if not items:
            return [f"{prefix}{{}}"]
        if len(items) == 1:
            key, child = items[0]
            key_token = _encode_key(
                str(key), options.document_delimiter, allow_dots=False
            )
            if isinstance(child, list):
                return _encode_array(
                    child,
                    indent,
                    options,
                    header_prefix=f"{prefix}{key_token}",
                )
            if isinstance(child, dict):
                lines = [f"{prefix}{key_token}"]
                lines.extend(_encode_object(child, indent + options.indent, options))
                return lines
            encoded_value = _encode_primitive(child, options.document_delimiter)
            return [f"{prefix}{key_token}: {encoded_value}"]

        first_key, first_value = items[0]
        if isinstance(first_value, list) and _tabular_fields(first_value) is not None:
            key_token = _encode_key(
                str(first_key), options.document_delimiter, allow_dots=False
            )
            lines = _encode_array(
                first_value,
                indent,
                options,
                header_prefix=f"{prefix}{key_token}",
            )
            for key, child in items[1:]:
                lines.extend(
                    _encode_object_entry(
                        str(key),
                        child,
                        indent + options.indent,
                        options,
                        allow_dots=False,
                    )
                )
            return lines

        lines = [_indent_line("-", indent)]
        for key, child in items:
            lines.extend(
                _encode_object_entry(
                    str(key),
                    child,
                    indent + options.indent,
                    options,
                    allow_dots=False,
                )
            )
        return lines

    if isinstance(value, list):
        return _encode_array(value, indent, options, header_prefix=prefix)
    encoded_value = _encode_primitive(value, delimiter)
    return [f"{prefix}{encoded_value}"]


def _select_array_delimiter(
    value: Sequence[Any],
    *,
    fields: Optional[Sequence[str]] = None,
) -> str:
    candidates: List[str] = []
    if fields:
        candidates.extend(str(field) for field in fields)
    for item in value:
        if isinstance(item, str):
            candidates.append(item)
        elif isinstance(item, dict):
            for cell in item.values():
                if isinstance(cell, str):
                    candidates.append(cell)
    return select_delimiter(candidates) if candidates else _DELIMITER_CANDIDATES[0]


def _tabular_fields(value: Sequence[Any]) -> Optional[List[str]]:
    if not value or len(value) < 2:
        return None
    if not all(isinstance(item, dict) for item in value):
        return None
    first = value[0]
    if not isinstance(first, dict):
        return None
    fields = [str(key) for key in first.keys()]
    field_set = set(fields)
    for item in value:
        if not isinstance(item, dict):
            return None
        if {str(key) for key in item.keys()} != field_set:
            return None
        if any(not _is_primitive(val) for val in item.values()):
            return None
    return fields


def _contains_tabular_array(value: Any) -> bool:
    if isinstance(value, list) and _tabular_fields(value) is not None:
        return True
    if (
        isinstance(value, list)
        and len(value) >= 3
        and all(_is_primitive(item) for item in value)
    ):
        return True
    if isinstance(value, dict):
        return any(_contains_tabular_array(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tabular_array(child) for child in value)
    return False


def _max_fold_chain_length(value: Any) -> int:
    if isinstance(value, dict):
        max_length = 1
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            current_length = 1
            if _is_identifier_segment(key):
                current = child
                while isinstance(current, dict) and len(current) == 1:
                    next_key, next_value = next(iter(current.items()))
                    if not isinstance(next_key, str) or not _is_identifier_segment(
                        next_key
                    ):
                        break
                    current_length += 1
                    current = next_value
            max_length = max(max_length, current_length, _max_fold_chain_length(child))
        return max_length
    if isinstance(value, list):
        return max((_max_fold_chain_length(item) for item in value), default=1)
    return 1


def _fold_key_chain(
    key: str,
    value: Any,
    siblings: Iterable[str],
    max_depth: int,
) -> Tuple[str, Any, bool]:
    if not _is_identifier_segment(key):
        return key, value, False
    chain = [key]
    current = value
    while isinstance(current, dict) and len(current) == 1 and len(chain) < max_depth:
        next_key, next_value = next(iter(current.items()))
        if not isinstance(next_key, str) or not _is_identifier_segment(next_key):
            break
        chain.append(next_key)
        current = next_value
    if len(chain) <= 1:
        return key, value, False
    folded_key = ".".join(chain)
    if folded_key in siblings:
        return key, value, False
    return folded_key, current, True


def _encode_key(value: str, delimiter: str, *, allow_dots: bool) -> str:
    if _needs_quotes(value, delimiter, is_key=True, allow_dots=allow_dots):
        return _quote(value)
    return value


def _encode_primitive(value: Any, delimiter: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        if _needs_quotes(value, delimiter):
            return _quote(value)
        return value
    return _quote(str(value))


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "null"
    if value == 0.0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def _needs_quotes(
    value: str,
    delimiter: str,
    *,
    is_key: bool = False,
    allow_dots: bool = False,
) -> bool:
    if value == "":
        return True
    if value[0].isspace() or value[-1].isspace():
        return True
    if value.startswith("-"):
        return True
    lowered = value.lower()
    if lowered in _RESERVED_LITERALS:
        return True
    if _NUMERIC_LIKE_PATTERN.match(value):
        return True
    if any(char in value for char in (":", "\n", "\r")):
        return True
    if any(char in value for char in ("[", "]", "{", "}", "(", ")")):
        return True
    if "\\" in value or '"' in value:
        return True
    if delimiter and delimiter in value:
        return True
    if is_key and not allow_dots and "." in value:
        return True
    return False


def _quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _is_identifier_segment(value: str) -> bool:
    if not _IDENTIFIER_PATTERN.match(value):
        return False
    if value.lower() in _RESERVED_LITERALS:
        return False
    if _NUMERIC_LIKE_PATTERN.match(value):
        return False
    if value.startswith("-"):
        return False
    return True


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _collect_alias_candidates(
    node: Any,
    key_counts: Dict[str, int],
    value_counts: Dict[str, int],
) -> None:
    if isinstance(node, dict):
        for key, child in node.items():
            key_text = str(key)
            key_counts[key_text] = key_counts.get(key_text, 0) + 1
            _collect_alias_candidates(child, key_counts, value_counts)
        return
    if isinstance(node, list):
        for child in node:
            _collect_alias_candidates(child, key_counts, value_counts)
        return
    if isinstance(node, str):
        value_counts[node] = value_counts.get(node, 0) + 1


def _build_alias_table(
    counts: Dict[str, int],
    *,
    prefix: str,
    skip_id_like_keys: bool,
) -> Dict[str, str]:
    candidates: List[Tuple[str, int]] = []
    for token, count in counts.items():
        if count < _ALIAS_MIN_REPETITIONS or len(token) < _ALIAS_MIN_LEN:
            continue
        if _NUMERIC_LIKE_PATTERN.match(token):
            continue
        if skip_id_like_keys and _is_id_like_key(token):
            continue
        if token.lower() in _RESERVED_LITERALS:
            continue
        candidates.append((token, count))
    if not candidates:
        return {}

    candidates.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    reserved_tokens = set(counts.keys())
    alias_table: Dict[str, str] = {}
    alias_index = 0
    for token, _ in candidates:
        alias = f"{prefix}{alias_index}"
        while alias in reserved_tokens or alias in alias_table.values():
            alias_index += 1
            alias = f"{prefix}{alias_index}"
        alias_table[token] = alias
        alias_index += 1
    return alias_table


def _is_id_like_key(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("id")
        or lowered in {"uuid", "guid", "identifier"}
    )


def _apply_aliases(
    node: Any,
    key_aliases: Dict[str, str],
    value_aliases: Dict[str, str],
) -> Any:
    if isinstance(node, dict):
        rebuilt: Dict[str, Any] = {}
        for key, child in node.items():
            key_text = str(key)
            rebuilt[key_aliases.get(key_text, key_text)] = _apply_aliases(
                child,
                key_aliases,
                value_aliases,
            )
        return rebuilt
    if isinstance(node, list):
        return [_apply_aliases(item, key_aliases, value_aliases) for item in node]
    if isinstance(node, str):
        return value_aliases.get(node, node)
    return node


def _indent_line(text: str, indent: int) -> str:
    if indent <= 0:
        return text
    return f"{' ' * indent}{text}"


def _delimiter_indicator(delimiter: str) -> str:
    if delimiter in {"\t", "|"}:
        return delimiter
    return ""


def _optimize_header_row(fields: Sequence[str]) -> List[str]:
    if not fields:
        return []

    header_abbreviations = {
        "identifier": "id",
        "description": "desc",
        "timestamp": "ts",
        "quantity": "qty",
        "reference": "ref",
        "configuration": "cfg",
        "enabled": "on",
        "disabled": "off",
    }

    optimized: List[str] = []
    seen: Set[str] = set()

    for field in fields:
        lower = field.lower()
        candidate = header_abbreviations.get(lower, field)
        if len(candidate) >= len(field):
            candidate = field
        elif field.isupper():
            candidate = candidate.upper()
        elif field[:1].isupper():
            candidate = candidate.capitalize()

        if candidate in seen:
            return list(fields)
        seen.add(candidate)
        optimized.append(candidate)

    return optimized


__all__ = [
    "ToonEncodeOptions",
    "build_alias_legend_line",
    "choose_flatten_depth",
    "compress_structure",
    "encode",
    "restore_structure_aliases",
    "select_delimiter",
    "should_convert_block",
    "should_fold_keys",
]
