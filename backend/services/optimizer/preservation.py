from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import (Any, Callable, Dict, Iterable, List, Match, Optional,
                    Pattern, Set, Tuple)

from . import config, toon_encoder
from .placeholders import \
    build_placeholder_normalization_map as _build_placeholder_normalization_map
from .placeholders import get_placeholder_ranges as _get_placeholder_ranges
from .placeholders import get_placeholder_tokens as _get_placeholder_tokens
from .protect import ProtectChunk, parse_protect_tags

_RESTORE_PATTERN_CACHE: Dict[Tuple[Tuple[str, str], ...], Pattern[str]] = {}
_DIGIT_TOKEN_PATTERN = re.compile(r"(?<!\w)(?=[\w-]*\d)[\w-]+")


_DEFAULT_JSON_POLICY: Dict[str, Any] = {
    "default": False,
    "overrides": {},
    "minify": False,
}

_ALIAS_MIN_KEY_LENGTH = 12
_ALIAS_MIN_REPETITIONS = 2
_ALIAS_PREFIX = "k"
_ALIAS_SKIP_SUBSTRINGS = ("__JSON", "__TOON")


class _JsonPlaceholderState:
    """Utility to generate stable JSON placeholder tokens."""

    def __init__(self, preserved: Dict[str, Any]) -> None:
        self._tokens: List[str] = preserved.setdefault("json_tokens", [])
        self._literals: List[str] = preserved.setdefault("json_literals", [])
        self._strings: List[Dict[str, Any]] = preserved.setdefault("json_strings", [])

    def add_token(self, value: str) -> str:
        placeholder = f"__JSONTOK_{len(self._tokens)}__"
        self._tokens.append(value)
        return placeholder

    def add_literal(self, value: str) -> str:
        placeholder = f"__JSONLIT_{len(self._literals)}__"
        self._literals.append(value)
        return placeholder

    def add_string_bounds(
        self,
        path: Tuple[str, ...],
        original: str,
        compressible: bool,
    ) -> Tuple[str, str]:
        index = len(self._strings)
        open_token = f"__JSONSTR_OPEN_{index}__"
        close_token = f"__JSONSTR_CLOSE_{index}__"
        self._strings.append(
            {
                "open_token": open_token,
                "close_token": close_token,
                "path": list(path),
                "original": original,
                "compressible": compressible,
            }
        )
        return open_token, close_token


def _resolve_prefix_map(opt) -> Dict[str, str]:
    defaults = dict(config.PLACEHOLDER_PREFIXES)
    prefix_map = getattr(opt, "PLACEHOLDER_PREFIXES", None)
    if isinstance(prefix_map, dict):
        merged = defaults.copy()
        merged.update(prefix_map)
        return merged
    return defaults


def _get_restore_pattern(prefix_map: Dict[str, str]) -> Pattern[str]:
    cache_key = tuple(sorted(prefix_map.items()))
    if cache_key not in _RESTORE_PATTERN_CACHE:
        if not prefix_map:
            pattern = re.compile(r"__NEVER_MATCH__")
        else:
            options = "|".join(
                re.escape(prefix)
                for prefix in sorted(prefix_map.values(), key=len, reverse=True)
            )
            pattern = re.compile(rf"__(?P<prefix>{options})_(?P<index>\d+)__")
        _RESTORE_PATTERN_CACHE[cache_key] = pattern
    return _RESTORE_PATTERN_CACHE[cache_key]


def restore_placeholders(text: str, preserved: Dict, prefix_map: Dict[str, str]) -> str:
    """Restore placeholders while supporting nested placeholder expansion."""

    if not text:
        return text

    pattern = _get_restore_pattern(prefix_map)
    prefix_to_key = {prefix: key for key, prefix in prefix_map.items()}

    def replacer(match: Match[str]) -> str:
        prefix = match.group("prefix")
        index = int(match.group("index"))
        key = prefix_to_key.get(prefix)
        if key is None:
            return match.group(0)

        values = preserved.get(key, []) if isinstance(preserved, dict) else []
        if 0 <= index < len(values):
            return values[index]
        return match.group(0)

    # Re-run substitution while newly inserted content still contains
    # placeholder tokens (e.g., citation bodies holding number placeholders).
    previous = None
    current = text
    while previous != current and pattern.search(current):
        previous = current
        current = pattern.sub(replacer, current)
    return current


def _json_path_matches(pattern: str, path: Tuple[str, ...]) -> bool:
    if not pattern:
        return not path

    segments = [segment for segment in pattern.split(".") if segment]
    path_segments = list(path)

    def _match(pattern_index: int, path_index: int) -> bool:
        while pattern_index < len(segments):
            token = segments[pattern_index]
            if token == "**":
                if pattern_index == len(segments) - 1:
                    return True
                for offset in range(path_index, len(path_segments) + 1):
                    if _match(pattern_index + 1, offset):
                        return True
                return False

            if path_index >= len(path_segments):
                return False

            if token != "*" and token != path_segments[path_index]:
                return False

            pattern_index += 1
            path_index += 1

        return path_index == len(path_segments)

    return _match(0, 0)


def _should_compress_json_value(path: Tuple[str, ...], policy: Dict[str, Any]) -> bool:
    overrides = policy.get("overrides", {}) if isinstance(policy, dict) else {}
    if overrides:
        sorted_items = sorted(
            overrides.items(), key=lambda item: len(item[0]), reverse=True
        )
        for pattern, allow in sorted_items:
            if not isinstance(pattern, str) or not isinstance(allow, bool):
                continue
            if _json_path_matches(pattern, path):
                return allow

    default = policy.get("default") if isinstance(policy, dict) else None
    return bool(default) if isinstance(default, bool) else False


def _find_json_string_literals(
    text: str, start_pos: int
) -> Tuple[Dict[Tuple[str, ...], str], Dict[Tuple[str, ...], str]]:
    """
    Recursively scan JSON structure to extract original string literals and key literals.
    Returns two mappings:
    1. literals: path tuples to their original JSON string value representations
    2. key_literals: path tuples to their original JSON key representations
    This preserves escape sequences exactly as they appear in the source.
    """
    literals: Dict[Tuple[str, ...], str] = {}
    key_literals: Dict[Tuple[str, ...], str] = {}

    def _scan_value(pos: int, path: Tuple[str, ...]) -> int:
        """Scan a JSON value starting at pos, return end position"""
        # Skip whitespace
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1

        if pos >= len(text):
            return pos

        char = text[pos]

        # String value - capture the original literal
        if char == '"':
            str_start = pos
            pos += 1
            while pos < len(text):
                if text[pos] == "\\":
                    pos += 2  # Skip escaped character
                    continue
                if text[pos] == '"':
                    pos += 1
                    # Store the original JSON string literal (including quotes)
                    literals[path] = text[str_start:pos]
                    return pos
                pos += 1
            return pos

        # Object
        if char == "{":
            pos += 1
            first = True
            while pos < len(text):
                # Skip whitespace
                while pos < len(text) and text[pos] in " \t\r\n":
                    pos += 1
                if pos >= len(text):
                    break
                if text[pos] == "}":
                    return pos + 1
                if not first:
                    if text[pos] == ",":
                        pos += 1
                    while pos < len(text) and text[pos] in " \t\r\n":
                        pos += 1

                # Parse key (must be a string)
                if text[pos] != '"':
                    break
                key_literal_start = (
                    pos  # Start of the full key literal including quotes
                )
                key_start = pos + 1
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    if text[pos] == "\\":
                        pos += 2
                        continue
                    pos += 1
                if pos >= len(text):
                    break
                raw_key = text[key_start:pos]
                pos += 1  # Skip closing quote
                key_literal_end = pos  # End of the full key literal including quotes

                # Decode the key to get the actual Python string (for path lookups)
                try:
                    import json as json_module

                    decoded_key = json_module.loads(
                        text[key_literal_start:key_literal_end]
                    )
                except Exception:
                    decoded_key = raw_key

                # Store the original key literal (including quotes) for this path
                child_path = path + (decoded_key,)
                key_literals[child_path] = text[key_literal_start:key_literal_end]

                # Skip whitespace and colon
                while pos < len(text) and text[pos] in " \t\r\n":
                    pos += 1
                if pos < len(text) and text[pos] == ":":
                    pos += 1

                # Parse value with decoded key in path
                pos = _scan_value(pos, child_path)
                first = False
            return pos

        # Array
        if char == "[":
            pos += 1
            index = 0
            first = True
            while pos < len(text):
                # Skip whitespace
                while pos < len(text) and text[pos] in " \t\r\n":
                    pos += 1
                if pos >= len(text):
                    break
                if text[pos] == "]":
                    return pos + 1
                if not first:
                    if text[pos] == ",":
                        pos += 1
                    while pos < len(text) and text[pos] in " \t\r\n":
                        pos += 1

                pos = _scan_value(pos, path + (str(index),))
                index += 1
                first = False
            return pos

        # Other literals (null, true, false, numbers)
        literal_match = re.match(
            r"null|true|false|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text[pos:]
        )
        if literal_match:
            return pos + len(literal_match.group(0))

        return pos

    _scan_value(start_pos, ())
    return literals, key_literals


def _infer_block_indent(text: str, start_pos: int) -> str:
    line_start = text.rfind("\n", 0, start_pos) + 1
    prefix = text[line_start:start_pos]
    if prefix.strip():
        return ""
    return prefix


def _apply_indent(text: str, indent: str) -> str:
    if not indent:
        return text
    return "\n".join(
        f"{indent}{line}" if line else indent for line in text.splitlines()
    )


def _collect_json_keys(value: Any, counts: Dict[str, int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            counts[key_text] = counts.get(key_text, 0) + 1
            _collect_json_keys(child, counts)
        return
    if isinstance(value, list):
        for item in value:
            _collect_json_keys(item, counts)


def _should_alias_json_key(key: str, count: int) -> bool:
    if not key:
        return False
    if any(substr in key for substr in _ALIAS_SKIP_SUBSTRINGS):
        return False
    # Skip keys containing reserved characters that would break the legend format
    reserved_chars = {",", "=", "}"}
    if any(char in key for char in reserved_chars):
        return False
    return count >= _ALIAS_MIN_REPETITIONS or len(key) >= _ALIAS_MIN_KEY_LENGTH


def _build_json_key_alias_map(
    value: Any,
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    counts: Dict[str, int] = {}
    _collect_json_keys(value, counts)
    if not counts:
        return {}, []

    candidates = [
        (key, count)
        for key, count in counts.items()
        if _should_alias_json_key(key, count)
    ]
    if not candidates:
        return {}, []

    candidates.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    reserved = set(counts.keys())
    alias_map: Dict[str, str] = {}
    alias_pairs: List[Tuple[str, str]] = []
    alias_index = 1
    for key, _ in candidates:
        if key in alias_map:
            continue
        alias = f"{_ALIAS_PREFIX}{alias_index}"
        while alias in reserved or alias in alias_map.values():
            alias_index += 1
            alias = f"{_ALIAS_PREFIX}{alias_index}"
        alias_map[key] = alias
        alias_pairs.append((alias, key))
        alias_index += 1
    return alias_map, alias_pairs


def _apply_json_key_aliases(
    value: Any, alias_map: Dict[str, str]
) -> Tuple[Any, Dict[Tuple[str, ...], Tuple[str, ...]]]:
    path_map: Dict[Tuple[str, ...], Tuple[str, ...]] = {(): ()}

    def _walk(
        node: Any,
        original_path: Tuple[str, ...],
        alias_path: Tuple[str, ...],
    ) -> Any:
        if isinstance(node, dict):
            rebuilt: Dict[str, Any] = {}
            for key, child in node.items():
                key_text = str(key)
                alias_key = alias_map.get(key_text, key_text)
                child_original_path = original_path + (key_text,)
                child_alias_path = alias_path + (alias_key,)
                path_map[child_alias_path] = child_original_path
                rebuilt[alias_key] = _walk(child, child_original_path, child_alias_path)
            return rebuilt
        if isinstance(node, list):
            rebuilt_list: List[Any] = []
            for index, child in enumerate(node):
                index_text = str(index)
                child_original_path = original_path + (index_text,)
                child_alias_path = alias_path + (index_text,)
                path_map[child_alias_path] = child_original_path
                rebuilt_list.append(_walk(child, child_original_path, child_alias_path))
            return rebuilt_list
        return node

    return _walk(value, (), ()), path_map


def _build_json_alias_legend(alias_pairs: List[Tuple[str, str]]) -> str:
    if not alias_pairs:
        return ""
    entries = ",".join(f"{alias}={key}" for alias, key in alias_pairs)
    return f"@keys{{{entries}}}"


def _serialize_for_alias_comparison(
    value: Any,
    policy: Dict[str, Any],
    original_literals: Dict[Tuple[str, ...], str],
    key_literals: Dict[Tuple[str, ...], str],
    separators: Tuple[str, str],
    path_map: Optional[Dict[Tuple[str, ...], Tuple[str, ...]]] = None,
) -> str:
    local_preserved: Dict[str, Any] = {
        "json_tokens": [],
        "json_literals": [],
        "json_strings": [],
    }
    local_state = _JsonPlaceholderState(local_preserved)
    return _serialize_json_value(
        value,
        local_state,
        policy,
        (),
        original_literals,
        key_literals,
        separators,
        path_map,
    )


def _alias_map_saves_tokens(
    original_value: Any,
    alias_value: Any,
    original_literals: Dict[Tuple[str, ...], str],
    key_literals: Dict[Tuple[str, ...], str],
    separators: Tuple[str, str],
    token_counter: Callable[[str], int],
    alias_pairs: List[Tuple[str, str]],
    alias_path_map: Optional[Dict[Tuple[str, ...], Tuple[str, ...]]],
    policy: Dict[str, Any],
) -> bool:
    original_serialized = _serialize_for_alias_comparison(
        original_value,
        policy,
        original_literals,
        key_literals,
        separators,
    )
    alias_serialized = _serialize_for_alias_comparison(
        alias_value,
        policy,
        original_literals,
        key_literals,
        separators,
        alias_path_map,
    )
    legend = _build_json_alias_legend(alias_pairs)
    if not legend:
        return False
    original_tokens = token_counter(original_serialized)
    alias_tokens = token_counter(f"{legend}{alias_serialized}")
    return alias_tokens < original_tokens


def _serialize_json_value(
    value: Any,
    state: _JsonPlaceholderState,
    policy: Dict[str, Any],
    path: Tuple[str, ...],
    original_literals: Dict[Tuple[str, ...], str],
    key_literals: Dict[Tuple[str, ...], str],
    separators: Tuple[str, str],
    path_map: Optional[Dict[Tuple[str, ...], Tuple[str, ...]]] = None,
) -> str:
    lookup_path = path_map.get(path, path) if path_map else path
    if isinstance(value, dict):
        open_token = state.add_token("{")
        close_token = state.add_token("}")
        parts: List[str] = [open_token]
        items = list(value.items())
        key_separator, kv_separator = separators
        for index, (key, child) in enumerate(items):
            child_path = path + (str(key),)
            lookup_child_path = (
                path_map.get(child_path, child_path) if path_map else child_path
            )
            # Use original key literal only when the key is unchanged.
            original_key_literal = None
            if lookup_child_path and lookup_child_path[-1] == str(key):
                original_key_literal = key_literals.get(lookup_child_path)
            if original_key_literal:
                key_token = state.add_token(original_key_literal)
            else:
                key_literal = json.dumps(key, ensure_ascii=False)
                key_token = state.add_token(key_literal)
            parts.append(key_token)
            parts.append(kv_separator)
            parts.append(
                _serialize_json_value(
                    child,
                    state,
                    policy,
                    child_path,
                    original_literals,
                    key_literals,
                    separators,
                    path_map,
                )
            )
            if index < len(items) - 1:
                parts.append(key_separator)
        parts.append(close_token)
        return "".join(parts)

    if isinstance(value, list):
        open_token = state.add_token("[")
        close_token = state.add_token("]")
        parts = [open_token]
        key_separator, _ = separators
        for index, child in enumerate(value):
            parts.append(
                _serialize_json_value(
                    child,
                    state,
                    policy,
                    path + (str(index),),
                    original_literals,
                    key_literals,
                    separators,
                    path_map,
                )
            )
            if index < len(value) - 1:
                parts.append(key_separator)
        parts.append(close_token)
        return "".join(parts)

    if isinstance(value, str):
        allow_compress = _should_compress_json_value(lookup_path, policy)
        if not allow_compress:
            # Use the original JSON string literal if available to preserve escape sequences
            original_literal = original_literals.get(lookup_path)
            if original_literal:
                return state.add_literal(original_literal)
            # Fallback to re-serialization if original not found
            literal = json.dumps(value, ensure_ascii=False)
            return state.add_literal(literal)

        open_token, close_token = state.add_string_bounds(lookup_path, value, True)
        return f"{open_token}{value}{close_token}"

    literal = json.dumps(value, ensure_ascii=False)
    return state.add_literal(literal)


def _preserve_json_blocks(
    text: str,
    policy: Optional[Dict[str, Any]],
    preserved: Dict[str, Any],
    *,
    enable_toon_conversion: bool = False,
    enable_alias_json_keys: bool = False,
    token_counter: Optional[Callable[[str], int]] = None,
) -> str:
    if not text:
        return text

    if isinstance(policy, dict):
        default_value = (
            policy.get("default") if isinstance(policy.get("default"), bool) else False
        )
        override_map = (
            policy.get("overrides") if isinstance(policy.get("overrides"), dict) else {}
        )
        overrides = {
            str(path): bool(value)
            for path, value in override_map.items()
            if isinstance(path, str) and isinstance(value, bool)
        }
        minify_setting = policy.get("minify")
        if isinstance(minify_setting, bool):
            minify_value = minify_setting or default_value
        else:
            minify_value = default_value
        effective_policy = {
            "default": default_value,
            "overrides": overrides,
            "minify": minify_value,
        }
    else:
        effective_policy = dict(_DEFAULT_JSON_POLICY)

    decoder = json.JSONDecoder()
    state = _JsonPlaceholderState(preserved)
    pieces: List[str] = []
    last_index = 0
    position = 0
    length = len(text)
    found_block = False
    toon_blocks = preserved.setdefault("toon_blocks", [])
    toon_stats = preserved.setdefault(
        "toon_stats", {"conversions": 0, "bytes_saved": 0}
    )
    toon_prefix = config.PLACEHOLDER_PREFIXES.get("toon_blocks", "TOON")

    while position < length:
        char = text[position]
        if char not in "{[":
            position += 1
            continue

        try:
            parsed, end = decoder.raw_decode(text, position)
        except JSONDecodeError:
            position += 1
            continue
        consumed = end - position

        pieces.append(text[last_index:position])

        raw_json = text[position : position + consumed]
        if enable_toon_conversion and toon_encoder.should_convert_block(
            parsed, len(raw_json)
        ):
            toon_text = toon_encoder.encode(parsed)
            indent_prefix = _infer_block_indent(text, position)
            if indent_prefix:
                toon_text = _apply_indent(toon_text, indent_prefix)
            if len(toon_text) < len(raw_json):
                placeholder = f"__{toon_prefix}_{len(toon_blocks)}__"
                toon_blocks.append(toon_text)
                pieces.append(placeholder)
                position += consumed
                last_index = position
                found_block = True
                toon_stats["conversions"] += 1
                toon_stats["bytes_saved"] += len(raw_json) - len(toon_text)
                continue

        # Extract original string literals and key literals to preserve escape sequences
        original_literals, key_literals = _find_json_string_literals(raw_json, 0)

        separators = (",", ":") if effective_policy.get("minify") else (", ", ": ")
        use_aliasing = False
        alias_map: Dict[str, str] = {}
        alias_pairs: List[Tuple[str, str]] = []
        alias_value: Any = parsed
        alias_path_map: Optional[Dict[Tuple[str, ...], Tuple[str, ...]]] = None

        if enable_alias_json_keys and token_counter:
            alias_map, alias_pairs = _build_json_key_alias_map(parsed)
            if alias_map:
                alias_value, alias_path_map = _apply_json_key_aliases(parsed, alias_map)
                if _alias_map_saves_tokens(
                    parsed,
                    alias_value,
                    original_literals,
                    key_literals,
                    separators,
                    token_counter,
                    alias_pairs,
                    alias_path_map,
                    effective_policy,
                ):
                    use_aliasing = True

        serialized = _serialize_json_value(
            alias_value if use_aliasing else parsed,
            state,
            effective_policy,
            (),
            original_literals,
            key_literals,
            separators,
            alias_path_map if use_aliasing else None,
        )
        if use_aliasing:
            legend = _build_json_alias_legend(alias_pairs)
            serialized = f"{legend}{serialized}"
        pieces.append(serialized)
        position += consumed
        last_index = position
        found_block = True

    if not found_block:
        return text

    pieces.append(text[last_index:])
    return "".join(pieces)


def _restore_json_strings(text: str, preserved: Dict[str, Any]) -> str:
    entries = preserved.get("json_strings") if isinstance(preserved, dict) else None
    if not entries:
        return text

    for entry in entries:
        open_token = entry.get("open_token")
        close_token = entry.get("close_token")
        if not open_token or not close_token:
            continue

        start = text.find(open_token)
        if start == -1:
            continue
        content_start = start + len(open_token)
        end = text.find(close_token, content_start)
        if end == -1:
            continue

        compressible = bool(entry.get("compressible"))
        if compressible:
            raw_content = text[content_start:end]
            replacement = json.dumps(raw_content, ensure_ascii=False)
        else:
            replacement = json.dumps(entry.get("original", ""), ensure_ascii=False)

        text = f"{text[:start]}{replacement}{text[end + len(close_token):]}"

    return text


# These helpers operate on the PromptOptimizer instance to reuse its
# placeholder configuration and tokenizer/count methods where needed.


def _normalize_force_patterns(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []

    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        if not raw:
            continue
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def _compile_force_patterns(
    patterns: Iterable[str],
) -> List[Tuple[Pattern[str], int, bool]]:
    compiled: List[Tuple[Pattern[str], int, bool]] = []
    for index, raw in enumerate(patterns):
        is_regex = False
        pattern_text = raw
        if raw.lower().startswith("regex:"):
            is_regex = True
            pattern_text = raw[len("regex:") :].lstrip()
        try:
            if is_regex:
                compiled.append((re.compile(pattern_text), index, True))
            else:
                compiled.append((re.compile(re.escape(pattern_text)), index, False))
        except re.error:
            continue
    return compiled


def _resolve_force_configuration(
    opt, force_digits: Optional[bool]
) -> Tuple[List[Tuple[Pattern[str], int, bool]], bool]:
    base_patterns = _normalize_force_patterns(
        getattr(opt, "FORCE_PRESERVE_PATTERNS", config.FORCE_PRESERVE_PATTERNS)
    )
    compiled = _compile_force_patterns(base_patterns)

    default_force_digits = getattr(
        opt, "FORCE_PRESERVE_DIGITS", config.FORCE_PRESERVE_DIGITS
    )
    should_force_digits = (
        force_digits if force_digits is not None else default_force_digits
    )
    return compiled, bool(should_force_digits)


def _collect_forced_matches(
    text: str,
    compiled_patterns: Iterable[Tuple[Pattern[str], int, bool]],
    force_digits: bool,
) -> List[Tuple[int, int, str, int]]:
    matches: List[Tuple[int, int, str, int]] = []
    compiled_list = list(compiled_patterns)
    for pattern, order, _ in compiled_list:
        for match in pattern.finditer(text):
            value = match.group(0)
            if not value or config.PLACEHOLDER_PATTERN.fullmatch(value):
                continue
            matches.append((match.start(), match.end(), value, order))

    if force_digits:
        if compiled_list:
            digit_order = max(order for _, order, _ in compiled_list) + 1
        else:
            digit_order = 0

        for match in _DIGIT_TOKEN_PATTERN.finditer(text):
            value = match.group(0)
            if not value or config.PLACEHOLDER_PATTERN.fullmatch(value):
                continue
            matches.append((match.start(), match.end(), value, digit_order))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[3]))

    filtered: List[Tuple[int, int, str, int]] = []
    last_end = -1
    for start, end, value, order in matches:
        if start < last_end:
            continue
        filtered.append((start, end, value, order))
        last_end = end

    return filtered


def extract_and_preserve(
    opt,
    text: str,
    *,
    force_digits: Optional[bool] = None,
    json_policy: Optional[Dict[str, Any]] = None,
    enable_toon_conversion: bool = False,
    enable_alias_json_keys: bool = False,
) -> Tuple[str, Dict]:
    """Extract and preserve critical elements that should not be modified.

    Preserves:
    - Code blocks (```code``` and `code`)
    - Quoted text ("..." and '...')
    - Numbers (integers, decimals, percentages)
    - Citations and brackets [...]
    - URLs (http://, https://)
    - Explicit <protect>...</protect> regions
    - Configured forced tokens and digit-bearing tokens when enabled
    """
    preserved: Dict[str, Any] = {
        "code_blocks": [],
        "urls": [],
        "quotes": [],
        "numbers": [],
        "citations": [],
        "protected": [],
        "forced": [],
        "json_tokens": [],
        "json_literals": [],
        "json_strings": [],
        "toon_blocks": [],
        "toon_stats": {"conversions": 0, "bytes_saved": 0},
    }

    prefix_map = _resolve_prefix_map(opt)
    protect_prefix = prefix_map.get("protected", "PROTECT")

    parsed_chunks: List[ProtectChunk] = parse_protect_tags(text)
    if not parsed_chunks:
        result = text
    else:
        result_parts: List[str] = []
        for chunk in parsed_chunks:
            if chunk.type == "text":
                result_parts.append(chunk.value)
            else:
                placeholder = f"__{protect_prefix}_{len(preserved['protected'])}__"
                preserved["protected"].append(chunk.value)
                result_parts.append(placeholder)
        result = "".join(result_parts)

    # Preserve code blocks (multi-line and inline)
    code_prefix = prefix_map.get("code_blocks", "CODE")
    code_pattern = re.compile(
        r"(?P<fence>```|~~~)[\s\S]*?(?P=fence)|"
        r"(?P<tick>`{1,})(?=\S)(?:(?!(?P=tick)).)+?(?<=\S)(?P=tick)"
    )

    def code_replacer(match: Match[str]) -> str:
        placeholder = f"__{code_prefix}_{len(preserved['code_blocks'])}__"
        preserved["code_blocks"].append(match.group(0))
        return placeholder

    result = code_pattern.sub(code_replacer, result)

    # Preserve JSON structures
    result = _preserve_json_blocks(
        result,
        json_policy,
        preserved,
        enable_toon_conversion=enable_toon_conversion,
        enable_alias_json_keys=enable_alias_json_keys,
        token_counter=opt.count_tokens,
    )

    # Preserve URLs
    url_prefix = prefix_map.get("urls", "URL")
    url_pattern = re.compile(r'https?://[^\s<>"]+')

    def url_replacer(match: Match[str]) -> str:
        placeholder = f"__{url_prefix}_{len(preserved['urls'])}__"
        preserved["urls"].append(match.group(0))
        return placeholder

    result = url_pattern.sub(url_replacer, result)

    # Preserve quoted text
    quote_prefix = prefix_map.get("quotes", "QUOTE")
    quote_pattern = re.compile(r'"[^"]+"|\'[^\']+\'')
    quote_placeholders: Dict[str, str] = {}

    def quote_replacer(match: Match[str]) -> str:
        quoted_value = match.group(0)
        existing = quote_placeholders.get(quoted_value)
        if existing is not None:
            return existing

        placeholder = f"__{quote_prefix}_{len(preserved['quotes'])}__"
        preserved["quotes"].append(quoted_value)
        quote_placeholders[quoted_value] = placeholder
        return placeholder

    result = quote_pattern.sub(quote_replacer, result)

    compiled_patterns, should_force_digits = _resolve_force_configuration(
        opt,
        force_digits,
    )

    if compiled_patterns or should_force_digits:
        forced_prefix = prefix_map.get("forced", "FORCE")
        forced_matches = _collect_forced_matches(
            result, compiled_patterns, should_force_digits
        )
        if forced_matches:
            rebuilt: List[str] = []
            cursor = 0
            for start, end, value, _ in forced_matches:
                if start < cursor:
                    continue
                rebuilt.append(result[cursor:start])
                placeholder = f"__{forced_prefix}_{len(preserved['forced'])}__"
                preserved["forced"].append(value)
                rebuilt.append(placeholder)
                cursor = end
            rebuilt.append(result[cursor:])
            result = "".join(rebuilt)

    # Preserve numbers (including decimals, percentages, negative numbers)
    number_prefix = prefix_map.get("numbers", "NUM")
    number_pattern = re.compile(r"-?\b\d+(?:\.\d+)?%?\b")

    def number_replacer(match: Match[str]) -> str:
        placeholder = f"__{number_prefix}_{len(preserved['numbers'])}__"
        preserved["numbers"].append(match.group(0))
        return placeholder

    result = number_pattern.sub(number_replacer, result)

    # Preserve citations [1], [Author, Year], etc.
    citation_prefix = prefix_map.get("citations", "CIT")
    citation_pattern = re.compile(r"\[[^\]]+\]")

    def citation_replacer(match: Match[str]) -> str:
        placeholder = f"__{citation_prefix}_{len(preserved['citations'])}__"
        preserved["citations"].append(match.group(0))
        return placeholder

    result = citation_pattern.sub(citation_replacer, result)

    return result, preserved


def restore(opt, text: str, preserved: Dict) -> str:
    """Restore preserved elements using the configured placeholder prefixes."""
    prefix_map = _resolve_prefix_map(opt)
    restored = restore_placeholders(text, preserved, prefix_map)
    return _restore_json_strings(restored, preserved)


def get_placeholder_tokens(opt, preserved: Dict) -> Set[str]:
    """Return placeholder tokens generated during preservation."""
    prefix_map = _resolve_prefix_map(opt)
    return _get_placeholder_tokens(
        preserved or {},
        prefix_map=prefix_map,
        include_json_strings=True,
    )


def build_placeholder_normalization_map(opt, preserved: Dict) -> Dict[str, str]:
    """Return a deterministic mapping for placeholders based on their values."""
    prefix_map = _resolve_prefix_map(opt)
    return _build_placeholder_normalization_map(
        preserved or {},
        prefix_map=prefix_map,
        include_json_strings=True,
    )


def get_placeholder_ranges(
    opt, text: str, preserved: Optional[Dict]
) -> List[Tuple[int, int]]:
    """Locate preserved placeholder spans to avoid modifying them."""
    prefix_map = _resolve_prefix_map(opt)
    ranges = _get_placeholder_ranges(
        text,
        preserved or {},
        prefix_map=prefix_map,
        include_json_strings=True,
    )
    ranges.sort()
    return ranges
