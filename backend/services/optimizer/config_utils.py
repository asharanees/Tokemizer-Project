from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_env_float(variable: str, default: float) -> float:
    raw_value = os.environ.get(variable)
    if raw_value in (None, ""):
        return default

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s value '%s'; using default %.3f", variable, raw_value, default
        )
        return default


def get_env_int(variable: str, default: int) -> int:
    raw_value = os.environ.get(variable)
    if raw_value in (None, ""):
        return default

    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s value '%s'; using default %d", variable, raw_value, default
        )
        return default


def get_env_bool(variable: str, default: bool) -> bool:
    raw_value = os.environ.get(variable)
    if raw_value in (None, ""):
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning(
        "Invalid %s value '%s'; using default %s", variable, raw_value, default
    )
    return default


def sanitize_canonical_map(mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not mapping:
        return {}

    sanitized = {}
    for long_form, short_form in mapping.items():
        if not isinstance(long_form, str) or not isinstance(short_form, str):
            continue

        normalized_long = long_form.strip().lower()
        normalized_short = short_form.strip()

        if not normalized_long or not normalized_short:
            continue

        sanitized[normalized_long] = normalized_short

    return sanitized


def load_phrase_dictionary(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        logger.warning("Phrase dictionary not found at %s", path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive loading
        logger.warning("Failed to load phrase dictionary %s: %s", path, exc)
        return {}

    if isinstance(payload, dict):
        if "entries" in payload and isinstance(payload["entries"], list):
            entries = payload["entries"]
        elif "phrases" in payload and isinstance(payload["phrases"], dict):
            entries = list(payload["phrases"].items())
        else:
            entries = list(payload.items())
    elif isinstance(payload, list):
        entries = payload
    else:
        return {}

    phrase_map: Dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict):
            phrase = entry.get("phrase")
            symbol = entry.get("symbol")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            phrase, symbol = entry
        else:
            continue

        if not isinstance(phrase, str) or not isinstance(symbol, str):
            continue
        phrase = phrase.strip()
        symbol = symbol.strip()
        if not phrase or not symbol:
            continue
        phrase_map[phrase] = symbol

    return phrase_map


__all__ = [
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "load_phrase_dictionary",
    "sanitize_canonical_map",
]
