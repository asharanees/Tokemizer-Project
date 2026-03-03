"""Helpers for configuring the uvicorn runtime.

This module centralizes how we read runtime environment variables for the API
server so Docker, docker-compose, and tests can validate the same logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict

try:  # Optional performance extras
    import httptools  # noqa: F401

    _HAS_HTTPTOOLS = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_HTTPTOOLS = False

try:  # Optional performance extras
    import uvloop  # noqa: F401

    _HAS_UVLOOP = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_UVLOOP = False

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_KEEP_ALIVE = 120
DEFAULT_GRACEFUL_TIMEOUT = 120
DEFAULT_LIMIT_CONCURRENCY = 32
DEFAULT_WORKERS = 1

TIMEOUT_GRACEFUL_SHUTDOWN_KEY = "timeout_graceful_shutdown"

_ENV_ERROR_TEMPLATE = "{name} must be a positive integer, got '{raw}'"


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_positive_int(name: str, default: int) -> int:
    """Return a positive integer from *name* or *default* if unset."""

    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default

    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(_ENV_ERROR_TEMPLATE.format(name=name, raw=raw_value)) from exc

    if parsed <= 0:
        raise ValueError(_ENV_ERROR_TEMPLATE.format(name=name, raw=raw_value))

    return parsed


def build_uvicorn_kwargs() -> Dict[str, Any]:
    """Return keyword arguments for :func:`uvicorn.run` based on env vars."""

    kwargs: Dict[str, Any] = {
        "host": os.environ.get("UVICORN_HOST", DEFAULT_HOST),
        "port": _read_positive_int("PORT", DEFAULT_PORT),
        "timeout_keep_alive": DEFAULT_TIMEOUT_KEEP_ALIVE,
        TIMEOUT_GRACEFUL_SHUTDOWN_KEY: DEFAULT_GRACEFUL_TIMEOUT,
        "limit_concurrency": DEFAULT_LIMIT_CONCURRENCY,
        "workers": _read_positive_int("UVICORN_WORKERS", DEFAULT_WORKERS),
        "access_log": _read_bool("UVICORN_ACCESS_LOG", False),
    }

    if _HAS_UVLOOP:
        kwargs["loop"] = "uvloop"
    if _HAS_HTTPTOOLS:
        kwargs["http"] = "httptools"

    return kwargs
