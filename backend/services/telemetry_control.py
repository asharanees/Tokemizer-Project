"""Runtime toggle for performance telemetry collection."""

from __future__ import annotations

import threading

__all__ = ["is_enabled", "set_enabled"]

_telemetry_enabled = False
_telemetry_lock = threading.Lock()


def is_enabled() -> bool:
    """Return whether telemetry is currently enabled."""
    with _telemetry_lock:
        return _telemetry_enabled


def set_enabled(value: bool) -> None:
    """Toggle telemetry collection on or off."""
    global _telemetry_enabled
    with _telemetry_lock:
        _telemetry_enabled = value
