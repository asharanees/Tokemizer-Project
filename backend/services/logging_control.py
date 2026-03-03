"""Runtime control for logging verbosity."""

from __future__ import annotations

import logging
import threading
from typing import Optional

__all__ = ["get_level", "set_level"]

_log_level = "INFO"
_lock = threading.Lock()


def get_level() -> str:
    """Return current log level name."""
    with _lock:
        return _log_level


def set_level(level_name: str) -> None:
    """Set the logging level globally."""
    global _log_level
    
    level_name = level_name.upper().strip()
    if level_name not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level_name = "INFO"

    with _lock:
        _log_level = level_name
        
    # Update root logger
    logging.getLogger().setLevel(level_name)
    
    # Update specific loggers if necessary
    logging.getLogger("uvicorn").setLevel(level_name)
    logging.getLogger("uvicorn.access").setLevel("WARNING")
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")
    logging.getLogger("huggingface_hub").setLevel("WARNING")
    logging.getLogger("tokemizer").setLevel(level_name)
