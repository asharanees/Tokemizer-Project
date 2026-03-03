"""Deployment preflight checks for runtime-critical infrastructure paths."""

from __future__ import annotations

import os
from typing import Optional

from services.model_cache_manager import resolve_hf_home


def validate_hf_home_ready(hf_home: Optional[str] = None) -> str:
    """Return resolved HF_HOME if it exists and is writable; raise otherwise."""

    resolved = str(hf_home or resolve_hf_home()).strip()
    if not resolved:
        raise RuntimeError(
            "Deployment preflight failed: HF_HOME resolved to an empty path."
        )

    if not os.path.isdir(resolved):
        raise RuntimeError(
            f"Deployment preflight failed: HF_HOME directory not found at {resolved}."
        )

    if not os.access(resolved, os.W_OK):
        raise RuntimeError(
            f"Deployment preflight failed: HF_HOME is not writable at {resolved}."
        )

    return resolved
