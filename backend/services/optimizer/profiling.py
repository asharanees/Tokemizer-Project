"""Lightweight helpers for collecting profiling information during optimization."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional


class ProfilingScope:
    """Context manager that records elapsed time for a named pipeline step."""

    __slots__ = ("_profiler", "_name", "_start")

    def __init__(self, profiler: "PipelineProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "ProfilingScope":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        _exc_type: Optional[type],
        _exc: Optional[BaseException],
        _tb: Optional[Any],
    ) -> bool:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._profiler._record(self._name, elapsed_ms)
        return False


class PipelineProfiler:
    """Thread-safe accumulator that stores timing for named pipeline stages."""

    __slots__ = ("enabled", "_records", "_lock")

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._records: Dict[str, float] = {}
        self._lock = threading.Lock()

    def step(self, name: str):
        if not self.enabled:
            return nullcontext()
        return ProfilingScope(self, name)

    def _record(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._records[name] = self._records.get(name, 0.0) + elapsed_ms

    def record_flag(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._records[name] = value

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {key: round(value, 2) for key, value in self._records.items()}

    @property
    def records(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._records)
