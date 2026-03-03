"""Optimizer package: exposes PromptOptimizer and singleton instance.

This package reorganizes the previous monolithic optimizer.py into a package
without changing the public API. Existing imports like

    from services.optimizer import PromptOptimizer, optimizer

continue to work.
"""

from .core import PromptOptimizer, optimizer
from .lexical import QUANTULUM_AVAILABLE  # re-export for tests

__all__ = [
    "PromptOptimizer",
    "optimizer",
    "QUANTULUM_AVAILABLE",
]
