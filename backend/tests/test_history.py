from __future__ import annotations

import importlib

import pytest
import services.optimizer.history as history_module
from services.optimizer import config as optimizer_config


@pytest.fixture(autouse=True)
def restore_history_module() -> None:
    original_keywords = list(optimizer_config.DIRECTIVE_KEYWORDS)
    yield
    optimizer_config.DIRECTIVE_KEYWORDS = original_keywords
    importlib.reload(history_module)


def test_build_history_notes_uses_updated_directive_keywords() -> None:
    optimizer_config.DIRECTIVE_KEYWORDS = ["highlight"]
    importlib.reload(history_module)

    notes = history_module.build_history_notes(
        None,
        [
            {
                "role": "user",
                "content": "Please highlight the acceptance criteria and deliverables clearly.",
            }
        ],
    )

    assert notes == [
        "- User: Please highlight the acceptance criteria and deliverables clearly"
    ]


def test_build_history_notes_falls_back_without_keywords() -> None:
    optimizer_config.DIRECTIVE_KEYWORDS = []
    importlib.reload(history_module)

    notes = history_module.build_history_notes(
        None,
        [
            {
                "role": "assistant",
                "content": "First sentence without directives. Additional context follows.",
            }
        ],
    )

    assert notes == ["- Assistant: First sentence without directives"]


def test_score_history_turn_counts_reloaded_keywords() -> None:
    optimizer_config.DIRECTIVE_KEYWORDS = ["compliance"]
    importlib.reload(history_module)

    score = history_module.score_history_turn(
        None,
        "user",
        "Ensure compliance with the latest standards.",
        index=0,
        total=2,
    )

    baseline = history_module.score_history_turn(
        None,
        "user",
        "Follow the latest standards strictly.",
        index=0,
        total=2,
    )

    assert score > baseline
