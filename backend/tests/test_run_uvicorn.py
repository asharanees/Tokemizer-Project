"""Tests for the run_uvicorn entrypoint."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import run_uvicorn


def test_main_strips_legacy_timeout_kwarg(monkeypatch):
    """``run_uvicorn.main`` must not pass the legacy timeout kwarg to uvicorn."""

    captured_kwargs: Dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured_kwargs["app"] = app
        captured_kwargs["kwargs"] = kwargs

    monkeypatch.setattr(run_uvicorn.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        run_uvicorn, "validate_hf_home_ready", lambda hf_home=None: "/tmp/hf-home"
    )

    run_uvicorn.main()

    assert captured_kwargs["app"] == "server:app"
    assert "timeout_graceful" not in captured_kwargs["kwargs"]
    assert "timeout_graceful_shutdown" in captured_kwargs["kwargs"]


def test_main_fails_before_uvicorn_when_hf_home_preflight_fails(monkeypatch):
    called = {"uvicorn_run": False}

    def fake_run(app: str, **kwargs: Any) -> None:
        called["uvicorn_run"] = True

    monkeypatch.setattr(run_uvicorn.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        run_uvicorn,
        "validate_hf_home_ready",
        lambda hf_home=None: (_ for _ in ()).throw(RuntimeError("hf preflight failed")),
    )

    with pytest.raises(RuntimeError, match="hf preflight failed"):
        run_uvicorn.main()

    assert called["uvicorn_run"] is False
