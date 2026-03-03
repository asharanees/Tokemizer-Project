from __future__ import annotations

import importlib

import pytest
import uvicorn_config


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "UVICORN_HOST",
        "PORT",
        "UVICORN_WORKERS",
        "UVICORN_ACCESS_LOG",
    ]:
        monkeypatch.delenv(key, raising=False)


def _reload_config() -> dict:
    importlib.reload(uvicorn_config)
    return uvicorn_config.build_uvicorn_kwargs()


def test_defaults_match_long_running_expectations() -> None:
    kwargs = _reload_config()
    assert kwargs["timeout_keep_alive"] == uvicorn_config.DEFAULT_TIMEOUT_KEEP_ALIVE
    assert (
        kwargs["timeout_graceful_shutdown"] == uvicorn_config.DEFAULT_GRACEFUL_TIMEOUT
    )
    assert kwargs["limit_concurrency"] == uvicorn_config.DEFAULT_LIMIT_CONCURRENCY
    assert kwargs["access_log"] is False


def test_env_overrides_are_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UVICORN_HOST", "127.0.0.1")
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("UVICORN_ACCESS_LOG", "true")
    kwargs = _reload_config()
    assert kwargs["workers"] == 2
    assert kwargs["port"] == 9000
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["access_log"] is True


def test_invalid_values_raise_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UVICORN_WORKERS", "-10")
    with pytest.raises(ValueError):
        _reload_config()
