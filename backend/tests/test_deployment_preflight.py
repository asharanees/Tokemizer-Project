from __future__ import annotations

from pathlib import Path

import pytest
from deployment_preflight import validate_hf_home_ready


def test_validate_hf_home_ready_succeeds_for_existing_writable_dir(tmp_path: Path):
    resolved = validate_hf_home_ready(str(tmp_path))
    assert resolved == str(tmp_path)


def test_validate_hf_home_ready_fails_when_directory_missing(tmp_path: Path):
    missing = tmp_path / "missing-hf-home"
    with pytest.raises(RuntimeError, match="directory not found"):
        validate_hf_home_ready(str(missing))


def test_validate_hf_home_ready_fails_when_directory_not_writable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr("os.access", lambda path, mode: False)
    with pytest.raises(RuntimeError, match="not writable"):
        validate_hf_home_ready(str(tmp_path))
