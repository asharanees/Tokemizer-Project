import io
import os
import tarfile
import time
import zipfile

import pytest

import services.model_cache_manager as mcm


def test_token_fingerprint_and_build_message():
    assert mcm._token_fingerprint(None) == "none"
    assert len(mcm._token_fingerprint("abc123")) == 16

    msg = mcm._build_auth_issue_message("repo/x", attempts=1, blocked=False)
    assert "Authentication failure" in msg or "access denied" in msg
    msg_blocked = mcm._build_auth_issue_message("repo/x", attempts=3, blocked=True)
    assert "download attempts are paused" in msg_blocked


def test_resolve_hf_and_spacy_home_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home_env"))
    assert mcm.resolve_hf_home() == str(tmp_path / "hf_home_env")

    monkeypatch.setenv("SPACY_HOME", str(tmp_path / "spacy_home_env"))
    assert mcm.resolve_spacy_home() == str(tmp_path / "spacy_home_env")


def test_resolve_allow_patterns_override_and_infer():
    cfg = {"allow_patterns": ["*.bin", "custom.txt"]}
    patterns = mcm._resolve_allow_patterns(cfg, ["model.safetensors"])
    assert "*.bin" in patterns
    assert "custom.txt" in patterns

    # infer from expected files
    cfg2 = {}
    patterns2 = mcm._resolve_allow_patterns(cfg2, ["model.onnx", "vocab.json"])
    assert "*.onnx" in patterns2 or "model.onnx" in patterns2
    assert "vocab.json" in patterns2


def test_find_file_in_model_path_direct_recursive_and_snapshots(tmp_path):
    root = tmp_path / "modelroot"
    root.mkdir()
    (root / "a.txt").write_text("x")
    sub = root / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")

    assert mcm._find_file_in_model_path(str(root), "a.txt") is not None
    assert mcm._find_file_in_model_path(str(root), "b.txt") is not None

    # snapshots
    snaps = root / "snapshots" / "rev1"
    snaps.mkdir(parents=True)
    (snaps / "c.txt").write_text("z")
    assert mcm._find_file_in_model_path(str(root), "c.txt") is not None


def test_safe_extract_zip_and_tar_path_traversal(tmp_path):
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../evil.txt", "evil")

    with pytest.raises(ValueError):
        mcm._safe_extract_zip(str(bad_zip), str(tmp_path / "outzip"))

    bad_tar = tmp_path / "bad.tar"
    with tarfile.open(bad_tar, "w") as tf:
        ti = tarfile.TarInfo("../evil.txt")
        data = io.BytesIO(b"evil")
        ti.size = len(data.getvalue())
        tf.addfile(ti, data)

    with pytest.raises(ValueError):
        mcm._safe_extract_tar(str(bad_tar), str(tmp_path / "outtar"))


def test_extract_model_archive_unsupported(tmp_path):
    somefile = tmp_path / "file.unknown"
    somefile.write_text("x")
    with pytest.raises(ValueError):
        mcm._extract_model_archive(str(somefile), str(tmp_path / "out"))


def test_safe_extract_zip_success(tmp_path):
    zpath = tmp_path / "ok.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("good.txt", "ok")
    out = tmp_path / "outok"
    out.mkdir()
    mcm._safe_extract_zip(str(zpath), str(out))
    assert (out / "good.txt").read_text() == "ok"


def test_acquire_release_lock_and_is_model_locked(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    # short ttl to allow cleanup checks
    monkeypatch.setattr(mcm, "_LOCK_TTL_SECONDS", 1)

    # acquire
    lock = mcm._acquire_model_lock(str(hub), "m1")
    assert lock is not None
    assert mcm._is_model_locked(str(hub), "m1") is True

    # release
    mcm._release_model_lock(lock)
    assert mcm._is_model_locked(str(hub), "m1") is False

    # create stale lock file
    lock2 = os.path.join(str(hub), mcm._LOCK_DIRNAME, "m2.lock")
    os.makedirs(os.path.dirname(lock2), exist_ok=True)
    with open(lock2, "w", encoding="utf-8") as fh:
        fh.write(str(time.time() - 3600))
    # should clean up stale lock and return False
    assert mcm._is_model_locked(str(hub), "m2") is False
    assert not os.path.exists(lock2)
