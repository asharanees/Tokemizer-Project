import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import services.model_cache_manager as model_cache_manager
from services.model_cache_manager import (
    _MANIFEST_VERSION,
    ModelCacheValidator,
    _download_model,
    cache_uploaded_model_archive,
    clear_model_download_issue,
    ensure_models_cached,
    get_model_download_issues,
    reset_model_download_issues,
    resolve_cached_model_artifact,
    resolve_tokenizer_root_from_artifact,
)


def _build_validator(tmp_path: Path) -> ModelCacheValidator:
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir(exist_ok=True)
    validator = ModelCacheValidator(str(hf_home))
    validator.configs = {
        "test_model": {
            "model_name": "example/model",
            "min_size_bytes": 0,
            "expected_files": ["config.json"],
            "revision": "rev-a",
            "allow_patterns": [],
        }
    }
    return validator


def test_refresh_mode_download_missing_skips_download_when_cached(
    monkeypatch, tmp_path
):
    validator = _build_validator(tmp_path)
    download_calls: list = []

    monkeypatch.setattr(
        model_cache_manager,
        "_download_model",
        lambda model_type, config, validator=None: download_calls.append(model_type)
        or True,
    )
    validate_calls: list = []

    def stub_validate(self, model_type, use_cache=True, generate_manifest=False):
        validate_calls.append({"model_type": model_type, "use_cache": use_cache})
        return {"cached_ok": True, "cached_reason": "cached_ok"}

    monkeypatch.setattr(ModelCacheValidator, "validate_model_cache", stub_validate)

    available, missing = ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="download_missing",
    )

    assert available == ["test_model"]
    assert missing == []
    assert download_calls == []
    assert (
        validate_calls
    ), "Validation should be called before deciding whether to download"


def test_refresh_mode_download_missing_triggers_download(monkeypatch, tmp_path):
    validator = _build_validator(tmp_path)
    download_calls: list = []

    def stub_download(model_type, config, validator=None):
        download_calls.append(model_type)
        return True

    monkeypatch.setattr(model_cache_manager, "_download_model", stub_download)

    def stub_validate(self, model_type, use_cache=True, generate_manifest=False):
        return {"cached_ok": False, "cached_reason": "cache_missing"}

    monkeypatch.setattr(ModelCacheValidator, "validate_model_cache", stub_validate)

    available, missing = ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="download_missing",
    )

    assert download_calls == ["test_model"]
    assert available == []
    assert missing == ["test_model"]


def test_refresh_mode_recovery_calls_cleanup(monkeypatch, tmp_path):
    validator = _build_validator(tmp_path)
    download_calls: list = []
    cleanup_called = []

    def stub_download(model_type, config, validator=None):
        download_calls.append(model_type)
        return True

    def stub_cleanup(self, dry_run=True):
        cleanup_called.append(dry_run)
        return {"removed_count": 0, "freed_bytes": 0}

    monkeypatch.setattr(ModelCacheValidator, "cleanup_incomplete_models", stub_cleanup)
    monkeypatch.setattr(model_cache_manager, "_download_model", stub_download)

    monkeypatch.setattr(
        ModelCacheValidator,
        "validate_model_cache",
        lambda self, model_type, use_cache=True, generate_manifest=False: {
            "cached_ok": False,
            "cached_reason": "cache_missing",
        },
    )

    available, missing = ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="recovery",
    )

    assert cleanup_called == [False]
    assert download_calls == ["test_model"]
    assert missing == ["test_model"]
    assert available == []


def test_manifest_validation_success(tmp_path):
    model_root = tmp_path / "cached_model"
    model_root.mkdir()
    config_path = model_root / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    sha256 = ModelCacheValidator._get_file_hash(str(config_path), "sha256")
    manifest = {
        "version": _MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "revision": "rev-a",
        "snapshot": ".",
        "expected_files": ["config.json"],
        "total_size_bytes": config_path.stat().st_size,
        "files": [
            {
                "path": "config.json",
                "size": config_path.stat().st_size,
                "sha256": sha256,
            }
        ],
    }
    (model_root / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    validator = ModelCacheValidator(str(tmp_path))
    result = validator._validate_manifest(
        str(model_root), expected_files=["config.json"], revision="rev-a"
    )
    assert result == (True, "cached_ok")


def test_manifest_validation_revision_mismatch(tmp_path):
    model_root = tmp_path / "cached_model_again"
    model_root.mkdir()
    config_path = model_root / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    manifest = {
        "version": _MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "revision": "rev-a",
        "snapshot": ".",
        "expected_files": ["config.json"],
        "total_size_bytes": config_path.stat().st_size,
        "files": [
            {
                "path": "config.json",
                "size": config_path.stat().st_size,
                "sha256": ModelCacheValidator._get_file_hash(
                    str(config_path), "sha256"
                ),
            }
        ],
    }
    (model_root / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    validator = ModelCacheValidator(str(tmp_path))
    result = validator._validate_manifest(
        str(model_root), expected_files=["config.json"], revision="other-revision"
    )
    assert result == (False, "revision_mismatch")


def test_cache_uploaded_model_archive_success(tmp_path):
    validator = _build_validator(tmp_path)
    archive_path = tmp_path / "test-model.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_ref:
        zip_ref.writestr("config.json", "{}")

    status = cache_uploaded_model_archive(
        str(tmp_path / "hf_home"),
        model_type="test_model",
        archive_path=str(archive_path),
        validator=validator,
    )

    assert status["cached_ok"] is True
    validate_status = validator.validate_model_cache(
        "test_model", use_cache=False, generate_manifest=True
    )
    assert validate_status.get("cached_ok") is True


def test_resolve_cached_model_artifact_finds_nested_file(tmp_path):
    hf_home = tmp_path / "hf_home"
    snapshot = (
        hf_home
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / "rev123"
        / "onnx"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    onnx_file = snapshot / "model.onnx"
    onnx_file.write_bytes(b"onnx")

    model_path = resolve_cached_model_artifact(
        "semantic_guard",
        "sentence-transformers/all-MiniLM-L6-v2",
        "model.onnx",
        hf_home=str(hf_home),
    )

    assert model_path == str(onnx_file)


def test_resolve_tokenizer_root_from_artifact_uses_parent_when_needed(tmp_path):
    snapshot_root = tmp_path / "snapshot"
    onnx_dir = snapshot_root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    artifact = onnx_dir / "model.onnx"
    artifact.write_bytes(b"onnx")
    (snapshot_root / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolved = resolve_tokenizer_root_from_artifact(str(artifact))

    assert resolved == str(snapshot_root)


def test_download_model_continues_when_revision_validation_fails(
    monkeypatch,
    tmp_path,
):
    validator = _build_validator(tmp_path)

    config = {
        "model_name": "example/model",
        "min_size_bytes": 0,
        "expected_files": ["config.json"],
        "revision": "main",
        "allow_patterns": [],
    }

    monkeypatch.setattr(
        model_cache_manager,
        "_validate_revision_online",
        lambda repo_id, revision: False,
    )

    def fake_snapshot_download(*args, **kwargs):
        cache_dir = Path(kwargs["cache_dir"])
        model_root = cache_dir / "models--example--model"
        snapshot_dir = model_root / "snapshots" / "rev-a"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    downloaded = _download_model("test_model", config=config, validator=validator)

    assert downloaded is True


def test_download_model_retries_anonymous_after_auth_failure(
    monkeypatch,
    tmp_path,
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    config = {
        "model_name": "example/model",
        "min_size_bytes": 0,
        "expected_files": ["config.json"],
        "revision": "main",
        "allow_patterns": [],
    }

    monkeypatch.setattr(
        model_cache_manager,
        "_validate_revision_online",
        lambda repo_id, revision: True,
    )
    monkeypatch.setattr(
        model_cache_manager, "_resolve_hf_token", lambda: "hf_bad_token"
    )

    calls = {"count": 0, "tokens": []}

    def fake_snapshot_download(*args, **kwargs):
        calls["count"] += 1
        calls["tokens"].append(kwargs.get("token"))
        token = kwargs.get("token")
        if calls["count"] == 1 and token == "hf_bad_token":
            raise RuntimeError("401 unauthorized")
        cache_dir = Path(kwargs["cache_dir"])
        model_root = cache_dir / "models--example--model"
        snapshot_dir = model_root / "snapshots" / "rev-b"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    downloaded = _download_model("test_model", config=config, validator=validator)

    assert downloaded is True
    assert calls["count"] == 2
    assert calls["tokens"] == ["hf_bad_token", False]


def test_download_model_records_auth_issue_on_failure(
    monkeypatch,
    tmp_path,
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    config = {
        "model_name": "example/model",
        "min_size_bytes": 0,
        "expected_files": ["config.json"],
        "revision": "main",
        "allow_patterns": [],
    }

    monkeypatch.setattr(
        model_cache_manager,
        "_validate_revision_online",
        lambda repo_id, revision: True,
    )
    monkeypatch.setattr(
        model_cache_manager, "_resolve_hf_token", lambda: "hf_bad_token"
    )

    def fake_snapshot_download(*args, **kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    downloaded = _download_model("test_model", config=config, validator=validator)

    assert downloaded is False
    issues = get_model_download_issues()
    issue = issues.get("test_model") or {}
    assert issue.get("category") == "auth_blocked"
    assert int(issue.get("auth_failure_attempts") or 0) == 3
    assert "HF_TOKEN" in str(issue.get("message") or "")


def test_download_model_blocks_after_three_auth_failures(
    monkeypatch,
    tmp_path,
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    config = {
        "model_name": "example/model",
        "min_size_bytes": 0,
        "expected_files": ["config.json"],
        "revision": "main",
        "allow_patterns": [],
    }

    monkeypatch.setattr(
        model_cache_manager,
        "_validate_revision_online",
        lambda repo_id, revision: True,
    )
    monkeypatch.setattr(
        model_cache_manager, "_resolve_hf_token", lambda: "hf_bad_token"
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("401 unauthorized")),
    )

    downloaded = _download_model("test_model", config=config, validator=validator)
    assert downloaded is False

    issues = get_model_download_issues()
    issue = issues.get("test_model") or {}
    assert issue.get("category") == "auth_blocked"
    assert int(issue.get("auth_failure_attempts") or 0) == 3
    assert "download attempts are paused" in str(issue.get("message") or "")


def test_download_model_stops_after_three_non_auth_failures(
    monkeypatch,
    tmp_path,
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    config = {
        "model_name": "example/model",
        "min_size_bytes": 0,
        "expected_files": ["config.json"],
        "revision": "main",
        "allow_patterns": [],
    }

    monkeypatch.setattr(
        model_cache_manager,
        "_validate_revision_online",
        lambda repo_id, revision: True,
    )
    monkeypatch.setattr(model_cache_manager, "_resolve_hf_token", lambda: None)

    calls = {"count": 0}

    def fake_snapshot_download(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("network timeout")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    downloaded = _download_model("test_model", config=config, validator=validator)

    assert downloaded is False
    assert calls["count"] == 3
    issues = get_model_download_issues()
    issue = issues.get("test_model") or {}
    assert issue.get("category") == "download_failed"
    assert int(issue.get("download_failure_attempts") or 0) == 3


def test_refresh_mode_skips_download_when_auth_is_blocked_for_private_repo(
    monkeypatch, tmp_path
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)
    validator.configs["test_model"]["model_name"] = "tokemizer/private-model"

    monkeypatch.setattr(
        ModelCacheValidator,
        "validate_model_cache",
        lambda self, model_type, use_cache=True, generate_manifest=False: {
            "cached_ok": False,
            "cached_reason": "cache_missing",
        },
    )
    monkeypatch.setattr(
        model_cache_manager,
        "_resolve_hf_token",
        lambda: "hf_blocked_token",
    )

    fingerprint = model_cache_manager._token_fingerprint("hf_blocked_token")
    model_cache_manager._MODEL_DOWNLOAD_ISSUES["test_model"] = {
        "model_repo": "tokemizer/private-model",
        "category": "auth_blocked",
        "message": "blocked",
        "auth_failure_attempts": 3,
        "auth_token_fingerprint": fingerprint,
    }

    download_calls = []
    monkeypatch.setattr(
        model_cache_manager,
        "_download_model",
        lambda model_type, config, validator=None: download_calls.append(model_type)
        or True,
    )

    available, missing = ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="download_missing",
    )

    assert download_calls == []
    assert available == []
    assert missing == ["test_model"]


def test_refresh_mode_does_not_skip_public_repo_when_auth_issue_blocked(
    monkeypatch, tmp_path
):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    monkeypatch.setattr(
        ModelCacheValidator,
        "validate_model_cache",
        lambda self, model_type, use_cache=True, generate_manifest=False: {
            "cached_ok": False,
            "cached_reason": "cache_missing",
        },
    )
    monkeypatch.setattr(
        model_cache_manager,
        "_resolve_hf_token",
        lambda: "hf_blocked_token",
    )

    fingerprint = model_cache_manager._token_fingerprint("hf_blocked_token")
    model_cache_manager._MODEL_DOWNLOAD_ISSUES["test_model"] = {
        "model_repo": "example/model",
        "category": "auth_blocked",
        "message": "blocked",
        "auth_failure_attempts": 3,
        "auth_token_fingerprint": fingerprint,
    }

    download_calls = []
    monkeypatch.setattr(
        model_cache_manager,
        "_download_model",
        lambda model_type, config, validator=None: download_calls.append(model_type)
        or True,
    )

    ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="download_missing",
    )

    assert download_calls == ["test_model"]


def test_reset_download_issues_allows_retries_again(monkeypatch, tmp_path):
    clear_model_download_issue("test_model")
    validator = _build_validator(tmp_path)

    monkeypatch.setattr(
        ModelCacheValidator,
        "validate_model_cache",
        lambda self, model_type, use_cache=True, generate_manifest=False: {
            "cached_ok": False,
            "cached_reason": "cache_missing",
        },
    )
    monkeypatch.setattr(
        model_cache_manager,
        "_resolve_hf_token",
        lambda: "hf_blocked_token",
    )

    fingerprint = model_cache_manager._token_fingerprint("hf_blocked_token")
    model_cache_manager._MODEL_DOWNLOAD_ISSUES["test_model"] = {
        "model_repo": "example/model",
        "category": "auth_blocked",
        "message": "blocked",
        "auth_failure_attempts": 3,
        "auth_token_fingerprint": fingerprint,
    }

    cleared = reset_model_download_issues(["test_model"])
    assert cleared == 1

    download_calls = []
    monkeypatch.setattr(
        model_cache_manager,
        "_download_model",
        lambda model_type, config, validator=None: download_calls.append(model_type)
        or True,
    )

    ensure_models_cached(
        str(tmp_path / "hf_home"),
        ["test_model"],
        validator=validator,
        refresh_mode="download_missing",
    )

    assert download_calls == ["test_model"]


def test_import_spacy_model_package_adds_home_user_site_to_sys_path(
    monkeypatch,
    tmp_path,
):
    home_dir = tmp_path / "home"
    site_packages = (
        home_dir
        / ".local"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package_dir = site_packages / "fake_spacy_model_pkg"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr("site.getusersitepackages", lambda: str(tmp_path / "missing"))

    module_name = "fake_spacy_model_pkg"
    if module_name in sys.modules:
        sys.modules.pop(module_name, None)

    module = model_cache_manager._import_spacy_model_package(
        module_name,
        home_hint=str(home_dir),
    )

    assert getattr(module, "VALUE", None) == 1


def test_ensure_spacy_model_cached_accepts_nested_config_path(tmp_path):
    model_root = tmp_path / "spacy" / "en_core_web_md"
    nested = model_root / "en_core_web_md-3.8.0"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "config.cfg").write_text('[nlp]\nlang = "en"\n', encoding="utf-8")

    cached = model_cache_manager.ensure_spacy_model_cached(
        model_name="en_core_web_md",
        target_path=str(model_root),
        allow_downloads=False,
    )

    assert cached is True


def test_get_spacy_cache_status_resolves_nested_config_path(tmp_path):
    model_root = tmp_path / "spacy" / "en_core_web_md"
    nested = model_root / "en_core_web_md-3.8.0"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "config.cfg").write_text('[nlp]\nlang = "en"\n', encoding="utf-8")

    status = model_cache_manager.get_spacy_cache_status(
        "en_core_web_md",
        model_path=str(model_root),
    )

    assert status["cached_ok"] is True
    assert status["cached_reason"] == "cached_ok"
    assert status["path"] == str(nested)


def test_get_spacy_cache_status_accepts_installed_package_when_cache_missing(
    monkeypatch,
    tmp_path,
):
    home_dir = tmp_path / "home"
    site_packages = (
        home_dir
        / ".local"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    model_name = "fake_spacy_model_pkg_status"
    package_dir = site_packages / model_name
    nested = package_dir / "fake_spacy_model_pkg_status-1.0.0"
    nested.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (nested / "config.cfg").write_text('[nlp]\nlang = "en"\n', encoding="utf-8")

    monkeypatch.syspath_prepend(str(site_packages))

    if model_name in sys.modules:
        sys.modules.pop(model_name, None)

    status = model_cache_manager.get_spacy_cache_status(
        model_name,
        model_path=str(tmp_path / "missing-cache" / model_name),
    )

    assert status["cached_ok"] is True
    assert status["cached_reason"] == "cached_ok"
    assert status["cache_source"] == "package"
    assert status["path"] == str(nested)


def test_ensure_spacy_model_cached_serializes_concurrent_downloads(
    monkeypatch,
    tmp_path,
):
    target_path = tmp_path / "spacy" / "en_core_web_sm"
    package_root = tmp_path / "site-packages" / "en_core_web_sm"
    nested = package_root / "en_core_web_sm-3.8.0"
    init_file = package_root / "__init__.py"
    download_calls = {"count": 0}

    def _fake_import(_model_name, home_hint=None):
        if not init_file.is_file():
            raise ImportError("not installed")
        return SimpleNamespace(__file__=str(init_file))

    def _fake_download(_model_name):
        download_calls["count"] += 1
        nested.mkdir(parents=True, exist_ok=True)
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text("VALUE = 1\n", encoding="utf-8")
        (nested / "config.cfg").write_text('[nlp]\nlang = "en"\n', encoding="utf-8")

    monkeypatch.setattr(model_cache_manager, "_import_spacy_model_package", _fake_import)
    monkeypatch.setattr("spacy.cli.download", _fake_download)

    def _worker():
        return model_cache_manager.ensure_spacy_model_cached(
            model_name="en_core_web_sm",
            target_path=str(target_path),
            allow_downloads=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_worker) for _ in range(2)]
        results = [future.result() for future in futures]

    assert results == [True, True]
    assert download_calls["count"] == 1
