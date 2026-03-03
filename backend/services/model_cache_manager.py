"""
Model cache management for HuggingFace models.

Handles checking local cache before downloading, validating model sizes,
and managing model lifecycle for baked or cached models.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import site
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database import get_db, init_db

logger = logging.getLogger(__name__)

_COREF_MINILM_MODEL_ALIAS = "coref_minilm"
_COREF_MINILM_MODEL_ID = (
    "talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
)
_MODEL_DOWNLOAD_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
)
_HF_TOKEN_ENV_CANDIDATES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
)
_MANIFEST_FILENAME = "model_manifest.json"
_MANIFEST_BACKUP_SUFFIX = ".bak"
_MANIFEST_VERSION = 1
_LOCK_DIRNAME = ".model_locks"
_VALIDATION_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_VALIDATION_TTL", "600"))
_LOCK_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_LOCK_TTL_SECONDS", "1800"))
_MAX_AUTH_FAILURE_ATTEMPTS = int(os.environ.get("MODEL_AUTH_FAILURE_MAX_ATTEMPTS", "3"))
_MAX_DOWNLOAD_ATTEMPTS = max(1, int(os.environ.get("MODEL_DOWNLOAD_MAX_ATTEMPTS", "3")))
_DOWNLOAD_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("MODEL_DOWNLOAD_RETRY_BACKOFF_SECONDS", "1.0")
)
_FULL_HASH_MAX_BYTES = int(
    os.environ.get("MODEL_CACHE_FULL_HASH_BYTES", str(250 * 1024 * 1024))
)
_MODEL_DOWNLOAD_ISSUES: Dict[str, Dict[str, Any]] = {}
_SPACY_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_SPACY_CACHE_LOCKS_GUARD = threading.Lock()


def _get_spacy_cache_lock(model_name: str, target_path: str) -> threading.Lock:
    lock_key = f"{model_name}::{target_path}"
    with _SPACY_CACHE_LOCKS_GUARD:
        lock = _SPACY_CACHE_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _SPACY_CACHE_LOCKS[lock_key] = lock
        return lock


def _resolve_hf_token() -> Optional[str]:
    for key in _HF_TOKEN_ENV_CANDIDATES:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _is_hf_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    indicators = (
        "401",
        "unauthorized",
        "invalid username or password",
        "authentication",
        "private",
        "gated",
    )
    return any(indicator in text for indicator in indicators)


def _record_model_download_issue(
    model_type: str,
    *,
    model_repo: str,
    category: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model_repo": model_repo,
        "category": category,
        "message": message,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(metadata, dict):
        payload.update(metadata)
    _MODEL_DOWNLOAD_ISSUES[model_type] = payload


def clear_model_download_issue(model_type: str) -> None:
    _MODEL_DOWNLOAD_ISSUES.pop(model_type, None)


def get_model_download_issues() -> Dict[str, Dict[str, Any]]:
    return {key: dict(value) for key, value in _MODEL_DOWNLOAD_ISSUES.items()}


def reset_model_download_issues(model_types: Optional[List[str]] = None) -> int:
    if not model_types:
        cleared = len(_MODEL_DOWNLOAD_ISSUES)
        _MODEL_DOWNLOAD_ISSUES.clear()
        return cleared

    cleared = 0
    for model_type in {str(item).strip() for item in model_types if str(item).strip()}:
        if model_type in _MODEL_DOWNLOAD_ISSUES:
            _MODEL_DOWNLOAD_ISSUES.pop(model_type, None)
            cleared += 1
    return cleared


def _token_fingerprint(token: Optional[str]) -> str:
    value = (token or "").strip()
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _build_auth_issue_message(
    model_repo: str,
    *,
    attempts: int,
    blocked: bool,
) -> str:
    if blocked:
        return (
            f"HuggingFace access denied for {model_repo}. Authentication failed "
            f"{attempts}/{_MAX_AUTH_FAILURE_ATTEMPTS} times; download attempts are paused. "
            "Admin must set a valid HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in backend env/.env "
            "and run model refresh again."
        )
    return (
        f"HuggingFace access denied for {model_repo}. Authentication failure "
        f"{attempts}/{_MAX_AUTH_FAILURE_ATTEMPTS}. Set a valid HF_TOKEN (or "
        "HUGGINGFACE_HUB_TOKEN) in backend env/.env."
    )


def _build_download_issue_message(model_repo: str, *, attempts: int) -> str:
    return (
        f"Failed to download {model_repo}. Download failed "
        f"{attempts}/{_MAX_DOWNLOAD_ATTEMPTS} attempts. "
        "Check internet connectivity, HuggingFace access/token, and model availability."
    )


def _should_pause_auth_download(model_type: str, model_repo: str) -> bool:
    issue = _MODEL_DOWNLOAD_ISSUES.get(model_type)
    if not isinstance(issue, dict):
        return False

    issue_repo = str(issue.get("model_repo") or "").strip()
    if issue_repo and issue_repo != model_repo:
        clear_model_download_issue(model_type)
        return False

    # Public repos should never be blocked by prior auth-failure guard state.
    # Keep auth pause behavior only for explicitly private org repos.
    if not model_repo.lower().startswith("tokemizer/"):
        return False

    category = str(issue.get("category") or "").strip().lower()
    if category not in {"auth_failed", "auth_blocked"}:
        return False

    try:
        attempts = int(issue.get("auth_failure_attempts") or 0)
    except (TypeError, ValueError):
        attempts = 0
    if attempts < _MAX_AUTH_FAILURE_ATTEMPTS:
        return False

    current_fingerprint = _token_fingerprint(_resolve_hf_token())
    recorded_fingerprint = str(issue.get("auth_token_fingerprint") or "").strip()
    if recorded_fingerprint and recorded_fingerprint != current_fingerprint:
        logger.info(
            "Detected HuggingFace token change for %s; resetting auth failure guard",
            model_type,
        )
        clear_model_download_issue(model_type)
        return False

    return True


def _sync_hf_runtime_network_state(disable_network: bool) -> None:
    try:
        from huggingface_hub import constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = disable_network
    except Exception:
        pass

    try:
        from transformers.utils import hub as transformers_hub

        if hasattr(transformers_hub, "HF_HUB_OFFLINE"):
            transformers_hub.HF_HUB_OFFLINE = disable_network
        if hasattr(transformers_hub, "TRANSFORMERS_OFFLINE"):
            transformers_hub.TRANSFORMERS_OFFLINE = disable_network
    except Exception:
        pass


@contextmanager
def _allow_model_downloads() -> None:
    managed_env_vars = _MODEL_DOWNLOAD_ENV_VARS + _HF_TOKEN_ENV_CANDIDATES
    previous = {key: os.environ.get(key) for key in managed_env_vars}
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    resolved_token = _resolve_hf_token()
    if resolved_token:
        os.environ["HF_TOKEN"] = resolved_token
    _sync_hf_runtime_network_state(False)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        disable_network = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        _sync_hf_runtime_network_state(disable_network)


def _acquire_model_lock(hub_cache: str, model_type: str) -> Optional[str]:
    lock_dir = os.path.join(hub_cache, _LOCK_DIRNAME)
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{model_type}.lock")
    for attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(time.time()))
            return lock_path
        except FileExistsError:
            if attempt == 0 and not _is_model_locked(hub_cache, model_type):
                continue
            return None
        except OSError as exc:
            logger.warning("Failed to create lock for %s: %s", model_type, exc)
            return None
    return None


def _release_model_lock(lock_path: Optional[str]) -> None:
    if not lock_path:
        return
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _is_model_locked(hub_cache: str, model_type: str) -> bool:
    lock_path = os.path.join(hub_cache, _LOCK_DIRNAME, f"{model_type}.lock")
    if not os.path.isfile(lock_path):
        return False
    if _LOCK_TTL_SECONDS <= 0:
        return True
    lock_age = _lock_age_seconds(lock_path)
    if lock_age is None:
        return True
    if lock_age > _LOCK_TTL_SECONDS:
        _cleanup_stale_lock(lock_path, model_type, lock_age)
        return False
    return True


def _lock_age_seconds(lock_path: str) -> Optional[float]:
    try:
        with open(lock_path, "r", encoding="utf-8") as fh:
            timestamp = float((fh.read() or "").strip())
    except (OSError, ValueError):
        timestamp = None

    if timestamp is None:
        try:
            timestamp = os.path.getmtime(lock_path)
        except OSError:
            return None

    return max(0.0, time.time() - timestamp)


def _cleanup_stale_lock(lock_path: str, model_type: str, lock_age: float) -> None:
    try:
        os.remove(lock_path)
        logger.info(
            "Removed stale model cache lock for %s (age %.0fs > ttl %ss)",
            model_type,
            lock_age,
            _LOCK_TTL_SECONDS,
        )
    except OSError as exc:
        logger.warning("Failed to remove stale lock for %s: %s", model_type, exc)


def _resolve_allow_patterns(
    config: Dict[str, Any], expected_files: List[str]
) -> List[str]:
    override_patterns = config.get("allow_patterns") or []
    if override_patterns:
        return list(
            dict.fromkeys(str(item) for item in override_patterns if str(item).strip())
        )

    expected_names = [str(item).strip() for item in expected_files if str(item).strip()]
    expects_onnx = any(
        name in {"model.onnx", "model.int8.onnx"} for name in expected_names
    )
    allow_patterns = [
        "*.json",
        "tokenizer*",
        "vocab*",
        "merges*",
        "special_tokens_map.json",
    ]
    file_patterns = {
        ".onnx": "*.onnx",
        ".bin": "*.bin",
        ".safetensors": "*.safetensors",
        ".pt": "*.pt",
        ".model": "*.model",
    }
    for filename in expected_names:
        for suffix, pattern in file_patterns.items():
            if filename.endswith(suffix):
                allow_patterns.append(pattern)
                break
    if expects_onnx:
        allow_patterns.extend(["*.safetensors", "*.bin"])
    if expected_names:
        allow_patterns.extend(expected_names)
    return list(dict.fromkeys(allow_patterns))


def _validate_revision_online(model_repo: str, revision: str) -> bool:
    if not revision:
        return True
    token = _resolve_hf_token()
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).model_info(
            repo_id=model_repo, revision=revision, token=token
        )
        return True
    except Exception as exc:
        if _is_hf_auth_error(exc):
            logger.info(
                "Revision validation for %s@%s requires authentication",
                model_repo,
                revision,
            )
        else:
            logger.warning(
                "Revision validation failed for %s@%s: %s",
                model_repo,
                revision,
                exc,
            )
        if token and _is_hf_auth_error(exc):
            try:
                from huggingface_hub import HfApi

                HfApi(token=False).model_info(
                    repo_id=model_repo,
                    revision=revision,
                    token=False,
                )
                logger.info(
                    "Revision validation succeeded for %s@%s after anonymous retry",
                    model_repo,
                    revision,
                )
                return True
            except Exception:
                pass
        return False


def resolve_hf_home(hf_home: Optional[str] = None) -> str:
    value = str(hf_home or "").strip()
    if value:
        return value

    env_value = os.environ.get("HF_HOME")
    if env_value and env_value.strip():
        return env_value.strip()

    docker_default = "/app/.cache/huggingface"
    try:
        if os.name != "nt" and os.path.isdir("/app"):
            return docker_default
    except OSError:
        pass

    try:
        from huggingface_hub import constants as hf_constants

        resolved = str(getattr(hf_constants, "HF_HOME", "")).strip()
        if resolved:
            return resolved
    except Exception:
        pass

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "huggingface")

    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def resolve_spacy_home(
    spacy_home: Optional[str] = None, hf_home: Optional[str] = None
) -> str:
    value = str(spacy_home or "").strip()
    if value:
        return value

    env_value = os.environ.get("SPACY_HOME")
    if env_value and env_value.strip():
        return env_value.strip()

    return os.path.join(resolve_hf_home(hf_home), "spacy")


def _resolve_model_repo(model_type: str, model_name: str) -> str:
    if model_type == "coreference" and model_name == _COREF_MINILM_MODEL_ALIAS:
        return _COREF_MINILM_MODEL_ID
    return model_name


def resolve_cached_model_path(
    model_type: str, model_name: str, hf_home: Optional[str] = None
) -> Optional[str]:
    if not model_name:
        return None
    hf_home = resolve_hf_home(hf_home)
    validator = ModelCacheValidator(hf_home)
    model_repo = _resolve_model_repo(model_type, model_name)
    model_root = validator._find_model_path(model_repo)
    if not model_root:
        return None
    snapshots_path = os.path.join(model_root, "snapshots")
    refs_path = os.path.join(model_root, "refs")
    ref_hash = None
    if os.path.isdir(refs_path):
        main_ref = os.path.join(refs_path, "main")
        if os.path.isfile(main_ref):
            try:
                with open(main_ref, "r", encoding="utf-8") as fh:
                    ref_hash = fh.read().strip()
            except OSError:
                ref_hash = None
        if not ref_hash:
            try:
                for entry in sorted(os.listdir(refs_path)):
                    ref_path = os.path.join(refs_path, entry)
                    if not os.path.isfile(ref_path):
                        continue
                    try:
                        with open(ref_path, "r", encoding="utf-8") as fh:
                            ref_hash = fh.read().strip()
                    except OSError:
                        ref_hash = None
                    if ref_hash:
                        break
            except OSError:
                ref_hash = None
    if ref_hash and os.path.isdir(snapshots_path):
        candidate = os.path.join(snapshots_path, ref_hash)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(snapshots_path):
        try:
            snapshot_dirs = [
                os.path.join(snapshots_path, name)
                for name in os.listdir(snapshots_path)
                if os.path.isdir(os.path.join(snapshots_path, name))
            ]
        except OSError:
            snapshot_dirs = []
        if snapshot_dirs:
            try:
                return max(snapshot_dirs, key=os.path.getmtime)
            except OSError:
                return sorted(snapshot_dirs)[-1]
    return None


def _model_cache_dir_name(model_repo: str) -> str:
    sanitized = model_repo.replace("/", "--")
    return f"models--{sanitized}"


def _find_model_path_in_cache(cache_root: str, model_repo: str) -> Optional[str]:
    if not os.path.isdir(cache_root):
        return None
    target_dir = _model_cache_dir_name(model_repo)
    try:
        for entry in os.listdir(cache_root):
            if entry.startswith(target_dir):
                full_path = os.path.join(cache_root, entry)
                if os.path.isdir(full_path):
                    return full_path
    except OSError:
        return None
    return None


def _cleanup_stale_model_dirs(
    cache_root: str, model_repo: str, keep_path: Optional[str] = None
) -> None:
    target_dir = _model_cache_dir_name(model_repo)
    try:
        for entry in os.listdir(cache_root):
            if not entry.startswith(target_dir):
                continue
            full_path = os.path.join(cache_root, entry)
            if not os.path.isdir(full_path):
                continue
            if keep_path and os.path.abspath(full_path) == os.path.abspath(keep_path):
                continue
            shutil.rmtree(full_path)
    except OSError as exc:
        logger.debug("Failed to cleanup stale cache dirs for %s: %s", model_repo, exc)


def _resolve_snapshot_dir(model_root: str) -> Optional[str]:
    snapshots_path = os.path.join(model_root, "snapshots")
    refs_path = os.path.join(model_root, "refs")
    ref_hash = None
    if os.path.isdir(refs_path):
        main_ref = os.path.join(refs_path, "main")
        if os.path.isfile(main_ref):
            try:
                with open(main_ref, "r", encoding="utf-8") as fh:
                    ref_hash = fh.read().strip()
            except OSError:
                ref_hash = None
        if not ref_hash:
            try:
                for entry in sorted(os.listdir(refs_path)):
                    ref_path = os.path.join(refs_path, entry)
                    if not os.path.isfile(ref_path):
                        continue
                    try:
                        with open(ref_path, "r", encoding="utf-8") as fh:
                            ref_hash = fh.read().strip()
                    except OSError:
                        ref_hash = None
                    if ref_hash:
                        break
            except OSError:
                ref_hash = None
    if ref_hash and os.path.isdir(snapshots_path):
        candidate = os.path.join(snapshots_path, ref_hash)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(snapshots_path):
        try:
            snapshot_dirs = [
                os.path.join(snapshots_path, name)
                for name in os.listdir(snapshots_path)
                if os.path.isdir(os.path.join(snapshots_path, name))
            ]
        except OSError:
            snapshot_dirs = []
        if snapshot_dirs:
            try:
                return max(snapshot_dirs, key=os.path.getmtime)
            except OSError:
                return sorted(snapshot_dirs)[-1]
    return None


def _candidate_user_site_packages(home_hint: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidates.append(user_site)
        elif isinstance(user_site, list):
            candidates.extend(str(item) for item in user_site if str(item).strip())
    except Exception:
        pass

    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    home_candidates = [
        str(home_hint or "").strip(),
        str(os.environ.get("HOME") or "").strip(),
        str(os.path.expanduser("~") or "").strip(),
    ]
    for home_dir in home_candidates:
        if not home_dir:
            continue
        candidates.append(
            os.path.join(home_dir, ".local", "lib", version_tag, "site-packages")
        )
        candidates.append(
            os.path.join(home_dir, ".local", "lib64", version_tag, "site-packages")
        )
    return list(dict.fromkeys(path for path in candidates if path))


def _import_spacy_model_package(model_name: str, home_hint: Optional[str] = None):
    import importlib

    try:
        return importlib.import_module(model_name)
    except Exception:
        importlib.invalidate_caches()
        try:
            return importlib.import_module(model_name)
        except Exception:
            pass
        for site_path in _candidate_user_site_packages(home_hint):
            if os.path.isdir(site_path) and site_path not in sys.path:
                sys.path.append(site_path)
        importlib.invalidate_caches()
        return importlib.import_module(model_name)


def _resolve_spacy_model_dir(model_path: str) -> Optional[str]:
    """Resolve the concrete spaCy model directory that contains config.cfg."""
    config_path = os.path.join(model_path, "config.cfg")
    if os.path.isfile(config_path):
        return model_path
    if not os.path.isdir(model_path):
        return None

    try:
        entries = sorted(os.listdir(model_path))
    except OSError:
        return None

    for entry in entries:
        candidate = os.path.join(model_path, entry)
        if not os.path.isdir(candidate):
            continue
        if os.path.isfile(os.path.join(candidate, "config.cfg")):
            return candidate
    return None


def ensure_spacy_model_cached(
    model_name: str = "en_core_web_sm",
    target_path: Optional[str] = None,
    allow_downloads: bool = False,
) -> bool:
    target_path = (
        target_path or os.environ.get("PROMPT_OPTIMIZER_SPACY_MODEL_PATH", "").strip()
    )
    if not target_path:
        spacy_home = resolve_spacy_home()
        target_path = os.path.join(spacy_home, model_name)

    lock = _get_spacy_cache_lock(model_name, target_path)
    with lock:
        if _resolve_spacy_model_dir(target_path):
            return True
        try:
            home_hint = (os.environ.get("HOME") or "").strip() or None
            try:
                package = _import_spacy_model_package(model_name, home_hint=home_hint)
            except Exception:
                if not allow_downloads:
                    logger.warning(
                        "spaCy model %s not available locally; run refresh to download.",
                        model_name,
                    )
                    return False
                from spacy.cli import download as spacy_download

                with _allow_model_downloads():
                    home_dir = (os.environ.get("HOME") or "").strip()
                    fallback_home: Optional[str] = None
                    if (
                        not home_dir
                        or not os.path.isdir(home_dir)
                        or not os.access(home_dir, os.W_OK)
                    ):
                        fallback_home = os.path.join(resolve_hf_home(), ".spacy_home")
                        os.makedirs(fallback_home, exist_ok=True)
                        logger.info(
                            "Using fallback HOME for spaCy download: %s", fallback_home
                        )
                        home_hint = fallback_home

                    env_backup = {
                        "HOME": os.environ.get("HOME"),
                        "PYTHONUSERBASE": os.environ.get("PYTHONUSERBASE"),
                        "PIP_CACHE_DIR": os.environ.get("PIP_CACHE_DIR"),
                    }
                    try:
                        if fallback_home:
                            os.environ["HOME"] = fallback_home
                            os.environ["PYTHONUSERBASE"] = os.path.join(
                                fallback_home, ".local"
                            )
                            pip_cache_dir = os.path.join(fallback_home, ".cache", "pip")
                            os.makedirs(pip_cache_dir, exist_ok=True)
                            os.environ["PIP_CACHE_DIR"] = pip_cache_dir
                        spacy_download(model_name)
                    except SystemExit as exc:
                        raise RuntimeError(
                            f"spaCy download exited with code {getattr(exc, 'code', exc)}"
                        ) from exc
                    finally:
                        for key, value in env_backup.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                package = _import_spacy_model_package(model_name, home_hint=home_hint)

            package_file = str(getattr(package, "__file__", "") or "").strip()
            if not package_file:
                raise RuntimeError(f"spaCy model package has no file path: {model_name}")
            source = os.path.dirname(os.path.abspath(package_file))
            source_model_dir = _resolve_spacy_model_dir(source) or source
            os.makedirs(target_path, exist_ok=True)
            shutil.copytree(source_model_dir, target_path, dirs_exist_ok=True)
            return _resolve_spacy_model_dir(target_path) is not None
        except Exception as exc:
            logger.warning("Failed to prepare spaCy model cache: %s", exc)
            return False


def get_spacy_cache_status(
    model_name: str,
    model_path: Optional[str] = None,
    validator: Optional["ModelCacheValidator"] = None,
) -> Dict[str, Any]:
    if not model_path:
        spacy_home = resolve_spacy_home()
        model_path = os.path.join(spacy_home, model_name)

    resolved_model_path = _resolve_spacy_model_dir(model_path)
    cache_source = "filesystem"
    cached_ok = resolved_model_path is not None

    if not cached_ok:
        try:
            package = _import_spacy_model_package(model_name)
            package_file = str(getattr(package, "__file__", "") or "").strip()
            if package_file:
                package_root = os.path.dirname(os.path.abspath(package_file))
                package_model_dir = _resolve_spacy_model_dir(package_root)
                if package_model_dir:
                    resolved_model_path = package_model_dir
                    cached_ok = True
                    cache_source = "package"
        except Exception:
            pass

    size_target = resolved_model_path or model_path
    if not os.path.isdir(size_target):
        size = 0
        last_modified = None
    else:
        if validator is None:
            validator = ModelCacheValidator(resolve_hf_home())
        size = validator._get_cached_directory_size(size_target)
        try:
            mtime = os.path.getmtime(size_target)
            from datetime import datetime

            last_modified = datetime.fromtimestamp(mtime).isoformat()
        except OSError:
            last_modified = None

    if validator is None:
        validator = ModelCacheValidator(resolve_hf_home())
    if cached_ok:
        cached_reason = "cached_ok"
        status = "cached"
    elif os.path.isdir(model_path):
        cached_reason = "cache_invalid"
        status = "invalid"
    else:
        cached_reason = "cache_missing"
        status = "missing"

    return {
        "model_name": model_name,
        "path": resolved_model_path,
        "size_bytes": size,
        "size_formatted": validator._format_size(size),
        "cached_ok": cached_ok,
        "cached_reason": cached_reason,
        "cache_source": cache_source,
        "status": status,
        "last_modified": last_modified,
    }


def _find_file_in_model_path(model_path: str, target: str) -> Optional[str]:
    def _pattern_matches(base_dir: str, pattern: str) -> Optional[str]:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            try:
                for name in os.listdir(base_dir):
                    if fnmatch.fnmatch(name, pattern):
                        return os.path.join(base_dir, name)
            except OSError:
                return None
            return None
        candidate = os.path.join(base_dir, pattern)
        return candidate if os.path.isfile(candidate) else None

    direct_match = _pattern_matches(model_path, target)
    if direct_match is not None:
        return direct_match

    if os.path.isdir(model_path):
        for root, _dirs, _files in os.walk(model_path):
            recursive_match = _pattern_matches(root, target)
            if recursive_match is not None:
                return recursive_match

    snapshots_path = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots_path):
        for snapshot in os.listdir(snapshots_path):
            snapshot_dir = os.path.join(snapshots_path, snapshot)
            match = _pattern_matches(snapshot_dir, target)
            if match is not None:
                return match
    return None


def resolve_cached_model_artifact(
    model_type: str,
    model_name: str,
    filename: str,
    hf_home: Optional[str] = None,
) -> Optional[str]:
    model_path = resolve_cached_model_path(model_type, model_name, hf_home=hf_home)
    if not model_path:
        return None
    return _find_file_in_model_path(model_path, filename)


def resolve_tokenizer_root_from_artifact(artifact_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(artifact_path))

    def _has_tokenizer_files(path: str) -> bool:
        candidates = (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        )
        return any(os.path.isfile(os.path.join(path, name)) for name in candidates)

    if _has_tokenizer_files(base_dir):
        return base_dir

    parent_dir = os.path.dirname(base_dir)
    if parent_dir and parent_dir != base_dir and _has_tokenizer_files(parent_dir):
        return parent_dir

    return base_dir


def get_model_configs() -> Dict[str, dict]:
    """
    Fetch model configurations from the database.
    Returns a dictionary keyed by model_type.
    """
    init_db()
    configs = {}
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT model_type, model_name, min_size_bytes, expected_files, revision, allow_patterns
                FROM model_inventory
                WHERE is_active = 1
                """).fetchall()

            for row in rows:
                try:
                    expected_files = json.loads(row["expected_files"])
                except json.JSONDecodeError:
                    logger.error(
                        "Invalid JSON in expected_files for model %s", row["model_type"]
                    )
                    continue

                allow_patterns = []
                if row["allow_patterns"]:
                    try:
                        parsed_allow = json.loads(row["allow_patterns"])
                        if isinstance(parsed_allow, list):
                            allow_patterns = parsed_allow
                    except json.JSONDecodeError:
                        logger.error(
                            "Invalid JSON in allow_patterns for model %s",
                            row["model_type"],
                        )

                configs[row["model_type"]] = {
                    "model_name": row["model_name"],
                    "min_size_bytes": row["min_size_bytes"],
                    "expected_files": expected_files,
                    "revision": row["revision"] or "",
                    "allow_patterns": allow_patterns,
                }

    except Exception as exc:
        logger.error(f"Failed to fetch model configs from DB: {exc}")

    return configs


_ONNX_EXPORTER_MODULE: Optional[object] = None


def _load_onnx_exporter() -> Optional[object]:
    global _ONNX_EXPORTER_MODULE
    if _ONNX_EXPORTER_MODULE is not None:
        return _ONNX_EXPORTER_MODULE
    try:
        backend_root = os.path.dirname(os.path.dirname(__file__))
        script_path = os.path.join(backend_root, "scripts", "export_onnx_models.py")
        if not os.path.isfile(script_path):
            return None
        spec = importlib.util.spec_from_file_location("export_onnx_models", script_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ONNX_EXPORTER_MODULE = module
        return module
    except Exception as exc:
        logger.warning("Failed to load ONNX export helper: %s", exc)
        return None


def _needs_onnx_export(model_path: str, expected_files: List[str]) -> bool:
    if "model.onnx" not in expected_files:
        return False
    if _find_file_in_model_path(model_path, "model.onnx") is not None:
        return False
    if _find_file_in_model_path(model_path, "model.int8.onnx") is not None:
        return False
    return True


def _try_export_onnx_model(model_type: str, model_name: str, model_path: str) -> bool:
    exporter = _load_onnx_exporter()
    if exporter is None:
        return False
    try:
        if model_type in {"semantic_guard", "semantic_rank"}:
            ok = exporter.export_semantic_guard(
                model_name, model_path, model_type=model_type
            )
        elif model_type == "token_classifier":
            ok = exporter.export_token_classifier(model_name, model_path)
        else:
            return False
    except Exception as exc:
        logger.warning("Failed to export ONNX for %s: %s", model_type, exc)
        return False
    if ok:
        logger.info("Exported ONNX model for %s", model_type)
    else:
        logger.warning("ONNX export failed for %s", model_type)
    return ok


def _ensure_onnx_artifacts(
    validator: "ModelCacheValidator", model_types: List[str]
) -> None:
    for model_type in model_types:
        config = validator.configs.get(model_type)
        if not config:
            continue
        expected_files = config.get("expected_files") or []
        if "model.onnx" not in expected_files:
            continue
        model_name = config.get("model_name") or ""
        if not model_name:
            continue
        model_path = resolve_cached_model_path(
            model_type, model_name, hf_home=validator.hf_home
        )
        if not model_path:
            continue
        if not _needs_onnx_export(model_path, expected_files):
            continue
        logger.info("Exporting ONNX artifacts for %s", model_type)
        _try_export_onnx_model(model_type, model_name, model_path)


class ModelCacheValidator:
    """Validates and manages HuggingFace model cache."""

    def __init__(self, hf_home: str) -> None:
        """
        Initialize the validator with HuggingFace home directory.

        Args:
            hf_home: Path to HuggingFace cache directory (usually /app/.cache/huggingface)
        """
        self.hf_home = hf_home
        self.hub_cache = os.path.join(hf_home, "hub")
        self.configs = get_model_configs()
        self._size_cache: Dict[str, int] = {}
        self._validation_cache: Dict[str, Dict[str, Any]] = {}

    def model_exists(self, model_type: str) -> bool:
        """
        Check if a model exists in cache.

        Args:
            model_type: Type of model ('semantic_guard', 'semantic_rank', 'entropy',
                'token_classifier', 'coreference')

        Returns:
            True if model exists and size is valid, False otherwise
        """
        status = self.validate_model_cache(model_type, use_cache=True)
        if not status.get("cached_ok"):
            return False
        logger.info("Model %s validated in cache: %s", model_type, status.get("path"))
        return True

    def get_missing_models(self, model_types: Optional[List[str]] = None) -> List[str]:
        """
        Get list of missing or invalid models.

        Args:
            model_types: Specific model types to check. If None, checks all.

        Returns:
            List of model types that are missing or invalid
        """
        if model_types is None:
            model_types = list(self.configs.keys())

        missing = []
        for model_type in model_types:
            if not self.model_exists(model_type):
                missing.append(model_type)

        return missing

    def validate_model_cache(
        self,
        model_type: str,
        use_cache: bool = True,
        generate_manifest: bool = False,
    ) -> Dict[str, Any]:
        config = self.configs.get(model_type)
        if not config:
            return {"cached_ok": False, "cached_reason": "config_missing", "path": None}

        start_time = time.time()
        model_repo = _resolve_model_repo(model_type, config.get("model_name", ""))
        model_root = self._find_model_path(model_repo)
        if not model_root:
            return {"cached_ok": False, "cached_reason": "cache_missing", "path": None}

        now = time.time()
        if use_cache:
            cached = self._validation_cache.get(model_root)
            if (
                cached
                and now - float(cached.get("checked_at", 0)) < _VALIDATION_TTL_SECONDS
            ):
                return cached

        expected_files = config.get("expected_files") or []
        min_size_bytes = int(config.get("min_size_bytes") or 0)
        revision = (config.get("revision") or "").strip()

        size_valid = self._validate_model_size(model_root, min_size_bytes)
        manifest_status = "missing"
        if not size_valid:
            cached_ok, cached_reason = False, "size_too_small"
            manifest_status = "skipped"
        else:
            manifest_result = self._validate_manifest(
                model_root,
                expected_files=expected_files,
                revision=revision,
            )
            if manifest_result is not None:
                cached_ok, cached_reason = manifest_result
                manifest_status = "valid" if cached_ok else cached_reason
            else:
                files_valid, files_reason = self._validate_model_files(
                    model_root, expected_files
                )
                cached_ok, cached_reason = files_valid, files_reason

        if cached_ok and generate_manifest:
            self._ensure_manifest(model_root, expected_files, revision)
            manifest_status = "generated"

        result = {
            "cached_ok": cached_ok,
            "cached_reason": cached_reason,
            "path": model_root,
            "manifest_status": manifest_status,
            "checked_at": now,
        }
        self._validation_cache[model_root] = result
        duration = time.time() - start_time
        if duration >= 0.5:
            logger.info("Validated %s cache in %.2fs", model_type, duration)
        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached models.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "hub_cache_path": self.hub_cache,
            "hub_cache_exists": os.path.isdir(self.hub_cache),
            "models": {},
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        }

        if not os.path.isdir(self.hub_cache):
            return stats

        try:
            for model_type, config in self.configs.items():
                model_repo = _resolve_model_repo(
                    model_type, config.get("model_name", "")
                )
                model_path = self._find_model_path(model_repo)
                cached_ok = False
                cached_reason = "cache_missing"

                if model_path and os.path.isdir(model_path):
                    validation = self.validate_model_cache(
                        model_type, use_cache=True, generate_manifest=False
                    )
                    cached_ok = bool(validation.get("cached_ok"))
                    cached_reason = validation.get("cached_reason") or "cache_invalid"
                    manifest_status = validation.get("manifest_status")
                    if not cached_ok and _is_model_locked(self.hub_cache, model_type):
                        cached_reason = "locked"
                    size = self._get_cached_directory_size(model_path)
                    try:
                        mtime = os.path.getmtime(model_path)
                        from datetime import datetime

                        last_modified = datetime.fromtimestamp(mtime).isoformat()
                    except OSError:
                        last_modified = None

                    stats["models"][model_type] = {
                        "model_name": config["model_name"],
                        "path": model_path,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "size_formatted": self._format_size(size),
                        "cached_ok": cached_ok,
                        "cached_reason": cached_reason,
                        "manifest_status": manifest_status,
                        "status": "cached" if cached_ok else "invalid",
                        "last_modified": last_modified,
                        "revision": (config.get("revision") or "").strip() or None,
                    }
                    stats["total_size_bytes"] += size
                else:
                    stats["models"][model_type] = {
                        "model_name": config["model_name"],
                        "path": None,
                        "size_bytes": 0,
                        "size_mb": 0,
                        "size_formatted": "0 B",
                        "cached_ok": False,
                        "cached_reason": "cache_missing",
                        "manifest_status": "missing",
                        "status": "missing",
                        "last_modified": None,
                        "revision": (config.get("revision") or "").strip() or None,
                    }
        except Exception as exc:
            logger.error(f"Error gathering cache stats: {exc}")

        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        stats["total_size_formatted"] = self._format_size(stats["total_size_bytes"])
        return stats

    def _get_cached_directory_size(self, path: str) -> int:
        cached = self._size_cache.get(path)
        if cached is not None:
            return cached
        size = self._get_directory_size(path)
        self._size_cache[path] = size
        return size

    def _format_size(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def cleanup_incomplete_models(self, dry_run: bool = True) -> Dict[str, any]:
        """
        Find and optionally remove incomplete/corrupted model directories.

        Args:
            dry_run: If True, only report what would be deleted. If False, delete.

        Returns:
            Dictionary with cleanup results
        """
        results = {
            "dry_run": dry_run,
            "candidates_found": 0,
            "removed_count": 0,
            "freed_bytes": 0,
            "candidates": [],
        }

        if not os.path.isdir(self.hub_cache):
            return results

        valid_model_dirs = set()
        for model_type, config in self.configs.items():
            model_repo = _resolve_model_repo(model_type, config.get("model_name", ""))
            model_path = self._find_model_path(model_repo)
            if model_path:
                valid_model_dirs.add(os.path.basename(model_path))

        try:
            for entry in os.listdir(self.hub_cache):
                full_path = os.path.join(self.hub_cache, entry)
                if not os.path.isdir(full_path):
                    continue
                if entry == _LOCK_DIRNAME or entry.startswith(".tmp-download-"):
                    continue

                if entry not in valid_model_dirs:
                    size = self._get_cached_directory_size(full_path)
                    results["candidates_found"] += 1
                    results["candidates"].append(
                        {"path": full_path, "name": entry, "size_bytes": size}
                    )

                    if not dry_run:
                        try:
                            import shutil

                            shutil.rmtree(full_path)
                            results["removed_count"] += 1
                            results["freed_bytes"] += size
                            logger.info(f"Removed incomplete model: {entry}")
                        except Exception as exc:
                            logger.error(f"Failed to remove {entry}: {exc}")

        except Exception as exc:
            logger.error(f"Error during cleanup scan: {exc}")

        return results

    def _find_model_path(self, model_name: str) -> Optional[str]:
        """
        Find model path in HuggingFace cache.

        HF models are stored with sanitized directory names like:
        /hub/models--<org>--<model-name>

        Args:
            model_name: Model identifier (e.g., 'BAAI/bge-small-en-v1.5')

        Returns:
            Full path to model directory or None if not found
        """
        if not os.path.isdir(self.hub_cache):
            return None

        # Normalize model name to directory format
        # 'org/model-name' -> 'models--org--model-name'
        sanitized = model_name.replace("/", "--")
        target_dir = f"models--{sanitized}"

        try:
            for entry in os.listdir(self.hub_cache):
                if entry.startswith(target_dir):
                    full_path = os.path.join(self.hub_cache, entry)
                    if os.path.isdir(full_path):
                        return full_path
        except Exception as exc:
            logger.debug(f"Error searching for model {model_name}: {exc}")

        return None

    def _validate_model_size(self, model_path: str, min_size_bytes: int) -> bool:
        """
        Validate that model directory meets minimum size requirement.

        Args:
            model_path: Path to model directory
            min_size_bytes: Minimum acceptable size in bytes

        Returns:
            True if size is >= min_size_bytes
        """
        try:
            total_size = self._get_cached_directory_size(model_path)
            is_valid = total_size >= min_size_bytes

            if not is_valid:
                logger.debug(
                    f"Model size validation failed: {model_path} "
                    f"({total_size} bytes < {min_size_bytes} bytes minimum)"
                )
            return is_valid
        except Exception as exc:
            logger.error(f"Error validating model size at {model_path}: {exc}")
            return False

    def _validate_model_files(
        self, model_path: str, expected_files: List[str]
    ) -> Tuple[bool, str]:
        """
        Validate that expected files exist in model directory.

        Args:
            model_path: Path to model directory
            expected_files: List of expected file names

        Returns:
            True if all expected files exist
        """
        try:
            alternative_files = {
                "pytorch_model.bin": [
                    "model.safetensors",
                    "pytorch_model-*.bin",
                    "model-*.safetensors",
                ],
                "model.safetensors": [
                    "pytorch_model.bin",
                    "pytorch_model-*.bin",
                    "model-*.safetensors",
                ],
                "model.onnx": ["model.int8.onnx"],
            }

            for filename in expected_files:
                if _find_file_in_model_path(model_path, filename):
                    continue

                alternatives = alternative_files.get(filename, [])
                if any(
                    _find_file_in_model_path(model_path, alt) for alt in alternatives
                ):
                    continue

                logger.debug("Expected file not found: %s in %s", filename, model_path)
                return False, "missing_files"

            return True, "cached_ok"
        except Exception as exc:
            logger.error("Error validating model files at %s: %s", model_path, exc)
            return False, "validation_error"

    def _manifest_path(self, model_root: str) -> str:
        return os.path.join(model_root, _MANIFEST_FILENAME)

    def _load_manifest(self, model_root: str) -> Optional[Dict[str, Any]]:
        manifest_path = self._manifest_path(model_root)
        backup_path = manifest_path + _MANIFEST_BACKUP_SUFFIX
        for candidate in (manifest_path, backup_path):
            if not os.path.isfile(candidate):
                continue
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    if candidate == backup_path:
                        logger.info("Loaded manifest backup for %s", model_root)
                    return data
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read manifest %s: %s", candidate, exc)
                continue
        return None

    def _write_manifest(self, model_root: str, manifest: Dict[str, Any]) -> None:
        manifest_path = self._manifest_path(model_root)
        backup_path = manifest_path + _MANIFEST_BACKUP_SUFFIX
        if os.path.isfile(manifest_path):
            try:
                shutil.copy2(manifest_path, backup_path)
            except OSError as exc:
                logger.debug("Failed to backup manifest for %s: %s", model_root, exc)
        temp_path = f"{manifest_path}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        os.replace(temp_path, manifest_path)

    def _build_manifest(
        self,
        model_root: str,
        snapshot_dir: str,
        expected_files: List[str],
        revision: str,
    ) -> Dict[str, Any]:
        files: List[Dict[str, Any]] = []
        total_size = 0
        expected_patterns = [str(item) for item in expected_files if str(item)]
        for root, _, filenames in os.walk(snapshot_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    continue
                rel_path = os.path.relpath(file_path, model_root)
                digest = None
                if expected_patterns and any(
                    fnmatch.fnmatch(filename, pattern) for pattern in expected_patterns
                ):
                    digest = self._get_file_hash(file_path, "sha256")
                files.append({"path": rel_path, "size": size, "sha256": digest})
                total_size += size

        if total_size <= _FULL_HASH_MAX_BYTES:
            for entry in files:
                if entry.get("sha256"):
                    continue
                abs_path = os.path.join(model_root, entry["path"])
                entry["sha256"] = self._get_file_hash(abs_path, "sha256")
        return {
            "version": _MANIFEST_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "revision": revision,
            "snapshot": os.path.relpath(snapshot_dir, model_root),
            "expected_files": expected_files,
            "total_size_bytes": total_size,
            "files": files,
        }

    def _ensure_manifest(
        self, model_root: str, expected_files: List[str], revision: str
    ) -> None:
        if self._load_manifest(model_root) is not None:
            return
        snapshot_dir = _resolve_snapshot_dir(model_root)
        if not snapshot_dir:
            return
        manifest = self._build_manifest(
            model_root=model_root,
            snapshot_dir=snapshot_dir,
            expected_files=expected_files,
            revision=revision,
        )
        self._write_manifest(model_root, manifest)

    def _validate_manifest(
        self,
        model_root: str,
        expected_files: List[str],
        revision: str,
    ) -> Optional[Tuple[bool, str]]:
        manifest = self._load_manifest(model_root)
        if manifest is None:
            return None
        if int(manifest.get("version") or 0) != _MANIFEST_VERSION:
            return False, "manifest_version_mismatch"

        manifest_revision = (manifest.get("revision") or "").strip()
        if revision and not manifest_revision:
            return False, "revision_mismatch"
        if revision and manifest_revision and revision != manifest_revision:
            return False, "revision_mismatch"

        if expected_files:
            files_ok, files_reason = self._validate_model_files(
                model_root, expected_files
            )
            if not files_ok:
                return False, files_reason

        snapshot_rel = manifest.get("snapshot")
        if snapshot_rel:
            snapshot_dir = os.path.join(model_root, snapshot_rel)
            if not os.path.isdir(snapshot_dir):
                return False, "snapshot_missing"

        file_entries = manifest.get("files")
        if not isinstance(file_entries, list):
            return False, "manifest_invalid"

        expected_patterns = [str(item) for item in expected_files if str(item)]
        should_hash_all = bool(
            int(manifest.get("total_size_bytes") or 0) <= _FULL_HASH_MAX_BYTES
        )

        for entry in file_entries:
            if not isinstance(entry, dict):
                return False, "manifest_invalid"
            rel_path = entry.get("path")
            if not rel_path:
                return False, "manifest_invalid"
            abs_path = os.path.join(model_root, rel_path)
            if not os.path.isfile(abs_path):
                return False, "manifest_file_missing"
            expected_size = entry.get("size")
            try:
                current_size = os.path.getsize(abs_path)
            except OSError:
                return False, "manifest_file_missing"
            if expected_size is not None and current_size != expected_size:
                return False, "manifest_size_mismatch"

        for entry in file_entries:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("path")
            if not rel_path:
                continue
            filename = os.path.basename(rel_path)
            if not should_hash_all and expected_patterns:
                if not any(
                    fnmatch.fnmatch(filename, pattern) for pattern in expected_patterns
                ):
                    continue
            expected_hash = entry.get("sha256")
            if not expected_hash:
                continue
            abs_path = os.path.join(model_root, rel_path)
            current_hash = self._get_file_hash(abs_path, "sha256")
            if current_hash != expected_hash:
                return False, "manifest_hash_mismatch"

        return True, "cached_ok"

    @staticmethod
    def _get_directory_size(path: str) -> int:
        """
        Calculate total size of directory recursively.

        Args:
            path: Directory path

        Returns:
            Total size in bytes
        """
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file(follow_symlinks=True):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += ModelCacheValidator._get_directory_size(entry.path)
        except Exception as exc:
            logger.debug(f"Error calculating directory size for {path}: {exc}")
        return total

    @staticmethod
    def _get_file_hash(filepath: str, algorithm: str = "md5") -> Optional[str]:
        """
        Calculate hash of a file.

        Args:
            filepath: Path to file
            algorithm: Hash algorithm (default: md5)

        Returns:
            Hex digest of file hash or None if error
        """
        try:
            hash_obj = hashlib.new(algorithm)
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as exc:
            logger.debug(f"Error calculating hash for {filepath}: {exc}")
            return None


def ensure_models_cached(
    hf_home: str,
    model_types: Optional[List[str]] = None,
    validator: Optional[ModelCacheValidator] = None,
    export_onnx: bool = True,
    refresh_mode: str = "download_missing",
) -> Tuple[List[str], List[str]]:
    """
    Ensure models are cached, optionally triggering downloads for missing models.

    This is the main entry point for model validation.

    Args:
        hf_home: Path to HuggingFace cache directory
        model_types: Specific models to check. If None, checks all required models.
        validator: Optional shared validator instance for caching across calls.
        export_onnx: If True, generate missing ONNX artifacts when applicable.

    Returns:
        Tuple of (available_models, missing_models)
    """
    if validator is None:
        validator = ModelCacheValidator(hf_home)

    if model_types is None:
        model_types = list(validator.configs.keys())

    refresh_mode = refresh_mode.strip().lower()
    valid_modes = {"download_missing", "force_redownload", "recovery"}
    if refresh_mode not in valid_modes:
        raise ValueError(f"Invalid refresh_mode: {refresh_mode}")

    if refresh_mode == "recovery":
        cleanup = validator.cleanup_incomplete_models(dry_run=False)
        logger.info(
            "Recovery cleanup removed %s entries (freed %s bytes)",
            cleanup.get("removed_count"),
            cleanup.get("freed_bytes"),
        )

    with _allow_model_downloads():
        for model_type in model_types:
            config = validator.configs.get(model_type)
            if not config:
                continue
            model_name = config.get("model_name", "")
            model_repo = _resolve_model_repo(model_type, model_name)
            validation = validator.validate_model_cache(
                model_type, use_cache=False, generate_manifest=False
            )
            cached_ok = bool(validation.get("cached_ok"))
            should_download = refresh_mode == "force_redownload"
            if refresh_mode in {"download_missing", "recovery"} and not cached_ok:
                should_download = True
            if not should_download:
                continue
            if _should_pause_auth_download(model_type, model_repo):
                issue = _MODEL_DOWNLOAD_ISSUES.get(model_type, {})
                logger.warning(
                    "Skipping download for %s: %s",
                    model_type,
                    issue.get("message")
                    or "authentication failures exceeded retry limit",
                )
                continue
            logger.info("Refreshing model %s with mode %s", model_type, refresh_mode)
            _download_model(
                model_type,
                config,
                validator=validator,
            )

    validator._size_cache.clear()
    validator._validation_cache.clear()

    if export_onnx:
        _ensure_onnx_artifacts(validator, list(dict.fromkeys(model_types)))

    missing = [
        model_type
        for model_type in model_types
        if not validator.validate_model_cache(
            model_type, use_cache=False, generate_manifest=True
        ).get("cached_ok")
    ]
    available = [m for m in model_types if m not in missing]
    for model_type in available:
        clear_model_download_issue(model_type)

    return available, missing


def _download_model(
    model_type: str,
    config: Optional[dict] = None,
    validator: Optional[ModelCacheValidator] = None,
) -> bool:
    """
    Download a specific model from HuggingFace using atomic staging.

    Args:
        model_type: Type of model to download
        config: Model configuration (optional, fetched if None)
        validator: Optional cache validator instance
    Returns:
        True if download successful, False otherwise
    """
    if config is None:
        configs = get_model_configs()
        config = configs.get(model_type)

    if not config:
        logger.error("Unknown model type: %s", model_type)
        return False

    if validator is None:
        validator = ModelCacheValidator(resolve_hf_home())

    model_name = config.get("model_name", "")
    model_repo = _resolve_model_repo(model_type, model_name)
    revision = (config.get("revision") or "").strip() or None
    resolved_token = _resolve_hf_token()

    if not resolved_token and model_repo.lower().startswith("tokemizer/"):
        issue_message = (
            f"Model {model_repo} requires HuggingFace authentication. "
            "Provide HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in backend env/.env and retry refresh."
        )
        _record_model_download_issue(
            model_type,
            model_repo=model_repo,
            category="auth_failed",
            message=issue_message,
            metadata={
                "auth_failure_attempts": 1,
                "auth_token_fingerprint": _token_fingerprint(None),
            },
        )
        logger.warning("Skipping download for %s: %s", model_type, issue_message)
        return False

    lock_path = _acquire_model_lock(validator.hub_cache, model_type)
    if not lock_path:
        logger.warning("Skipping download for %s: cache lock held", model_type)
        return False

    temp_root = os.path.join(
        validator.hub_cache, f".tmp-download-{model_type}-{uuid.uuid4().hex}"
    )
    os.makedirs(temp_root, exist_ok=True)
    attempts_made = 0

    try:
        logger.info("Starting download for %s: %s", model_type, model_repo)

        if revision and not _validate_revision_online(model_repo, revision):
            logger.warning(
                "Revision check failed for %s@%s; proceeding with snapshot download",
                model_repo,
                revision,
            )

        from huggingface_hub import snapshot_download

        expected_files = config.get("expected_files") or []
        allow_patterns = _resolve_allow_patterns(config, expected_files)
        snapshot_kwargs: Dict[str, Any] = {
            "repo_id": model_repo,
            "allow_patterns": allow_patterns,
            "revision": revision,
            "cache_dir": temp_root,
        }
        if resolved_token:
            snapshot_kwargs["token"] = resolved_token

        start_time = time.time()
        last_download_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
            attempts_made = attempt
            try:
                try:
                    snapshot_download(**snapshot_kwargs)
                except Exception as exc:
                    if resolved_token and _is_hf_auth_error(exc):
                        logger.warning(
                            "Authenticated download failed for %s; retrying anonymously: %s",
                            model_repo,
                            exc,
                        )
                        snapshot_kwargs["token"] = False
                        snapshot_download(**snapshot_kwargs)
                    else:
                        raise
                last_download_exc = None
                break
            except Exception as exc:
                last_download_exc = exc
                if attempt >= _MAX_DOWNLOAD_ATTEMPTS:
                    raise
                logger.warning(
                    "Download attempt %s/%s failed for %s: %s. Retrying...",
                    attempt,
                    _MAX_DOWNLOAD_ATTEMPTS,
                    model_type,
                    exc,
                )
                if _DOWNLOAD_RETRY_BACKOFF_SECONDS > 0:
                    time.sleep(min(_DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt, 5.0))
        if last_download_exc is not None:
            raise last_download_exc
        duration = time.time() - start_time
        logger.info("Downloaded %s in %.2fs", model_type, duration)

        temp_model_root = _find_model_path_in_cache(temp_root, model_repo)
        if not temp_model_root:
            logger.error(
                "Download completed but cache path not found for %s", model_type
            )
            return False

        target_dir_name = os.path.basename(temp_model_root)
        final_model_root = os.path.join(validator.hub_cache, target_dir_name)

        if os.path.isdir(final_model_root):
            shutil.rmtree(final_model_root)

        os.replace(temp_model_root, final_model_root)
        _cleanup_stale_model_dirs(
            validator.hub_cache,
            model_repo,
            keep_path=final_model_root,
        )

        validator._size_cache.pop(final_model_root, None)
        validator._validation_cache.pop(final_model_root, None)
        validator._ensure_manifest(
            final_model_root,
            expected_files=expected_files,
            revision=revision or "",
        )
        clear_model_download_issue(model_type)
        logger.info("Cached %s at %s", model_type, final_model_root)
        return True
    except Exception as exc:
        issue_message = str(exc)
        if _is_hf_auth_error(exc):
            current_fingerprint = _token_fingerprint(_resolve_hf_token())
            prior_issue = _MODEL_DOWNLOAD_ISSUES.get(model_type, {})
            prior_fingerprint = str(prior_issue.get("auth_token_fingerprint") or "")
            try:
                prior_attempts = int(prior_issue.get("auth_failure_attempts") or 0)
            except (TypeError, ValueError):
                prior_attempts = 0
            attempts = (
                prior_attempts + max(1, attempts_made)
                if prior_fingerprint == current_fingerprint
                else max(1, attempts_made)
            )
            attempts = min(_MAX_AUTH_FAILURE_ATTEMPTS, attempts)
            blocked = attempts >= _MAX_AUTH_FAILURE_ATTEMPTS
            issue_category = "auth_blocked" if blocked else "auth_failed"
            issue_message = _build_auth_issue_message(
                model_repo,
                attempts=attempts,
                blocked=blocked,
            )
            if blocked:
                logger.error(
                    "Paused download attempts for %s after %s/%s authentication "
                    "failures for %s. Admin action required to fix HF token.",
                    model_type,
                    attempts,
                    _MAX_AUTH_FAILURE_ATTEMPTS,
                    model_repo,
                )
            else:
                logger.warning(
                    "Authentication failed for %s (%s/%s) on %s. "
                    "Provide a valid HF_TOKEN/HUGGINGFACE_HUB_TOKEN in backend env/.env.",
                    model_type,
                    attempts,
                    _MAX_AUTH_FAILURE_ATTEMPTS,
                    model_repo,
                )
            metadata = {
                "auth_failure_attempts": attempts,
                "auth_token_fingerprint": current_fingerprint,
            }
        else:
            issue_category = "download_failed"
            attempts = max(1, attempts_made)
            issue_message = _build_download_issue_message(
                model_repo,
                attempts=attempts,
            )
            metadata = {"download_failure_attempts": attempts}
        _record_model_download_issue(
            model_type,
            model_repo=model_repo,
            category=issue_category,
            message=issue_message,
            metadata=metadata,
        )
        if _is_hf_auth_error(exc):
            logger.warning("Failed to download %s due to authentication", model_type)
        else:
            logger.error("Failed to download %s: %s", model_type, exc)
        return False
    finally:
        _release_model_lock(lock_path)
        try:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root)
        except OSError:
            pass


def _safe_extract_zip(archive_path: str, target_dir: str) -> None:
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_name = member.filename
            if not member_name:
                continue
            destination = os.path.abspath(os.path.join(target_dir, member_name))
            target_abs = os.path.abspath(target_dir)
            if (
                not destination.startswith(target_abs + os.sep)
                and destination != target_abs
            ):
                raise ValueError("Archive contains invalid path traversal entries")
        zip_ref.extractall(target_dir)


def _safe_extract_tar(archive_path: str, target_dir: str) -> None:
    with tarfile.open(archive_path, "r:*") as tar_ref:
        members = tar_ref.getmembers()
        target_abs = os.path.abspath(target_dir)
        for member in members:
            member_name = member.name
            if not member_name:
                continue
            destination = os.path.abspath(os.path.join(target_dir, member_name))
            if (
                not destination.startswith(target_abs + os.sep)
                and destination != target_abs
            ):
                raise ValueError("Archive contains invalid path traversal entries")
        tar_ref.extractall(target_dir, members=members)


def _extract_model_archive(archive_path: str, target_dir: str) -> None:
    if zipfile.is_zipfile(archive_path):
        _safe_extract_zip(archive_path, target_dir)
        return
    if tarfile.is_tarfile(archive_path):
        _safe_extract_tar(archive_path, target_dir)
        return
    raise ValueError("Unsupported archive format. Use .zip, .tar, .tar.gz, or .tgz")


def cache_uploaded_model_archive(
    hf_home: str,
    model_type: str,
    archive_path: str,
    validator: Optional[ModelCacheValidator] = None,
) -> Dict[str, Any]:
    if validator is None:
        validator = ModelCacheValidator(hf_home)

    config = validator.configs.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}")

    model_name = config.get("model_name", "")
    model_repo = _resolve_model_repo(model_type, model_name)
    revision = (config.get("revision") or "").strip()
    expected_files = config.get("expected_files") or []
    target_dir_name = f"models--{model_repo.replace('/', '--')}"
    final_model_root = os.path.join(validator.hub_cache, target_dir_name)
    lock_path = _acquire_model_lock(validator.hub_cache, model_type)
    if not lock_path:
        raise RuntimeError(f"Upload lock is held for {model_type}; try again shortly")

    temp_model_root = os.path.join(
        validator.hub_cache, f".tmp-upload-{model_type}-{uuid.uuid4().hex}"
    )
    snapshot_hash = f"manual-{uuid.uuid4().hex[:16]}"
    temp_snapshot_dir = os.path.join(temp_model_root, "snapshots", snapshot_hash)
    temp_refs_dir = os.path.join(temp_model_root, "refs")
    os.makedirs(temp_snapshot_dir, exist_ok=True)
    os.makedirs(temp_refs_dir, exist_ok=True)
    with open(os.path.join(temp_refs_dir, "main"), "w", encoding="utf-8") as fh:
        fh.write(snapshot_hash)

    try:
        _extract_model_archive(archive_path, temp_snapshot_dir)

        has_files = False
        for _, _, filenames in os.walk(temp_snapshot_dir):
            if filenames:
                has_files = True
                break
        if not has_files:
            raise ValueError("Uploaded archive is empty")

        if os.path.isdir(final_model_root):
            shutil.rmtree(final_model_root)
        os.replace(temp_model_root, final_model_root)
        _cleanup_stale_model_dirs(
            validator.hub_cache,
            model_repo,
            keep_path=final_model_root,
        )

        validator._size_cache.pop(final_model_root, None)
        validator._validation_cache.pop(final_model_root, None)
        validator._ensure_manifest(
            final_model_root,
            expected_files=expected_files,
            revision=revision,
        )
        status = validator.validate_model_cache(
            model_type, use_cache=False, generate_manifest=True
        )
        if not status.get("cached_ok"):
            raise RuntimeError(
                f"Uploaded model failed validation: {status.get('cached_reason') or 'cache_invalid'}"
            )

        return {
            "model_type": model_type,
            "cached_ok": True,
            "cached_reason": status.get("cached_reason", "cached_ok"),
            "path": final_model_root,
            "revision": revision or None,
        }
    finally:
        _release_model_lock(lock_path)
        try:
            if os.path.isdir(temp_model_root):
                shutil.rmtree(temp_model_root)
        except OSError:
            pass
