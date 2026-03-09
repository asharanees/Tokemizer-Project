import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import auth_utils
from auth import require_admin
from database import (
    Customer,
    aggregate_history_stats,
    create_customer,
    create_subscription_plan,
    delete_subscription_plan,
    disable_customer,
    get_admin_setting,
    get_customer_by_id,
    get_llm_system_context,
    get_subscription_plan_by_id,
    get_usage,
    list_all_customers,
    list_recent_history,
    list_recent_telemetry,
    list_subscription_plans,
    list_usage_breakdown,
    set_admin_setting,
    set_llm_system_context,
    try_acquire_admin_setting_lock,
    update_customer,
)
from database_extensions import (
    add_or_update_model_inventory,
    delete_model_inventory,
    get_model_inventory_item,
    list_model_inventory,
)
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, EmailStr, Field
from services import logging_control, telemetry_control
from services.email import email_service
from services.model_cache_manager import (
    ModelCacheValidator,
    cache_uploaded_model_archive,
    ensure_models_cached,
    get_model_download_issues,
    reset_model_download_issues,
    resolve_hf_home,
)
from services.optimizer.model_capabilities import (
    build_model_readiness,
    list_capabilities_for_model,
    model_lookup_from_status,
)
from services.quota_manager import quota_manager
from services.rate_limiter import api_rate_limiter
from services.secret_store import encrypt_value

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)

PROTECTED_MODEL_TYPES = {
    "semantic_guard",
    "semantic_rank",
    "entropy",
    "entropy_fast",
    "token_classifier",
    "coreference",
}
_MODEL_REFRESH_STATE_KEY = "model_refresh_state"
_MODEL_REFRESH_TTL_SECONDS = 30 * 60
_MODEL_CACHE_SNAPSHOT_KEY = "model_cache_snapshot"
_MODEL_CACHE_VALIDATION_VERSION_KEY = "model_cache_validation_version"
_MODEL_REFRESH_MODES = {"download_missing", "force_redownload", "recovery"}
_ADMIN_SETTINGS_ALLOWED_KEYS = {
    "smtp_host",
    "smtp_port",
    "smtp_user",
    "smtp_from_email",
    "smtp_password",
    "stripe_secret_key",
    "stripe_publishable_key",
    "log_level",
    "telemetry_enabled",
    "access_token_expire_minutes",
    "refresh_token_expire_days",
    "history_enabled",
    "learned_abbreviations_enabled",
    "optimizer_prewarm_models",
    "cors_origins",
    "llm_system_context",
}
_ADMIN_SETTINGS_BOOLEAN_KEYS = {
    "telemetry_enabled",
    "history_enabled",
    "learned_abbreviations_enabled",
    "optimizer_prewarm_models",
}
_ADMIN_SETTINGS_INTEGER_KEYS = {
    "smtp_port",
    "access_token_expire_minutes",
    "refresh_token_expire_days",
}
_ADMIN_SETTINGS_STRING_KEYS = {
    "smtp_host",
    "smtp_user",
    "smtp_from_email",
    "stripe_publishable_key",
    "cors_origins",
}
_ADMIN_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _read_admin_int_setting(key: str, default: int) -> int:
    try:
        return int(get_admin_setting(key, default))
    except (TypeError, ValueError):
        return default


def _read_admin_log_level_setting(default: str = "INFO") -> str:
    raw_value = str(get_admin_setting("log_level", default) or "").strip().upper()
    if raw_value in _ADMIN_LOG_LEVELS:
        return raw_value
    return default


def _normalize_admin_setting(key: str, value: Any) -> Any:
    if key == "log_level":
        normalized = str(value or "").strip().upper()
        if normalized not in _ADMIN_LOG_LEVELS:
            raise HTTPException(
                status_code=400,
                detail="log_level must be one of DEBUG|INFO|WARNING|ERROR|CRITICAL",
            )
        return normalized

    if key in _ADMIN_SETTINGS_BOOLEAN_KEYS:
        if isinstance(value, bool):
            return value
        raise HTTPException(status_code=400, detail=f"{key} must be a boolean")

    if key in _ADMIN_SETTINGS_INTEGER_KEYS:
        try:
            normalized_int = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{key} must be an integer")
        if key == "smtp_port" and (normalized_int < 1 or normalized_int > 65535):
            raise HTTPException(
                status_code=400,
                detail="smtp_port must be between 1 and 65535",
            )
        if key != "smtp_port" and normalized_int < 1:
            raise HTTPException(status_code=400, detail=f"{key} must be >= 1")
        return normalized_int

    if key in _ADMIN_SETTINGS_STRING_KEYS:
        return str(value or "").strip()

    return value


def _refresh_optimizer_models() -> None:
    from services.optimizer.core import optimizer

    optimizer.refresh_model_configs()


def _update_model_cache_snapshot(
    validator: ModelCacheValidator,
    stats: Dict[str, Any],
    model_status: Dict[str, Any],
    scoped_model_type: Optional[str] = None,
) -> None:
    all_model_types = (
        list(validator.configs.keys()) if isinstance(validator.configs, dict) else []
    )
    missing_models = (
        validator.get_missing_models(all_model_types) if all_model_types else []
    )
    available_models = [
        model_type for model_type in all_model_types if model_type not in missing_models
    ]
    stats_models = stats.get("models") if isinstance(stats.get("models"), dict) else {}
    issues = get_model_download_issues()
    snapshot_warnings: List[str] = []
    for model_type, issue in issues.items():
        model_entry = stats_models.get(model_type)
        if not isinstance(model_entry, dict):
            continue
        if bool(model_entry.get("cached_ok")):
            continue
        issue_category = str(issue.get("category") or "").strip().lower()
        issue_message = str(issue.get("message") or "").strip()
        if issue_category in {"auth_failed", "auth_blocked"}:
            model_entry["cached_reason"] = "auth_failed"
        if issue_message:
            model_entry["cached_error_detail"] = issue_message
            snapshot_warnings.append(f"{model_type}: {issue_message}")
    readiness_snapshot = _build_model_status_snapshot(
        stats=stats,
        model_status=model_status,
    )
    if scoped_model_type:
        existing_snapshot = _get_model_cache_snapshot()
        existing_status = existing_snapshot.get("model_status")
        merged_status = (
            dict(existing_status) if isinstance(existing_status, dict) else {}
        )
        if scoped_model_type in readiness_snapshot:
            merged_status[scoped_model_type] = readiness_snapshot[scoped_model_type]
        readiness_snapshot = merged_status

    _set_model_cache_snapshot(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
            "model_status": readiness_snapshot,
            "available_models": available_models,
            "missing_models": missing_models,
            "warnings": list(dict.fromkeys(snapshot_warnings)),
        }
    )


def bump_model_cache_validation_version() -> int:
    """Signal request-path model cache validators to invalidate local TTL caches."""
    next_value = int(time.time() * 1000)
    set_admin_setting(_MODEL_CACHE_VALIDATION_VERSION_KEY, next_value)
    return next_value


def get_model_cache_validation_version() -> int:
    raw = get_admin_setting(_MODEL_CACHE_VALIDATION_VERSION_KEY, 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def refresh_model_cache_snapshot(
    *,
    scoped_model_type: Optional[str] = None,
    validator: Optional[ModelCacheValidator] = None,
    model_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recompute and persist model cache/readiness snapshot from live cache and optimizer state."""
    from services.model_cache_manager import get_model_configs, get_spacy_cache_status
    from services.optimizer.core import optimizer

    working_validator = validator or ModelCacheValidator(resolve_hf_home())
    working_validator.configs = get_model_configs()

    live_status = model_status
    if live_status is None:
        live_status = optimizer.model_status()
    if not isinstance(live_status, dict):
        live_status = {}

    stats = working_validator.get_cache_stats()
    spacy_model_name = getattr(optimizer, "_spacy_model_name", "en_core_web_sm")
    spacy_stat = get_spacy_cache_status(spacy_model_name, validator=working_validator)
    stats.setdefault("models", {})["spacy"] = spacy_stat
    try:
        total_without_spacy = int(stats.get("total_size_bytes") or 0)
        stats["total_size_bytes"] = total_without_spacy + int(
            spacy_stat.get("size_bytes") or 0
        )
        stats["total_size_formatted"] = working_validator._format_size(
            stats["total_size_bytes"]
        )
    except (TypeError, ValueError):
        pass

    _update_model_cache_snapshot(
        validator=working_validator,
        stats=stats,
        model_status=live_status,
        scoped_model_type=scoped_model_type,
    )
    return _get_model_cache_snapshot()


def _default_model_refresh_state() -> Dict[str, Any]:
    return {
        "state": "idle",
        "started_at": None,
        "started_at_epoch": None,
        "finished_at": None,
        "error": None,
        "mode": None,
        "available_models": [],
        "missing_models": [],
        "target_models": [],
        "warnings": [],
    }


def _normalize_model_refresh_state(state: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_model_refresh_state()
    base.update(state)

    # Clear stale/invalid in-progress states so UI does not remain blocked after
    # process restarts or interrupted refresh workers.
    if base.get("state") == "running":
        now_epoch = int(time.time())
        started_at_epoch_raw = base.get("started_at_epoch")
        try:
            started_at_epoch = int(started_at_epoch_raw)
        except (TypeError, ValueError):
            started_at_epoch = None
        is_stale = (
            started_at_epoch is None
            or (now_epoch - started_at_epoch) > _MODEL_REFRESH_TTL_SECONDS
        )
        if is_stale:
            return _default_model_refresh_state()

    return base


def _default_model_cache_snapshot() -> Dict[str, Any]:
    return {
        "generated_at": None,
        "stats": {
            "models": {},
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {},
        "available_models": [],
        "missing_models": [],
        "warnings": [],
    }


def _get_model_cache_snapshot() -> Dict[str, Any]:
    snapshot = get_admin_setting(
        _MODEL_CACHE_SNAPSHOT_KEY, _default_model_cache_snapshot()
    )
    if not isinstance(snapshot, dict):
        return _default_model_cache_snapshot()
    base = _default_model_cache_snapshot()
    base.update(snapshot)
    stats = base.get("stats")
    if not isinstance(stats, dict):
        base["stats"] = _default_model_cache_snapshot()["stats"]
    else:
        stats_base = _default_model_cache_snapshot()["stats"]
        stats_base.update(stats)
        base["stats"] = stats_base
    return base


def _set_model_cache_snapshot(snapshot: Dict[str, Any]) -> None:
    set_admin_setting(_MODEL_CACHE_SNAPSHOT_KEY, snapshot)


def _invalidate_model_cache_snapshot(model_type: Optional[str] = None) -> None:
    if not model_type:
        _set_model_cache_snapshot(_default_model_cache_snapshot())
        return
    snapshot = _get_model_cache_snapshot()
    stats = snapshot.get("stats") if isinstance(snapshot.get("stats"), dict) else {}
    models = stats.get("models") if isinstance(stats.get("models"), dict) else {}
    models.pop(model_type, None)
    stats["models"] = models
    snapshot["stats"] = stats
    for key in ("available_models", "missing_models"):
        values = snapshot.get(key)
        if isinstance(values, list):
            snapshot[key] = [item for item in values if item != model_type]
    model_status = snapshot.get("model_status")
    if isinstance(model_status, dict):
        model_status.pop(model_type, None)
        snapshot["model_status"] = model_status
    _set_model_cache_snapshot(snapshot)


def _derive_loaded_status(
    model_type: str, model_status: Dict[str, Any]
) -> Tuple[Optional[bool], str]:
    if not model_status:
        return None, "not_refreshed"
    status = model_status.get(model_type)
    if not isinstance(status, dict):
        return None, "not_warmed"
    if status.get("loaded"):
        return True, "loaded_ok"
    return False, "load_failed"


def _has_warmup_epoch(live_status: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(live_status, dict):
        return False
    return live_status.get("__warmup_epoch") is not None


def _normalize_live_loaded_status(
    live_status: Optional[Dict[str, Any]], model_type: str
) -> Tuple[Optional[bool], str]:
    """Map optimizer warm-up status into API-facing loaded fields."""
    status = live_status.get(model_type) if isinstance(live_status, dict) else None
    warmup_ran = _has_warmup_epoch(live_status)

    if not isinstance(status, dict) or "loaded" not in status:
        return None, "not_warmed"

    loaded = status.get("loaded")
    if loaded is True:
        return True, "loaded_ok"
    if loaded is False:
        return (False, "load_failed") if warmup_ran else (False, "not_warmed")
    return None, "not_warmed"


def _live_loaded_status_with_presence(
    live_status: Optional[Dict[str, Any]], model_type: str
) -> Tuple[bool, Optional[bool], str]:
    """Return normalized live loaded status plus whether live data is authoritative."""
    status = live_status.get(model_type) if isinstance(live_status, dict) else None
    warmup_ran = _has_warmup_epoch(live_status)
    loaded_ok, loaded_reason = _normalize_live_loaded_status(live_status, model_type)

    if not isinstance(status, dict) or "loaded" not in status:
        return False, loaded_ok, loaded_reason

    loaded = status.get("loaded")
    has_live_entry = loaded is True or warmup_ran
    return has_live_entry, loaded_ok, loaded_reason


def _capability_lookup_from_statuses(
    model_status_snapshot: Optional[Dict[str, Any]],
    live_status: Optional[Dict[str, Any]],
) -> Dict[str, bool]:
    """Build capability lookup from live warm-up status with snapshot fallback."""
    lookup: Dict[str, bool] = {}

    if isinstance(live_status, dict) and live_status:
        live_lookup = model_lookup_from_status(live_status)
        # Keep only authoritative model entries from live status to avoid
        # overwriting snapshot readiness with pre-warmup default values.
        for model_type, status in live_status.items():
            if not isinstance(status, dict):
                continue
            has_live_entry, _loaded_ok, _loaded_reason = (
                _live_loaded_status_with_presence(live_status, model_type)
            )
            if not has_live_entry:
                continue
            lookup[model_type] = bool(live_lookup.get(model_type, False))

    snapshot = model_status_snapshot if isinstance(model_status_snapshot, dict) else {}
    for model_type, status in snapshot.items():
        if model_type in lookup or not isinstance(status, dict):
            continue
        loaded_ok = status.get("loaded_ok")
        if loaded_ok is None:
            continue
        lookup[model_type] = bool(loaded_ok)

    return lookup


def _build_model_status_snapshot(
    stats: Dict[str, Any], model_status: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    models = stats.get("models") if isinstance(stats.get("models"), dict) else {}
    readiness: Dict[str, Dict[str, Any]] = {}
    for model_type, info in models.items():
        if not isinstance(info, dict):
            continue
        cached_ok = bool(info.get("cached_ok"))
        cached_reason = info.get("cached_reason") or (
            "cached_ok" if cached_ok else "cache_missing"
        )
        loaded_ok, loaded_reason = _derive_loaded_status(model_type, model_status)
        readiness[model_type] = {
            "cached_ok": cached_ok,
            "cached_reason": cached_reason,
            "loaded_ok": loaded_ok,
            "loaded_reason": loaded_reason,
        }
    return readiness


def _targeted_live_status(
    live_status: Dict[str, Any], model_type: str
) -> Dict[str, Any]:
    """Return a fresh status payload scoped to the model that was just probed."""

    if not isinstance(live_status, dict):
        return {}

    targeted: Dict[str, Any] = {}
    model_entry = live_status.get(model_type)
    if isinstance(model_entry, dict):
        warmup_epoch = live_status.get("__warmup_epoch")
        if warmup_epoch is not None:
            targeted["__warmup_epoch"] = warmup_epoch
        targeted[model_type] = model_entry

    return targeted


def _get_model_refresh_state() -> Dict[str, Any]:
    state = get_admin_setting(_MODEL_REFRESH_STATE_KEY, _default_model_refresh_state())
    if not isinstance(state, dict):
        return _default_model_refresh_state()
    normalized = _normalize_model_refresh_state(state)
    if normalized != state:
        _set_model_refresh_state(normalized)
    return normalized


def _set_model_refresh_state(state: Dict[str, Any]) -> None:
    set_admin_setting(_MODEL_REFRESH_STATE_KEY, state)


def _try_start_model_refresh(state: Dict[str, Any]) -> Dict[str, Any]:
    stale_before = int(time.time()) - _MODEL_REFRESH_TTL_SECONDS
    acquired, existing = try_acquire_admin_setting_lock(
        _MODEL_REFRESH_STATE_KEY,
        state,
        stale_before_epoch=stale_before,
    )
    if acquired:
        return state
    if isinstance(existing, dict):
        base = _default_model_refresh_state()
        base.update(existing)
        return base
    return _default_model_refresh_state()


def _run_model_refresh(
    hf_home: str,
    refresh_mode: str,
    model_types: Optional[List[str]] = None,
) -> None:
    from services.model_cache_manager import (
        ensure_spacy_model_cached,
        get_model_configs,
        get_spacy_cache_status,
    )
    from services.optimizer.core import optimizer

    state_snapshot = _get_model_refresh_state()
    started_at = state_snapshot.get("started_at")
    started_at_epoch = state_snapshot.get("started_at_epoch")
    target_models = state_snapshot.get("target_models") or []

    try:
        validator = ModelCacheValidator(hf_home)
        validator.configs = get_model_configs()
        all_model_types = list(validator.configs.keys())
        requested_model_types = model_types or (all_model_types + ["spacy"])
        spacy_requested = "spacy" in requested_model_types
        hf_target_model_types = [
            model_type
            for model_type in requested_model_types
            if model_type in all_model_types
        ]
        if not hf_target_model_types and not spacy_requested:
            raise RuntimeError("No valid model types available for refresh.")

        available: List[str] = []
        missing: List[str] = []
        if hf_target_model_types:
            available, missing = ensure_models_cached(
                hf_home,
                hf_target_model_types,
                validator=validator,
                refresh_mode=refresh_mode,
            )
        validator._size_cache.clear()

        spacy_model_name = getattr(optimizer, "_spacy_model_name", "en_core_web_sm")
        allow_spacy_downloads = True
        spacy_stat: Optional[Dict[str, Any]] = None
        if spacy_requested:
            ensure_spacy_model_cached(
                spacy_model_name, allow_downloads=allow_spacy_downloads
            )
            spacy_stat = get_spacy_cache_status(
                spacy_model_name, validator=validator
            )

        missing = validator.get_missing_models(all_model_types)
        available = [model for model in all_model_types if model not in missing]
        optimizer.refresh_model_configs()
        optimizer.warm_up()
        model_status = optimizer.model_status()
        snapshot = refresh_model_cache_snapshot(
            validator=validator,
            model_status=model_status,
        )
        stats = snapshot.get("stats") if isinstance(snapshot.get("stats"), dict) else {}
        readiness_snapshot = (
            snapshot.get("model_status")
            if isinstance(snapshot.get("model_status"), dict)
            else {}
        )
        missing = list(snapshot.get("missing_models") or [])
        available = list(snapshot.get("available_models") or [])
        if spacy_requested:
            if spacy_stat is None:
                spacy_stat = get_spacy_cache_status(
                    spacy_model_name, validator=validator
                )
            if bool(spacy_stat.get("cached_ok")):
                if "spacy" not in available:
                    available.append("spacy")
                missing = [model for model in missing if model != "spacy"]
            else:
                if "spacy" not in missing:
                    missing.append("spacy")
                available = [model for model in available if model != "spacy"]
        readiness_lookup = model_lookup_from_status(model_status)
        readiness_contract = build_model_readiness(readiness_lookup)
        required_not_ready = sorted(
            [
                model
                for model, info in readiness_contract.items()
                if info.get("hard_required") and not info.get("intended_usage_ready")
            ]
        )
        warnings: List[str] = []
        if required_not_ready:
            warnings.append(
                "Pipeline required models not ready: " + ", ".join(required_not_ready)
            )
        if missing:
            warnings.append("Cache missing models: " + ", ".join(sorted(missing)))
        snapshot_warnings = [
            item for item in (snapshot.get("warnings") or []) if isinstance(item, str)
        ]
        warnings.extend(snapshot_warnings)
        warnings = list(dict.fromkeys(warnings))
        target_scope = [
            model_type for model_type in target_models if isinstance(model_type, str)
        ]
        scoped_missing = (
            [model for model in missing if model in target_scope]
            if target_scope
            else missing
        )
        scoped_not_ready: List[str] = []
        readiness_sources = [readiness_snapshot, readiness_contract]
        scoped_candidates = target_scope if target_scope else all_model_types
        for model_type in scoped_candidates:
            for readiness_source in readiness_sources:
                model_readiness = readiness_source.get(model_type)
                if not isinstance(model_readiness, dict):
                    continue
                if model_readiness.get("intended_usage_ready") is False:
                    scoped_not_ready.append(model_type)
                    break
        scoped_not_ready = sorted(set(scoped_not_ready))
        required_refresh_failures = (
            required_not_ready if not target_scope else []
        )
        refresh_failed = (
            len(scoped_missing) > 0
            or len(scoped_not_ready) > 0
            or len(required_refresh_failures) > 0
        )
        failure_reasons: List[str] = []
        if scoped_missing:
            failure_reasons.append(
                "failed model download(s): " + ", ".join(sorted(scoped_missing))
            )
        if scoped_not_ready:
            failure_reasons.append(
                "not-ready model(s) after warm-up: "
                + ", ".join(sorted(scoped_not_ready))
            )
        if required_refresh_failures:
            failure_reasons.append(
                "required models not ready: "
                + ", ".join(sorted(required_refresh_failures))
            )
        refresh_error = (
            "Refresh completed with errors: " + "; ".join(failure_reasons)
            if refresh_failed
            else None
        )
        bump_model_cache_validation_version()
        _set_model_refresh_state(
            {
                "state": "failed" if refresh_failed else "completed",
                "started_at": started_at,
                "started_at_epoch": started_at_epoch,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": refresh_error,
                "mode": refresh_mode,
                "available_models": available,
                "missing_models": missing,
                "target_models": target_models,
                "warnings": warnings,
            }
        )
    except Exception as exc:
        _set_model_refresh_state(
            {
                "state": "failed",
                "started_at": started_at,
                "started_at_epoch": started_at_epoch,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "mode": refresh_mode,
                "available_models": [],
                "missing_models": [],
                "target_models": target_models,
                "warnings": [],
            }
        )


def _parse_expected_files(value: Any) -> List[str]:
    if isinstance(value, list):
        parsed = [str(item).strip() for item in value if str(item).strip()]
        return parsed

    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    items = [segment.strip() for segment in str(value).split(",")]
    return [item for item in items if item]


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    subscription_status: Optional[str] = None
    subscription_tier: Optional[str] = None
    quota_override: Optional[int] = None
    quota_overage_bonus: Optional[int] = Field(None, ge=0)


class UserResponse(BaseModel):
    id: str
    name: Optional[str]
    email: Optional[str]
    role: str
    is_active: bool
    subscription_status: str
    subscription_tier: str
    created_at: str
    quota_overage_bonus: int = 0
    last_active: Optional[str] = None  # TODO: Track last login


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    offset: int
    limit: int


class PlanResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    monthly_price_cents: int
    annual_price_cents: Optional[int] = None
    monthly_quota: int
    rate_limit_rpm: int
    concurrent_optimization_jobs: int = Field(5, ge=1)
    batch_size_limit: int = Field(1000, ge=1)
    optimization_history_retention_days: int = Field(365, ge=1)
    telemetry_retention_days: int = Field(365, ge=1)
    audit_log_retention_days: int = Field(365, ge=1)
    custom_canonical_mappings_limit: int = Field(1000, ge=1)
    max_api_keys: int
    features: List[str]
    is_active: bool
    is_public: bool = True
    plan_term: str = "monthly"
    monthly_discount_percent: int = 0
    yearly_discount_percent: int = 0


class PlanInput(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    monthly_price_cents: int
    annual_price_cents: Optional[int] = None
    monthly_quota: int
    rate_limit_rpm: int
    concurrent_optimization_jobs: int = Field(5, ge=1)
    batch_size_limit: int = Field(1000, ge=1)
    optimization_history_retention_days: int = Field(365, ge=1)
    telemetry_retention_days: int = Field(365, ge=1)
    audit_log_retention_days: int = Field(365, ge=1)
    custom_canonical_mappings_limit: int = Field(1000, ge=1)
    max_api_keys: int
    features: List[str]
    is_active: bool
    is_public: bool = True
    plan_term: str = "monthly"
    monthly_discount_percent: int = 0
    yearly_discount_percent: int = 0


class TestEmailRequest(BaseModel):
    to_email: EmailStr


def _ensure_subscription_tier_for_customers(
    role: Optional[str], subscription_tier: Optional[str]
) -> None:
    """Ensure customer users always have a subscription plan assigned."""
    if (role or "").lower() == "customer" and not subscription_tier:
        raise HTTPException(
            status_code=400,
            detail="Customers must have a subscription plan selected",
        )
    if (role or "").lower() == "customer" and subscription_tier:
        plan = get_subscription_plan_by_id(
            subscription_tier,
            include_inactive=True,
            include_non_public=True,
        )
        if not plan:
            raise HTTPException(
                status_code=400,
                detail="Selected subscription plan does not exist",
            )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    offset: int = 0,
    limit: int = 50,
    admin: Customer = Depends(require_admin),
):
    """List all users (Admin only)."""
    users, total = list_all_customers(offset, limit)
    return {
        "users": [
            UserResponse(
                id=u.id,
                name=u.name,
                email=u.email,
                role=u.role,
                is_active=u.is_active,
                subscription_status=u.subscription_status,
                subscription_tier=u.subscription_tier,
                created_at=u.created_at,
                quota_overage_bonus=u.quota_overage_bonus,
                last_active=u.updated_at,
            )
            for u in users
        ],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.post("/users", response_model=UserResponse)
async def create_user(
    user: UserUpdate, password: str, admin: Customer = Depends(require_admin)
):
    """Create a new user (Admin only)."""
    if not user.email:
        raise HTTPException(status_code=400, detail="Email required")
    if not password:
        raise HTTPException(status_code=400, detail="Password required")

    password_hash = auth_utils.get_password_hash(password)
    role = (user.role or "customer").lower()
    _ensure_subscription_tier_for_customers(role, user.subscription_tier)

    new_user = create_customer(
        name=user.name or "",
        email=user.email,
    )

    updates: Dict[str, Any] = {
        "password_hash": password_hash,
        "role": role,
        "is_active": user.is_active if user.is_active is not None else True,
        "subscription_status": user.subscription_status or "active",
    }

    if user.subscription_tier:
        updates["subscription_tier"] = user.subscription_tier
    elif role != "customer":
        updates["subscription_tier"] = "free"

    if user.quota_override is not None:
        updates["quota_override"] = user.quota_override
    if user.quota_overage_bonus is not None:
        updates["quota_overage_bonus"] = user.quota_overage_bonus

    updated = update_customer(new_user.id, **updates)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to create user")

    return UserResponse(
        id=updated.id,
        name=updated.name,
        email=updated.email,
        role=updated.role,
        is_active=updated.is_active,
        subscription_status=updated.subscription_status,
        subscription_tier=updated.subscription_tier,
        created_at=updated.created_at,
        quota_overage_bonus=updated.quota_overage_bonus,
        last_active=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, admin: Customer = Depends(require_admin)):
    """Get user details."""
    user = get_customer_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        subscription_status=user.subscription_status,
        subscription_tier=user.subscription_tier,
        created_at=user.created_at,
        quota_overage_bonus=user.quota_overage_bonus,
        last_active=user.updated_at,
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user_details(
    user_id: str, update: UserUpdate, admin: Customer = Depends(require_admin)
):
    """Update user details."""
    user = get_customer_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    final_role = (update.role or user.role).lower()
    final_subscription_tier = (
        update.subscription_tier
        if update.subscription_tier is not None
        else user.subscription_tier
    )
    _ensure_subscription_tier_for_customers(final_role, final_subscription_tier)

    updates: Dict[str, Any] = {}
    if update.name is not None:
        updates["name"] = update.name
    if update.email is not None:
        updates["email"] = update.email
    if update.role is not None:
        updates["role"] = update.role.lower()
    if update.is_active is not None:
        updates["is_active"] = update.is_active
    if update.subscription_status is not None:
        updates["subscription_status"] = update.subscription_status
    if update.subscription_tier is not None:
        updates["subscription_tier"] = update.subscription_tier
    if update.quota_override is not None:
        updates["quota_override"] = update.quota_override
    if update.quota_overage_bonus is not None:
        updates["quota_overage_bonus"] = update.quota_overage_bonus

    updated = update_customer(user_id, **updates)

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update user")

    return UserResponse(
        id=updated.id,
        name=updated.name,
        email=updated.email,
        role=updated.role,
        is_active=updated.is_active,
        subscription_status=updated.subscription_status,
        subscription_tier=updated.subscription_tier,
        created_at=updated.created_at,
        quota_overage_bonus=updated.quota_overage_bonus,
    )


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin: Customer = Depends(require_admin)):
    """Disable a user."""
    success = disable_customer(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User disabled"}


@router.get("/plans", response_model=List[PlanResponse])
async def list_plans(
    public_only: bool = Query(
        False,
        description="When true, return only plans with is_public=true. Default returns all plans.",
    ),
    admin: Customer = Depends(require_admin),
):
    """List all subscription plans (Admin only)."""
    plans = list_subscription_plans(
        include_inactive=True,
        include_non_public=not public_only,
    )
    return [
        PlanResponse(
            id=p.id,
            name=p.name,
            monthly_price_cents=p.monthly_price_cents,
            annual_price_cents=p.annual_price_cents,
            monthly_quota=p.monthly_quota,
            rate_limit_rpm=p.rate_limit_rpm,
            concurrent_optimization_jobs=p.concurrent_optimization_jobs,
            batch_size_limit=p.batch_size_limit,
            optimization_history_retention_days=p.optimization_history_retention_days,
            telemetry_retention_days=p.telemetry_retention_days,
            audit_log_retention_days=p.audit_log_retention_days,
            custom_canonical_mappings_limit=p.custom_canonical_mappings_limit,
            max_api_keys=p.max_api_keys,
            features=p.features,
            is_active=p.is_active,
            is_public=p.is_public,
            plan_term=p.plan_term,
            monthly_discount_percent=p.monthly_discount_percent,
            yearly_discount_percent=p.yearly_discount_percent,
        )
        for p in plans
    ]


@router.post("/plans", response_model=PlanResponse)
async def create_plan(plan: PlanInput, admin: Customer = Depends(require_admin)):
    """Create or update a subscripton plan (Admin only)."""
    new_plan = create_subscription_plan(
        id=plan.id,
        name=plan.name,
        description=plan.description,
        monthly_price_cents=plan.monthly_price_cents,
        annual_price_cents=plan.annual_price_cents,
        monthly_quota=plan.monthly_quota,
        rate_limit_rpm=plan.rate_limit_rpm,
        concurrent_optimization_jobs=plan.concurrent_optimization_jobs,
        batch_size_limit=plan.batch_size_limit,
        optimization_history_retention_days=plan.optimization_history_retention_days,
        telemetry_retention_days=plan.telemetry_retention_days,
        audit_log_retention_days=plan.audit_log_retention_days,
        custom_canonical_mappings_limit=plan.custom_canonical_mappings_limit,
        max_api_keys=plan.max_api_keys,
        features=plan.features,
        is_active=plan.is_active,
        is_public=plan.is_public,
        plan_term=plan.plan_term,
        monthly_discount_percent=plan.monthly_discount_percent,
        yearly_discount_percent=plan.yearly_discount_percent,
    )
    return PlanResponse(
        id=new_plan.id,
        name=new_plan.name,
        description=new_plan.description,
        monthly_price_cents=new_plan.monthly_price_cents,
        annual_price_cents=new_plan.annual_price_cents,
        monthly_quota=new_plan.monthly_quota,
        rate_limit_rpm=new_plan.rate_limit_rpm,
        concurrent_optimization_jobs=new_plan.concurrent_optimization_jobs,
        batch_size_limit=new_plan.batch_size_limit,
        optimization_history_retention_days=new_plan.optimization_history_retention_days,
        telemetry_retention_days=new_plan.telemetry_retention_days,
        audit_log_retention_days=new_plan.audit_log_retention_days,
        custom_canonical_mappings_limit=new_plan.custom_canonical_mappings_limit,
        max_api_keys=new_plan.max_api_keys,
        features=new_plan.features,
        is_active=new_plan.is_active,
        is_public=new_plan.is_public,
        plan_term=new_plan.plan_term,
        monthly_discount_percent=new_plan.monthly_discount_percent,
        yearly_discount_percent=new_plan.yearly_discount_percent,
    )


@router.delete("/plans/{plan_id}")
async def delete_plan(plan_id: str, admin: Customer = Depends(require_admin)):
    """Delete a subscription plan (Admin only).

    A plan can only be deleted if no customers have active subscriptions to it.
    """
    success = delete_subscription_plan(plan_id)
    if not success:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete plan: customers have active subscriptions to this plan",
        )
    return {"message": "Plan deleted successfully"}


@router.get("/tenant-health")
async def tenant_health(admin: Customer = Depends(require_admin)):
    """Per-tenant operational health metrics for production operations."""
    users, _ = list_all_customers(0, 5000)
    payload: List[Dict[str, Any]] = []
    for user in users:
        if user.role != "customer":
            continue
        stats = aggregate_history_stats(limit=500, customer_id=user.id)
        history = list_recent_history(limit=500, customer_id=user.id)
        telemetry = list_recent_telemetry(limit=1000, customer_id=user.id)
        latencies = sorted(
            [
                float(row.get("duration_ms") or 0.0)
                for row in telemetry
                if float(row.get("duration_ms") or 0.0) > 0
            ]
        )

        def percentile(values: List[float], pct: float) -> float:
            if not values:
                return 0.0
            idx = int(round((pct / 100.0) * (len(values) - 1)))
            idx = max(0, min(idx, len(values) - 1))
            return round(values[idx], 2)

        total = max(len(history), 1)
        failures = sum(
            1 for rec in history if float(rec.semantic_similarity or 1.0) < 0.5
        )
        error_rate = round((failures / total) * 100.0, 2)
        _, remaining, quota_total = quota_manager.check_quota(user.id)
        quota_used = max(0, quota_total - remaining)
        quota_burndown_pct = (
            0.0 if quota_total <= 0 else round((quota_used / quota_total) * 100.0, 2)
        )

        payload.append(
            {
                "tenant_id": user.id,
                "email": user.email,
                "subscription_tier": user.subscription_tier,
                "error_rate_pct": error_rate,
                "p95_latency_ms": percentile(latencies, 95.0),
                "p99_latency_ms": percentile(latencies, 99.0),
                "token_throughput": int(stats.get("tokens_saved", 0.0)),
                "quota_limit": quota_total,
                "quota_used": quota_used,
                "quota_burndown_pct": quota_burndown_pct,
            }
        )
    return {"tenants": payload}


@router.get("/settings")
async def get_global_settings(admin: Customer = Depends(require_admin)):
    """Get global admin settings (SMTP, Stripe, Logging, Telemetry, Security, Optimization)."""
    return {
        "smtp_host": get_admin_setting("smtp_host", "localhost"),
        "smtp_port": get_admin_setting("smtp_port", 1025),
        "smtp_user": get_admin_setting("smtp_user", ""),
        "smtp_from_email": get_admin_setting(
            "smtp_from_email", "noreply@tokemizer.com"
        ),
        "smtp_password_set": bool(get_admin_setting("smtp_password", "")),
        "stripe_secret_key_set": bool(get_admin_setting("stripe_secret_key", "")),
        "stripe_publishable_key": get_admin_setting("stripe_publishable_key", ""),
        "log_level": _read_admin_log_level_setting("INFO"),
        "telemetry_enabled": bool(get_admin_setting("telemetry_enabled", False)),
        "access_token_expire_minutes": _read_admin_int_setting(
            "access_token_expire_minutes", 30
        ),
        "refresh_token_expire_days": _read_admin_int_setting(
            "refresh_token_expire_days", 7
        ),
        "history_enabled": bool(get_admin_setting("history_enabled", True)),
        "learned_abbreviations_enabled": bool(
            get_admin_setting("learned_abbreviations_enabled", True)
        ),
        "optimizer_prewarm_models": bool(
            get_admin_setting("optimizer_prewarm_models", True)
        ),
        "cors_origins": get_admin_setting("cors_origins", "*"),
        "llm_system_context": get_llm_system_context(),
    }


@router.patch("/settings")
async def update_global_settings(
    settings: dict, admin: Customer = Depends(require_admin)
):
    """Update global admin settings."""
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Settings payload must be an object")

    normalized_settings: Dict[str, Any] = {}
    for key, value in settings.items():
        if key not in _ADMIN_SETTINGS_ALLOWED_KEYS:
            raise HTTPException(status_code=400, detail=f"Cannot update {key}")
        if key in ("smtp_password", "stripe_secret_key"):
            # Sensitive fields are encrypted
            if value:
                set_admin_setting(key, encrypt_value(str(value)))
            else:
                set_admin_setting(key, "")
            continue
        if key == "llm_system_context":
            set_llm_system_context(str(value or ""))
            continue
        normalized_settings[key] = _normalize_admin_setting(key, value)

    for key, value in normalized_settings.items():
        set_admin_setting(key, value)

    # Apply runtime changes immediately
    if "log_level" in normalized_settings:
        logging_control.set_level(str(normalized_settings["log_level"]))
    if "telemetry_enabled" in normalized_settings:
        telemetry_control.set_enabled(bool(normalized_settings["telemetry_enabled"]))

    return {"status": "success"}


@router.post("/settings/test-email")
async def send_test_email(
    payload: TestEmailRequest, admin: Customer = Depends(require_admin)
):
    """Send a test email using current SMTP configuration."""
    email_service.send_email(
        to_email=payload.to_email,
        subject="Tokemizer SMTP Test",
        body="This is a test email from Tokemizer.",
    )
    return {"status": "sent"}


@router.get("/usage/{user_id}")
async def get_user_usage(user_id: str, admin: Customer = Depends(require_admin)):
    """Admin: Get a specific user's current usage."""
    user = get_customer_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.now(timezone.utc)
    period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()
    usage_record = get_usage(user.id, period_start)
    _, remaining, total = quota_manager.check_quota(user.id)
    breakdown = list_usage_breakdown(user.id, period_start)

    return {
        "user_id": user.id,
        "calls_used": usage_record.calls_used if usage_record else 0,
        "quota_limit": total,
        "remaining": remaining,
        "period_start": period_start,
        "subscription_tier": user.subscription_tier,
        "subscription_status": user.subscription_status,
        "breakdown": [
            {
                "source": entry.source,
                "calls": entry.calls_used,
                "api_key_id": entry.api_key_id,
                "name": entry.api_key_name,
            }
            for entry in breakdown
        ],
    }


class ModelInfo(BaseModel):
    model_type: str
    model_name: str
    component: str
    library_type: str
    usage: str
    min_size_bytes: int = 0
    expected_files: List[str] = Field(default_factory=list)
    revision: Optional[str] = None
    allow_patterns: List[str] = Field(default_factory=list)
    size_bytes: int
    size_formatted: str
    download_date: Optional[str] = None
    cached_ok: bool
    cached_reason: Optional[str] = None
    cached_error_detail: Optional[str] = None
    loaded_ok: Optional[bool] = None
    loaded_reason: Optional[str] = None
    intended_usage_ready: Optional[bool] = None
    intended_usage_reason: Optional[str] = None
    intended_features: List[str] = Field(default_factory=list)
    required_mode_gates: List[str] = Field(default_factory=list)
    required_profile_gates: List[str] = Field(default_factory=list)
    hard_required: bool = False
    last_refresh: Optional[str] = None
    path: Optional[str] = None
    refresh_state: Optional[str] = None
    refresh_mode: Optional[str] = None
    refresh_target_models: List[str] = Field(default_factory=list)


class ModelUpdate(BaseModel):
    model_name: str
    component: Optional[str] = None
    library_type: Optional[str] = None
    usage: Optional[str] = None
    min_size_bytes: Optional[int] = Field(default=None, ge=0)
    expected_files: Optional[List[str]] = None
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None


class ModelCreate(BaseModel):
    model_type: str
    model_name: str
    component: Optional[str] = None
    library_type: Optional[str] = None
    usage: Optional[str] = None
    min_size_bytes: int = Field(default=0, ge=0)
    expected_files: List[str] = Field(default_factory=lambda: ["config.json"])
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total_size_bytes: int
    total_size_formatted: str
    warnings: List[str] = Field(default_factory=list)


class ProtectedModelTypesResponse(BaseModel):
    protected_model_types: List[str]


class ModelUploadResponse(BaseModel):
    model_type: str
    cached_ok: bool
    cached_reason: Optional[str] = None
    loaded_ok: Optional[bool] = None
    loaded_reason: Optional[str] = None
    intended_usage_ready: Optional[bool] = None
    intended_usage_reason: Optional[str] = None
    message: str


@router.get("/models/protected", response_model=ProtectedModelTypesResponse)
async def list_protected_model_types(admin: Customer = Depends(require_admin)):
    return {"protected_model_types": sorted(PROTECTED_MODEL_TYPES)}


@router.get("/models", response_model=ModelListResponse)
async def list_models(admin: Customer = Depends(require_admin)):
    """List all model configurations and their cache status (Admin only)."""
    snapshot = _get_model_cache_snapshot()
    stats = snapshot.get("stats") if isinstance(snapshot.get("stats"), dict) else {}
    stats_models = stats.get("models") if isinstance(stats.get("models"), dict) else {}
    model_status = (
        snapshot.get("model_status")
        if isinstance(snapshot.get("model_status"), dict)
        else {}
    )
    last_refresh = snapshot.get("generated_at")
    snapshot_warnings = [
        item for item in (snapshot.get("warnings") or []) if isinstance(item, str)
    ]

    live_status: Optional[Dict[str, Any]] = None
    fresh_spacy_stat: Optional[Dict[str, Any]] = None
    try:
        from services.optimizer.core import optimizer
        from services.model_cache_manager import get_spacy_cache_status

        candidate_status = optimizer.model_status()
        if isinstance(candidate_status, dict):
            live_status = candidate_status

        try:
            spacy_model_name = getattr(optimizer, "_spacy_model_name", "en_core_web_sm")
            fresh_spacy_stat = get_spacy_cache_status(spacy_model_name)
            if fresh_spacy_stat:
                stats_models["spacy"] = fresh_spacy_stat
        except Exception:
            fresh_spacy_stat = None

        has_spacy_live_entry, live_spacy_loaded_ok, _live_spacy_loaded_reason = (
            _live_loaded_status_with_presence(live_status, "spacy")
        )
        should_probe_spacy = bool(fresh_spacy_stat and fresh_spacy_stat.get("cached_ok")) and (
            not has_spacy_live_entry
            or live_spacy_loaded_ok is False
            or live_spacy_loaded_ok is None
        )
        if should_probe_spacy and hasattr(optimizer, "probe_model_readiness"):
            try:
                probe = optimizer.probe_model_readiness("spacy")
                if isinstance(probe, dict):
                    refreshed_status = optimizer.model_status()
                    if isinstance(refreshed_status, dict):
                        live_status = refreshed_status
            except Exception:
                pass
    except Exception:
        live_status = None

    can_use_snapshot_loaded_state = _has_warmup_epoch(live_status)

    inventory = list_model_inventory()
    capability_lookup = _capability_lookup_from_statuses(model_status, live_status)
    readiness_by_model = build_model_readiness(capability_lookup)
    models_list = []
    inventory_types = set()
    for row in inventory:
        model_type = row.get("model_type", "")
        inventory_types.add(model_type)
        try:
            min_size_bytes = int(row.get("min_size_bytes") or 0)
        except (TypeError, ValueError):
            min_size_bytes = 0
        expected_files = _parse_expected_files(row.get("expected_files"))
        model_stat = stats_models.get(model_type)
        safe_model_stat = model_stat or {}
        readiness_stat = model_status.get(model_type, {}) if model_status else {}
        has_live_entry, live_loaded_ok, live_loaded_reason = (
            _live_loaded_status_with_presence(live_status, model_type)
        )
        cached_ok = bool(safe_model_stat.get("cached_ok", False))
        cached_reason = safe_model_stat.get("cached_reason")
        loaded_ok = (
            live_loaded_ok
            if has_live_entry
            else (
                (readiness_stat.get("loaded_ok") if readiness_stat else None)
                if can_use_snapshot_loaded_state
                else None
            )
        )
        loaded_reason = (
            live_loaded_reason
            if has_live_entry
            else (
                (readiness_stat.get("loaded_reason") if readiness_stat else None)
                if can_use_snapshot_loaded_state
                else "not_warmed"
            )
        )
        allow_patterns = _parse_expected_files(row.get("allow_patterns"))
        capability = list_capabilities_for_model(model_type)
        intended_readiness = readiness_by_model.get(model_type, {})
        models_list.append(
            ModelInfo(
                model_type=model_type,
                model_name=row.get("model_name", ""),
                component=row.get("component", "") or model_type,
                library_type=row.get("library_type", ""),
                usage=row.get("usage", ""),
                min_size_bytes=min_size_bytes,
                expected_files=expected_files,
                revision=(row.get("revision") or "").strip() or None,
                allow_patterns=allow_patterns,
                size_bytes=safe_model_stat.get("size_bytes", 0),
                size_formatted=safe_model_stat.get("size_formatted", "0 B"),
                download_date=safe_model_stat.get("last_modified"),
                cached_ok=cached_ok,
                cached_reason=cached_reason,
                cached_error_detail=safe_model_stat.get("cached_error_detail"),
                loaded_ok=loaded_ok,
                loaded_reason=loaded_reason,
                intended_usage_ready=intended_readiness.get("intended_usage_ready"),
                intended_usage_reason=intended_readiness.get("intended_usage_reason"),
                intended_features=list(capability.get("intended_features") or []),
                required_mode_gates=list(capability.get("required_mode_gates") or []),
                required_profile_gates=list(
                    capability.get("required_profile_gates") or []
                ),
                hard_required=bool(capability.get("hard_required")),
                last_refresh=last_refresh,
                path=(
                    str(safe_model_stat.get("path"))
                    if safe_model_stat.get("path")
                    else None
                ),
            )
        )

    if "spacy" not in inventory_types:
        spacy_status = model_status.get("spacy") if model_status else None
        (
            has_spacy_live_entry,
            live_spacy_loaded,
            live_spacy_loaded_reason,
        ) = _live_loaded_status_with_presence(live_status, "spacy")
        spacy_loaded = (
            live_spacy_loaded
            if has_spacy_live_entry
            else (
                (spacy_status.get("loaded_ok") if spacy_status else None)
                if can_use_snapshot_loaded_state
                else None
            )
        )
        spacy_loaded_reason = (
            live_spacy_loaded_reason
            if has_spacy_live_entry
            else (
                (spacy_status.get("loaded_reason") if spacy_status else None)
                if can_use_snapshot_loaded_state
                else "not_warmed"
            )
        )
        spacy_stat = stats_models.get("spacy") or {}
        spacy_capability = list_capabilities_for_model("spacy")
        spacy_readiness = readiness_by_model.get("spacy", {})
        models_list.append(
            ModelInfo(
                model_type="spacy",
                model_name=spacy_stat.get("model_name", "en_core_web_sm"),
                component="spaCy",
                library_type="spacy",
                usage="Semantic deduplication and linguistic passes",
                min_size_bytes=0,
                expected_files=[],
                revision=None,
                allow_patterns=[],
                size_bytes=spacy_stat.get("size_bytes", 0),
                size_formatted=spacy_stat.get("size_formatted", "0 B"),
                download_date=spacy_stat.get("last_modified"),
                cached_ok=bool(spacy_stat.get("cached_ok")),
                cached_reason=spacy_stat.get("cached_reason"),
                cached_error_detail=spacy_stat.get("cached_error_detail"),
                loaded_ok=spacy_loaded,
                loaded_reason=spacy_loaded_reason,
                intended_usage_ready=spacy_readiness.get("intended_usage_ready"),
                intended_usage_reason=spacy_readiness.get("intended_usage_reason"),
                intended_features=list(spacy_capability.get("intended_features") or []),
                required_mode_gates=list(
                    spacy_capability.get("required_mode_gates") or []
                ),
                required_profile_gates=list(
                    spacy_capability.get("required_profile_gates") or []
                ),
                hard_required=bool(spacy_capability.get("hard_required")),
                last_refresh=last_refresh,
                path=(str(spacy_stat.get("path")) if spacy_stat.get("path") else None),
            )
        )

    return {
        "models": models_list,
        "total_size_bytes": stats.get("total_size_bytes", 0),
        "total_size_formatted": stats.get("total_size_formatted", "0 B"),
        "warnings": snapshot_warnings,
    }


@router.get("/models/refresh", response_model=Dict[str, Any])
async def get_model_refresh_status(admin: Customer = Depends(require_admin)):
    return _get_model_refresh_state()


@router.post("/models/refresh", response_model=Dict[str, Any])
async def refresh_models(
    mode: str = Query("download_missing"),
    admin: Customer = Depends(require_admin),
):
    hf_home = resolve_hf_home()
    refresh_mode = mode.strip().lower()
    if refresh_mode not in _MODEL_REFRESH_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid refresh mode: {mode}")

    now_epoch = int(time.time())
    requested_state = {
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_at_epoch": now_epoch,
        "finished_at": None,
        "error": None,
        "mode": refresh_mode,
        "available_models": [],
        "missing_models": [],
        "target_models": [],
        "warnings": [],
    }
    current_state = _try_start_model_refresh(requested_state)
    if current_state != requested_state:
        return current_state

    reset_count = reset_model_download_issues()
    if reset_count:
        logger.info(
            "Admin-triggered refresh reset %s model download failure state(s).",
            reset_count,
        )

    Thread(
        target=_run_model_refresh,
        args=(hf_home, refresh_mode, None),
        daemon=True,
    ).start()
    bump_model_cache_validation_version()
    return current_state


@router.post("/models/{model_type}/refresh", response_model=Dict[str, Any])
async def refresh_model(
    model_type: str,
    mode: str = Query("download_missing"),
    admin: Customer = Depends(require_admin),
):
    hf_home = resolve_hf_home()
    refresh_mode = mode.strip().lower()
    if refresh_mode not in _MODEL_REFRESH_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid refresh mode: {mode}")

    if model_type != "spacy" and not get_model_inventory_item(model_type):
        raise HTTPException(
            status_code=404, detail=f"Model type {model_type} not found"
        )

    now_epoch = int(time.time())
    requested_state = {
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_at_epoch": now_epoch,
        "finished_at": None,
        "error": None,
        "mode": refresh_mode,
        "available_models": [],
        "missing_models": [],
        "target_models": [model_type],
        "warnings": [],
    }
    current_state = _try_start_model_refresh(requested_state)
    if current_state != requested_state:
        return current_state

    if model_type != "spacy":
        reset_count = reset_model_download_issues([model_type])
        if reset_count:
            logger.info(
                "Admin-triggered model refresh reset failure state for %s.",
                model_type,
            )

    Thread(
        target=_run_model_refresh,
        args=(hf_home, refresh_mode, [model_type]),
        daemon=True,
    ).start()
    bump_model_cache_validation_version()
    return current_state


@router.get("/models/airgap", response_model=Dict[str, Any])
async def check_airgap_readiness(admin: Customer = Depends(require_admin)):
    hf_home = resolve_hf_home()
    validator = ModelCacheValidator(hf_home)
    from services.model_cache_manager import get_model_configs, get_spacy_cache_status

    validator.configs = get_model_configs()
    missing_models: List[str] = []
    invalid_models: List[str] = []
    manifest_failures: Dict[str, str] = {}

    for model_type in validator.configs.keys():
        status = validator.validate_model_cache(model_type, use_cache=False)
        if status.get("cached_ok"):
            continue
        reason = status.get("cached_reason") or "cache_invalid"
        if reason == "cache_missing":
            missing_models.append(model_type)
        else:
            invalid_models.append(model_type)
        if reason.startswith("manifest") or reason == "revision_mismatch":
            manifest_failures[model_type] = reason

    from services.optimizer.core import optimizer

    spacy_model_name = getattr(optimizer, "_spacy_model_name", "en_core_web_sm")
    spacy_status = get_spacy_cache_status(spacy_model_name, validator=validator)
    if not spacy_status.get("cached_ok"):
        missing_models.append("spacy")

    return {
        "ready": not missing_models and not invalid_models,
        "missing_models": missing_models,
        "invalid_models": invalid_models,
        "manifest_failures": manifest_failures,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _start_targeted_model_refresh(
    *,
    hf_home: str,
    model_type: str,
    refresh_mode: str,
) -> Dict[str, Any]:
    now_epoch = int(time.time())
    requested_state = {
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_at_epoch": now_epoch,
        "finished_at": None,
        "error": None,
        "mode": refresh_mode,
        "available_models": [],
        "missing_models": [],
        "target_models": [model_type],
        "warnings": [],
    }
    current_state = _try_start_model_refresh(requested_state)
    if current_state == requested_state:
        if model_type != "spacy":
            reset_model_download_issues([model_type])
        Thread(
            target=_run_model_refresh,
            args=(hf_home, refresh_mode, [model_type]),
            daemon=True,
        ).start()
        bump_model_cache_validation_version()
    return current_state


@router.post("/models", response_model=ModelInfo)
async def create_model_config(
    model: ModelCreate,
    admin: Customer = Depends(require_admin),
):
    """Create a new model configuration (Admin only)."""

    # Check if already exists
    existing = get_model_inventory_item(model.model_type)
    if existing:
        raise HTTPException(
            status_code=409, detail=f"Model type {model.model_type} already exists"
        )

    updated_record = add_or_update_model_inventory(
        model_type=model.model_type,
        model_name=model.model_name,
        min_size_bytes=model.min_size_bytes or 0,
        expected_files=(
            model.expected_files
            if model.expected_files is not None
            else ["config.json"]
        ),
        component=model.component or model.model_type,
        library_type=model.library_type or "",
        usage=model.usage or "",
        revision=model.revision,
        allow_patterns=model.allow_patterns,
    )

    if not updated_record:
        raise HTTPException(
            status_code=500, detail="Failed to create model configuration"
        )
    _invalidate_model_cache_snapshot(model.model_type)

    hf_home = resolve_hf_home()
    _refresh_optimizer_models()

    refresh_state = _start_targeted_model_refresh(
        hf_home=hf_home,
        model_type=model.model_type,
        refresh_mode="download_missing",
    )

    created_expected_files = _parse_expected_files(updated_record.get("expected_files"))
    created_allow_patterns = _parse_expected_files(updated_record.get("allow_patterns"))

    return ModelInfo(
        model_type=model.model_type,
        model_name=updated_record["model_name"],
        component=updated_record.get("component", model.model_type),
        library_type=updated_record.get("library_type", ""),
        usage=updated_record.get("usage", ""),
        min_size_bytes=updated_record.get("min_size_bytes", 0),
        expected_files=created_expected_files,
        revision=(updated_record.get("revision") or "").strip() or None,
        allow_patterns=created_allow_patterns,
        size_bytes=0,
        size_formatted="0 B",
        download_date=None,
        cached_ok=False,
        cached_reason="refresh_pending",
        loaded_ok=None,
        loaded_reason="not_warmed",
        last_refresh=_get_model_cache_snapshot().get("generated_at"),
        path=None,
        refresh_state=refresh_state.get("state"),
        refresh_mode=refresh_state.get("mode"),
        refresh_target_models=list(refresh_state.get("target_models") or []),
    )


@router.post("/models/{model_type}/upload", response_model=ModelUploadResponse)
async def upload_model_archive(
    model_type: str,
    archive: UploadFile = File(...),
    admin: Customer = Depends(require_admin),
):
    if model_type == "spacy":
        raise HTTPException(
            status_code=400,
            detail="spaCy upload is not supported through archive upload; use refresh download.",
        )

    if not get_model_inventory_item(model_type):
        raise HTTPException(
            status_code=404, detail=f"Model type {model_type} not found"
        )

    filename = (archive.filename or "").strip().lower()
    allowed_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    if filename and not any(filename.endswith(ext) for ext in allowed_suffixes):
        raise HTTPException(
            status_code=400,
            detail="Unsupported archive format. Upload .zip, .tar, .tar.gz, or .tgz",
        )

    hf_home = resolve_hf_home()
    validator = ModelCacheValidator(hf_home)
    from services.model_cache_manager import get_model_configs

    validator.configs = get_model_configs()
    temp_file_path: Optional[str] = None
    try:
        import tempfile

        suffix = ""
        if filename.endswith(".tar.gz"):
            suffix = ".tar.gz"
        elif filename.endswith(".tgz"):
            suffix = ".tgz"
        elif filename.endswith(".tar"):
            suffix = ".tar"
        elif filename.endswith(".zip"):
            suffix = ".zip"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await archive.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        upload_status = cache_uploaded_model_archive(
            hf_home,
            model_type=model_type,
            archive_path=temp_file_path,
            validator=validator,
        )
        _invalidate_model_cache_snapshot(model_type)
        _refresh_optimizer_models()

        from services.optimizer.core import optimizer

        optimizer.probe_model_readiness(model_type)
        stats = validator.get_cache_stats()
        live_status = _targeted_live_status(optimizer.model_status(), model_type)
        status_snapshot = _build_model_status_snapshot(stats, live_status)
        _update_model_cache_snapshot(
            validator,
            stats,
            live_status,
            scoped_model_type=model_type,
        )
        capability_lookup = _capability_lookup_from_statuses(
            status_snapshot, live_status
        )
        readiness_by_model = build_model_readiness(capability_lookup)
        readiness = readiness_by_model.get(model_type, {})
        loaded_info = status_snapshot.get(model_type, {})

        return ModelUploadResponse(
            model_type=model_type,
            cached_ok=bool(upload_status.get("cached_ok")),
            cached_reason=upload_status.get("cached_reason"),
            loaded_ok=loaded_info.get("loaded_ok"),
            loaded_reason=loaded_info.get("loaded_reason"),
            intended_usage_ready=readiness.get("intended_usage_ready"),
            intended_usage_reason=readiness.get("intended_usage_reason"),
            message=f"Model archive uploaded and validated for {model_type}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            await archive.close()
        except Exception:
            pass
        if temp_file_path and os.path.isfile(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


@router.delete("/models/{model_type}")
async def delete_model_config(
    model_type: str,
    override_core_models: bool = Query(
        False,
        description=(
            "Set to true to remove protected core optimizer models "
            "(semantic_guard, entropy, token_classifier)."
        ),
    ),
    admin: Customer = Depends(require_admin),
):
    """Delete (disable) a model configuration (Admin only)."""
    if model_type in PROTECTED_MODEL_TYPES and not override_core_models:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Deleting '{model_type}' is protected because it powers core optimizer"
                " capabilities. Add '?override_core_models=true' to confirm if you"
                " understand the impact."
            ),
        )
    success = delete_model_inventory(model_type)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Model type {model_type} not found"
        )
    _invalidate_model_cache_snapshot(model_type)

    # Trigger re-validation
    hf_home = resolve_hf_home()
    validator = ModelCacheValidator(hf_home)
    from services.model_cache_manager import get_model_configs

    validator.configs = get_model_configs()
    _refresh_optimizer_models()
    bump_model_cache_validation_version()

    return {"status": "success", "message": f"Model {model_type} deleted"}


@router.put("/models/{model_type}", response_model=ModelInfo)
async def update_model_config(
    model_type: str,
    update: ModelUpdate,
    admin: Customer = Depends(require_admin),
):
    """Update configuration for a specific model type (Admin only)."""
    # Get current config to preserve values if not provided
    current = get_model_inventory_item(model_type)
    if not current:
        raise HTTPException(
            status_code=404, detail=f"Model type {model_type} not found"
        )

    # Parse existing expected files
    current_files = _parse_expected_files(current.get("expected_files"))
    if not current_files:
        current_files = ["config.json"]
    current_allow_patterns = _parse_expected_files(current.get("allow_patterns"))

    updated_record = add_or_update_model_inventory(
        model_type=model_type,
        model_name=update.model_name,
        min_size_bytes=(
            update.min_size_bytes
            if update.min_size_bytes is not None
            else current["min_size_bytes"]
        ),
        expected_files=(
            update.expected_files
            if update.expected_files is not None
            else current_files
        ),
        revision=(
            update.revision if update.revision is not None else current.get("revision")
        ),
        allow_patterns=(
            update.allow_patterns
            if update.allow_patterns is not None
            else current_allow_patterns
        ),
        component=(
            update.component
            if update.component is not None
            else current.get("component", "")
        ),
        library_type=(
            update.library_type
            if update.library_type is not None
            else current.get("library_type", "")
        ),
        usage=update.usage if update.usage is not None else current.get("usage", ""),
    )

    if not updated_record:
        raise HTTPException(
            status_code=500, detail="Failed to update model configuration"
        )
    _invalidate_model_cache_snapshot(model_type)

    hf_home = resolve_hf_home()
    _refresh_optimizer_models()

    refresh_state = _start_targeted_model_refresh(
        hf_home=hf_home,
        model_type=model_type,
        refresh_mode="download_missing",
    )

    updated_expected_files = _parse_expected_files(updated_record.get("expected_files"))
    updated_allow_patterns = _parse_expected_files(updated_record.get("allow_patterns"))

    return ModelInfo(
        model_type=model_type,
        model_name=updated_record["model_name"],
        component=updated_record.get("component", ""),
        library_type=updated_record.get("library_type", ""),
        usage=updated_record.get("usage", ""),
        min_size_bytes=updated_record.get("min_size_bytes", 0),
        expected_files=updated_expected_files,
        revision=(updated_record.get("revision") or "").strip() or None,
        allow_patterns=updated_allow_patterns,
        size_bytes=0,
        size_formatted="0 B",
        download_date=None,
        cached_ok=False,
        cached_reason="refresh_pending",
        loaded_ok=None,
        loaded_reason="not_warmed",
        last_refresh=_get_model_cache_snapshot().get("generated_at"),
        path=None,
        refresh_state=refresh_state.get("state"),
        refresh_mode=refresh_state.get("mode"),
        refresh_target_models=list(refresh_state.get("target_models") or []),
    )
