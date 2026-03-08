import hashlib
import json
import logging
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from auth import get_current_customer, require_admin, track_usage
from database import (
    Customer,
    aggregate_history_stats,
    bulk_create_canonical_mappings,
    create_batch_job,
    create_canonical_mapping,
    create_customer,
    customer_scope,
    delete_canonical_mappings,
    get_admin_setting,
    get_canonical_mappings_cache,
    get_customer_by_id,
    get_llm_profiles,
    get_llm_optimization_job,
    get_llm_system_context,
    get_optimization_history_record,
    get_subscription_plan_by_id,
    get_usage,
    init_db,
    is_telemetry_enabled,
    list_batch_jobs,
    list_canonical_mappings,
    list_recent_history,
    list_recent_telemetry,
    reap_stale_llm_optimization_jobs,
    record_optimization_history,
    set_admin_setting,
    set_llm_profiles,
    set_llm_system_context,
    create_llm_optimization_job,
    update_llm_optimization_job,
    update_batch_job,
    update_canonical_mapping,
)

try:
    from database import get_canonical_mappings_cache_version
except ImportError:

    def get_canonical_mappings_cache_version() -> int:
        return 0


try:
    from database import increment_canonical_mappings_cache_version
except ImportError:

    def increment_canonical_mappings_cache_version() -> int:
        return get_canonical_mappings_cache_version()


from dotenv import load_dotenv

try:
    import boto3
except Exception:  # pragma: no cover - boto3 may be absent in lightweight test envs
    boto3 = None  # type: ignore[assignment]

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware

try:
    from fastapi.responses import JSONResponse, ORJSONResponse
except ImportError:  # pragma: no cover - lightweight test stubs may omit JSONResponse
    from fastapi.responses import ORJSONResponse

    try:
        from starlette.responses import JSONResponse
    except Exception:

        class JSONResponse:  # type: ignore[no-redef]
            def __init__(
                self,
                content: Any,
                status_code: int = 200,
                headers: Optional[Dict[str, str]] = None,
                media_type: Optional[str] = None,
            ) -> None:
                self.content = content
                self.status_code = status_code
                self.headers = headers or {}
                self.media_type = media_type


from fastapi.staticfiles import StaticFiles
from models.canonical_mapping import (
    CanonicalMappingBulkCreate,
    CanonicalMappingBulkDelete,
    CanonicalMappingCreate,
    CanonicalMappingDeleteResponse,
    CanonicalMappingListResponse,
    CanonicalMappingResponse,
    CanonicalMappingUpdate,
)
from models.optimization import (
    LLMOptimizationJobResponse,
    LLMOptimizationSubmitResponse,
    OptimizationBatchResponse,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationStats,
)
from pydantic import BaseModel, ConfigDict, Field
from routers import (
    admin_routes,
    api_key_routes,
    auth_routes,
    billing_routes,
    mapping_routes,
    subscription_routes,
    usage_routes,
    webhook_routes,
)
from routers.admin_routes import (
    bump_model_cache_validation_version,
    get_model_cache_validation_version,
    refresh_model_cache_snapshot,
)
from services import logging_control
from services.billing import create_checkout_session, create_stripe_customer
from services.llm_proxy import LLMProviderError, LLMResult, call_llm, get_llm_providers
from services import tracing as tracing_service
from services.model_cache_manager import ensure_models_cached, resolve_hf_home
from services.optimizer import config as optimizer_config
from services.optimizer import optimizer
from services.optimizer.config_utils import sanitize_canonical_map
from services.optimizer.model_capabilities import (
    build_model_readiness,
    build_not_ready_warnings,
    build_not_used_warnings,
    model_lookup_from_status,
)
from services.optimizer.pipeline_config import resolve_optimization_config
from services.optimizer.protect import ProtectTagError
from services.optimizer.router import (
    ContentProfile,
    SmartContext,
    get_profile,
    get_profile_for_text,
    merge_disabled_passes,
    resolve_smart_context,
)
from services.quota_manager import quota_manager
from services.telemetry_control import set_enabled as set_telemetry_enabled
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response


class SignUpRequest(BaseModel):
    name: str
    email: str


class CheckoutRequest(BaseModel):
    price_id: str
    success_url: str
    cancel_url: str


class KeyResponse(BaseModel):
    api_key: str


ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = ROOT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")


def _json_response(
    *,
    content: Any,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
    media_type: Optional[str] = None,
) -> Response:
    try:
        return ORJSONResponse(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )
    except AssertionError:
        return JSONResponse(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )


try:
    import orjson as _orjson  # noqa: F401

    _DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:  # pragma: no cover
    _DEFAULT_RESPONSE_CLASS = JSONResponse


def _env_truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _format_bytes(value: int) -> str:
    """Format bytes to human-readable format."""
    if value < 1024:
        return f"{value} B"
    elif value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    elif value < 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MB"
    else:
        return f"{value / (1024 * 1024 * 1024):.1f} GB"


def _get_dir_size(path: str) -> int:
    """Calculate total size of directory recursively."""
    try:
        total = 0
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _get_dir_size(entry.path)
        return total
    except Exception:
        return 0


def _detect_hf_volume(include_size: bool = True) -> Dict[str, Any]:
    """
    Auto-detect HuggingFace cache status.

    Returns dict with cache directory information.
    """
    hf_home = resolve_hf_home().strip()

    result = {
        "path": hf_home,
        "exists": os.path.isdir(hf_home),
        "is_mounted": False,  # Kept for backward compatibility in response structure
        "writable": False,
        "size_bytes": 0,
    }

    if result["exists"]:
        # We no longer strictly check for mount points as models are baked in
        result["is_mounted"] = True
        result["writable"] = os.access(hf_home, os.W_OK)
        if include_size:
            result["size_bytes"] = _get_dir_size(hf_home)

    return result


def _model_cache_info_from_snapshot() -> Dict[str, Any]:
    """Read lightweight model cache status from persisted admin snapshot."""
    snapshot = get_admin_setting("model_cache_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}

    available = snapshot.get("available_models")
    missing = snapshot.get("missing_models")

    available_models = [m for m in (available or []) if isinstance(m, str)]
    missing_models = [m for m in (missing or []) if isinstance(m, str)]

    cached_count = 0
    stats = snapshot.get("stats")
    if isinstance(stats, dict):
        models = stats.get("models")
        if isinstance(models, dict):
            cached_count = sum(
                1
                for model_state in models.values()
                if isinstance(model_state, dict) and bool(model_state.get("cached_ok"))
            )

    if cached_count == 0 and available_models:
        cached_count = len(available_models)

    return {
        "cached_count": cached_count,
        "available": available_models,
        "missing": missing_models,
    }


def _detect_cached_models(hf_home: str) -> Dict[str, Any]:
    """
    Check which HuggingFace models are cached locally.

    Returns dict with cached, available, and missing models.
    """
    from services.model_cache_manager import (
        ModelCacheValidator,
        _resolve_model_repo,
        get_model_configs,
    )

    # Fetch required models dynamically from the database/inventory
    db_configs = get_model_configs()
    required_models = {
        model_type: config["model_name"] for model_type, config in db_configs.items()
    }

    result = {
        "cached": set(),
        "available": [],
        "missing": list(required_models.keys()),
        "details": {},
    }

    validator = ModelCacheValidator(hf_home)
    validator.configs = db_configs

    if not os.path.isdir(hf_home):
        return result

    # Check HuggingFace hub cache for informational purposes
    hub_cache = os.path.join(hf_home, "hub")
    if os.path.isdir(hub_cache):
        try:
            for model_dir in os.listdir(hub_cache):
                model_path = os.path.join(hub_cache, model_dir)
                if os.path.isdir(model_path):
                    result["cached"].add(model_dir)
                    result["details"][model_dir] = {
                        "path": model_path,
                        "size_bytes": _get_dir_size(model_path),
                    }
        except Exception:
            pass

    # Validate each required model using the shared validator logic
    for model_type, model_name in required_models.items():
        model_repo = _resolve_model_repo(model_type, model_name)
        model_path = validator._find_model_path(model_repo)
        if model_path:
            cached_dir = os.path.basename(model_path)
            result["cached"].add(cached_dir)
            result["details"][cached_dir] = {
                "path": model_path,
                "size_bytes": _get_dir_size(model_path),
            }

        if validator.model_exists(model_type):
            if model_type not in result["available"]:
                result["available"].append(model_type)
            if model_type in result["missing"]:
                result["missing"].remove(model_type)

    return result


tags_metadata = [
    {"name": "Health", "description": "System health and dependency status."},
    {
        "name": "Optimizer",
        "description": "Endpoints for prompt optimization and related configuration.",
    },
    {
        "name": "Canonical Mappings",
        "description": "Manage canonicalization mappings used during optimization.",
    },
    {
        "name": "LLM",
        "description": "Utilities for testing provider connectivity and listing models.",
    },
]


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    startup_event()
    try:
        yield
    finally:
        _llm_async_worker_stop.set()


app = FastAPI(
    title="Tokemizer API",
    version="1.0.0",
    description=(
        "Enterprise-grade prompt optimization: deterministic compression with semantic safeguards, "
        "canonicalization, entropy pruning, and section ranking."
    ),
    openapi_tags=tags_metadata,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    default_response_class=_DEFAULT_RESPONSE_CLASS,
    lifespan=_app_lifespan,
)

app.include_router(auth_routes.router)
app.include_router(admin_routes.router)
app.include_router(api_key_routes.router)
app.include_router(billing_routes.router)
app.include_router(subscription_routes.router)
app.include_router(webhook_routes.router)
app.include_router(mapping_routes.router)
app.include_router(usage_routes.router)

api_router = APIRouter(prefix="/api/v1")


# HTTP headers must be latin-1 encodable. Strip any characters that cannot be encoded
# (e.g., emoji prefixes used for visual emphasis in logs) before adding warnings to
# custom response headers.
def _sanitize_header_value(value: str) -> str:
    """Return a latin-1 safe representation of *value* suitable for HTTP headers."""

    sanitized = value.encode("latin-1", "ignore").decode("latin-1")
    if not sanitized:
        sanitized = value.encode("ascii", "ignore").decode("ascii", "ignore")

    # Strip leading/trailing whitespace after encoding cleanup
    sanitized = sanitized.strip()

    return sanitized


_HEALTH_CACHE_TTL_SECONDS = 30
_health_cache: Optional[Tuple[float, Dict[str, Any]]] = None
_health_cache_lock = threading.Lock()


def _build_health_payload() -> Dict[str, Any]:
    from services.optimizer.core import (
        TIKTOKEN_AVAILABLE,
        MinHash,
        np,
        optimizer,
        spacy,
    )

    model_status = optimizer.model_status()

    # Keep health endpoint lightweight: avoid deep filesystem scans and
    # per-request model validation.
    volume_info = _detect_hf_volume(include_size=False)
    model_cache_info = _model_cache_info_from_snapshot()

    # Check dependency status
    dependencies = {
        "tiktoken": TIKTOKEN_AVAILABLE and optimizer.tokenizer is not None,
        "numpy": np is not None,
        "spacy": spacy is not None,
        "datasketch": MinHash is not None,
    }

    # Add tiktoken cache source path for debugging and ops visibility
    tiktoken_cache_info = {
        "enabled": TIKTOKEN_AVAILABLE and optimizer.tokenizer is not None,
        "cache_dir": os.environ.get("TIKTOKEN_CACHE_DIR", "default (~/.tiktoken)"),
    }

    # Model readiness (from warm_up)
    nlp_loaded = bool(model_status.get("spacy", {}).get("loaded"))
    model_lookup = model_lookup_from_status(model_status)
    model_readiness = build_model_readiness(model_lookup)
    coref_loaded = bool(model_lookup.get("coreference"))
    semantic_loaded = bool(model_lookup.get("semantic_guard"))
    entropy_teacher_loaded = bool(model_lookup.get("entropy"))
    entropy_fast_loaded = bool(model_lookup.get("entropy_fast"))
    token_classifier_loaded = bool(model_lookup.get("token_classifier"))

    # Categorize techniques by availability
    always_enabled = {
        "Content Preservation": True,
        "Whitespace Compression": True,
        "Politeness Removal": True,
        "Instruction Simplification": True,
        "Redundancy Removal": True,
        "Number & Unit Normalization": True,
        "Entity Canonicalization": True,
        "Synonym Shortening": True,
        "Filler Word Removal": True,
        "Repeated Fragment Compression": True,
        "Background Outlining": True,
        "Punctuation Compression": True,
        "Format Simplification": True,
    }

    dependency_gated = {
        "Coreference Compression": coref_loaded,
        "Content Deduplication": nlp_loaded,
        "Semantic Safeguard": semantic_loaded,
    }

    maximum_level_only = {
        "Example Compression": True,
        "History Summarization": np is not None,
        "Entropy Pruning": entropy_fast_loaded,
        "Token Classifier Fast Path": token_classifier_loaded,
        "TOON Conversion": True,
    }

    conditional_on_flags = {
        "Adaptive Abbreviation Learning": True,
    }

    # Count techniques
    base_enabled = (
        sum(always_enabled.values())
        + sum(1 for v in dependency_gated.values() if v)
        + sum(conditional_on_flags.values())
    )
    base_total = len(always_enabled) + len(dependency_gated) + len(conditional_on_flags)

    maximum_available = sum(1 for v in maximum_level_only.values() if v)
    maximum_total = len(maximum_level_only)

    total_default_enabled = base_enabled
    total_default_total = base_total + maximum_total

    # Generate warnings based on actual state
    warnings = []

    if not volume_info["exists"]:
        warnings.append(
            "HF_HOME directory does not exist; models cached to ephemeral storage"
        )
    elif not volume_info["is_mounted"]:
        warnings.append(
            "HF_HOME is not mounted; models will not persist across restarts"
        )

    if not volume_info["writable"]:
        warnings.append("HF_HOME is not writable; model downloads will fail")

    for model_type in model_cache_info["missing"]:
        warnings.append(f"{model_type} model not cached; will download on first use")

    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies,
        "tiktoken_cache": tiktoken_cache_info,
        "models": {
            "spacy_nlp": nlp_loaded,
            "coreference": coref_loaded,
            "semantic_guard": semantic_loaded,
            "semantic_rank": bool(model_lookup.get("semantic_rank")),
            "entropy_model": entropy_teacher_loaded,
            "entropy_fast": entropy_fast_loaded,
            "token_classifier": token_classifier_loaded,
        },
        "model_capabilities": model_readiness,
        "volume": {
            "path": volume_info["path"],
            "exists": volume_info["exists"],
            "mounted": volume_info["is_mounted"],
            "writable": volume_info["writable"],
            "size_bytes": volume_info["size_bytes"],
        },
        "cache": {
            "total_models_cached": model_cache_info["cached_count"],
            "available": model_cache_info["available"],
            "missing": model_cache_info["missing"],
        },
        "techniques": {
            "total": {
                "enabled": total_default_enabled,
                "total": total_default_total,
                "percentage": (
                    round((total_default_enabled / total_default_total) * 100, 1)
                    if total_default_total > 0
                    else 0
                ),
                "note": "Total techniques enabled with default settings (optimization_mode='balanced')",
            },
            "base_techniques": {
                "enabled": base_enabled,
                "total": base_total,
                "percentage": round((base_enabled / base_total) * 100, 1),
                "details": {
                    **always_enabled,
                    **dependency_gated,
                    **conditional_on_flags,
                },
                "note": "Core techniques available at all optimization levels",
            },
            "maximum_level_techniques": {
                "available": maximum_available,
                "total": maximum_total,
                "percentage": (
                    round((maximum_available / maximum_total) * 100, 1)
                    if maximum_total > 0
                    else 0
                ),
                "details": maximum_level_only,
                "note": "Advanced techniques available only in optimization_mode='maximum'",
            },
        },
        "configuration": {
            "semantic_deduplication_enabled": optimizer.enable_semantic_deduplication,
            "lsh_deduplication_enabled": optimizer.enable_lsh_deduplication,
            "semantic_guard_enabled": optimizer.semantic_guard_enabled,
            "chunk_size": optimizer.chunk_size,
            "chunk_threshold": optimizer.chunk_threshold,
        },
        "warnings": (
            [
                f"Missing dependency: {dep}"
                for dep, status in dependencies.items()
                if not status
            ]
            + [
                (
                    "spaCy NLP model not loaded - semantic deduplication disabled"
                    if not nlp_loaded
                    else None
                ),
                (
                    "Coreference model not loaded - coreference compression disabled"
                    if not coref_loaded
                    else None
                ),
                (
                    "Semantic guard model not loaded - strict mode requests will fail"
                    if optimizer.semantic_guard_enabled and not semantic_loaded
                    else None
                ),
                (
                    "Entropy model not loaded - teacher quality guard unavailable for maximum mode"
                    if not entropy_teacher_loaded
                    else None
                ),
                (
                    "Entropy-fast model not loaded - strict mode requests will fail"
                    if not entropy_fast_loaded
                    else None
                ),
                (
                    "Token classifier model not loaded - strict mode maximum requests will fail"
                    if not token_classifier_loaded
                    else None
                ),
            ]
            if (
                not all(dependencies.values())
                or not nlp_loaded
                or not coref_loaded
                or (optimizer.semantic_guard_enabled and not semantic_loaded)
                or not entropy_fast_loaded
                or not token_classifier_loaded
            )
            else []
        )
        + warnings,
        "recommendations": [
            "Ensure all dependencies are installed: tiktoken, numpy, spacy, datasketch, sentence-transformers",
            (
                f"Total techniques enabled: {total_default_enabled}/{total_default_total} "
                f"({round((total_default_enabled/total_default_total)*100, 1) if total_default_total > 0 else 0}%)"
            ),
            "Use optimization_mode='conservative' or 'balanced' for faster processing with reduced compression",
            (
                f"Advanced techniques: {maximum_available}/{maximum_total} available "
                f"({round((maximum_available/maximum_total)*100, 1) if maximum_total > 0 else 0}%)"
            ),
        ],
    }


# Custom exception handler to ensure UTF-8 encoding for all error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handle HTTPException with proper UTF-8 encoding.

    Preserves custom headers (e.g., WWW-Authenticate, Retry-After) that may be set
    by endpoints raising exceptions with authentication or rate-limiting requirements.
    """
    return _json_response(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,  # Preserve custom headers from the exception
        media_type="application/json; charset=utf-8",
    )


@api_router.get("/")
async def root() -> dict:
    return {"message": "Tokemizer optimizer ready"}


@api_router.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Service status including enabled techniques and dependency availability",
)
async def health_check() -> dict:
    """
    Health check endpoint that reports system status and available optimization techniques.
    """
    global _health_cache
    now = time.monotonic()
    with _health_cache_lock:
        if _health_cache and (now - _health_cache[0]) < _HEALTH_CACHE_TTL_SECONDS:
            return _health_cache[1]

    payload = _build_health_payload()
    with _health_cache_lock:
        _health_cache = (now, payload)
    return payload


# Canonical Mappings CRUD Endpoints
@api_router.get(
    "/canonical-mappings",
    response_model=CanonicalMappingListResponse,
    tags=["Canonical Mappings"],
    summary="List canonical mappings",
)
async def get_canonical_mappings(
    offset: int = 0,
    limit: int = 100,
    admin: Customer = Depends(require_admin),
):
    """List all canonical mappings with pagination."""
    try:
        mappings, total = await run_in_threadpool(
            list_canonical_mappings, offset, limit
        )
        return CanonicalMappingListResponse(
            mappings=[
                CanonicalMappingResponse(
                    id=m.id,
                    source_token=m.source_token,
                    target_token=m.target_token,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
                for m in mappings
            ],
            total=total,
            offset=offset,
            limit=limit,
        )
    except Exception as e:
        logging.error(f"Failed to list canonical mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/canonical-mappings",
    response_model=CanonicalMappingResponse,
    status_code=201,
    tags=["Canonical Mappings"],
    summary="Create or update a canonical mapping",
)
async def create_or_update_canonical_mapping(
    mapping: CanonicalMappingCreate, admin: Customer = Depends(require_admin)
):
    """Create or update a single canonical mapping (upsert)."""
    try:
        result = await run_in_threadpool(
            create_canonical_mapping, mapping.source_token, mapping.target_token
        )
        # Invalidate caches to ensure new mappings are immediately reflected
        _invalidate_canonical_mappings_cache()
        return CanonicalMappingResponse(
            id=result.id,
            source_token=result.source_token,
            target_token=result.target_token,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )
    except Exception as e:
        logging.error(f"Failed to create canonical mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/canonical-mappings/bulk",
    response_model=List[CanonicalMappingResponse],
    status_code=201,
    tags=["Canonical Mappings"],
    summary="Bulk create or update canonical mappings",
)
async def bulk_create_or_update_canonical_mappings(
    request: CanonicalMappingBulkCreate,
    admin: Customer = Depends(require_admin),
):
    """Bulk create or update canonical mappings (upsert)."""
    try:
        mappings_data = [(m.source_token, m.target_token) for m in request.mappings]
        results = await run_in_threadpool(bulk_create_canonical_mappings, mappings_data)
        # Invalidate caches to ensure new mappings are immediately reflected
        _invalidate_canonical_mappings_cache()
        return [
            CanonicalMappingResponse(
                id=r.id,
                source_token=r.source_token,
                target_token=r.target_token,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in results
        ]
    except Exception as e:
        logging.error(f"Failed to bulk create canonical mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put(
    "/canonical-mappings/{mapping_id}",
    response_model=CanonicalMappingResponse,
    tags=["Canonical Mappings"],
    summary="Update a canonical mapping by ID",
)
async def update_mapping(
    mapping_id: int,
    mapping: CanonicalMappingUpdate,
    admin: Customer = Depends(require_admin),
):
    """Update a canonical mapping by ID."""
    try:
        result = await run_in_threadpool(
            update_canonical_mapping,
            mapping_id,
            mapping.source_token,
            mapping.target_token,
        )
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Mapping with id {mapping_id} not found"
            )
        # Invalidate caches to ensure updated mappings are immediately reflected
        _invalidate_canonical_mappings_cache()
        return CanonicalMappingResponse(
            id=result.id,
            source_token=result.source_token,
            target_token=result.target_token,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update canonical mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete(
    "/canonical-mappings",
    response_model=CanonicalMappingDeleteResponse,
    tags=["Canonical Mappings"],
    summary="Bulk delete canonical mappings",
)
async def delete_mappings(
    request: CanonicalMappingBulkDelete,
    admin: Customer = Depends(require_admin),
):
    """Bulk delete canonical mappings by IDs."""
    try:
        deleted_count = await run_in_threadpool(delete_canonical_mappings, request.ids)
        # Invalidate caches to ensure deleted mappings are immediately reflected
        _invalidate_canonical_mappings_cache()
        return CanonicalMappingDeleteResponse(deleted_count=deleted_count)
    except Exception as e:
        logging.error(f"Failed to delete canonical mappings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_cache_size() -> int:
    try:
        size = int(os.environ.get("OPTIMIZER_CACHE_SIZE", "1000"))
        return max(size, 0)
    except (TypeError, ValueError):
        return 1000


def _get_cache_ttl_seconds() -> Optional[int]:
    try:
        ttl = int(os.environ.get("OPTIMIZER_CACHE_TTL_SECONDS", "0"))
    except (TypeError, ValueError):
        ttl = 0
    return ttl if ttl > 0 else None


@dataclass
class OptimizationCacheEntry:
    payload: Dict[str, Any]
    expires_at: Optional[float]


class OptimizationCache:
    def __init__(self, max_size: int, ttl_seconds: Optional[int]):
        self._max_size = max(0, max_size)
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, OptimizationCacheEntry]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            if entry.expires_at and entry.expires_at <= time.time():
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return entry.payload

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        if self._max_size <= 0:
            return
        expires_at = (
            time.time() + self._ttl_seconds if self._ttl_seconds is not None else None
        )
        with self._lock:
            if key in self._cache:
                self._cache.pop(key, None)
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = OptimizationCacheEntry(
                payload=payload,
                expires_at=expires_at,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def set_max_size(self, max_size: int) -> None:
        normalized = max(0, max_size)
        with self._lock:
            self._max_size = normalized
            if self._max_size <= 0:
                self._cache.clear()
                return
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)


_CACHE_SIZE = _get_cache_size()
_CACHE_TTL_SECONDS = _get_cache_ttl_seconds()
_CACHE_TOKEN_LIMIT = 20000
_PROMPT_INTERN_MAX_BYTES = 256
_cache_lock = threading.Lock()
_optimization_cache = OptimizationCache(_CACHE_SIZE, _CACHE_TTL_SECONDS)
_CANONICAL_CACHE_VERSION = 0
_CANONICAL_CACHE_DB_VERSION = 0
_CANONICAL_CACHE_DB_VERSION_LAST_CHECK = 0.0
_CANONICAL_CACHE_DB_VERSION_TTL_SECONDS = 2.0

_MODEL_AVAILABILITY_CACHE_TTL_SECONDS = max(
    1, int(os.environ.get("MODEL_AVAILABILITY_CACHE_TTL_SECONDS", "60") or 60)
)
_model_availability_cache_lock = threading.Lock()
_model_availability_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
_model_availability_checks_total = 0
_model_availability_cache_hits = 0
_model_availability_cache_misses = 0
_model_availability_check_time_ms = 0.0


_tenant_active_optimizations: Dict[str, int] = {}
_tenant_active_optimizations_lock = threading.Lock()


def _refresh_canonical_cache_version() -> int:
    global _CANONICAL_CACHE_DB_VERSION, _CANONICAL_CACHE_DB_VERSION_LAST_CHECK
    now = time.monotonic()
    if (
        now - _CANONICAL_CACHE_DB_VERSION_LAST_CHECK
        < _CANONICAL_CACHE_DB_VERSION_TTL_SECONDS
    ):
        return _CANONICAL_CACHE_DB_VERSION
    _CANONICAL_CACHE_DB_VERSION_LAST_CHECK = now
    try:
        _CANONICAL_CACHE_DB_VERSION = get_canonical_mappings_cache_version()
    except Exception:
        return _CANONICAL_CACHE_DB_VERSION
    return _CANONICAL_CACHE_DB_VERSION


def _set_cache_size(cache_size: int) -> None:
    global _CACHE_SIZE
    normalized = max(0, cache_size)
    with _cache_lock:
        _CACHE_SIZE = normalized
        _optimization_cache.set_max_size(normalized)


def _detect_cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the active cgroup memory limit, if any."""

    # Check common cgroup v2 then v1 paths
    candidate_paths = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            raw_value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if not raw_value or raw_value == "max":
            continue

        try:
            limit = int(raw_value)
        except ValueError:
            continue

        # Ignore unbounded/sentinel values from cgroups
        # cgroup v2 uses "max" (handled above), while cgroup v1 often uses 9223372036854771712 (LLONG_MAX-4096)
        # Treat any extremely large value as "no limit" to force fallback to /proc/meminfo.
        if limit in (9223372036854771712, 9223372036854775807):
            continue
        if limit >= (
            1 << 60
        ):  # ~1 EiB threshold; far above any realistic container limit
            continue

        if limit > 0:
            return limit

    return None


def _detect_system_memory_bytes() -> Optional[int]:
    """Best-effort detection of available memory (cgroup limit or physical)."""

    cgroup_limit = _detect_cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        return cgroup_limit

    # /proc/meminfo is the most reliable on Linux containers
    meminfo_path = Path("/proc/meminfo")
    if sys.platform.startswith("linux") and meminfo_path.exists():
        try:
            with meminfo_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kib = int(parts[1])
                            return total_kib * 1024
        except (OSError, ValueError):
            pass

    # Fall back to POSIX sysconf if available
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * phys_pages
        except (OSError, ValueError):
            return None

    return None


def _format_bytes_to_gib(value: Optional[int]) -> str:
    if not value:
        return "unknown"
    gib_value = value / (1024**3)
    return f"{gib_value:.1f} GiB"


_TOTAL_SYSTEM_MEMORY_BYTES = _detect_system_memory_bytes()
_LONG_PROMPT_MIN_MEMORY_GB = 8.0
_LONG_PROMPT_TOKEN_THRESHOLD = 200000
_LONG_PROMPT_MEMORY_THRESHOLD_BYTES = int(_LONG_PROMPT_MIN_MEMORY_GB * (1024**3))
_MEMORY_GUARD_ACTIVE = (
    _LONG_PROMPT_TOKEN_THRESHOLD > 0
    and _TOTAL_SYSTEM_MEMORY_BYTES is not None
    and _TOTAL_SYSTEM_MEMORY_BYTES < _LONG_PROMPT_MEMORY_THRESHOLD_BYTES
)


def _normalize_segment_spans(
    spans: Optional[Sequence[Any]],
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[Tuple[Tuple[int, int, Any, Any], ...]],
]:
    if not spans:
        return None, None

    payload: List[Dict[str, Any]] = []
    key_items: List[Tuple[int, int, Any, Any]] = []

    for span in spans:
        if hasattr(span, "model_dump"):
            span = span.model_dump(exclude_none=True)
        if not isinstance(span, dict):
            continue

        payload.append(span)
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        label = span.get("label")
        weight = span.get("weight")
        weight_value = None
        if weight is not None:
            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                weight_value = None
        key_items.append((start, end, label, weight_value))

    if not payload:
        return None, None
    return payload, tuple(key_items) if key_items else None


def _enforce_memory_guard_for_prompts(prompts: Sequence[str]) -> None:
    """Prevent processing of ultra-long prompts when system memory is insufficient."""

    if not _MEMORY_GUARD_ACTIVE:
        return

    for prompt in prompts:
        if not prompt:
            continue

        if _LONG_PROMPT_TOKEN_THRESHOLD > 0:
            if prompt.isascii():
                if len(prompt) < _LONG_PROMPT_TOKEN_THRESHOLD:
                    continue
            else:
                if len(prompt.encode("utf-8")) < _LONG_PROMPT_TOKEN_THRESHOLD:
                    continue

        token_count = optimizer.count_tokens(prompt)
        if token_count >= _LONG_PROMPT_TOKEN_THRESHOLD:
            detected_label = _format_bytes_to_gib(_TOTAL_SYSTEM_MEMORY_BYTES)
            required_label = _format_bytes_to_gib(_LONG_PROMPT_MEMORY_THRESHOLD_BYTES)
            detail_msg = (
                f"Prompt contains ~{token_count:,} tokens but the host only reports {detected_label} RAM "
                f"(<{required_label} required for long prompts). "
                "Increase BACKEND_MEMORY_LIMIT/BACKEND_MEMORY_RESERVATION "
                "or reduce the prompt size."
            )
            raise HTTPException(
                status_code=400,
                detail=detail_msg,
            )


def _cache_guard(prompt: str) -> bool:
    if not _CACHE_SIZE:
        return True

    tokens = optimizer.count_tokens(prompt)
    if _CACHE_TOKEN_LIMIT and tokens > _CACHE_TOKEN_LIMIT:
        return True

    chunk_threshold = optimizer.chunk_threshold or optimizer.chunk_size or 0
    if chunk_threshold and tokens > chunk_threshold:
        return True

    return False


def _resolve_cache_flags(prompt: str) -> Tuple[str, ContentProfile, SmartContext]:
    content_type, profile = get_profile_for_text(prompt)
    smart_context = resolve_smart_context(prompt, profile)
    return content_type, profile, smart_context


def _build_cache_key(
    prompt: str,
    optimization_mode: str,
    segment_spans_key: Optional[Tuple[Tuple[int, int, Any, Any], ...]],
    query: Optional[str],
    preserve_digits: bool,
    customer_id: Optional[str],
    custom_canonicals_key: Optional[Tuple[Tuple[str, str], ...]] = None,
    force_disabled_passes: Optional[Sequence[str]] = None,
) -> str:
    global _CANONICAL_CACHE_VERSION

    persistent_version = _refresh_canonical_cache_version()
    if persistent_version != _CANONICAL_CACHE_VERSION:
        _CANONICAL_CACHE_VERSION = persistent_version
        if _CACHE_SIZE:
            _optimization_cache.clear()
        get_canonical_mappings_cache().clear()

    hasher = hashlib.blake2s(digest_size=16)
    hasher.update(prompt.encode("utf-8", "surrogatepass"))
    hasher.update(b"\0")
    hasher.update(optimization_mode.encode("utf-8", "surrogatepass"))
    hasher.update(b"\0")
    hasher.update(b"1" if preserve_digits else b"0")
    hasher.update(b"\0")
    hasher.update(str(persistent_version).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(get_canonical_mappings_cache().version()).encode("ascii"))
    hasher.update(b"\0")
    if customer_id:
        hasher.update(customer_id.encode("utf-8", "surrogatepass"))
    hasher.update(b"\0")

    if query:
        hasher.update(query.encode("utf-8", "surrogatepass"))
    hasher.update(b"\0")

    if custom_canonicals_key:
        for key, value in custom_canonicals_key:
            hasher.update(key.encode("utf-8", "surrogatepass"))
            hasher.update(b"\0")
            hasher.update(value.encode("utf-8", "surrogatepass"))
            hasher.update(b"\0")

    if force_disabled_passes:
        for pass_name in sorted(force_disabled_passes):
            hasher.update(pass_name.encode("utf-8", "surrogatepass"))
            hasher.update(b"\0")

    if segment_spans_key:
        for start, end, label, weight in segment_spans_key:
            hasher.update(struct.pack("<II", int(start), int(end)))
            hasher.update(b"\0")
            if label is not None:
                hasher.update(str(label).encode("utf-8", "surrogatepass"))
            hasher.update(b"\0")
            if weight is not None:
                hasher.update(repr(weight).encode("ascii", "backslashreplace"))
            hasher.update(b"\0")

    return f"b2s:{hasher.hexdigest()}"


def _sanitize_custom_canonicals(
    mapping: Optional[Dict[str, str]],
) -> Dict[str, str]:
    return sanitize_canonical_map(mapping)


def _canonical_map_key(
    mapping: Optional[Dict[str, str]],
) -> Optional[Tuple[Tuple[str, str], ...]]:
    if not mapping:
        return None
    return tuple(sorted(mapping.items()))


def _resolve_request_disabled_passes(
    request: OptimizationRequest,
) -> Optional[Tuple[str, ...]]:
    # Request-level pass toggles are intentionally not part of the public API.
    return None


def _optimize_direct(
    prompt: str,
    optimization_mode: str,
    segment_spans: Optional[List[Dict[str, Any]]],
    query: Optional[str],
    custom_canonicals: Optional[Dict[str, str]],
    force_disabled_passes: Optional[Sequence[str]],
    background_tasks: Optional[BackgroundTasks] = None,
    skip_db_write: bool = False,
    customer_id: Optional[str] = None,
    content_type: Optional[str] = None,
    content_profile: Optional[ContentProfile] = None,
    smart_context: Optional[SmartContext] = None,
):
    return optimizer.optimize(
        prompt,
        mode="basic",
        optimization_mode=optimization_mode,
        segment_spans=segment_spans,
        query=query,
        custom_canonicals=custom_canonicals,
        chat_metadata=None,
        background_tasks=background_tasks,
        skip_db_write=skip_db_write,
        customer_id=customer_id,
        force_disabled_passes=force_disabled_passes,
        content_type=content_type,
        content_profile=content_profile,
        smart_context=smart_context,
    )


def _optimize_with_cache(
    prompt: str,
    optimization_mode: str,
    segment_spans: Optional[List[Dict[str, Any]]],
    segment_spans_key: Optional[Tuple[Tuple[int, int, Any, Any], ...]],
    query: Optional[str],
    custom_canonicals: Optional[Dict[str, str]],
    force_disabled_passes: Optional[Sequence[str]],
    background_tasks: Optional[BackgroundTasks] = None,
    skip_db_write: bool = False,
    customer_id: Optional[str] = None,
):
    if skip_db_write or not _CACHE_SIZE:
        return _optimize_direct(
            prompt,
            optimization_mode,
            segment_spans,
            query,
            custom_canonicals,
            force_disabled_passes,
            background_tasks=background_tasks,
            skip_db_write=skip_db_write,
            customer_id=customer_id,
        )

    if prompt:
        prompt_bytes = prompt.encode("utf-8", "surrogatepass")
        if len(prompt_bytes) <= _PROMPT_INTERN_MAX_BYTES:
            # Intern only small prompts to avoid unbounded memory growth in long-lived services.
            prompt = sys.intern(prompt)

    normalized_query = (
        query.strip() if isinstance(query, str) and query.strip() else None
    )

    content_type, content_profile, smart_context = _resolve_cache_flags(prompt)
    preserve_digits = smart_context.preserve_digits
    cache_key = _build_cache_key(
        prompt,
        optimization_mode,
        segment_spans_key,
        normalized_query,
        preserve_digits=preserve_digits,
        customer_id=customer_id,
        custom_canonicals_key=_canonical_map_key(custom_canonicals),
        force_disabled_passes=force_disabled_passes,
    )
    cached = _optimization_cache.get(cache_key)
    if cached is not None:
        if not skip_db_write and customer_id:
            try:
                stats = (
                    cached.get("stats", {})
                    if isinstance(cached.get("stats"), dict)
                    else {}
                )
                try:
                    original_tokens = int(stats.get("original_tokens") or 0)
                except (TypeError, ValueError):
                    original_tokens = 0
                try:
                    optimized_tokens = int(stats.get("optimized_tokens") or 0)
                except (TypeError, ValueError):
                    optimized_tokens = 0
                prompt_rate = float(
                    getattr(optimizer, "prompt_cost_per_1k", 0.0) or 0.0
                )
                cost_before = (original_tokens / 1000.0) * prompt_rate
                cost_after = (optimized_tokens / 1000.0) * prompt_rate
                cost_saved = max(cost_before - cost_after, 0.0)
                record_optimization_history(
                    mode="basic",
                    raw_prompt=prompt,
                    optimized_prompt=cached.get("optimized_output") or "",
                    raw_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    processing_time_ms=0.0,
                    estimated_cost_before=cost_before,
                    estimated_cost_after=cost_after,
                    estimated_cost_saved=cost_saved,
                    customer_id=customer_id,
                    compression_percentage=stats.get("compression_percentage"),
                    semantic_similarity=stats.get("semantic_similarity"),
                    techniques_applied=cached.get("techniques_applied"),
                )
            except Exception:
                pass
        return cached

    result = _optimize_direct(
        prompt,
        optimization_mode,
        segment_spans,
        normalized_query,
        custom_canonicals,
        force_disabled_passes,
        background_tasks=None,
        skip_db_write=skip_db_write,
        customer_id=customer_id,
        content_type=content_type,
        content_profile=content_profile,
        smart_context=smart_context,
    )
    _optimization_cache.set(cache_key, result)
    return result


def _invalidate_canonical_mappings_cache(
    increment_db_version: bool = True,
) -> None:
    """
    Invalidate all caches related to canonical mappings.

    This function should be called whenever canonical mappings are created, updated, or deleted
    to ensure that optimization results reflect the latest mappings and prevent stale cache hits.

    Clears:
    1. Optimization LRU cache - Contains cached optimization results keyed by prompt and settings
    2. Database canonical mappings cache - Contains loaded mappings from the database
    """
    global _CANONICAL_CACHE_DB_VERSION, _CANONICAL_CACHE_DB_VERSION_LAST_CHECK
    global _CANONICAL_CACHE_VERSION

    try:
        if increment_db_version:
            new_version = increment_canonical_mappings_cache_version()
        else:
            new_version = get_canonical_mappings_cache_version()
        _CANONICAL_CACHE_DB_VERSION = new_version
        _CANONICAL_CACHE_DB_VERSION_LAST_CHECK = time.monotonic()
    except Exception:
        new_version = None

    # Clear optimization results cache (if caching is enabled)
    if _CACHE_SIZE:
        _optimization_cache.clear()
        if new_version is None:
            _CANONICAL_CACHE_VERSION += 1
        else:
            _CANONICAL_CACHE_VERSION = new_version
        logging.info("Cleared optimization cache due to canonical mappings change")

    # Clear database canonical mappings cache to force reload from DB
    get_canonical_mappings_cache().clear()
    logging.info("Cleared canonical mappings database cache")


@api_router.post(
    "/signup",
    tags=["Billing"],
    summary="Customer signup",
)
async def signup(request: SignUpRequest):
    """Register a new customer and create a Stripe identity."""
    try:
        # 1. Create Stripe Customer
        stripe_id = await run_in_threadpool(
            create_stripe_customer, request.email, request.name
        )
        # 2. Create local customer record
        customer = await run_in_threadpool(
            create_customer, request.name, request.email, stripe_customer_id=stripe_id
        )
        return {"customer_id": customer.id, "stripe_customer_id": stripe_id}
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to register customer")


@api_router.post(
    "/checkout",
    tags=["Billing"],
    summary="Start checkout session",
)
async def checkout(
    request: CheckoutRequest, customer: Customer = Security(get_current_customer)
):
    """Create a checkout session for the authenticated customer."""
    try:
        if not customer.stripe_customer_id:
            raise HTTPException(status_code=400, detail="Customer missing Stripe ID")

        checkout_url = await run_in_threadpool(
            create_checkout_session,
            customer.stripe_customer_id,
            request.success_url,
            request.cancel_url,
            request.price_id,
        )
        return {"checkout_url": checkout_url}
    except Exception as e:
        logger.error(f"Checkout error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@api_router.get(
    "/usage",
    tags=["Usage"],
    summary="Get current usage and quota",
)
async def get_current_usage(customer: Customer = Security(get_current_customer)):
    """Retrieve the current monthly usage and quota for the authenticated customer."""
    now = datetime.now(timezone.utc)
    period_start = now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

    usage_rec = await run_in_threadpool(get_usage, customer.id, period_start)
    _, remaining, total = quota_manager.check_quota(customer.id)

    plan = get_subscription_plan_by_id(
        customer.subscription_tier,
        include_inactive=True,
        include_non_public=True,
    )
    return {
        "customer_id": customer.id,
        "subscription_status": customer.subscription_status,
        "period_start": period_start,
        "calls_used": usage_rec.calls_used if usage_rec else 0,
        "quota_limit": total,
        "remaining": remaining,
        "quota_overage_bonus": int(customer.quota_overage_bonus or 0),
        "plan_limits": {
            "rate_limit_rpm": getattr(plan, "rate_limit_rpm", 1000),
            "concurrent_optimization_jobs": getattr(
                plan, "concurrent_optimization_jobs", 5
            ),
            "batch_size_limit": getattr(plan, "batch_size_limit", 1000),
            "optimization_history_retention_days": getattr(
                plan, "optimization_history_retention_days", 365
            ),
            "telemetry_retention_days": getattr(plan, "telemetry_retention_days", 365),
            "audit_log_retention_days": getattr(plan, "audit_log_retention_days", 365),
            "custom_canonical_mappings_limit": getattr(
                plan, "custom_canonical_mappings_limit", 1000
            ),
        },
    }


@api_router.post(
    "/optimize/async",
    response_model=LLMOptimizationSubmitResponse,
    status_code=202,
    tags=["Optimizer"],
    summary="Submit async LLM optimization job",
    description=(
        "Queue a single llm_based optimization request for asynchronous processing via SQS. "
        "Use GET /api/v1/optimize/jobs/{job_id} to poll for completion."
    ),
)
async def optimize_text_async(
    request: OptimizationRequest,
    customer: Customer = Depends(get_current_customer),
):
    if request.prompts:
        raise HTTPException(
            status_code=400,
            detail="Async LLM optimization supports a single prompt only",
        )
    if request.optimization_technique != "llm_based":
        raise HTTPException(
            status_code=400,
            detail="Async endpoint requires optimization_technique='llm_based'",
        )
    if not _llm_async_enabled():
        raise HTTPException(status_code=503, detail="Async LLM optimization is disabled")
    if not _get_llm_async_queue_url():
        raise HTTPException(
            status_code=503,
            detail="Async LLM queue is not configured",
        )

    payload = request.model_dump(mode="json")
    job = await run_in_threadpool(
        create_llm_optimization_job,
        customer_id=customer.id,
        request_payload=payload,
    )
    try:
        await run_in_threadpool(
            _enqueue_llm_optimization_job,
            job.id,
            tracing_service.inject_context_to_carrier(),
        )
    except Exception as exc:
        await run_in_threadpool(
            update_llm_optimization_job,
            job.id,
            customer_id=customer.id,
            status="failed",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to enqueue LLM optimization job",
        ) from exc

    track_usage(customer, count=1)
    return LLMOptimizationSubmitResponse(job_id=job.id, status="queued")


@api_router.get(
    "/optimize/jobs/{job_id}",
    response_model=LLMOptimizationJobResponse,
    tags=["Optimizer"],
    summary="Get async LLM optimization job status",
)
async def get_optimize_job_status(
    job_id: str,
    customer: Customer = Depends(get_current_customer),
):
    job = await run_in_threadpool(
        get_llm_optimization_job, job_id, customer_id=customer.id
    )
    if not job:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    result_model: Optional[OptimizationResponse] = None
    if isinstance(job.result_payload, dict):
        try:
            result_model = OptimizationResponse(**job.result_payload)
        except Exception:
            result_model = None

    return LLMOptimizationJobResponse(
        job_id=job.id,
        status=job.status,
        attempts=job.attempts,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        result=result_model,
        error_message=job.error_message,
    )


@api_router.post(
    "/optimize",
    response_model=OptimizationResponse | OptimizationBatchResponse,
    tags=["Optimizer"],
    summary="Optimize prompt(s)",
    description=(
        "Optimize a single prompt or a batch. Supply exactly one of 'prompt' or 'prompts'. "
        "Use 'optimization_mode' to control intensity. Optional 'segment_spans' or '<protect>' tags preserve "
        "critical spans. Provide 'query' only for RAG contexts. Smart selection automatically applies "
        "technical defaults based on content."
    ),
)
async def optimize_text(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    customer: Customer = Depends(get_current_customer),
):
    enforce_plan_limits = customer.role != "admin"
    slot_acquired = (
        _acquire_optimization_slot(customer.id) if enforce_plan_limits else False
    )
    try:
        if request.prompts:
            if enforce_plan_limits:
                plan_batch_limit = max(
                    1,
                    _customer_limit(customer.id, "batch_size_limit", 1000),
                )
                if len(request.prompts) > plan_batch_limit:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Batch size exceeds your plan limit ({plan_batch_limit})",
                    )
            _enforce_memory_guard_for_prompts(request.prompts)
            batch_results = await run_in_threadpool(
                _optimize_batch, request, background_tasks, customer.id
            )
            # Track usage for batch: count of prompts
            track_usage(customer, count=len(request.prompts))
            return _json_response(content=batch_results.model_dump())

        assert request.prompt is not None
        _enforce_memory_guard_for_prompts((request.prompt,))
        result = await run_in_threadpool(
            _optimize_single, request.prompt, request, background_tasks, customer.id
        )

        # Track usage for single prompt
        track_usage(customer, count=1)

        # Add warnings to response headers for browser visibility
        headers: Dict[str, str] = {}
        if result.warnings:
            # Sanitize warnings and filter out empty results
            sanitized_warnings = [
                sanitized
                for warning in result.warnings
                if (sanitized := _sanitize_header_value(warning))
            ]

            # Only add header if there are non-empty sanitized warnings
            if sanitized_warnings:
                headers["X-Tokemizer-Warnings"] = "; ".join(sanitized_warnings)

            # Also log to server console for visibility
            for warning in result.warnings:
                logger.warning(f"⚠️ {warning}")

        return _json_response(content=result.model_dump(), headers=headers)
    except ProtectTagError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        # Preserve client-facing HTTP status/message from guards and validators
        raise

    except Exception as error:  # pragma: no cover - defensive catch
        logger.exception("Optimization error: %s", error)
        raise HTTPException(
            status_code=500, detail="Internal server error during optimization"
        ) from error
    finally:
        _release_optimization_slot(customer.id, slot_acquired)


def _invalidate_model_availability_cache() -> None:
    with _model_availability_cache_lock:
        _model_availability_cache.clear()


def _validate_cached_model_availability(
    model_types: Sequence[str],
) -> Tuple[Dict[str, bool], Optional[str]]:
    global _model_availability_checks_total, _model_availability_cache_hits
    global _model_availability_cache_misses, _model_availability_check_time_ms

    key = tuple(sorted(model_types))
    now = time.time()
    current_version = get_model_cache_validation_version()

    with _model_availability_cache_lock:
        cached = _model_availability_cache.get(key)
        if cached:
            expires_at = float(cached.get("expires_at") or 0.0)
            cached_version = int(cached.get("version") or 0)
            if expires_at > now and cached_version == current_version:
                _model_availability_cache_hits += 1
                return dict(cached.get("availability") or {}), cached.get("warning")

    _model_availability_cache_misses += 1
    started = time.perf_counter()
    hf_home = resolve_hf_home()
    try:
        from services.model_cache_manager import ModelCacheValidator

        validator = ModelCacheValidator(hf_home)
        requested_models = list(model_types)
        availability = {
            model_type: bool(
                validator.validate_model_cache(
                    model_type, use_cache=False, generate_manifest=False
                ).get("cached_ok")
            )
            for model_type in requested_models
        }
        warning = None
    except Exception as exc:
        logger.warning("Model cache validation failed: %s", exc)
        availability = {model_type: False for model_type in model_types}
        warning = (
            "model cache validation failed - model readiness unknown; "
            "treating gated models as unavailable in strict mode."
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _model_availability_check_time_ms += elapsed_ms
    _model_availability_checks_total += 1

    with _model_availability_cache_lock:
        _model_availability_cache[key] = {
            "availability": dict(availability),
            "warning": warning,
            "expires_at": now + _MODEL_AVAILABILITY_CACHE_TTL_SECONDS,
            "version": current_version,
        }

    logger.debug(
        "Model availability validation completed in %.2fms (checks=%d hits=%d misses=%d)",
        elapsed_ms,
        _model_availability_checks_total,
        _model_availability_cache_hits,
        _model_availability_cache_misses,
    )

    return availability, warning


_STRICT_MODE_REQUIRED_MODELS: Dict[str, Tuple[str, ...]] = {
    "conservative": ("semantic_guard", "entropy_fast"),
    "balanced": (
        "semantic_guard",
        "spacy",
        "coreference",
        "semantic_rank",
        "entropy_fast",
    ),
    "maximum": (
        "semantic_guard",
        "spacy",
        "coreference",
        "semantic_rank",
        "entropy_fast",
        "token_classifier",
    ),
}


def _validate_optimization_mode(
    requested_mode: str,
    model_status: Dict[str, Any],
    cached_availability: Optional[Dict[str, bool]] = None,
) -> Tuple[str, bool]:
    """
    Validate requested mode against strict runtime model readiness.

    Returns:
        (actual_mode, strict_marker) where strict_marker is always False.
    """

    def _is_loaded(model_type: str) -> bool:
        if model_status:
            return bool(model_status.get(model_type, {}).get("loaded", False))
        if cached_availability is not None:
            # Unknown model status defaults to unavailable so gated features fail safe.
            return bool(cached_availability.get(model_type, False))
        return False

    required_models = _STRICT_MODE_REQUIRED_MODELS.get(requested_mode, ())
    missing = [
        model_type for model_type in required_models if not _is_loaded(model_type)
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Optimization mode '{requested_mode}' unavailable. Missing required "
                f"runtime models: {', '.join(sorted(missing))}. "
                "Admin must refresh/prewarm models before retrying."
            ),
        )

    return requested_mode, False


def _optimize_single(
    prompt: str,
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    customer_id: Optional[str] = None,
) -> OptimizationResponse:
    if request.optimization_technique == "llm_based":
        return _optimize_single_llm(prompt, request)

    from services.optimizer.core import TIKTOKEN_AVAILABLE, np

    if not TIKTOKEN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required dependency 'tiktoken' is unavailable. "
                "Strict mode requires exact tokenizer support."
            ),
        )
    if request.optimization_mode in {"balanced", "maximum"} and np is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required dependency 'numpy' is unavailable. "
                "Strict mode requires numpy-backed semantic components."
            ),
        )

    start_time = time.time()
    segment_spans_payload, segment_spans_key = _normalize_segment_spans(
        request.segment_spans
    )
    query = request.query
    custom_canonicals = _sanitize_custom_canonicals(request.custom_canonicals) or None
    force_disabled_passes = _resolve_request_disabled_passes(request)

    model_status = optimizer.model_status()
    if not model_status:
        try:
            for model_type in sorted(
                _STRICT_MODE_REQUIRED_MODELS.get(request.optimization_mode, ())
            ):
                optimizer.probe_model_readiness(model_type)
            model_status = optimizer.model_status()
        except Exception:
            logger.debug("Initial model readiness probe failed", exc_info=True)
    if not model_status:
        raise HTTPException(
            status_code=503,
            detail=(
                "Optimizer model readiness is unavailable. "
                "Run admin model refresh/prewarm and retry."
            ),
        )
    actual_mode, _ = _validate_optimization_mode(
        request.optimization_mode, model_status
    )

    bypass_cache = _cache_guard(prompt)

    with customer_scope(customer_id):
        if _CACHE_SIZE and not bypass_cache:
            raw_result = _optimize_with_cache(
                prompt,
                actual_mode,
                segment_spans_payload,
                segment_spans_key,
                query,
                custom_canonicals,
                force_disabled_passes,
                background_tasks=None,
                skip_db_write=False,
                customer_id=customer_id,
            )
        else:
            raw_result = optimizer.optimize(
                prompt,
                mode="basic",
                optimization_mode=actual_mode,
                segment_spans=segment_spans_payload,
                query=query,
                custom_canonicals=custom_canonicals,
                chat_metadata=None,
                background_tasks=None,
                skip_db_write=False,
                customer_id=customer_id,
                force_disabled_passes=force_disabled_passes,
            )

    stats_payload = raw_result.get("stats", {})
    stats = OptimizationStats(**stats_payload)

    if stats.processing_time_ms <= 0:
        stats.processing_time_ms = (time.time() - start_time) * 1000

    # Generate warnings for disabled techniques
    warnings = []
    if model_status:
        model_lookup = model_lookup_from_status(model_status)
    else:
        model_lookup = {}

    # Check if semantic guard caused a rollback
    techniques_applied = raw_result.get("techniques_applied", [])
    if "Semantic Guard Rollback" in techniques_applied:
        warnings.append(
            "Optimization was reverted due to low semantic similarity. "
            "The compressed version differed too much from the original."
        )

    if request.optimization_mode == "conservative":
        warnings.append(
            "Tip: Set optimization_mode='maximum' for up to 30% better compression (enables 3 more techniques)"
        )

    warnings.extend(
        build_not_ready_warnings(
            model_lookup,
            actual_mode,
            query_present=bool(query and query.strip()),
            profile_name=stats.content_profile,
            segment_spans_present=bool(request.segment_spans),
            disabled_passes=_resolve_disabled_passes_for_warning(
                optimization_mode=actual_mode,
                content_profile=stats.content_profile,
                force_disabled_passes=force_disabled_passes,
            ),
        )
    )
    warnings.extend(
        build_not_used_warnings(
            model_lookup,
            actual_mode,
            techniques_applied,
            query_present=bool(query and query.strip()),
            profile_name=stats.content_profile,
            segment_spans_present=bool(request.segment_spans),
            disabled_passes=_resolve_disabled_passes_for_warning(
                optimization_mode=actual_mode,
                content_profile=stats.content_profile,
                force_disabled_passes=force_disabled_passes,
            ),
            semantic_guard_enabled=bool(optimizer.semantic_guard_enabled),
            token_classifier_post_enabled=bool(
                optimizer_config.TOKEN_CLASSIFIER_POST_PASS_ENABLED
            ),
        )
    )
    if warnings:
        warnings = list(dict.fromkeys(warnings))

    router = None
    if stats.content_profile:
        router = {
            "content_type": stats.content_profile,
            "profile": stats.content_profile,
        }

    return OptimizationResponse(
        optimized_output=raw_result["optimized_output"],
        stats=stats,
        router=router,
        techniques_applied=raw_result.get("techniques_applied"),
        warnings=warnings if warnings else None,
    )


def _optimize_single_llm(
    prompt: str,
    request: OptimizationRequest,
) -> OptimizationResponse:
    started_at = time.perf_counter()

    provider = os.getenv("LLM_OPTIMIZER_PROVIDER", "ollama").strip() or "ollama"
    model = (
        os.getenv("LLM_OPTIMIZER_MODEL", "tokemizer-q4_k_m").strip()
        or "tokemizer-q4_k_m"
    )
    api_key = os.getenv("LLM_OPTIMIZER_API_KEY", "").strip()

    if provider == "ollama" and not api_key:
        api_key = os.getenv("LLM_OPTIMIZER_OLLAMA_BASE_URL", "").strip()

    max_input_chars = _env_int(
        "LLM_OPTIMIZER_MAX_INPUT_CHARS",
        120000,
        minimum=2000,
        maximum=1000000,
    )
    max_input_tokens = _env_int(
        "LLM_OPTIMIZER_MAX_INPUT_TOKENS",
        30000,
        minimum=1000,
        maximum=200000,
    )
    max_composed_tokens = _env_int(
        "LLM_OPTIMIZER_MAX_COMPOSED_TOKENS",
        45000,
        minimum=1500,
        maximum=240000,
    )

    prompt_char_len = len(prompt)
    if prompt_char_len > max_input_chars:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too large for LLM async optimization ({prompt_char_len} chars > "
                f"limit {max_input_chars})."
            ),
        )

    prompt_token_estimate = optimizer.count_tokens(prompt)
    if prompt_token_estimate > max_input_tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too large for LLM async optimization (~{prompt_token_estimate} tokens > "
                f"limit {max_input_tokens})."
            ),
        )

    user_prompt_parts: List[str] = [
        f"Optimization mode: {request.optimization_mode}",
        "Input:",
        prompt,
    ]
    if request.query:
        user_prompt_parts.insert(1, f"Query hint: {request.query}")
    user_prompt = "\n\n".join(part for part in user_prompt_parts if part)

    system_context = get_llm_system_context()
    composed_prompt = f"System Context:\n{system_context}\n\nUser Prompt:\n{user_prompt}"
    composed_token_estimate = optimizer.count_tokens(composed_prompt)
    if composed_token_estimate > max_composed_tokens:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Composed LLM request is too large (~{composed_token_estimate} tokens > "
                f"limit {max_composed_tokens})."
            ),
        )

    with tracing_service.start_span(
        "llm.optimize.single",
        kind=tracing_service.SpanKind.INTERNAL,
        attributes={
            "llm.provider": provider,
            "llm.model": model,
            "prompt.input_chars": len(prompt),
            "prompt.input_tokens_estimate": prompt_token_estimate,
            "prompt.composed_tokens_estimate": composed_token_estimate,
        },
    ):
        try:
            with tracing_service.start_span(
                "llm.call",
                kind=tracing_service.SpanKind.CLIENT,
                attributes={
                    "llm.provider": provider,
                    "llm.model": model,
                },
            ):
                llm_result = call_llm(provider, model, composed_prompt, api_key)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LLMProviderError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    optimized_output = (llm_result.text or "").strip()
    if not optimized_output:
        raise HTTPException(status_code=502, detail="LLM optimizer returned empty output")

    processing_time_ms = max(
        llm_result.duration_ms,
        (time.perf_counter() - started_at) * 1000.0,
    )
    original_chars = len(prompt)
    optimized_chars = len(optimized_output)
    original_tokens = max(1, (original_chars + 3) // 4)
    optimized_tokens = max(1, (optimized_chars + 3) // 4)
    token_savings = max(0, original_tokens - optimized_tokens)
    compression_percentage = (
        (token_savings / original_tokens) * 100.0 if original_tokens else 0.0
    )

    return OptimizationResponse(
        optimized_output=optimized_output,
        stats=OptimizationStats(
            original_chars=original_chars,
            optimized_chars=optimized_chars,
            compression_percentage=compression_percentage,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            token_savings=token_savings,
            processing_time_ms=processing_time_ms,
            fast_path=False,
            content_profile="llm_based",
            smart_context_description=f"LLM-based optimization via {provider}:{model}",
            semantic_similarity=None,
            semantic_similarity_source=None,
            deduplication=None,
        ),
        router={"content_type": "llm_based", "profile": "llm_based"},
        techniques_applied=[
            "LLM Based Optimization",
            f"Provider: {provider}",
            f"Model: {model}",
        ],
        warnings=None,
    )


def _resolve_disabled_passes_for_warning(
    *,
    optimization_mode: str,
    content_profile: Optional[str],
    force_disabled_passes: Optional[Sequence[str]],
) -> List[str]:
    mode_config = resolve_optimization_config(optimization_mode)
    mode_disabled = set(mode_config.get("disabled_passes", []))

    if content_profile:
        profile = get_profile(content_profile)
    else:
        profile = get_profile("general_prose")

    resolved = merge_disabled_passes(mode_disabled, profile)
    if force_disabled_passes:
        resolved.update(force_disabled_passes)
    return sorted(resolved)


def _get_customer_plan(customer_id: str):
    customer_obj = get_customer_by_id(customer_id)
    if not customer_obj:
        return None
    return get_subscription_plan_by_id(
        customer_obj.subscription_tier,
        include_inactive=True,
        include_non_public=True,
    )


def _customer_limit(customer_id: Optional[str], field: str, default: int) -> int:
    if not customer_id:
        return default
    plan = _get_customer_plan(customer_id)
    if not plan:
        return default
    value = getattr(plan, field, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _acquire_optimization_slot(customer_id: Optional[str]) -> bool:
    if not customer_id:
        return False
    limit = max(1, _customer_limit(customer_id, "concurrent_optimization_jobs", 5))
    with _tenant_active_optimizations_lock:
        active = _tenant_active_optimizations.get(customer_id, 0)
        if active >= limit:
            raise HTTPException(
                status_code=429,
                detail="Concurrent optimization job limit reached for your plan",
            )
        _tenant_active_optimizations[customer_id] = active + 1
    return True


def _release_optimization_slot(customer_id: Optional[str], acquired: bool) -> None:
    if not customer_id or not acquired:
        return
    with _tenant_active_optimizations_lock:
        active = _tenant_active_optimizations.get(customer_id, 0)
        if active <= 1:
            _tenant_active_optimizations.pop(customer_id, None)
            return
        _tenant_active_optimizations[customer_id] = active - 1


def _optimize_batch(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    customer_id: Optional[str] = None,
) -> OptimizationBatchResponse:
    assert request.prompts is not None

    if not request.prompts:
        return OptimizationBatchResponse(
            batch_job_id="",
            results=[],
            summary={
                "total_items": 0,
                "avg_compression": 0.0,
                "total_processing_time_ms": 0.0,
                "throughput_prompts_per_second": 0.0,
            },
        )

    job = None
    if customer_id is not None:
        job = create_batch_job(
            request.name or f"Batch {datetime.now(timezone.utc).isoformat()}",
            total_items=len(request.prompts),
            customer_id=customer_id,
        )
    start_time = time.time()

    def _optimize_prompt(prompt: str) -> OptimizationResponse:
        return _optimize_single(prompt, request, background_tasks, customer_id)

    executor = _get_batch_executor()
    try:
        chunk_size = int(os.environ.get("BATCH_OPTIMIZE_CHUNK_SIZE", "64"))
    except (TypeError, ValueError):
        chunk_size = 64
    chunk_size = max(chunk_size, 1)

    prompt_to_indices: Dict[str, List[int]] = {}
    unique_prompts: List[str] = []
    for index, prompt in enumerate(request.prompts):
        indices = prompt_to_indices.get(prompt)
        if indices is None:
            prompt_to_indices[prompt] = [index]
            unique_prompts.append(prompt)
        else:
            indices.append(index)

    results: List[Optional[OptimizationResponse]] = [None] * len(request.prompts)

    total_compression = 0.0
    total_items = len(request.prompts)
    try:
        update_every = int(os.environ.get("BATCH_PROGRESS_UPDATE_EVERY", "10"))
    except (TypeError, ValueError):
        update_every = 10
    update_every = max(update_every, 1)

    processed_items = 0
    next_update_at = update_every
    for chunk_start in range(0, len(unique_prompts), chunk_size):
        chunk = unique_prompts[chunk_start : chunk_start + chunk_size]
        chunk_results = list(executor.map(_optimize_prompt, chunk))
        for prompt, response in zip(chunk, chunk_results):
            indices = prompt_to_indices.get(prompt, [])
            for idx in indices:
                results[idx] = response
            total_compression += response.stats.compression_percentage * len(indices)
            processed_items += len(indices)
            if processed_items >= total_items or processed_items >= next_update_at:
                if job is not None and customer_id is not None:
                    update_batch_job(
                        job.id,
                        customer_id=customer_id,
                        processed_items=processed_items,
                    )
                next_update_at = processed_items + update_every

    finalized_results: List[OptimizationResponse] = []
    for item in results:
        if item is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Batch optimization result missing for an item")
        finalized_results.append(item)

    processing_time = (time.time() - start_time) * 1000
    avg_compression = total_compression / total_items if total_items else 0.0
    throughput = (
        (total_items / (processing_time / 1000)) if processing_time > 0 else 0.0
    )

    if job is not None and customer_id is not None:
        update_batch_job(
            job.id,
            customer_id=customer_id,
            status="completed",
            total_savings_percentage=avg_compression,
            processing_time_ms=processing_time,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    return OptimizationBatchResponse(
        batch_job_id=job.id if job is not None else "",
        results=finalized_results,
        summary={
            "total_items": total_items,
            "avg_compression": round(avg_compression, 2),
            "total_processing_time_ms": round(processing_time, 2),
            "throughput_prompts_per_second": round(throughput, 2),
        },
    )


class LLMTestRequest(BaseModel):
    provider: str
    model: str
    prompt: str
    api_key: Optional[str] = None
    profile_name: Optional[str] = None


class LLMTestResponse(BaseModel):
    text: str
    duration_ms: float


class LLMModelOptionResponse(BaseModel):
    value: str
    label: str


class LLMProviderInfoResponse(BaseModel):
    key: str
    label: str
    models: Tuple[LLMModelOptionResponse, ...]


class LLMProviderListResponse(BaseModel):
    providers: List[LLMProviderInfoResponse]


class LLMProfile(BaseModel):
    name: str
    provider: str
    model: str
    api_key: Optional[str] = Field(default=None, exclude=True)
    has_api_key: Optional[bool] = None


class TelemetryPassResponse(BaseModel):
    optimization_id: str
    pass_name: str
    pass_order: int
    duration_ms: float
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    reduction_percent: float
    created_at: str


class SettingsResponse(BaseModel):
    semantic_guard_threshold: float
    semantic_guard_enabled: bool
    semantic_guard_model: Optional[str]
    guard_latency_ms: float
    guard_tokens_saved: float
    telemetry_baseline_window_days: int
    optimizer_cache_size: int
    telemetry_enabled: bool
    lsh_enabled: bool
    lsh_similarity_threshold: float
    llm_system_context: str
    llm_profiles: List[LLMProfile]


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_guard_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    semantic_guard_enabled: Optional[bool] = None
    semantic_guard_model: Optional[str] = None
    guard_latency_ms: Optional[float] = Field(None, ge=0.0)
    guard_tokens_saved: Optional[float] = Field(None, ge=0.0)
    telemetry_baseline_window_days: Optional[int] = Field(None, ge=1)
    optimizer_cache_size: Optional[int] = Field(None, ge=0)
    telemetry_enabled: Optional[bool] = None
    lsh_enabled: Optional[bool] = None
    lsh_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    llm_system_context: Optional[str] = None
    llm_profiles: Optional[List[LLMProfile]] = None


def _build_settings_response(customer_id: Optional[str] = None) -> SettingsResponse:
    profiles = get_llm_profiles(customer_id) if customer_id else []
    return SettingsResponse(
        semantic_guard_threshold=optimizer.semantic_guard_threshold,
        semantic_guard_enabled=optimizer.semantic_guard_enabled,
        semantic_guard_model=optimizer.semantic_guard_model,
        guard_latency_ms=optimizer_config.SEMANTIC_GUARD_LATENCY_GUARD_MS,
        guard_tokens_saved=optimizer_config.SEMANTIC_GUARD_TOKEN_SAVINGS_BASELINE,
        telemetry_baseline_window_days=optimizer_config.TELEMETRY_BASELINE_WINDOW_DAYS,
        optimizer_cache_size=_CACHE_SIZE,
        telemetry_enabled=is_telemetry_enabled(),
        lsh_enabled=getattr(optimizer, "enable_lsh_deduplication", False),
        lsh_similarity_threshold=getattr(optimizer, "lsh_similarity_threshold", 0.0),
        llm_system_context=get_llm_system_context(),
        llm_profiles=[
            LLMProfile(
                name=profile["name"],
                provider=profile["provider"],
                model=profile["model"],
                has_api_key=bool(profile.get("api_key")),
            )
            for profile in profiles
        ],
    )


def _apply_settings_update(
    request: SettingsUpdateRequest, customer_id: Optional[str] = None
) -> None:
    updated_fields = set(request.model_fields_set)

    if (
        "semantic_guard_threshold" in updated_fields
        and request.semantic_guard_threshold is not None
    ):
        optimizer.semantic_guard_threshold = max(
            0.0, min(request.semantic_guard_threshold, 1.0)
        )
    if (
        "semantic_guard_enabled" in updated_fields
        and request.semantic_guard_enabled is not None
    ):
        optimizer.semantic_guard_enabled = request.semantic_guard_enabled
    if "semantic_guard_model" in updated_fields:
        model_name = (request.semantic_guard_model or "").strip() or None
        optimizer.semantic_guard_model = model_name
        optimizer.semantic_chunk_model = model_name
    if "guard_latency_ms" in updated_fields and request.guard_latency_ms is not None:
        optimizer_config.SEMANTIC_GUARD_LATENCY_GUARD_MS = request.guard_latency_ms
    if (
        "guard_tokens_saved" in updated_fields
        and request.guard_tokens_saved is not None
    ):
        optimizer_config.SEMANTIC_GUARD_TOKEN_SAVINGS_BASELINE = (
            request.guard_tokens_saved
        )
    if (
        "telemetry_baseline_window_days" in updated_fields
        and request.telemetry_baseline_window_days is not None
    ):
        optimizer_config.TELEMETRY_BASELINE_WINDOW_DAYS = (
            request.telemetry_baseline_window_days
        )
    if (
        "optimizer_cache_size" in updated_fields
        and request.optimizer_cache_size is not None
    ):
        _set_cache_size(request.optimizer_cache_size)
    if "telemetry_enabled" in updated_fields and request.telemetry_enabled is not None:
        set_admin_setting("telemetry_enabled", request.telemetry_enabled)
        set_telemetry_enabled(request.telemetry_enabled)
    if "lsh_enabled" in updated_fields and request.lsh_enabled is not None:
        optimizer.enable_lsh_deduplication = request.lsh_enabled
    if (
        "lsh_similarity_threshold" in updated_fields
        and request.lsh_similarity_threshold is not None
    ):
        optimizer.lsh_similarity_threshold = max(
            0.0, min(request.lsh_similarity_threshold, 1.0)
        )
    if "llm_system_context" in updated_fields and request.llm_system_context is not None:
        set_llm_system_context(request.llm_system_context)
    if (
        "llm_profiles" in updated_fields
        and request.llm_profiles is not None
        and customer_id
    ):
        existing_by_name = {
            profile.get("name", "").strip(): profile
            for profile in get_llm_profiles(customer_id)
        }
        merged = []
        for profile in request.llm_profiles:
            name = profile.name.strip()
            provider = profile.provider.strip()
            model = profile.model.strip()
            api_key = (profile.api_key or "").strip()
            if api_key:
                merged.append(
                    {
                        "name": name,
                        "provider": provider,
                        "model": model,
                        "api_key": api_key,
                    }
                )
                continue

            existing = existing_by_name.get(name)
            if not existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"API key required for new profile '{name}'",
                )

            merged.append(
                {
                    "name": name,
                    "provider": provider,
                    "model": model,
                    "api_key": existing.get("api_key", ""),
                }
            )

        set_llm_profiles(customer_id, merged)

    cache_sensitive_fields = {
        "semantic_guard_threshold",
        "semantic_guard_enabled",
        "semantic_guard_model",
        "lsh_enabled",
        "lsh_similarity_threshold",
    }
    if updated_fields & cache_sensitive_fields:
        _optimization_cache.clear()


@api_router.post(
    "/llm/test",
    response_model=LLMTestResponse,
    tags=["LLM"],
    summary="Test upstream LLM provider",
    description="Proxy a single prompt to an upstream provider for connectivity testing",
)
async def proxy_llm_test(
    request: LLMTestRequest, customer: Customer = Depends(get_current_customer)
) -> LLMTestResponse:
    api_key = (request.api_key or "").strip()
    if not api_key and request.profile_name:
        for profile in get_llm_profiles(customer.id):
            if profile.get("name") == request.profile_name:
                api_key = str(profile.get("api_key", "")).strip()
                break

    if not api_key and request.provider != "ollama":
        raise HTTPException(status_code=400, detail="API key required")

    try:
        result: LLMResult = await run_in_threadpool(
            call_llm, request.provider, request.model, request.prompt, api_key
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive catch
        logger.exception("Provider proxy error: %s", error)
        raise HTTPException(
            status_code=500, detail="Failed to reach upstream provider"
        ) from error

    return LLMTestResponse(text=result.text, duration_ms=result.duration_ms)


@api_router.get(
    "/llm/providers",
    response_model=LLMProviderListResponse,
    tags=["LLM"],
    summary="List supported providers and models",
)
async def list_llm_providers() -> LLMProviderListResponse:
    providers_payload = [
        LLMProviderInfoResponse(
            key=provider["key"],
            label=provider["label"],
            models=tuple(
                LLMModelOptionResponse(**model)
                for model in provider.get("models", [])
                if isinstance(model, dict)
            ),
        )
        for provider in get_llm_providers()
    ]

    return LLMProviderListResponse(providers=providers_payload)


@api_router.get(
    "/history",
    tags=["Optimizer"],
    summary="Recent optimizations",
)
async def list_history(
    limit: int = 50, customer: Customer = Security(get_current_customer)
) -> dict:
    safe_limit = max(1, min(limit, 500))
    retention_days = max(
        1, _customer_limit(customer.id, "optimization_history_retention_days", 365)
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    history = []
    for record in list_recent_history(safe_limit, customer_id=customer.id):
        try:
            created_at = datetime.fromisoformat(
                str(record.created_at).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if created_at >= cutoff:
            history.append(record)
    payload = []
    for rec in history:
        tokens_saved = max(rec.raw_tokens - rec.optimized_tokens, 0)
        compression = rec.compression_percentage or (
            (tokens_saved / rec.raw_tokens * 100) if rec.raw_tokens > 0 else 0.0
        )
        payload.append(
            {
                "id": rec.id,
                "original_tokens": rec.raw_tokens,
                "optimized_tokens": rec.optimized_tokens,
                "tokens_saved": tokens_saved,
                "compression_percentage": round(compression, 2),
                "semantic_similarity": rec.semantic_similarity,
                "mode": rec.mode,
                "created_at": rec.created_at,
                "techniques_applied": rec.techniques_applied,
            }
        )
    return {"optimizations": payload}


@api_router.get(
    "/history/{optimization_id}",
    tags=["Optimizer"],
    summary="Optimization detail",
)
async def get_history_detail(
    optimization_id: str, customer: Customer = Security(get_current_customer)
) -> dict:
    rec = get_optimization_history_record(optimization_id, customer_id=customer.id)
    if not rec:
        raise HTTPException(status_code=404, detail="Optimization not found")
    retention_days = max(
        1, _customer_limit(customer.id, "optimization_history_retention_days", 365)
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    created_at = datetime.fromisoformat(str(rec.created_at).replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if created_at < cutoff:
        raise HTTPException(status_code=404, detail="Optimization not found")

    tokens_saved = max(rec.raw_tokens - rec.optimized_tokens, 0)
    compression = rec.compression_percentage or (
        (tokens_saved / rec.raw_tokens * 100) if rec.raw_tokens > 0 else 0.0
    )
    return {
        "id": rec.id,
        "mode": rec.mode,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
        "raw_prompt": rec.raw_prompt,
        "optimized_prompt": rec.optimized_prompt,
        "original_tokens": rec.raw_tokens,
        "optimized_tokens": rec.optimized_tokens,
        "tokens_saved": tokens_saved,
        "compression_percentage": round(compression, 2),
        "semantic_similarity": rec.semantic_similarity,
        "processing_time_ms": rec.processing_time_ms,
        "estimated_cost_before": rec.estimated_cost_before,
        "estimated_cost_after": rec.estimated_cost_after,
        "estimated_cost_saved": rec.estimated_cost_saved,
        "techniques_applied": rec.techniques_applied,
    }


@api_router.get(
    "/stats",
    tags=["Optimizer"],
    summary="Aggregate stats",
)
async def aggregate_stats(
    customer: Customer = Security(get_current_customer),
) -> dict:
    stats = aggregate_history_stats(customer_id=customer.id)
    input_cost_savings = (stats["tokens_saved"] / 1_000_000) * 0.015
    history = list_recent_history(limit=1000, customer_id=customer.id)
    by_date: Dict[str, Dict[str, int]] = {}
    for rec in history:
        created_at = getattr(rec, "created_at", None)
        if not created_at:
            continue
        try:
            parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except Exception:
            continue
        day_key = parsed.astimezone(timezone.utc).date().isoformat()
        bucket = by_date.get(day_key)
        if bucket is None:
            bucket = {"original": 0, "optimized": 0}
            by_date[day_key] = bucket
        bucket["original"] += int(getattr(rec, "raw_tokens", 0) or 0)
        bucket["optimized"] += int(getattr(rec, "optimized_tokens", 0) or 0)

    today = datetime.now(timezone.utc).date()
    trend = []
    for delta in range(6, -1, -1):
        day = today - timedelta(days=delta)
        day_key = day.isoformat()
        bucket = by_date.get(day_key) or {"original": 0, "optimized": 0}
        trend.append(
            {
                "name": day.strftime("%a"),
                "original": bucket["original"],
                "optimized": bucket["optimized"],
            }
        )

    return {
        "tokens_saved": stats["tokens_saved"],
        "cost_savings": round(input_cost_savings, 2),
        "avg_compression_percentage": stats["avg_compression_percentage"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "avg_quality_score": stats["avg_quality_score"],
        "total_optimizations": stats["total_optimizations"],
        "estimated_monthly_savings": stats["estimated_monthly_savings"],
        "trend": trend,
    }


@api_router.get(
    "/batch-jobs",
    tags=["Optimizer"],
    summary="Batch jobs",
)
async def list_batch_jobs_endpoint(
    limit: int = 20, customer: Customer = Security(get_current_customer)
) -> dict:
    safe_limit = max(1, min(limit, 100))
    jobs = list_batch_jobs(safe_limit, customer_id=customer.id)

    payload = []
    for job in jobs:
        progress = (
            int((job.processed_items / job.total_items) * 100)
            if job.total_items > 0
            else 0
        )
        payload.append(
            {
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "progress_percentage": progress,
                "savings_percentage": job.total_savings_percentage,
                "processing_time_ms": job.processing_time_ms,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
            }
        )

    return {"batch_jobs": payload}


@api_router.get(
    "/telemetry/recent",
    response_model=List[TelemetryPassResponse],
    tags=["Optimizer"],
    summary="Recent per-pass telemetry",
)
async def recent_telemetry(
    limit: int = 100, customer: Customer = Security(get_current_customer)
) -> List[TelemetryPassResponse]:
    safe_limit = max(1, min(limit, 500))
    retention_days = max(
        1, _customer_limit(customer.id, "telemetry_retention_days", 365)
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    rows = []
    for row in list_recent_telemetry(safe_limit, customer_id=customer.id):
        created_raw = str(row.get("created_at") or "")
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if created_at >= cutoff:
            rows.append(row)
    return [TelemetryPassResponse(**row) for row in rows]


@api_router.get(
    "/settings",
    response_model=SettingsResponse,
    tags=["Optimizer"],
    summary="Runtime configuration defaults",
)
async def get_settings(
    customer: Customer = Security(get_current_customer),
) -> SettingsResponse:
    return _build_settings_response(customer.id)


@api_router.patch(
    "/settings",
    response_model=SettingsResponse,
    tags=["Optimizer"],
    summary="Update runtime configuration defaults",
)
async def update_settings(
    request: SettingsUpdateRequest, customer: Customer = Security(get_current_customer)
) -> SettingsResponse:
    _apply_settings_update(request, customer.id)
    return _build_settings_response(customer.id)


# Include the router in the main app
app.include_router(api_router)

# Serve built frontend if available
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend",
    )


# Middleware to ensure UTF-8 encoding for all responses
@app.middleware("http")
async def add_utf8_charset(request: Request, call_next):
    """
    Ensure all responses use UTF-8 charset.

    Preserves any additional Content-Type parameters (e.g., profile, boundary)
    while ensuring charset=utf-8 is set or updated for JSON responses.
    """
    extracted_ctx = tracing_service.extract_context_from_headers(request.headers)
    token = tracing_service.attach_context(extracted_ctx)

    try:
        with tracing_service.start_span(
            "http.request",
            kind=tracing_service.SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.target": request.url.path,
            },
        ):
            response = await call_next(request)
    finally:
        tracing_service.detach_context(token)
    content_type = response.headers.get("content-type", "")

    # Only modify JSON responses
    if content_type.startswith("application/json"):
        # Parse the content-type header to preserve other parameters
        parts = [part.strip() for part in content_type.split(";")]
        media_type = parts[0]
        params = {}

        # Extract existing parameters
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.strip()] = value.strip()

        # Set or update charset to utf-8
        params["charset"] = "utf-8"

        # Reconstruct the content-type header
        param_str = "; ".join(f"{k}={v}" for k, v in params.items())
        response.headers["content-type"] = f"{media_type}; {param_str}"

    return response


app.add_middleware(
    GZipMiddleware,
    minimum_size=500,
)

# Determine CORS origins
try:
    cors_setting = get_admin_setting("cors_origins", None)
    if cors_setting:
        origins = [o.strip() for o in str(cors_setting).split(",")]
    else:
        origins = os.environ.get("CORS_ORIGINS", "*").split(",")
except Exception:
    origins = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global executor pool for batch optimization (reused across requests)
_batch_executor: Optional[ThreadPoolExecutor] = None
_batch_executor_lock = threading.Lock()
_llm_async_worker_thread: Optional[threading.Thread] = None
_llm_async_worker_lock = threading.Lock()
_llm_async_worker_stop = threading.Event()
_ollama_keepalive_thread: Optional[threading.Thread] = None
_ollama_keepalive_lock = threading.Lock()


def _llm_async_enabled() -> bool:
    return _env_truthy("LLM_OPTIMIZE_ASYNC_ENABLED", "true")


def _get_llm_async_queue_url() -> str:
    return os.environ.get("LLM_OPTIMIZE_SQS_QUEUE_URL", "").strip()


def _get_sqs_client():
    if boto3 is None:
        raise RuntimeError("boto3 is unavailable for SQS operations")
    region = (
        os.environ.get("LLM_OPTIMIZE_SQS_REGION", "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or "us-east-1"
    )
    return boto3.client("sqs", region_name=region)


def _message_attributes_from_trace_context(trace_context: Optional[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    if not trace_context:
        return {}
    attrs: Dict[str, Dict[str, str]] = {}
    traceparent = trace_context.get("traceparent", "").strip()
    tracestate = trace_context.get("tracestate", "").strip()
    if traceparent:
        attrs["traceparent"] = {"StringValue": traceparent, "DataType": "String"}
    if tracestate:
        attrs["tracestate"] = {"StringValue": tracestate, "DataType": "String"}
    return attrs


def _trace_context_from_message_attributes(attrs: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(attrs, dict):
        return {}
    out: Dict[str, str] = {}
    traceparent = str(attrs.get("traceparent", {}).get("StringValue", "")).strip()
    tracestate = str(attrs.get("tracestate", {}).get("StringValue", "")).strip()
    if traceparent:
        out["traceparent"] = traceparent
    if tracestate:
        out["tracestate"] = tracestate
    return out


def _enqueue_llm_optimization_job(
    job_id: str,
    trace_context: Optional[Dict[str, str]] = None,
) -> None:
    queue_url = _get_llm_async_queue_url()
    if not queue_url:
        raise RuntimeError("LLM_OPTIMIZE_SQS_QUEUE_URL is not configured")
    client = _get_sqs_client()
    message = {"job_id": job_id, "source": "tokemizer-backend"}
    attrs = _message_attributes_from_trace_context(trace_context)
    with tracing_service.start_span(
        "sqs.send_message",
        kind=tracing_service.SpanKind.CLIENT,
        attributes={
            "messaging.system": "aws.sqs",
            "messaging.destination": queue_url,
            "messaging.operation": "send",
            "job.id": job_id,
        },
    ):
        send_kwargs: Dict[str, Any] = {
            "QueueUrl": queue_url,
            "MessageBody": json.dumps(message),
        }
        if attrs:
            send_kwargs["MessageAttributes"] = attrs
        client.send_message(**send_kwargs)


def _process_llm_optimization_job(job_id: str) -> None:
    with tracing_service.start_span(
        "llm.job.process",
        kind=tracing_service.SpanKind.INTERNAL,
        attributes={"job.id": job_id},
    ):
        job = get_llm_optimization_job(job_id)
        if not job:
            return
        if job.status in {"completed", "failed"}:
            return

        attempts = int(job.attempts or 0) + 1
        update_llm_optimization_job(
            job_id,
            status="processing",
            attempts=attempts,
            error_message="",
        )

        try:
            payload = dict(job.request_payload)
            request = OptimizationRequest(**payload)
            if not request.prompt:
                raise ValueError("Async LLM optimization requires a single prompt")
            if request.optimization_technique != "llm_based":
                raise ValueError("Async queue supports only llm_based optimization")

            result = _optimize_single_llm(request.prompt, request)
            update_llm_optimization_job(
                job_id,
                status="completed",
                result_payload=result.model_dump(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.exception("Async LLM job %s failed", job_id)
            update_llm_optimization_job(
                job_id,
                status="failed",
                error_message=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )


def _reap_stale_llm_jobs_if_needed(*, force: bool = False) -> None:
    if not _env_truthy("LLM_OPTIMIZE_STALE_REAPER_ENABLED", "true"):
        return

    processing_seconds = _env_int(
        "LLM_OPTIMIZE_STALE_PROCESSING_SECONDS",
        3600,
        minimum=120,
        maximum=172800,
    )
    queued_seconds = _env_int(
        "LLM_OPTIMIZE_STALE_QUEUED_SECONDS",
        21600,
        minimum=300,
        maximum=604800,
    )

    if processing_seconds <= 0 and queued_seconds <= 0:
        return

    now = datetime.now(timezone.utc)
    processing_before = (
        (now - timedelta(seconds=processing_seconds)).isoformat()
        if processing_seconds > 0
        else None
    )
    queued_before = (
        (now - timedelta(seconds=queued_seconds)).isoformat()
        if queued_seconds > 0
        else None
    )

    reaped = reap_stale_llm_optimization_jobs(
        stale_processing_before=processing_before,
        stale_queued_before=queued_before,
        processing_error_message=(
            f"Stale async job reaped after >{processing_seconds}s in processing"
        ),
        queued_error_message=(
            f"Stale async job reaped after >{queued_seconds}s in queued"
        ),
    )
    total = int(reaped.get("total_reaped") or 0)
    if total > 0 or force:
        logger.info(
            "LLM stale-job reaper: processing=%s queued=%s total=%s",
            reaped.get("processing_reaped", 0),
            reaped.get("queued_reaped", 0),
            total,
        )


def _process_llm_queue_message(*, client: Any, queue_url: str, message: Dict[str, Any]) -> None:
    receipt_handle = message.get("ReceiptHandle")
    body_raw = message.get("Body", "")
    job_id: Optional[str] = None
    try:
        payload = json.loads(body_raw)
        if isinstance(payload, dict):
            candidate = payload.get("job_id")
            if isinstance(candidate, str) and candidate.strip():
                job_id = candidate.strip()
    except Exception:
        job_id = None

    try:
        if job_id:
            trace_context = _trace_context_from_message_attributes(
                message.get("MessageAttributes")
            )
            extracted_ctx = tracing_service.extract_context_from_carrier(trace_context)
            token = tracing_service.attach_context(extracted_ctx)
            try:
                with tracing_service.start_span(
                    "sqs.receive.process",
                    kind=tracing_service.SpanKind.CONSUMER,
                    attributes={
                        "messaging.system": "aws.sqs",
                        "messaging.destination": queue_url,
                        "messaging.operation": "process",
                        "job.id": job_id,
                    },
                ):
                    _process_llm_optimization_job(job_id)
            finally:
                tracing_service.detach_context(token)
        else:
            logger.warning("Received invalid LLM async queue payload: %s", body_raw)
    finally:
        if receipt_handle:
            try:
                client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            except Exception:
                logger.exception("Failed to delete SQS message for job=%s", job_id)


def _llm_async_worker_loop() -> None:
    queue_url = _get_llm_async_queue_url()
    if not queue_url:
        logger.warning("LLM async worker disabled: missing LLM_OPTIMIZE_SQS_QUEUE_URL")
        return

    try:
        client = _get_sqs_client()
    except Exception as exc:
        logger.warning("LLM async worker disabled: failed to init SQS client (%s)", exc)
        return

    wait_seconds_raw = os.environ.get("LLM_OPTIMIZE_SQS_WAIT_SECONDS", "10").strip()
    visibility_raw = os.environ.get("LLM_OPTIMIZE_SQS_VISIBILITY_TIMEOUT", "120").strip()
    try:
        wait_seconds = max(1, min(int(wait_seconds_raw), 20))
    except (TypeError, ValueError):
        wait_seconds = 10
    try:
        visibility_timeout = max(30, min(int(visibility_raw), 900))
    except (TypeError, ValueError):
        visibility_timeout = 120

    worker_concurrency = _env_int(
        "LLM_OPTIMIZE_WORKER_CONCURRENCY",
        1,
        minimum=1,
        maximum=8,
    )
    reaper_interval_seconds = _env_int(
        "LLM_OPTIMIZE_REAPER_INTERVAL_SECONDS",
        60,
        minimum=10,
        maximum=600,
    )

    logger.info(
        "LLM async worker started (queue=%s, concurrency=%s)",
        queue_url,
        worker_concurrency,
    )

    _reap_stale_llm_jobs_if_needed(force=True)
    last_reaper_run = time.monotonic()

    with ThreadPoolExecutor(max_workers=worker_concurrency) as worker_pool:
        in_flight: List[Any] = []
        while not _llm_async_worker_stop.is_set():
            try:
                now_monotonic = time.monotonic()
                if now_monotonic - last_reaper_run >= reaper_interval_seconds:
                    _reap_stale_llm_jobs_if_needed()
                    last_reaper_run = now_monotonic

                in_flight = [future for future in in_flight if not future.done()]
                available_slots = worker_concurrency - len(in_flight)
                if available_slots <= 0:
                    time.sleep(0.1)
                    continue

                response = client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=min(10, available_slots),
                    WaitTimeSeconds=wait_seconds,
                    VisibilityTimeout=visibility_timeout,
                    MessageAttributeNames=["All"],
                )
                messages = response.get("Messages", [])
                if not messages:
                    continue

                for message in messages:
                    in_flight.append(
                        worker_pool.submit(
                            _process_llm_queue_message,
                            client=client,
                            queue_url=queue_url,
                            message=message,
                        )
                    )
            except Exception:
                logger.exception("LLM async worker loop error")
                time.sleep(1.0)

        for future in in_flight:
            try:
                future.result(timeout=5)
            except Exception:
                logger.exception("LLM async worker task failed during shutdown")

    logger.info("LLM async worker stopped")


def _start_llm_async_worker() -> None:
    global _llm_async_worker_thread
    if not _llm_async_enabled():
        logger.info("LLM async worker disabled by LLM_OPTIMIZE_ASYNC_ENABLED")
        return

    with _llm_async_worker_lock:
        if _llm_async_worker_thread and _llm_async_worker_thread.is_alive():
            return
        _llm_async_worker_stop.clear()
        _llm_async_worker_thread = threading.Thread(
            target=_llm_async_worker_loop,
            name="llm-async-worker",
            daemon=True,
        )
        _llm_async_worker_thread.start()


def _resolve_ollama_runtime_base_url() -> str:
    return (
        os.environ.get("LLM_OPTIMIZER_OLLAMA_BASE_URL", "").strip()
        or os.environ.get("OLLAMA_BASE_URL", "").strip()
        or "http://localhost:11434"
    ).rstrip("/")


def _resolve_ollama_runtime_model() -> str:
    return os.environ.get("LLM_OPTIMIZER_MODEL", "tokemizer-q4_k_m").strip() or "tokemizer-q4_k_m"


def _send_ollama_keepalive_ping(timeout_seconds: int = 20) -> bool:
    base_url = _resolve_ollama_runtime_base_url()
    model = _resolve_ollama_runtime_model()
    keep_alive = os.environ.get("LLM_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly: OK"}],
        "stream": False,
        "keep_alive": keep_alive,
    }
    request = urllib.request.Request(
        url=f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if int(getattr(response, "status", 200) or 200) >= 400:
                return False
            response.read(1024)
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _ollama_keepalive_worker_loop() -> None:
    interval_raw = os.environ.get("LLM_OLLAMA_KEEPALIVE_INTERVAL_SECONDS", "300").strip()
    timeout_raw = os.environ.get("LLM_OLLAMA_KEEPALIVE_TIMEOUT_SECONDS", "20").strip()
    try:
        interval_seconds = max(30, min(int(interval_raw), 3600))
    except (TypeError, ValueError):
        interval_seconds = 300
    try:
        timeout_seconds = max(5, min(int(timeout_raw), 60))
    except (TypeError, ValueError):
        timeout_seconds = 20

    startup_ok = _send_ollama_keepalive_ping(timeout_seconds=timeout_seconds)
    if startup_ok:
        logger.info("Ollama warm call completed successfully")
    else:
        logger.warning("Ollama warm call failed; will continue keepalive attempts")

    while not _llm_async_worker_stop.wait(interval_seconds):
        ok = _send_ollama_keepalive_ping(timeout_seconds=timeout_seconds)
        if not ok:
            logger.warning("Ollama keepalive ping failed")


def _start_ollama_keepalive_worker() -> None:
    global _ollama_keepalive_thread

    provider = os.environ.get("LLM_OPTIMIZER_PROVIDER", "ollama").strip().lower() or "ollama"
    if provider != "ollama":
        return

    warm_enabled = _env_truthy("LLM_OLLAMA_WARM_ON_STARTUP", "true")
    heartbeat_enabled = _env_truthy("LLM_OLLAMA_KEEPALIVE_ENABLED", "true")
    if not warm_enabled and not heartbeat_enabled:
        logger.info("Ollama warm/keepalive disabled by environment")
        return

    if warm_enabled and not heartbeat_enabled:
        ok = _send_ollama_keepalive_ping()
        if ok:
            logger.info("Ollama warm call completed successfully")
        else:
            logger.warning("Ollama warm call failed")
        return

    with _ollama_keepalive_lock:
        if _ollama_keepalive_thread and _ollama_keepalive_thread.is_alive():
            return
        _ollama_keepalive_thread = threading.Thread(
            target=_ollama_keepalive_worker_loop,
            name="ollama-keepalive-worker",
            daemon=True,
        )
        _ollama_keepalive_thread.start()


def _run_model_preparation_worker(
    hf_home: str,
    missing_models: Sequence[str],
    prewarm_enabled: bool,
    sync_spacy: bool = False,
) -> None:
    """Run HuggingFace model caching and optional warm-up in a background thread."""

    if not missing_models and not prewarm_enabled and not sync_spacy:
        return

    models_to_sync = list(missing_models)

    def _worker() -> None:
        available: List[str] = []
        remaining_missing: List[str] = list(models_to_sync)
        try:
            try:
                if models_to_sync:
                    logger.info(
                        "Background cache sync starting for models: %s",
                        ", ".join(models_to_sync),
                    )
                available, still_missing = ensure_models_cached(
                    hf_home, models_to_sync, refresh_mode="download_missing"
                )
                remaining_missing = list(still_missing)
            except Exception as exc:
                logger.warning("Background cache sync failed: %s", exc, exc_info=True)
                available = []
                remaining_missing = list(models_to_sync)

            if sync_spacy:
                try:
                    from services.model_cache_manager import ensure_spacy_model_cached

                    spacy_model_name = getattr(
                        optimizer, "_spacy_model_name", "en_core_web_sm"
                    )
                    spacy_cached = ensure_spacy_model_cached(
                        spacy_model_name, allow_downloads=True
                    )
                    if spacy_cached:
                        if "spacy" not in available:
                            available.append("spacy")
                    elif "spacy" not in remaining_missing:
                        remaining_missing.append("spacy")
                except Exception as exc:
                    logger.warning("Background spaCy cache preparation failed: %s", exc)
                    if "spacy" not in remaining_missing:
                        remaining_missing.append("spacy")

            if remaining_missing:
                logger.warning(
                    "Background cache still missing: %s",
                    ", ".join(remaining_missing),
                )
            else:
                cached_models = ", ".join(available) or "<none>"
                logger.info("Background cache ready: %s", cached_models)

            if prewarm_enabled:
                logger.info("Background pre-warming models...")
                optimizer.warm_up()

                final_status = optimizer.model_status()
                model_entries = {
                    k: v for k, v in final_status.items() if isinstance(v, dict)
                }
                loaded = [k for k, v in model_entries.items() if v.get("loaded")]
                failed = [k for k, v in model_entries.items() if not v.get("loaded")]

                if loaded:
                    logger.info("Loaded models: %s", ", ".join(loaded))
                if failed:
                    logger.warning(
                        "Failed to load models: %s (strict mode requires admin remediation)",
                        ", ".join(failed),
                    )
            try:
                refresh_model_cache_snapshot()
            except Exception:
                logger.warning(
                    "Failed to refresh model cache snapshot after background preparation",
                    exc_info=True,
                )
            bump_model_cache_validation_version()
            _invalidate_model_availability_cache()
        except Exception:
            logger.exception("Background HuggingFace model preparation failed")

    worker_thread = threading.Thread(
        target=_worker,
        name="hf-model-prep",
        daemon=True,
    )
    worker_thread.start()


def _get_batch_executor() -> ThreadPoolExecutor:
    """Get or create the global batch optimization executor pool."""
    global _batch_executor

    with _batch_executor_lock:
        if _batch_executor is None or getattr(_batch_executor, "_shutdown", False):
            cpu_limit = os.cpu_count() or 1
            env_limit: Optional[int] = None
            env_var = "PROMPT_OPTIMIZER_MAX_WORKERS"

            raw_value = os.environ.get(env_var)
            if raw_value:
                try:
                    parsed_value = int(raw_value)
                    if parsed_value > 0:
                        env_limit = parsed_value
                except ValueError:
                    logger.warning(
                        f"Ignoring {env_var}={raw_value!r} because it is not an integer"
                    )

            max_workers = env_limit if env_limit is not None else cpu_limit
            _batch_executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="batch-optimizer"
            )
            logger.info(
                f"Created global batch executor pool with {max_workers} workers"
            )

        return _batch_executor


def startup_event() -> None:
    tracing_service.configure_tracing("tokemizer-backend")
    init_db()

    # Apply saved admin settings
    try:
        log_level = get_admin_setting("log_level", "INFO")
        logging_control.set_level(log_level)

        telemetry_enabled = bool(get_admin_setting("telemetry_enabled", False))
        set_telemetry_enabled(telemetry_enabled)
        logger.info(
            f"Admin settings applied: log_level={log_level}, telemetry={telemetry_enabled}"
        )
    except Exception as e:
        logger.warning(f"Failed to apply admin settings on startup: {e}")

    get_canonical_mappings_cache().warm_up()
    # Start batch writers for async DB operations
    from database import get_history_writer

    get_history_writer()  # Initialize and start history batch writer

    # Run expensive model/cache checks in the background so auth endpoints are
    # immediately available after process startup.
    thread = threading.Thread(
        target=_run_startup_model_tasks,
        name="startup-model-tasks",
        daemon=True,
    )
    thread.start()

    _start_llm_async_worker()
    _start_ollama_keepalive_worker()


def _run_startup_model_tasks() -> None:
    """Execute slow model/cache startup checks without blocking API readiness."""
    try:
        _run_startup_model_tasks_safe()
    except Exception:
        logger.exception("Background startup model tasks failed")


def _run_startup_model_tasks_safe() -> None:
    """Run startup model tasks and log issues without failing API startup."""

    # AUTO-DETECT volume status (no env flag needed)
    volume_info = _detect_hf_volume()
    model_info = _detect_cached_models(volume_info["path"])
    try:
        refresh_model_cache_snapshot()
    except Exception:
        logger.warning(
            "Failed to refresh model cache snapshot during startup detection",
            exc_info=True,
        )

    # Check admin setting for prewarm, fallback to env
    admin_prewarm = get_admin_setting("optimizer_prewarm_models", None)
    if admin_prewarm is not None:
        prewarm_enabled = str(admin_prewarm).lower() in ("true", "1", "yes", "on")
    else:
        prewarm_enabled = _env_truthy("OPTIMIZER_PREWARM_MODELS", "true")

    spacy_missing = False
    spacy_model_name = getattr(optimizer, "_spacy_model_name", "en_core_web_sm")
    try:
        from services.model_cache_manager import get_spacy_cache_status

        spacy_missing = not bool(
            get_spacy_cache_status(spacy_model_name).get("cached_ok")
        )
    except Exception as exc:
        logger.warning("Failed to validate spaCy cache status during startup: %s", exc)
        spacy_missing = prewarm_enabled

    if prewarm_enabled and spacy_missing:
        try:
            from services.model_cache_manager import ensure_spacy_model_cached

            logger.info(
                "Preparing spaCy cache before startup pre-warm: %s",
                spacy_model_name,
            )
            if ensure_spacy_model_cached(spacy_model_name, allow_downloads=True):
                spacy_missing = False
                logger.info(
                    "spaCy cache prepared before startup pre-warm: %s",
                    spacy_model_name,
                )
            else:
                logger.warning(
                    "spaCy cache is still missing before startup pre-warm: %s",
                    spacy_model_name,
                )
        except Exception as exc:
            logger.warning("Failed to prepare spaCy cache before pre-warm: %s", exc)

    prewarm_completed = False
    if prewarm_enabled and (model_info["missing"] or spacy_missing):
        logger.info("Pre-warming currently available models at startup...")
        try:
            optimizer.warm_up()
            refresh_model_cache_snapshot()
        except Exception:
            logger.warning(
                "Failed to pre-warm currently available models during startup",
                exc_info=True,
            )

    if prewarm_enabled and not model_info["missing"] and not spacy_missing:
        logger.info("Pre-warming models at startup...")
        optimizer.warm_up()
        prewarm_completed = True
        try:
            refresh_model_cache_snapshot()
        except Exception:
            logger.warning(
                "Failed to refresh model cache snapshot after startup prewarm",
                exc_info=True,
            )

    _run_model_preparation_worker(
        volume_info["path"],
        model_info["missing"],
        prewarm_enabled=prewarm_enabled and not prewarm_completed,
        sync_spacy=spacy_missing,
    )
    logger.info("Scheduled background model preparation (prewarm=%s)", prewarm_enabled)

    # Log what we found (informational, never fails startup)
    logger.info("🔍 HuggingFace cache detection:")
    logger.info(f"   Path: {volume_info['path']}")
    logger.info(f"   Exists: {volume_info['exists']}")
    logger.info(f"   Mounted: {volume_info['is_mounted']}")
    logger.info(f"   Writable: {volume_info['writable']}")
    logger.info(f"   Size: {_format_bytes(volume_info['size_bytes'])}")

    if not volume_info["exists"]:
        logger.warning(
            f"⚠️ HF_HOME does not exist at {volume_info['path']}. "
            "Models will be cached in ephemeral storage (lost on restart)."
        )
    elif not volume_info["is_mounted"]:
        logger.warning(
            "⚠️ HF_HOME exists but is NOT mounted. "
            "For persistent model caching, mount a Docker volume or bind mount."
        )

    if model_info["missing"]:
        logger.warning(
            f"⚠️ Required models not cached: {', '.join(model_info['missing'])}. "
            "They will be downloaded on first use (ensure network access or pre-download)."
        )
    else:
        logger.info(
            f"✅ All required models cached: {', '.join(model_info['available'])}"
        )

    if not volume_info["writable"]:
        logger.warning(
            f"⚠️ HF_HOME is not writable at {volume_info['path']}. "
            "Model downloads and updates will fail."
        )

    if _MEMORY_GUARD_ACTIVE:
        logger.warning(
            "Detected %s RAM (< %s required). Prompts >= %s tokens will be rejected until more memory is provisioned.",
            _format_bytes_to_gib(_TOTAL_SYSTEM_MEMORY_BYTES),
            _format_bytes_to_gib(_LONG_PROMPT_MEMORY_THRESHOLD_BYTES),
            f"{_LONG_PROMPT_TOKEN_THRESHOLD:,}",
        )
    elif _LONG_PROMPT_TOKEN_THRESHOLD > 0 and _TOTAL_SYSTEM_MEMORY_BYTES is not None:
        logger.info(
            "Detected %s RAM (>= %s). Long-prompt guard armed at %s tokens.",
            _format_bytes_to_gib(_TOTAL_SYSTEM_MEMORY_BYTES),
            _format_bytes_to_gib(_LONG_PROMPT_MEMORY_THRESHOLD_BYTES),
            f"{_LONG_PROMPT_TOKEN_THRESHOLD:,}",
        )
