from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from types import MappingProxyType
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

from services.telemetry_control import is_enabled as telemetry_is_enabled

ROOT_DIR = Path(__file__).parent
_EMBEDDING_BASELINE_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDING_BENCHMARK_OUTPUT_PATH = (
    ROOT_DIR / "scripts" / "benchmark_outputs" / "embedding_model_benchmark.json"
)
_ADMIN_SETTING_EMBEDDING_PROFILE_OVERRIDES = "semantic_embedding_profile_overrides"


def _resolve_db_path() -> Path:
    override = os.environ.get("DB_PATH")
    if override:
        return Path(override)
    return ROOT_DIR / "app.db"


DB_PATH = _resolve_db_path()
DEFAULT_POOL_SIZE = 5
_DB_TIMEOUT_SECONDS = 5.0
_DB_CONNECT_RETRIES = 2
_HISTORY_BATCH_SIZE = 50
_HISTORY_FLUSH_INTERVAL_SECONDS = 5.0
_TELEMETRY_BATCH_SIZE = 50
_TELEMETRY_FLUSH_INTERVAL_SECONDS = 5.0
_SETTINGS_KEY_LLM_PROFILES = "llm_profiles"
_ADMIN_SETTING_CANONICAL_CACHE_VERSION = "canonical_cache_version"
_ADMIN_SETTING_LLM_SYSTEM_CONTEXT = "llm_system_context"
_LEARNED_PHRASE_MAX_ENTRIES_DEFAULT = 200
_LEARNED_PHRASE_TTL_DAYS_DEFAULT = 90


def _resolve_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


_LEARNED_PHRASE_MAX_ENTRIES = _resolve_env_int(
    "LEARNED_PHRASE_MAX_ENTRIES", _LEARNED_PHRASE_MAX_ENTRIES_DEFAULT
)
_LEARNED_PHRASE_TTL_DAYS = _resolve_env_int(
    "LEARNED_PHRASE_TTL_DAYS", _LEARNED_PHRASE_TTL_DAYS_DEFAULT
)

_SCOPED_CUSTOMER_ID: ContextVar[Optional[str]] = ContextVar(
    "tokemizer_scoped_customer_id", default=None
)


@contextmanager
def customer_scope(customer_id: Optional[str]) -> Generator[None, None, None]:
    token = _SCOPED_CUSTOMER_ID.set(customer_id)
    try:
        yield
    finally:
        _SCOPED_CUSTOMER_ID.reset(token)


def _resolve_pool_size() -> int:
    default_size = max(DEFAULT_POOL_SIZE, (os.cpu_count() or 2) * 2)
    raw_value = os.environ.get("DB_POOL_SIZE")
    if raw_value is None:
        return default_size
    try:
        return max(int(raw_value), 1)
    except (TypeError, ValueError):
        return default_size


def _get_available_memory_bytes() -> Optional[int]:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except (OSError, ValueError):
            pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None

    if (
        isinstance(pages, int)
        and isinstance(page_size, int)
        and pages > 0
        and page_size > 0
    ):
        return pages * page_size
    return None


@dataclass
class SubscriptionPlan:
    id: str
    name: str
    description: Optional[str]
    stripe_price_id: Optional[str]
    monthly_price_cents: int
    annual_price_cents: Optional[int]
    monthly_quota: int
    rate_limit_rpm: int
    concurrent_optimization_jobs: int
    batch_size_limit: int
    optimization_history_retention_days: int
    telemetry_retention_days: int
    audit_log_retention_days: int
    custom_canonical_mappings_limit: int
    max_api_keys: int
    features: List[str]
    is_active: bool
    is_public: bool
    plan_term: str  # 'monthly' or 'yearly'
    monthly_discount_percent: int  # 0-100
    yearly_discount_percent: int  # 0-100
    created_at: str
    updated_at: str


def _resolve_cache_size_kib() -> int:
    default_cache_kib = 64000
    available_bytes = _get_available_memory_bytes()
    if available_bytes is None:
        return default_cache_kib
    cache_kib = max(1024, min(int(available_bytes / 16384), 1_000_000))
    return cache_kib


DB_POOL_SIZE = _resolve_pool_size()
_DB_CACHE_SIZE_KIB = _resolve_cache_size_kib()

_connection_pool: "Queue[sqlite3.Connection]" = Queue(maxsize=DB_POOL_SIZE)
_init_lock = Lock()
_initialized = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _model_dominates_baseline(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> bool:
    candidate_recall = candidate.get("section_ranking_recall_at_k", {})
    baseline_recall = baseline.get("section_ranking_recall_at_k", {})
    try:
        quality_ok = (
            _safe_float(candidate.get("semantic_guard_acceptance_fidelity", 0.0))
            >= _safe_float(baseline.get("semantic_guard_acceptance_fidelity", 0.0))
            and _safe_float(candidate_recall.get("2", 0.0))
            >= _safe_float(baseline_recall.get("2", 0.0))
            and _safe_float(
                candidate.get("query_aware_compression_retention_quality", 0.0)
            )
            >= _safe_float(
                baseline.get("query_aware_compression_retention_quality", 0.0)
            )
        )
        latency_ok = _safe_float(
            candidate.get("latency_ms_per_1k_tokens", float("inf")),
            float("inf"),
        ) <= _safe_float(
            baseline.get("latency_ms_per_1k_tokens", float("inf")),
            float("inf"),
        )
    except Exception:
        return False
    return quality_ok and latency_ok


def _resolve_seeded_semantic_defaults() -> Tuple[str, Dict[str, str]]:
    default_model = _EMBEDDING_BASELINE_MODEL
    profile_overrides: Dict[str, str] = {}
    path = _EMBEDDING_BENCHMARK_OUTPUT_PATH
    if not path.exists():
        return default_model, profile_overrides

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default_model, profile_overrides
    if not isinstance(payload, dict):
        return default_model, profile_overrides

    models = payload.get("models")
    baseline_model = str(payload.get("baseline_model") or _EMBEDDING_BASELINE_MODEL)
    if not isinstance(models, dict):
        return default_model, profile_overrides
    baseline_metrics = models.get(baseline_model)
    if not isinstance(baseline_metrics, dict):
        return default_model, profile_overrides

    recs = payload.get("recommendations")
    if not isinstance(recs, dict):
        return default_model, profile_overrides

    candidate_default = str(recs.get("default") or baseline_model)
    candidate_default_metrics = models.get(candidate_default)
    if isinstance(candidate_default_metrics, dict) and _model_dominates_baseline(
        candidate_default_metrics, baseline_metrics
    ):
        default_model = candidate_default
    else:
        default_model = baseline_model

    overrides = recs.get("profile_overrides")
    if not isinstance(overrides, dict):
        return default_model, profile_overrides

    baseline_profiles = baseline_metrics.get("profiles")
    if not isinstance(baseline_profiles, dict):
        return default_model, profile_overrides

    for profile, model_name in overrides.items():
        profile_name = str(profile).strip()
        candidate_name = str(model_name).strip()
        if not profile_name or not candidate_name:
            continue
        candidate_metrics = models.get(candidate_name)
        if not isinstance(candidate_metrics, dict):
            continue
        candidate_profile = candidate_metrics.get("profiles", {}).get(profile_name)
        baseline_profile = baseline_profiles.get(profile_name)
        if not isinstance(candidate_profile, dict) or not isinstance(
            baseline_profile, dict
        ):
            continue
        profile_quality_ok = (
            _safe_float(
                candidate_profile.get("semantic_guard_acceptance_fidelity", 0.0)
            )
            >= _safe_float(
                baseline_profile.get("semantic_guard_acceptance_fidelity", 0.0)
            )
            and _safe_float(candidate_profile.get("section_ranking_recall_at_2", 0.0))
            >= _safe_float(baseline_profile.get("section_ranking_recall_at_2", 0.0))
            and _safe_float(
                candidate_profile.get("query_aware_compression_retention_quality", 0.0)
            )
            >= _safe_float(
                baseline_profile.get("query_aware_compression_retention_quality", 0.0)
            )
        )
        profile_latency_ok = _safe_float(
            candidate_metrics.get("latency_ms_per_1k_tokens", float("inf")),
            float("inf"),
        ) <= _safe_float(
            baseline_metrics.get("latency_ms_per_1k_tokens", float("inf")),
            float("inf"),
        )
        if profile_quality_ok and profile_latency_ok:
            profile_overrides[profile_name] = candidate_name

    return default_model, profile_overrides


def _ensure_required_entropy_inventory_entries(conn: sqlite3.Connection) -> None:
    """Ensure active entropy and entropy_fast inventory entries exist."""
    now_iso = datetime.now(timezone.utc).isoformat()
    required_models = [
        {
            "id": str(uuid.uuid4()),
            "model_type": "entropy",
            "model_name": "HuggingFaceTB/SmolLM2-360M",
            "min_size_bytes": 690 * 1024 * 1024,
            "expected_files": json.dumps(["model.safetensors", "config.json"]),
            "component": "Entropy Scoring",
            "library_type": "transformers",
            "usage": "Entropy pruning",
            "revision": "main",
            "allow_patterns": json.dumps([]),
            "is_active": True,
            "created_at": now_iso,
            "updated_at": now_iso,
        },
        {
            "id": str(uuid.uuid4()),
            "model_type": "entropy_fast",
            "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "min_size_bytes": 677 * 1024 * 1024,
            "expected_files": json.dumps(
                ["model.safetensors", "tokenizer.json", "config.json"]
            ),
            "component": "Entropy Scoring (Fast)",
            "library_type": "transformers",
            "usage": "Token-level drop probability scoring",
            "revision": "main",
            "allow_patterns": json.dumps([]),
            "is_active": True,
            "created_at": now_iso,
            "updated_at": now_iso,
        },
    ]

    for model in required_models:
        active_row = conn.execute(
            """
            SELECT 1
            FROM model_inventory
            WHERE model_type = ? AND is_active = 1
            LIMIT 1
            """,
            (model["model_type"],),
        ).fetchone()
        if active_row:
            continue

        conn.execute(
            """
            INSERT INTO model_inventory (
                id, model_type, model_name, min_size_bytes,
                expected_files, component, library_type, usage,
                revision, allow_patterns, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model["id"],
                model["model_type"],
                model["model_name"],
                model["min_size_bytes"],
                model["expected_files"],
                model["component"],
                model["library_type"],
                model["usage"],
                model["revision"],
                model["allow_patterns"],
                model["is_active"],
                model["created_at"],
                model["updated_at"],
            ),
        )


def _ensure_parent_directory(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _connect_with_retries() -> sqlite3.Connection:
    last_error: Optional[Exception] = None
    for attempt in range(_DB_CONNECT_RETRIES + 1):
        try:
            return sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,
                timeout=_DB_TIMEOUT_SECONDS,
            )
        except sqlite3.OperationalError as exc:
            last_error = exc
            if attempt >= _DB_CONNECT_RETRIES:
                break
            time.sleep(min(0.05 * (attempt + 1), 0.2))
    if last_error is not None:
        raise last_error
    raise sqlite3.OperationalError("Failed to establish database connection")


def get_db_connection() -> sqlite3.Connection:
    _ensure_parent_directory(DB_PATH)
    conn = _connect_with_retries()
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA cache_size=-{_DB_CACHE_SIZE_KIB}")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _acquire_connection() -> sqlite3.Connection:
    try:
        return _connection_pool.get_nowait()
    except Empty:
        return get_db_connection()


def _release_connection(conn: sqlite3.Connection) -> None:
    if conn is None:
        return
    try:
        _connection_pool.put_nowait(conn)
    except Exception:
        conn.close()


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = _acquire_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _release_connection(conn)


def init_db() -> None:
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    raw_prompt TEXT NOT NULL,
                    optimized_prompt TEXT NOT NULL,
                    raw_tokens INTEGER NOT NULL,
                    optimized_tokens INTEGER NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    estimated_cost_before REAL NOT NULL,
                    estimated_cost_after REAL NOT NULL,
                    estimated_cost_saved REAL NOT NULL,
                    compression_percentage REAL NOT NULL DEFAULT 0.0,
                    semantic_similarity REAL,
                    techniques_applied TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("DROP INDEX IF EXISTS idx_optimization_history_created_at")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_history_created_at_mode
                ON optimization_history(created_at, mode)
                """)
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_optimization_history_customer_id
                    ON optimization_history(customer_id)
                    """)
            except Exception:
                # Column might not exist yet; migration will handle it
                pass

            # Canonical mappings table for DB-backed configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS canonical_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_token TEXT NOT NULL COLLATE NOCASE,
                    target_token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            # Unique index on source_token for fast case-insensitive lookups and duplicate prevention
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_source_token
                ON canonical_mappings(source_token COLLATE NOCASE)
                """)

            # Index on updated_at for potential sorting/filtering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_canonical_updated_at
                ON canonical_mappings(updated_at)
                """)

            # User-specific canonical mappings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_canonical_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    source_token TEXT NOT NULL COLLATE NOCASE,
                    target_token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(id),
                    UNIQUE(customer_id, source_token)
                )
                """)

            # User OOTB mapping disabled status
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_ootb_mapping_disabled (
                    customer_id TEXT NOT NULL,
                    mapping_source_token TEXT NOT NULL COLLATE NOCASE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (customer_id, mapping_source_token),
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_phrase_dictionary (
                    customer_id TEXT NOT NULL,
                    phrase TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (customer_id, phrase),
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learned_phrase_customer_id
                ON learned_phrase_dictionary(customer_id)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learned_phrase_usage
                ON learned_phrase_dictionary(customer_id, usage_count DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learned_phrase_last_used
                ON learned_phrase_dictionary(customer_id, last_used_at DESC)
                """)

            # Performance telemetry table for per-pass optimization metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id TEXT NOT NULL,
                    pass_name TEXT NOT NULL,
                    pass_order INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    tokens_before INTEGER NOT NULL,
                    tokens_after INTEGER NOT NULL,
                    tokens_saved INTEGER NOT NULL,
                    reduction_percent REAL NOT NULL,
                    expected_utility REAL,
                    actual_utility REAL,
                    pass_skipped_reason TEXT,
                    content_profile TEXT,
                    optimization_mode TEXT,
                    token_bin TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (optimization_id) REFERENCES optimization_history(id)
                )
                """)

            # Migrate older telemetry schemas forward (CREATE TABLE IF NOT EXISTS does not
            # add missing columns on existing databases).
            try:
                existing_cols = {
                    str(row[1])
                    for row in conn.execute(
                        "PRAGMA table_info(performance_telemetry)"
                    ).fetchall()
                }
                required_cols = {
                    "expected_utility": "REAL",
                    "actual_utility": "REAL",
                    "pass_skipped_reason": "TEXT",
                    "content_profile": "TEXT",
                    "optimization_mode": "TEXT",
                    "token_bin": "TEXT",
                    "created_at": "TEXT",
                }
                for col_name, col_type in required_cols.items():
                    if col_name in existing_cols:
                        continue
                    conn.execute(
                        f"ALTER TABLE performance_telemetry ADD COLUMN {col_name} {col_type}"
                    )
            except Exception:
                # Never block startup on telemetry migrations.
                pass

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_optimization_id
                ON performance_telemetry(optimization_id)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_optimization_pass_order
                ON performance_telemetry(optimization_id, pass_order)
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_pass_name
                ON performance_telemetry(pass_name)
                """)

            # Admin Settings for global configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS admin_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            # Model Inventory table for dynamic model management
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_inventory (
                    id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    min_size_bytes INTEGER NOT NULL,
                    expected_files TEXT NOT NULL,
                    component TEXT NOT NULL DEFAULT '',
                    library_type TEXT NOT NULL DEFAULT '',
                    usage TEXT NOT NULL DEFAULT '',
                    revision TEXT NOT NULL DEFAULT '',
                    allow_patterns TEXT NOT NULL DEFAULT '[]',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_model_inventory_type_active
                ON model_inventory(model_type)
                WHERE is_active = 1
                """)

            # Seed default models if table is empty
            cursor = conn.execute("SELECT COUNT(*) FROM model_inventory")
            if cursor.fetchone()[0] == 0:
                seeded_semantic_model, profile_overrides = (
                    _resolve_seeded_semantic_defaults()
                )
                default_models = [
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "semantic_guard",
                        "model_name": seeded_semantic_model,
                        "min_size_bytes": 127 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.onnx", "tokenizer.json", "config.json"]
                        ),
                        "component": "Semantic Guard",
                        "library_type": "sentence-transformers",
                        "usage": "Semantic similarity scoring",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "semantic_rank",
                        "model_name": seeded_semantic_model,
                        "min_size_bytes": 127 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.onnx", "tokenizer.json", "config.json"]
                        ),
                        "component": "Semantic Ranker",
                        "library_type": "sentence-transformers",
                        "usage": "Section ranking & query-aware compression",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "entropy",
                        "model_name": "HuggingFaceTB/SmolLM2-360M",
                        "min_size_bytes": 690 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.safetensors", "config.json"]
                        ),
                        "component": "Entropy Scoring",
                        "library_type": "transformers",
                        "usage": "Entropy pruning",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "entropy_fast",
                        "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                        "min_size_bytes": 677 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.safetensors", "tokenizer.json", "config.json"]
                        ),
                        "component": "Entropy Scoring (Fast)",
                        "library_type": "transformers",
                        "usage": "Token-level drop probability scoring",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "token_classifier",
                        "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                        "min_size_bytes": 677 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.safetensors", "tokenizer.json", "config.json"]
                        ),
                        "component": "Token Classifier",
                        "library_type": "transformers",
                        "usage": "Maximum-mode compression",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "model_type": "coreference",
                        "model_name": "talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large",
                        "min_size_bytes": 498 * 1024 * 1024,
                        "expected_files": json.dumps(
                            ["model.onnx", "tokenizer.json", "config.json"]
                        ),
                        "component": "Coreference",
                        "library_type": "spacy-coref",
                        "usage": "Coreference compression",
                        "revision": "main",
                        "allow_patterns": json.dumps([]),
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                ]

                for model in default_models:
                    conn.execute(
                        """
                        INSERT INTO model_inventory (
                            id, model_type, model_name, min_size_bytes,
                            expected_files, component, library_type, usage,
                            revision, allow_patterns, is_active, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            model["id"],
                            model["model_type"],
                            model["model_name"],
                            model["min_size_bytes"],
                            model["expected_files"],
                            model["component"],
                            model["library_type"],
                            model["usage"],
                            model["revision"],
                            model["allow_patterns"],
                            model["is_active"],
                            model["created_at"],
                            model["updated_at"],
                        ),
                    )

                if profile_overrides:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO admin_settings (key, value, updated_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            _ADMIN_SETTING_EMBEDDING_PROFILE_OVERRIDES,
                            json.dumps(
                                profile_overrides,
                                ensure_ascii=True,
                                separators=(",", ":"),
                            ),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )

            _ensure_required_entropy_inventory_entries(conn)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_created_at
                ON performance_telemetry(created_at)
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_utility_priors
                ON performance_telemetry(content_profile, optimization_mode, token_bin, pass_name)
                """)

            # Batch jobs for reporting and dashboards
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    processed_items INTEGER NOT NULL DEFAULT 0,
                    total_savings_percentage REAL,
                    processing_time_ms REAL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at
                ON batch_jobs(created_at)
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_optimization_jobs (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    status TEXT NOT NULL,
                    request_payload TEXT NOT NULL,
                    result_payload TEXT,
                    error_message TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_optimization_jobs_created_at
                ON llm_optimization_jobs(created_at)
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_optimization_jobs_customer_status
                ON llm_optimization_jobs(customer_id, status)
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (customer_id, key)
                )
                """)

            # Subscription Plans table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plans (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    stripe_price_id TEXT UNIQUE,
                    monthly_price_cents INTEGER NOT NULL DEFAULT 0,
                    annual_price_cents INTEGER,
                    monthly_quota INTEGER NOT NULL,
                    rate_limit_rpm INTEGER NOT NULL DEFAULT 1000,
                    concurrent_optimization_jobs INTEGER NOT NULL DEFAULT 5,
                    batch_size_limit INTEGER NOT NULL DEFAULT 1000,
                    optimization_history_retention_days INTEGER NOT NULL DEFAULT 365,
                    telemetry_retention_days INTEGER NOT NULL DEFAULT 365,
                    audit_log_retention_days INTEGER NOT NULL DEFAULT 365,
                    custom_canonical_mappings_limit INTEGER NOT NULL DEFAULT 1000,
                    max_api_keys INTEGER NOT NULL DEFAULT 10,
                    features TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    is_public BOOLEAN NOT NULL DEFAULT 1,
                    plan_term TEXT NOT NULL DEFAULT 'monthly',
                    monthly_discount_percent INTEGER NOT NULL DEFAULT 0,
                    yearly_discount_percent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            # Customers table for identity and authentication
            conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT UNIQUE,
                    api_key_hash TEXT,
                    password_hash TEXT,
                    role TEXT NOT NULL DEFAULT 'customer',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    subscription_status TEXT NOT NULL DEFAULT 'active',
                    subscription_tier TEXT DEFAULT 'free',
                    quota_override INTEGER,
                    quota_overage_bonus INTEGER NOT NULL DEFAULT 0,
                    stripe_customer_id TEXT UNIQUE,
                    stripe_subscription_id TEXT UNIQUE,
                    stripe_subscription_item_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

            # Usage tracking table for quota enforcement
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    customer_id TEXT NOT NULL,
                    api_key_id TEXT,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    calls_used INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (customer_id, period_start, api_key_id),
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            # API Keys table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                )
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_customer_id
                ON api_keys(customer_id)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash
                ON api_keys(key_hash)
                """)

        _initialized = True
        _migrate_history_table()
        _migrate_canonical_mappings()
        _migrate_customers_table()
        _migrate_subscription_plans()
        _migrate_batch_jobs_table()
        _migrate_settings_table()
        _migrate_usage_table()


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_history_enabled() -> bool:
    admin_val = get_admin_setting("history_enabled", None)
    if admin_val is not None:
        return str(admin_val).lower() in ("true", "1", "yes", "on")
    return _env_flag("OPTIMIZATION_HISTORY_ENABLED", True)


def is_learned_abbreviations_enabled() -> bool:
    admin_val = get_admin_setting("learned_abbreviations_enabled", None)
    if admin_val is not None:
        return str(admin_val).lower() in ("true", "1", "yes", "on")
    return _env_flag("LEARNED_ABBREVIATIONS_ENABLED", True)


def is_telemetry_enabled() -> bool:
    """Return whether performance telemetry is enabled.

    Admin DB setting is authoritative so the value stays consistent across
    multiple worker processes. Runtime toggle remains as fallback.
    """
    admin_val = get_admin_setting("telemetry_enabled", None)
    if admin_val is not None:
        if isinstance(admin_val, bool):
            return admin_val
        return str(admin_val).strip().lower() in {"true", "1", "yes", "on"}
    return telemetry_is_enabled()


def get_llm_profiles(customer_id: str) -> List[Dict[str, str]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ? AND customer_id = ?",
            (_SETTINGS_KEY_LLM_PROFILES, customer_id),
        ).fetchone()

    if not row:
        # Optional fallback to system defaults?
        # row = conn.execute("SELECT value FROM admin_settings ...")
        return []

    raw_value = row["value"]
    try:
        payload = json.loads(raw_value)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    normalized: List[Dict[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        provider = str(entry.get("provider", "")).strip()
        model = str(entry.get("model", "")).strip()
        api_key = str(entry.get("api_key", "")).strip()
        if not (name and provider and model and api_key):
            continue
        normalized.append(
            {
                "name": name,
                "provider": provider,
                "model": model,
                "api_key": api_key,
            }
        )
    return normalized


def set_llm_profiles(customer_id: str, profiles: List[Dict[str, str]]) -> None:
    payload = json.dumps(profiles, ensure_ascii=True, separators=(",", ":"))
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, customer_id, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (_SETTINGS_KEY_LLM_PROFILES, customer_id, payload, timestamp),
        )


def _load_default_llm_system_context() -> str:
    default_path = ROOT_DIR.parent / "LLM compression context.txt"
    try:
        content = default_path.read_text(encoding="utf-8").strip()
        if content:
            return content
    except OSError:
        pass
    return (
        "You are an expert semantic compression engine. Compress text aggressively while "
        "preserving meaning, facts, numbers, quoted text, code blocks, constraints, and "
        "actionable instructions. Return only the compressed result."
    )


def get_llm_system_context() -> str:
    use_file_default = (
        os.environ.get("LLM_SYSTEM_CONTEXT_FROM_FILE", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if use_file_default:
        return _load_default_llm_system_context()

    value = get_admin_setting(_ADMIN_SETTING_LLM_SYSTEM_CONTEXT, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _load_default_llm_system_context()


def set_llm_system_context(value: str) -> None:
    cleaned_value = (value or "").strip()
    if not cleaned_value:
        cleaned_value = _load_default_llm_system_context()
    set_admin_setting(_ADMIN_SETTING_LLM_SYSTEM_CONTEXT, cleaned_value)


def _cleanup_learned_phrase_dictionary(customer_id: str) -> None:
    max_entries = max(_LEARNED_PHRASE_MAX_ENTRIES, 0)
    ttl_days = max(_LEARNED_PHRASE_TTL_DAYS, 0)
    if max_entries == 0 and ttl_days == 0:
        return
    init_db()
    with get_db() as conn:
        if ttl_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            conn.execute(
                """
                DELETE FROM learned_phrase_dictionary
                WHERE customer_id = ? AND last_used_at < ?
                """,
                (customer_id, cutoff.isoformat()),
            )

        if max_entries > 0:
            conn.execute(
                """
                DELETE FROM learned_phrase_dictionary
                WHERE customer_id = ?
                    AND phrase NOT IN (
                        SELECT phrase FROM learned_phrase_dictionary
                        WHERE customer_id = ?
                        ORDER BY usage_count DESC, last_used_at DESC
                        LIMIT ?
                    )
                """,
                (customer_id, customer_id, max_entries),
            )


def get_learned_phrase_dictionary(
    customer_id: str, limit: Optional[int] = None
) -> Dict[str, str]:
    init_db()
    _cleanup_learned_phrase_dictionary(customer_id)
    max_entries = limit if limit is not None else _LEARNED_PHRASE_MAX_ENTRIES
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT phrase, alias
            FROM learned_phrase_dictionary
            WHERE customer_id = ?
            ORDER BY usage_count DESC, last_used_at DESC
            LIMIT ?
            """,
            (customer_id, max_entries),
        )
        return {row["phrase"]: row["alias"] for row in cursor.fetchall()}


def upsert_learned_phrase_mappings(
    customer_id: str, mappings: Mapping[str, str]
) -> None:
    if not mappings:
        return
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        for phrase, alias in mappings.items():
            conn.execute(
                """
                INSERT INTO learned_phrase_dictionary (
                    customer_id, phrase, alias, created_at, last_used_at, usage_count
                )
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(customer_id, phrase) DO UPDATE SET
                    alias = excluded.alias,
                    last_used_at = excluded.last_used_at,
                    usage_count = learned_phrase_dictionary.usage_count + 1
                """,
                (customer_id, phrase, alias, timestamp, timestamp),
            )
    _cleanup_learned_phrase_dictionary(customer_id)


def update_learned_phrase_usage(customer_id: str, phrases: List[str]) -> None:
    if not phrases:
        return
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        for phrase in phrases:
            conn.execute(
                """
                UPDATE learned_phrase_dictionary
                SET last_used_at = ?, usage_count = usage_count + 1
                WHERE customer_id = ? AND phrase = ?
                """,
                (timestamp, customer_id, phrase),
            )
    _cleanup_learned_phrase_dictionary(customer_id)


@dataclass
class OptimizationHistoryRecord:
    id: str
    customer_id: Optional[str]
    created_at: str
    updated_at: str
    mode: str
    raw_prompt: str
    optimized_prompt: str
    raw_tokens: int
    optimized_tokens: int
    processing_time_ms: float
    estimated_cost_before: float
    estimated_cost_after: float
    estimated_cost_saved: float
    compression_percentage: float
    semantic_similarity: Optional[float]
    techniques_applied: list[str]


@dataclass
class BatchJobRecord:
    id: str
    name: str
    status: str
    total_items: int
    processed_items: int
    total_savings_percentage: Optional[float]
    processing_time_ms: Optional[float]
    created_at: str
    completed_at: Optional[str]


@dataclass
class LLMOptimizationJobRecord:
    id: str
    customer_id: Optional[str]
    status: str
    request_payload: Dict[str, Any]
    result_payload: Optional[Dict[str, Any]]
    error_message: Optional[str]
    attempts: int
    created_at: str
    updated_at: str
    completed_at: Optional[str]


def record_optimization_history(
    *,
    mode: str,
    raw_prompt: str,
    optimized_prompt: str,
    raw_tokens: int,
    optimized_tokens: int,
    processing_time_ms: float,
    estimated_cost_before: float,
    estimated_cost_after: float,
    estimated_cost_saved: float,
    customer_id: Optional[str] = None,
    compression_percentage: Optional[float] = None,
    semantic_similarity: Optional[float] = None,
    techniques_applied: Optional[list[str]] = None,
    record_id: Optional[str] = None,
) -> Optional[str]:
    """Record optimization history using async batch writer."""
    if not is_history_enabled():
        return None

    if customer_id is None:
        customer_id = _SCOPED_CUSTOMER_ID.get()

    record_uuid = record_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    compression_value = (
        float(compression_percentage)
        if compression_percentage is not None
        else (
            ((raw_tokens - optimized_tokens) / raw_tokens * 100)
            if raw_tokens > 0
            else 0.0
        )
    )

    normalized_techniques = (
        [str(item) for item in techniques_applied] if techniques_applied else []
    )

    history_record = HistoryRecord(
        id=record_uuid,
        customer_id=customer_id,
        created_at=timestamp,
        updated_at=timestamp,
        mode=mode,
        raw_prompt=raw_prompt,
        optimized_prompt=optimized_prompt,
        raw_tokens=raw_tokens,
        optimized_tokens=optimized_tokens,
        processing_time_ms=processing_time_ms,
        estimated_cost_before=estimated_cost_before,
        estimated_cost_after=estimated_cost_after,
        estimated_cost_saved=estimated_cost_saved,
        compression_percentage=compression_value,
        semantic_similarity=(
            float(semantic_similarity) if semantic_similarity is not None else None
        ),
        techniques_applied=normalized_techniques,
    )

    # Always use batch writer for async writes
    try:
        writer = get_history_writer()
        writer.submit(history_record)
    except Exception as e:
        logging.error(f"Failed to submit history to batch writer: {e}", exc_info=True)

    return history_record.id


def list_recent_history(
    limit: int = 50, customer_id: Optional[str] = None
) -> list[OptimizationHistoryRecord]:
    """Return recent optimization history entries for dashboards."""
    init_db()
    with get_db() as conn:
        query = """
            SELECT
                id,
                customer_id,
                created_at,
                updated_at,
                mode,
                raw_prompt,
                optimized_prompt,
                raw_tokens,
                optimized_tokens,
                processing_time_ms,
                estimated_cost_before,
                estimated_cost_after,
                estimated_cost_saved,
                compression_percentage,
                semantic_similarity,
                techniques_applied
            FROM optimization_history
        """
        params = []
        if customer_id:
            query += " WHERE customer_id = ?"
            params.append(customer_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, tuple(params))
        rows = cursor.fetchall()

    return [
        OptimizationHistoryRecord(
            id=row["id"],
            customer_id=row["customer_id"] if "customer_id" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            mode=row["mode"],
            raw_prompt=row["raw_prompt"],
            optimized_prompt=row["optimized_prompt"],
            raw_tokens=row["raw_tokens"],
            optimized_tokens=row["optimized_tokens"],
            processing_time_ms=row["processing_time_ms"],
            estimated_cost_before=row["estimated_cost_before"],
            estimated_cost_after=row["estimated_cost_after"],
            estimated_cost_saved=row["estimated_cost_saved"],
            compression_percentage=float(row["compression_percentage"] or 0.0),
            semantic_similarity=(
                float(row["semantic_similarity"])
                if row["semantic_similarity"] is not None
                else None
            ),
            techniques_applied=_decode_techniques(row["techniques_applied"]),
        )
        for row in rows
    ]


def get_optimization_history_record(
    optimization_id: str, customer_id: Optional[str] = None
) -> Optional[OptimizationHistoryRecord]:
    init_db()
    with get_db() as conn:
        query = """
            SELECT
                id,
                customer_id,
                created_at,
                updated_at,
                mode,
                raw_prompt,
                optimized_prompt,
                raw_tokens,
                optimized_tokens,
                processing_time_ms,
                estimated_cost_before,
                estimated_cost_after,
                estimated_cost_saved,
                compression_percentage,
                semantic_similarity,
                techniques_applied
            FROM optimization_history
            WHERE id = ?
        """
        params: list[Any] = [optimization_id]
        if customer_id:
            query += " AND customer_id = ?"
            params.append(customer_id)

        row = conn.execute(query, tuple(params)).fetchone()

    if not row:
        return None

    return OptimizationHistoryRecord(
        id=row["id"],
        customer_id=row["customer_id"] if "customer_id" in row.keys() else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        mode=row["mode"],
        raw_prompt=row["raw_prompt"],
        optimized_prompt=row["optimized_prompt"],
        raw_tokens=row["raw_tokens"],
        optimized_tokens=row["optimized_tokens"],
        processing_time_ms=row["processing_time_ms"],
        estimated_cost_before=row["estimated_cost_before"],
        estimated_cost_after=row["estimated_cost_after"],
        estimated_cost_saved=row["estimated_cost_saved"],
        compression_percentage=float(row["compression_percentage"] or 0.0),
        semantic_similarity=(
            float(row["semantic_similarity"])
            if row["semantic_similarity"] is not None
            else None
        ),
        techniques_applied=_decode_techniques(row["techniques_applied"]),
    )


def aggregate_history_stats(
    limit: int = 200, customer_id: Optional[str] = None
) -> dict[str, float]:
    """Compute lightweight aggregates for dashboard statistics."""
    history = list_recent_history(limit, customer_id)
    if not history:
        return {
            "tokens_saved": 0.0,
            "avg_compression_percentage": 0.0,
            "avg_latency_ms": 0.0,
            "avg_quality_score": 0.0,
            "total_optimizations": 0,
            "estimated_monthly_savings": 0.0,
        }

    total_tokens_saved = sum(
        max(rec.raw_tokens - rec.optimized_tokens, 0) for rec in history
    )
    compression_values = [rec.compression_percentage for rec in history]
    avg_compression = (
        sum(compression_values) / len(compression_values) if compression_values else 0.0
    )
    avg_latency = sum(rec.processing_time_ms for rec in history) / len(history)

    similarities = [
        rec.semantic_similarity
        for rec in history
        if rec.semantic_similarity is not None
    ]
    avg_quality = sum(similarities) / len(similarities) if similarities else 0.0

    input_cost_savings = (total_tokens_saved / 1_000_000) * 0.015
    estimated_monthly_savings = input_cost_savings * 30

    return {
        "tokens_saved": total_tokens_saved,
        "avg_compression_percentage": round(avg_compression, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_quality_score": round(avg_quality, 3),
        "total_optimizations": len(history),
        "estimated_monthly_savings": round(estimated_monthly_savings, 2),
    }


# ========== Canonical Mappings Management ==========


@dataclass
class CanonicalMapping:
    id: int
    source_token: str
    target_token: str
    created_at: str
    updated_at: str


class CanonicalMappingsCache:
    """Thread-safe in-memory cache for canonical mappings from database."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._view: MappingProxyType[str, str] = MappingProxyType(self._cache)
        self._lock = Lock()
        self._loaded = False
        self._version = 0

    def version(self) -> int:
        with self._lock:
            return self._version

    def get_all(self) -> Mapping[str, str]:
        """Get all canonical mappings from cache, loading from DB if not yet loaded."""
        with self._lock:
            if not self._loaded:
                self._load_from_db()
            return self._view

    def warm_up(self) -> None:
        """Preload mappings to avoid initial DB fetch during first optimization."""
        with self._lock:
            if not self._loaded:
                self._load_from_db()

    def _load_from_db(self) -> None:
        """Load all canonical mappings from database into cache (called with lock held)."""
        try:
            # Ensure database is initialized before attempting to query
            init_db()
            with get_db() as conn:
                _seed_default_canonical_mappings(conn)
                cursor = conn.execute(
                    "SELECT source_token, target_token FROM canonical_mappings"
                )
                # Store with lowercase keys for case-insensitive lookup
                self._cache = {
                    row["source_token"].lower(): row["target_token"]
                    for row in cursor.fetchall()
                }
            self._loaded = True
        except Exception as e:
            # Graceful fallback: If database is not available or table doesn't exist,
            # use empty cache. This allows optimizer to work without database in test/CLI contexts.
            logging.warning(
                f"Failed to load canonical mappings from database, using empty cache: {e}"
            )
            self._cache = {}
            self._loaded = True  # Mark as loaded to prevent repeated attempts
        finally:
            self._view = MappingProxyType(self._cache)

    def invalidate_and_reload(self) -> None:
        """Invalidate cache and reload from database (call after write operations)."""
        with self._lock:
            self._version += 1
            self._loaded = False
            self._load_from_db()

    def clear(self) -> None:
        """Clear the cache and mark as not loaded. Next access will reload from database."""
        with self._lock:
            self._version += 1
            self._cache = {}
            self._loaded = False
            self._view = MappingProxyType(self._cache)


# Global cache instance
_canonical_cache = CanonicalMappingsCache()


def get_canonical_mappings_cache() -> CanonicalMappingsCache:
    """Get the global canonical mappings cache instance."""
    return _canonical_cache


def _seed_default_canonical_mappings(conn: sqlite3.Connection) -> None:
    """Insert missing default canonical mappings without overwriting custom entries."""
    from services.optimizer import config

    cursor = conn.execute("SELECT source_token FROM canonical_mappings")
    existing = {
        row["source_token"].lower() for row in cursor.fetchall() if row["source_token"]
    }

    timestamp = datetime.now(timezone.utc).isoformat()
    defaults = [
        (source, target, timestamp, timestamp)
        for source, target in config.CANONICALIZATIONS.items()
        if source.lower() not in existing
    ]

    if defaults:
        conn.executemany(
            """
            INSERT INTO canonical_mappings (source_token, target_token, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            defaults,
        )


def _migrate_canonical_mappings() -> None:
    """Ensure default canonical mappings exist in the database."""
    with get_db() as conn:
        _seed_default_canonical_mappings(conn)


def _decode_techniques(raw_value: Optional[str]) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def _migrate_history_table() -> None:
    """Add missing history columns for compression %, semantic similarity, and techniques."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(optimization_history)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        if "compression_percentage" not in existing_columns:
            conn.execute(
                "ALTER TABLE optimization_history ADD COLUMN compression_percentage REAL NOT NULL DEFAULT 0.0"
            )

        if "semantic_similarity" not in existing_columns:
            conn.execute(
                "ALTER TABLE optimization_history ADD COLUMN semantic_similarity REAL"
            )

        if "techniques_applied" not in existing_columns:
            conn.execute(
                "ALTER TABLE optimization_history ADD COLUMN techniques_applied TEXT"
            )

        if "customer_id" not in existing_columns:
            conn.execute(
                "ALTER TABLE optimization_history ADD COLUMN customer_id TEXT REFERENCES customers(id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_optimization_history_customer_id ON optimization_history(customer_id)"
            )


def _migrate_customers_table() -> None:
    """Ensure customers table schema matches current model."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(customers)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        if "trial_ends_at" in existing_columns:
            conn.execute("ALTER TABLE customers RENAME TO customers_old")
            conn.execute("""
                CREATE TABLE customers (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT UNIQUE,
                    api_key_hash TEXT,
                    password_hash TEXT,
                    role TEXT NOT NULL DEFAULT 'customer',
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    subscription_status TEXT NOT NULL DEFAULT 'inactive',
                    subscription_tier TEXT DEFAULT 'free',
                    quota_override INTEGER,
                    quota_overage_bonus INTEGER NOT NULL DEFAULT 0,
                    stripe_customer_id TEXT UNIQUE,
                    stripe_subscription_id TEXT UNIQUE,
                    stripe_subscription_item_id TEXT,
                    phone_number TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            desired_columns = [
                "id",
                "name",
                "email",
                "api_key_hash",
                "password_hash",
                "role",
                "is_active",
                "subscription_status",
                "subscription_tier",
                "quota_override",
                "quota_overage_bonus",
                "stripe_customer_id",
                "stripe_subscription_id",
                "stripe_subscription_item_id",
                "phone_number",
                "created_at",
                "updated_at",
            ]
            copy_columns = [col for col in desired_columns if col in existing_columns]
            if copy_columns:
                columns_csv = ", ".join(copy_columns)
                conn.execute(
                    f"INSERT INTO customers ({columns_csv}) SELECT {columns_csv} FROM customers_old"
                )
            conn.execute("DROP TABLE customers_old")

            cursor = conn.execute("PRAGMA table_info(customers)")
            existing_columns = {row["name"] for row in cursor.fetchall()}

        if "password_hash" not in existing_columns:
            conn.execute("ALTER TABLE customers ADD COLUMN password_hash TEXT")

        if "role" not in existing_columns:
            conn.execute(
                "ALTER TABLE customers ADD COLUMN role TEXT NOT NULL DEFAULT 'customer'"
            )

        if "is_active" not in existing_columns:
            conn.execute(
                "ALTER TABLE customers ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"
            )

        if "subscription_tier" not in existing_columns:
            conn.execute(
                "ALTER TABLE customers ADD COLUMN subscription_tier TEXT DEFAULT 'free'"
            )

        if "quota_override" not in existing_columns:
            conn.execute("ALTER TABLE customers ADD COLUMN quota_override INTEGER")

        if "phone_number" not in existing_columns:
            conn.execute("ALTER TABLE customers ADD COLUMN phone_number TEXT")

        if "quota_overage_bonus" not in existing_columns:
            conn.execute(
                "ALTER TABLE customers ADD COLUMN quota_overage_bonus INTEGER NOT NULL DEFAULT 0"
            )


def list_canonical_mappings(
    offset: int = 0, limit: int = 1000
) -> tuple[list[CanonicalMapping], int]:
    """
    List canonical mappings with pagination.

    Returns:
        Tuple of (list of mappings, total count)
    """
    init_db()
    with get_db() as conn:
        _seed_default_canonical_mappings(conn)
        # Get total count
        cursor = conn.execute("SELECT COUNT(*) as count FROM canonical_mappings")
        total = cursor.fetchone()["count"]

        # Get page of results
        cursor = conn.execute(
            """
            SELECT id, source_token, target_token, created_at, updated_at
            FROM canonical_mappings
            ORDER BY source_token COLLATE NOCASE
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        mappings = [
            CanonicalMapping(
                id=row["id"],
                source_token=row["source_token"],
                target_token=row["target_token"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in cursor.fetchall()
        ]
        return mappings, total


def get_plan_by_stripe_price_id(price_id: str) -> Optional[SubscriptionPlan]:
    """Fetch a subscription plan by Stripe price ID."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM subscription_plans WHERE stripe_price_id = ?",
            (price_id,),
        ).fetchone()
    if not row:
        return None
    return SubscriptionPlan(
        id=row["id"],
        name=row["name"],
        description=row["description"] if "description" in row.keys() else None,
        stripe_price_id=row["stripe_price_id"],
        monthly_price_cents=row["monthly_price_cents"],
        annual_price_cents=row["annual_price_cents"],
        monthly_quota=row["monthly_quota"],
        rate_limit_rpm=row["rate_limit_rpm"],
        concurrent_optimization_jobs=row["concurrent_optimization_jobs"],
        batch_size_limit=row["batch_size_limit"],
        optimization_history_retention_days=row["optimization_history_retention_days"],
        telemetry_retention_days=row["telemetry_retention_days"],
        audit_log_retention_days=row["audit_log_retention_days"],
        custom_canonical_mappings_limit=row["custom_canonical_mappings_limit"],
        max_api_keys=row["max_api_keys"],
        features=json.loads(row["features"]) if row["features"] else [],
        is_active=bool(row["is_active"]),
        is_public=bool(row["is_public"]) if "is_public" in row.keys() else True,
        plan_term=row["plan_term"] if "plan_term" in row.keys() else "monthly",
        monthly_discount_percent=(
            row["monthly_discount_percent"]
            if "monthly_discount_percent" in row.keys()
            else 0
        ),
        yearly_discount_percent=(
            row["yearly_discount_percent"]
            if "yearly_discount_percent" in row.keys()
            else 0
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_canonical_mapping(source_token: str, target_token: str) -> CanonicalMapping:
    """
    Create or update a canonical mapping (upsert with case-insensitive source).

    Returns:
        The created or updated mapping
    """
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        # Try to find existing mapping (case-insensitive)
        cursor = conn.execute(
            "SELECT id FROM canonical_mappings WHERE LOWER(source_token) = LOWER(?)",
            (source_token,),
        )
        existing = cursor.fetchone()

        if existing:
            # Update existing
            conn.execute(
                """
                UPDATE canonical_mappings
                SET target_token = ?, updated_at = ?
                WHERE id = ?
                """,
                (target_token, timestamp, existing["id"]),
            )
            mapping_id = existing["id"]
        else:
            # Insert new
            cursor = conn.execute(
                """
                INSERT INTO canonical_mappings (source_token, target_token, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (source_token, target_token, timestamp, timestamp),
            )
            mapping_id = cursor.lastrowid

    # Invalidate cache after write
    _canonical_cache.invalidate_and_reload()

    # Fetch and return the created/updated mapping
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, source_token, target_token, created_at, updated_at
            FROM canonical_mappings
            WHERE id = ?
            """,
            (mapping_id,),
        )
        row = cursor.fetchone()

    return CanonicalMapping(
        id=row["id"],
        source_token=row["source_token"],
        target_token=row["target_token"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def bulk_create_canonical_mappings(
    mappings: list[tuple[str, str]],
) -> list[CanonicalMapping]:
    """
    Bulk create or update canonical mappings.

    Args:
        mappings: List of (source_token, target_token) tuples

    Returns:
        List of created/updated mappings
    """
    init_db()
    created_ids = []
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        for source_token, target_token in mappings:
            # Check for existing (case-insensitive)
            cursor = conn.execute(
                "SELECT id FROM canonical_mappings WHERE LOWER(source_token) = LOWER(?)",
                (source_token,),
            )
            existing = cursor.fetchone()

            if existing:
                # Update
                conn.execute(
                    """
                    UPDATE canonical_mappings
                    SET target_token = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (target_token, timestamp, existing["id"]),
                )
                created_ids.append(existing["id"])
            else:
                # Insert
                cursor = conn.execute(
                    """
                    INSERT INTO canonical_mappings (source_token, target_token, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_token, target_token, timestamp, timestamp),
                )
                created_ids.append(cursor.lastrowid)

    # Invalidate cache after bulk write
    _canonical_cache.invalidate_and_reload()

    # Fetch all created/updated mappings
    with get_db() as conn:
        placeholders = ",".join("?" * len(created_ids))
        cursor = conn.execute(
            f"""
            SELECT id, source_token, target_token, created_at, updated_at
            FROM canonical_mappings
            WHERE id IN ({placeholders})
            """,
            created_ids,
        )

        return [
            CanonicalMapping(
                id=row["id"],
                source_token=row["source_token"],
                target_token=row["target_token"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in cursor.fetchall()
        ]


def update_canonical_mapping(
    mapping_id: int, source_token: str, target_token: str
) -> Optional[CanonicalMapping]:
    """
    Update an existing canonical mapping.

    Returns:
        The updated mapping, or None if not found
    """
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE canonical_mappings
            SET source_token = ?, target_token = ?, updated_at = ?
            WHERE id = ?
            """,
            (source_token, target_token, timestamp, mapping_id),
        )

        if cursor.rowcount == 0:
            return None

    # Invalidate cache after write
    _canonical_cache.invalidate_and_reload()

    # Fetch updated mapping
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, source_token, target_token, created_at, updated_at
            FROM canonical_mappings
            WHERE id = ?
            """,
            (mapping_id,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return CanonicalMapping(
        id=row["id"],
        source_token=row["source_token"],
        target_token=row["target_token"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def delete_canonical_mappings(mapping_ids: list[int]) -> int:
    """
    Delete canonical mappings by IDs.

    Returns:
        Number of mappings deleted
    """
    init_db()
    if not mapping_ids:
        return 0

    with get_db() as conn:
        placeholders = ",".join("?" * len(mapping_ids))
        cursor = conn.execute(
            f"DELETE FROM canonical_mappings WHERE id IN ({placeholders})",
            mapping_ids,
        )
        deleted_count = cursor.rowcount

    # Invalidate cache after delete
    if deleted_count > 0:
        _canonical_cache.invalidate_and_reload()

    return deleted_count


# ============================================================================
# Batch Jobs
# ============================================================================


def create_batch_job(
    name: str,
    total_items: int,
    customer_id: str,
    status: str = "processing",
    processed_items: int = 0,
) -> BatchJobRecord:
    """Create a batch job record for tracking synchronous batches."""
    init_db()
    job_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO batch_jobs (
                id,
                name,
                customer_id,
                status,
                total_items,
                processed_items,
                total_savings_percentage,
                processing_time_ms,
                created_at,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                name,
                customer_id,
                status,
                total_items,
                processed_items,
                None,
                None,
                timestamp,
                None,
            ),
        )
    return BatchJobRecord(
        id=job_id,
        name=name,
        status=status,
        total_items=total_items,
        processed_items=processed_items,
        total_savings_percentage=None,
        processing_time_ms=None,
        created_at=timestamp,
        completed_at=None,
    )


def update_batch_job(
    job_id: str, *, customer_id: Optional[str] = None, **updates: Any
) -> Optional[BatchJobRecord]:
    """Update a batch job and return the updated record."""
    if not updates:
        return get_batch_job(job_id, customer_id=customer_id)

    init_db()
    columns = []
    values: list[Any] = []
    for key, value in updates.items():
        columns.append(f"{key} = ?")
        values.append(value)
    values.append(job_id)
    if customer_id:
        values.append(customer_id)

    with get_db() as conn:
        where_clause = "id = ?"
        if customer_id:
            where_clause += " AND customer_id = ?"
        conn.execute(
            f"UPDATE batch_jobs SET {', '.join(columns)} WHERE {where_clause}",
            values,
        )

    return get_batch_job(job_id, customer_id=customer_id)


def get_batch_job(
    job_id: str, *, customer_id: Optional[str] = None
) -> Optional[BatchJobRecord]:
    init_db()
    with get_db() as conn:
        where_clause = "id = ?"
        params: List[Any] = [job_id]
        if customer_id:
            where_clause += " AND customer_id = ?"
            params.append(customer_id)
        cursor = conn.execute(
            """
            SELECT
                id,
                name,
                status,
                total_items,
                processed_items,
                total_savings_percentage,
                processing_time_ms,
                created_at,
                completed_at
            FROM batch_jobs
            WHERE """
            + where_clause
            + """
            """,
            tuple(params),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return BatchJobRecord(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        total_items=row["total_items"],
        processed_items=row["processed_items"],
        total_savings_percentage=row["total_savings_percentage"],
        processing_time_ms=row["processing_time_ms"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def list_batch_jobs(
    limit: int = 20, *, customer_id: Optional[str] = None
) -> list[BatchJobRecord]:
    init_db()
    with get_db() as conn:
        params: List[Any] = []
        where_clause = ""
        if customer_id:
            where_clause = " WHERE customer_id = ?"
            params.append(customer_id)
        params.append(limit)
        cursor = conn.execute(
            """
            SELECT
                id,
                name,
                status,
                total_items,
                processed_items,
                total_savings_percentage,
                processing_time_ms,
                created_at,
                completed_at
            FROM batch_jobs
            """
            + where_clause
            + """
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = cursor.fetchall()

    return [
        BatchJobRecord(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            total_items=row["total_items"],
            processed_items=row["processed_items"],
            total_savings_percentage=row["total_savings_percentage"],
            processing_time_ms=row["processing_time_ms"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )
        for row in rows
    ]


def create_llm_optimization_job(
    *, customer_id: Optional[str], request_payload: Dict[str, Any]
) -> LLMOptimizationJobRecord:
    init_db()
    job_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    payload_json = json.dumps(request_payload, ensure_ascii=False)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO llm_optimization_jobs (
                id,
                customer_id,
                status,
                request_payload,
                result_payload,
                error_message,
                attempts,
                created_at,
                updated_at,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                customer_id,
                "queued",
                payload_json,
                None,
                None,
                0,
                timestamp,
                timestamp,
                None,
            ),
        )
    return LLMOptimizationJobRecord(
        id=job_id,
        customer_id=customer_id,
        status="queued",
        request_payload=request_payload,
        result_payload=None,
        error_message=None,
        attempts=0,
        created_at=timestamp,
        updated_at=timestamp,
        completed_at=None,
    )


def get_llm_optimization_job(
    job_id: str, *, customer_id: Optional[str] = None
) -> Optional[LLMOptimizationJobRecord]:
    init_db()
    with get_db() as conn:
        where_clause = "id = ?"
        params: List[Any] = [job_id]
        if customer_id is not None:
            where_clause += " AND customer_id = ?"
            params.append(customer_id)
        cursor = conn.execute(
            """
            SELECT
                id,
                customer_id,
                status,
                request_payload,
                result_payload,
                error_message,
                attempts,
                created_at,
                updated_at,
                completed_at
            FROM llm_optimization_jobs
            WHERE """
            + where_clause,
            tuple(params),
        )
        row = cursor.fetchone()

    if not row:
        return None

    request_payload: Dict[str, Any] = {}
    result_payload: Optional[Dict[str, Any]] = None
    try:
        parsed_request = json.loads(row["request_payload"])
        if isinstance(parsed_request, dict):
            request_payload = parsed_request
    except (TypeError, ValueError, json.JSONDecodeError):
        request_payload = {}

    if row["result_payload"]:
        try:
            parsed_result = json.loads(row["result_payload"])
            if isinstance(parsed_result, dict):
                result_payload = parsed_result
        except (TypeError, ValueError, json.JSONDecodeError):
            result_payload = None

    return LLMOptimizationJobRecord(
        id=row["id"],
        customer_id=row["customer_id"],
        status=row["status"],
        request_payload=request_payload,
        result_payload=result_payload,
        error_message=row["error_message"],
        attempts=int(row["attempts"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def update_llm_optimization_job(
    job_id: str,
    *,
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
    result_payload: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    attempts: Optional[int] = None,
    completed_at: Optional[str] = None,
) -> Optional[LLMOptimizationJobRecord]:
    init_db()
    updates: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if status is not None:
        updates["status"] = status
    if result_payload is not None:
        updates["result_payload"] = json.dumps(result_payload, ensure_ascii=False)
    if error_message is not None:
        updates["error_message"] = error_message
    if attempts is not None:
        updates["attempts"] = int(attempts)
    if completed_at is not None:
        updates["completed_at"] = completed_at

    if len(updates) == 1:  # only updated_at
        return get_llm_optimization_job(job_id, customer_id=customer_id)

    columns = []
    values: List[Any] = []
    for key, value in updates.items():
        columns.append(f"{key} = ?")
        values.append(value)

    values.append(job_id)
    where_clause = "id = ?"
    if customer_id is not None:
        where_clause += " AND customer_id = ?"
        values.append(customer_id)

    with get_db() as conn:
        conn.execute(
            f"UPDATE llm_optimization_jobs SET {', '.join(columns)} WHERE {where_clause}",
            tuple(values),
        )

    return get_llm_optimization_job(job_id, customer_id=customer_id)


def reap_stale_llm_optimization_jobs(
    *,
    stale_processing_before: Optional[str] = None,
    stale_queued_before: Optional[str] = None,
    processing_error_message: Optional[str] = None,
    queued_error_message: Optional[str] = None,
) -> Dict[str, int]:
    """Mark stale async LLM jobs as failed and return affected row counts."""
    init_db()

    now_iso = datetime.now(timezone.utc).isoformat()
    processing_message = (
        processing_error_message
        or "Async optimization timed out while processing; job reaped as stale"
    )
    queued_message = (
        queued_error_message
        or "Async optimization expired in queue; job reaped as stale"
    )

    processing_reaped = 0
    queued_reaped = 0

    with get_db() as conn:
        if stale_processing_before:
            cursor = conn.execute(
                """
                UPDATE llm_optimization_jobs
                SET
                    status = 'failed',
                    error_message = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE status = 'processing' AND updated_at < ?
                """,
                (
                    processing_message,
                    now_iso,
                    now_iso,
                    stale_processing_before,
                ),
            )
            processing_reaped = int(cursor.rowcount or 0)

        if stale_queued_before:
            cursor = conn.execute(
                """
                UPDATE llm_optimization_jobs
                SET
                    status = 'failed',
                    error_message = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE status = 'queued' AND created_at < ?
                """,
                (
                    queued_message,
                    now_iso,
                    now_iso,
                    stale_queued_before,
                ),
            )
            queued_reaped = int(cursor.rowcount or 0)

    return {
        "processing_reaped": processing_reaped,
        "queued_reaped": queued_reaped,
        "total_reaped": processing_reaped + queued_reaped,
    }


# ============================================================================
# Performance Telemetry System
# ============================================================================


@dataclass
class TelemetryPassMetric:
    """Metrics for a single optimization pass."""

    pass_name: str
    pass_order: int
    duration_ms: float
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    reduction_percent: float
    expected_utility: Optional[float] = None
    actual_utility: Optional[float] = None
    pass_skipped_reason: Optional[str] = None
    content_profile: Optional[str] = None
    optimization_mode: Optional[str] = None
    token_bin: Optional[str] = None


@dataclass
class TelemetryRecord:
    """Complete telemetry record for an optimization run."""

    optimization_id: str
    passes: List[TelemetryPassMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class HistoryRecord:
    """Optimization history record for batch writing."""

    id: str
    customer_id: Optional[str]
    created_at: str
    updated_at: str
    mode: str
    raw_prompt: str
    optimized_prompt: str
    raw_tokens: int
    optimized_tokens: int
    processing_time_ms: float
    estimated_cost_before: float
    estimated_cost_after: float
    estimated_cost_saved: float
    compression_percentage: float
    semantic_similarity: Optional[float]
    techniques_applied: list[str]


class HistoryBatchWriter:
    """
    Async batch writer for optimization history.
    Collects history records in memory and periodically flushes to database.
    """

    def __init__(self, batch_size: int = 50, flush_interval_seconds: float = 5.0):
        self._queue: Queue[HistoryRecord] = Queue()
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._worker_thread: Optional[Thread] = None
        self._running = False
        self._lock = Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = Thread(
                target=self._worker, daemon=True, name="HistoryWriter"
            )
            self._worker_thread.start()
            logging.info("History batch writer started")

    def stop(self) -> None:
        """Stop the background worker and flush remaining records."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Add sentinel to wake up worker
        self._queue.put(None)  # type: ignore

        if self._worker_thread:
            self._worker_thread.join(timeout=10.0)
            logging.info("History batch writer stopped")

    def submit(self, record: HistoryRecord) -> None:
        """Submit a history record for async processing."""
        if not self._running:
            logging.warning("History writer not running, discarding record")
            return

        try:
            self._queue.put_nowait(record)
        except Exception as e:
            logging.warning(f"Failed to queue history record: {e}")

    def _worker(self) -> None:
        """Background worker that batches and writes history records."""
        batch: List[HistoryRecord] = []
        last_flush_time = datetime.now(timezone.utc).timestamp()

        while self._running:
            try:
                # Wait for records with timeout to allow periodic flushing
                timeout = max(
                    0.1,
                    self._flush_interval
                    - (datetime.now(timezone.utc).timestamp() - last_flush_time),
                )
                record = self._queue.get(timeout=timeout)

                # Sentinel value to stop worker
                if record is None:
                    break

                batch.append(record)

                # Flush if batch is full or enough time has passed
                current_time = datetime.now(timezone.utc).timestamp()
                should_flush_size = len(batch) >= self._batch_size
                should_flush_time = (
                    current_time - last_flush_time
                ) >= self._flush_interval

                if should_flush_size or should_flush_time:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush_time = current_time

            except Empty:
                # Timeout expired, flush any pending records
                if batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush_time = datetime.now(timezone.utc).timestamp()
            except Exception as e:
                logging.error(f"History worker error: {e}", exc_info=True)

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[HistoryRecord]) -> None:
        """Write a batch of history records to the database using executemany."""
        if not batch:
            return

        try:
            init_db()
            with get_db() as conn:
                # Prepare batch data for executemany
                batch_data = [
                    (
                        record.id,
                        record.customer_id,
                        record.created_at,
                        record.updated_at,
                        record.mode,
                        record.raw_prompt,
                        record.optimized_prompt,
                        record.raw_tokens,
                        record.optimized_tokens,
                        record.processing_time_ms,
                        record.estimated_cost_before,
                        record.estimated_cost_after,
                        record.estimated_cost_saved,
                        record.compression_percentage,
                        record.semantic_similarity,
                        json.dumps(record.techniques_applied),
                    )
                    for record in batch
                ]

                # Use executemany for efficient batch insert
                conn.executemany(
                    """
                    INSERT INTO optimization_history (
                        id,
                        customer_id,
                        created_at,
                        updated_at,
                        mode,
                        raw_prompt,
                        optimized_prompt,
                        raw_tokens,
                        optimized_tokens,
                        processing_time_ms,
                        estimated_cost_before,
                        estimated_cost_after,
                        estimated_cost_saved,
                        compression_percentage,
                        semantic_similarity,
                        techniques_applied
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_data,
                )

            logging.debug(f"Flushed {len(batch)} history records to database")
        except Exception as e:
            logging.error(f"Failed to flush history batch: {e}", exc_info=True)


# Global history writer instance
_history_writer: Optional[HistoryBatchWriter] = None
_history_writer_lock = Lock()


def get_history_writer() -> HistoryBatchWriter:
    """Get or create the global history writer instance."""
    global _history_writer

    with _history_writer_lock:
        if _history_writer is None:
            _history_writer = HistoryBatchWriter(
                batch_size=_HISTORY_BATCH_SIZE,
                flush_interval_seconds=_HISTORY_FLUSH_INTERVAL_SECONDS,
            )
            _history_writer.start()

        return _history_writer


class TelemetryBatchWriter:
    """
    Async batch writer for performance telemetry.
    Collects telemetry records in memory and periodically flushes to database.
    """

    def __init__(self, batch_size: int = 50, flush_interval_seconds: float = 5.0):
        self._queue: Queue[TelemetryRecord] = Queue()
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._worker_thread: Optional[Thread] = None
        self._running = False
        self._lock = Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = Thread(
                target=self._worker, daemon=True, name="TelemetryWriter"
            )
            self._worker_thread.start()
            logging.info("Telemetry batch writer started")

    def stop(self) -> None:
        """Stop the background worker and flush remaining records."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Add sentinel to wake up worker
        self._queue.put(None)  # type: ignore

        if self._worker_thread:
            self._worker_thread.join(timeout=10.0)
            logging.info("Telemetry batch writer stopped")

    def submit(self, record: TelemetryRecord) -> None:
        """Submit a telemetry record for async processing."""
        if not self._running:
            logging.warning("Telemetry writer not running, discarding record")
            return

        try:
            self._queue.put_nowait(record)
        except Exception as e:
            logging.warning(f"Failed to queue telemetry record: {e}")

    def _worker(self) -> None:
        """Background worker that batches and writes telemetry records."""
        batch: List[TelemetryRecord] = []
        last_flush_time = datetime.now(timezone.utc).timestamp()

        while self._running:
            try:
                # Wait for records with timeout to allow periodic flushing
                timeout = max(
                    0.1,
                    self._flush_interval
                    - (datetime.now(timezone.utc).timestamp() - last_flush_time),
                )
                record = self._queue.get(timeout=timeout)

                # Sentinel value to stop worker
                if record is None:
                    break

                batch.append(record)

                # Flush if batch is full or enough time has passed
                current_time = datetime.now(timezone.utc).timestamp()
                should_flush_size = len(batch) >= self._batch_size
                should_flush_time = (
                    current_time - last_flush_time
                ) >= self._flush_interval

                if should_flush_size or should_flush_time:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush_time = current_time

            except Empty:
                # Timeout expired, flush any pending records
                if batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush_time = datetime.now(timezone.utc).timestamp()
            except Exception as e:
                logging.error(f"Telemetry worker error: {e}", exc_info=True)

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[TelemetryRecord]) -> None:
        """Write a batch of telemetry records to the database using executemany."""
        if not batch:
            return

        try:
            init_db()
            with get_db() as conn:
                # Flatten all passes from all records into a single batch for executemany
                batch_data = []
                for record in batch:
                    for pass_metric in record.passes:
                        batch_data.append(
                            (
                                record.optimization_id,
                                pass_metric.pass_name,
                                pass_metric.pass_order,
                                pass_metric.duration_ms,
                                pass_metric.tokens_before,
                                pass_metric.tokens_after,
                                pass_metric.tokens_saved,
                                pass_metric.reduction_percent,
                                pass_metric.expected_utility,
                                pass_metric.actual_utility,
                                pass_metric.pass_skipped_reason,
                                pass_metric.content_profile,
                                pass_metric.optimization_mode,
                                pass_metric.token_bin,
                                record.created_at,
                            )
                        )

                # Use executemany for efficient batch insert
                if batch_data:
                    conn.executemany(
                        """
                        INSERT INTO performance_telemetry (
                            optimization_id,
                            pass_name,
                            pass_order,
                            duration_ms,
                            tokens_before,
                            tokens_after,
                            tokens_saved,
                            reduction_percent,
                            expected_utility,
                            actual_utility,
                            pass_skipped_reason,
                            content_profile,
                            optimization_mode,
                            token_bin,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch_data,
                    )

            total_passes = sum(len(record.passes) for record in batch)
            logging.debug(
                f"Flushed {len(batch)} telemetry records ({total_passes} passes) to database"
            )
        except Exception as e:
            logging.error(f"Failed to flush telemetry batch: {e}", exc_info=True)


# Global telemetry writer instance
_telemetry_writer: Optional[TelemetryBatchWriter] = None
_telemetry_lock = Lock()


def get_telemetry_writer() -> TelemetryBatchWriter:
    """Get or create the global telemetry writer instance."""
    global _telemetry_writer

    with _telemetry_lock:
        if _telemetry_writer is None:
            _telemetry_writer = TelemetryBatchWriter(
                batch_size=_TELEMETRY_BATCH_SIZE,
                flush_interval_seconds=_TELEMETRY_FLUSH_INTERVAL_SECONDS,
            )
            _telemetry_writer.start()

        return _telemetry_writer


def submit_telemetry(record: TelemetryRecord) -> None:
    """Submit a telemetry record for async processing."""
    if not is_telemetry_enabled():
        return

    writer = get_telemetry_writer()
    writer.submit(record)


def list_recent_telemetry(
    limit: int = 100, *, customer_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Return recent telemetry passes for analytics dashboards."""
    init_db()
    safe_limit = max(1, min(limit, 1000))
    if not customer_id:
        return []
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT
                t.optimization_id,
                t.pass_name,
                t.pass_order,
                t.duration_ms,
                t.tokens_before,
                t.tokens_after,
                t.tokens_saved,
                t.reduction_percent,
                t.created_at
            FROM performance_telemetry t
            INNER JOIN optimization_history h
                ON h.id = t.optimization_id
            WHERE h.customer_id = ?
            ORDER BY t.created_at DESC
            LIMIT ?
            """,
            (customer_id, safe_limit),
        )
        rows = cursor.fetchall()

    return [
        {
            "optimization_id": row["optimization_id"],
            "pass_name": row["pass_name"],
            "pass_order": row["pass_order"],
            "duration_ms": row["duration_ms"],
            "tokens_before": row["tokens_before"],
            "tokens_after": row["tokens_after"],
            "tokens_saved": row["tokens_saved"],
            "reduction_percent": row["reduction_percent"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


# ========== Customer & Usage Management ==========


@dataclass
class Customer:
    id: str
    created_at: str
    updated_at: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    api_key_hash: Optional[str] = None
    subscription_status: str = "inactive"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_subscription_item_id: Optional[str] = None
    password_hash: Optional[str] = None
    role: str = "customer"
    is_active: bool = True
    subscription_tier: str = "free"
    quota_override: Optional[int] = None
    quota_overage_bonus: int = 0
    api_key_id: Optional[str] = None


@dataclass
class UsageRecord:
    customer_id: str
    period_start: str
    period_end: str
    calls_used: int


@dataclass
class UsageBreakdown:
    source: str
    calls_used: int
    api_key_id: Optional[str] = None
    api_key_name: Optional[str] = None


def get_customer_by_api_key_hash(key_hash: str) -> Optional[Customer]:
    """Retrieve customer details by their API key hash.

    Checks both:
    1. Legacy api_key_hash column on customers table (for backward compatibility)
    2. New api_keys table for multiple API keys per customer
    """
    init_db()
    with get_db() as conn:
        # First, check the new api_keys table
        api_key_row = conn.execute(
            "SELECT id, customer_id FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        ).fetchone()

        if api_key_row:
            customer_id = api_key_row["customer_id"]
            customer_row = conn.execute(
                "SELECT * FROM customers WHERE id = ?",
                (customer_id,),
            ).fetchone()
            if customer_row:
                # Update last_used_at for the API key
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                    (datetime.now(timezone.utc).isoformat(), key_hash),
                )
                customer = Customer(**dict(customer_row))
                customer.api_key_id = api_key_row["id"]
                return customer

        # Fallback: check legacy api_key_hash on customers table
        row = conn.execute(
            "SELECT * FROM customers WHERE api_key_hash = ?",
            (key_hash,),
        ).fetchone()

    if not row:
        return None
    return Customer(**dict(row))


def update_customer_subscription(
    stripe_customer_id: str,
    status: str,
    tier: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    stripe_subscription_item_id: Optional[str] = None,
) -> bool:
    """Update subscription status and IDs based on Stripe webhooks."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        fields = {"subscription_status": status, "updated_at": timestamp}
        if tier:
            fields["subscription_tier"] = tier
        if stripe_subscription_id:
            fields["stripe_subscription_id"] = stripe_subscription_id
        if stripe_subscription_item_id:
            fields["stripe_subscription_item_id"] = stripe_subscription_item_id

        columns = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values())
        values.append(stripe_customer_id)

        conn.execute(
            f"UPDATE customers SET {columns} WHERE stripe_customer_id = ?", values
        )
        return conn.total_changes > 0


def increment_usage(
    customer_id: str,
    period_start: str,
    period_end: str,
    count: int = 1,
    api_key_id: Optional[str] = None,
) -> int:
    """Increment API call usage for a customer in a specific period.

    Args:
        customer_id: Customer ID
        period_start: Start of the billing period (ISO format)
        period_end: End of the billing period (ISO format)
        count: Number of calls to increment
        api_key_id: Optional API key ID (None for UI requests)
    """
    init_db()
    # Use empty string for NULL api_key_id in the PK since SQLite handles NULL specially in UNIQUE
    # Or we use COALESCE approach. Let's use a sentinel value '_ui_' for UI requests.
    source_id = api_key_id if api_key_id else "_ui_"

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO usage (customer_id, period_start, period_end, calls_used, api_key_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(customer_id, period_start, api_key_id) DO UPDATE SET
                calls_used = calls_used + excluded.calls_used,
                period_end = excluded.period_end
            """,
            (customer_id, period_start, period_end, count, source_id),
        )
        # Get total usage across all sources for this period
        row = conn.execute(
            "SELECT SUM(calls_used) as total FROM usage WHERE customer_id = ? AND period_start = ?",
            (customer_id, period_start),
        ).fetchone()
        return row["total"] if row and row["total"] else 0


def get_usage(customer_id: str, period_start: str) -> Optional[UsageRecord]:
    """Get aggregated usage record for a customer across all sources."""
    init_db()
    with get_db() as conn:
        # Aggregate across all API keys and UI
        row = conn.execute(
            """
            SELECT customer_id, period_start, MAX(period_end) as period_end, SUM(calls_used) as calls_used
            FROM usage
            WHERE customer_id = ? AND period_start = ?
            GROUP BY customer_id, period_start
            """,
            (customer_id, period_start),
        ).fetchone()
    if not row:
        return None
    return UsageRecord(**dict(row))


def list_usage_breakdown(customer_id: str, period_start: str) -> list[UsageBreakdown]:
    """Return per-source usage for a given period."""
    init_db()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT u.api_key_id, u.calls_used, k.name as api_key_name
            FROM usage u
            LEFT JOIN api_keys k ON u.api_key_id = k.id
            WHERE u.customer_id = ? AND u.period_start = ?
            ORDER BY u.calls_used DESC
            """,
            (customer_id, period_start),
        ).fetchall()

    breakdown: list[UsageBreakdown] = []
    for row in rows:
        api_key_id = row["api_key_id"]
        if api_key_id in (None, "_ui_"):
            source = "ui"
            api_key_id = None
        else:
            source = f"api_key_{api_key_id}"
        breakdown.append(
            UsageBreakdown(
                source=source,
                calls_used=row["calls_used"],
                api_key_id=api_key_id,
                api_key_name=row["api_key_name"],
            )
        )

    return breakdown


def list_usage_history(customer_id: str, limit: int = 6) -> list[UsageRecord]:
    """Return usage totals for recent billing periods."""
    init_db()
    safe_limit = max(1, min(limit, 24))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT customer_id, period_start, MAX(period_end) as period_end, SUM(calls_used) as calls_used
            FROM usage
            WHERE customer_id = ?
            GROUP BY customer_id, period_start
            ORDER BY period_start DESC
            LIMIT ?
            """,
            (customer_id, safe_limit),
        ).fetchall()

    return [UsageRecord(**dict(row)) for row in rows]


def get_customer_by_stripe_id(stripe_customer_id: str) -> Optional[Customer]:
    """Retrieve customer by their Stripe customer ID."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE stripe_customer_id = ?",
            (stripe_customer_id,),
        ).fetchone()
    if not row:
        return None
    return Customer(**dict(row))


def create_customer(
    name: str,
    email: str,
    api_key_hash: Optional[str] = None,
    stripe_customer_id: Optional[str] = None,
    id: Optional[str] = None,
) -> Customer:
    """Create a new customer record."""
    init_db()
    customer_id = id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Get default trial plan
    subscription_tier = "free"  # Fallback
    with get_db() as conn:
        plan_row = conn.execute(
            "SELECT id FROM subscription_plans WHERE name = 'Trial Plan'"
        ).fetchone()
        if plan_row:
            subscription_tier = plan_row["id"]

        conn.execute(
            """
            INSERT INTO customers (
                id, name, email, api_key_hash, stripe_customer_id, subscription_tier, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                name,
                email,
                api_key_hash,
                stripe_customer_id,
                subscription_tier,
                timestamp,
                timestamp,
            ),
        )
    return get_customer_by_id(customer_id)


def get_customer_by_id(customer_id: str) -> Optional[Customer]:
    """Retrieve customer by their internal ID."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE id = ?",
            (customer_id,),
        ).fetchone()
    if not row:
        return None
    return Customer(**dict(row))


def update_customer_api_key(customer_id: str, api_key_hash: str) -> bool:
    """Update a customer's API key hash."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE customers
            SET api_key_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (api_key_hash, timestamp, customer_id),
        )
        return conn.total_changes > 0


def get_customer_by_email(email: str) -> Optional[Customer]:
    """Retrieve customer by email address."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE email = ?",
            (email,),
        ).fetchone()
    if not row:
        return None
    return Customer(**dict(row))


def update_customer_password(customer_id: str, password_hash: str) -> bool:
    """Update a customer's password hash."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE customers
            SET password_hash = ?, updated_at = ?
            WHERE id = ?
            """,
            (password_hash, timestamp, customer_id),
        )
        return conn.total_changes > 0


def list_all_customers(offset: int = 0, limit: int = 50) -> tuple[List[Customer], int]:
    """List all customers with pagination."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) as count FROM customers")
        total = cursor.fetchone()["count"]

        rows = conn.execute(
            "SELECT * FROM customers ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    return [Customer(**dict(row)) for row in rows], total


def update_customer(customer_id: str, **updates) -> Optional[Customer]:
    """Generic update for customer fields."""
    if not updates:
        return get_customer_by_id(customer_id)

    init_db()
    valid_fields = {
        "name",
        "email",
        "phone_number",
        "password_hash",
        "role",
        "is_active",
        "subscription_status",
        "subscription_tier",
        "quota_override",
        "quota_overage_bonus",
        "api_key_hash",
        "stripe_customer_id",
        "stripe_subscription_id",
        "stripe_subscription_item_id",
    }

    filtered_updates = {k: v for k, v in updates.items() if k in valid_fields}
    if not filtered_updates:
        return get_customer_by_id(customer_id)

    timestamp = datetime.now(timezone.utc).isoformat()
    filtered_updates["updated_at"] = timestamp

    columns = ", ".join(f"{k} = ?" for k in filtered_updates.keys())
    values = list(filtered_updates.values())
    values.append(customer_id)

    with get_db() as conn:
        conn.execute(f"UPDATE customers SET {columns} WHERE id = ?", values)

    return get_customer_by_id(customer_id)


def disable_customer(customer_id: str) -> bool:
    """Soft delete/disable a customer."""
    return bool(update_customer(customer_id, is_active=False))


def _migrate_batch_jobs_table() -> None:
    """Add customer_id to batch_jobs table."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(batch_jobs)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        if "customer_id" not in existing_columns:
            conn.execute(
                "ALTER TABLE batch_jobs ADD COLUMN customer_id TEXT REFERENCES customers(id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_batch_jobs_customer_id ON batch_jobs(customer_id)"
            )


def _migrate_settings_table() -> None:
    """Migrate settings table to support per-customer keys (PK change)."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(settings)")
        rows = cursor.fetchall()
        columns = {row["name"] for row in rows}

        # Check if customer_id is part of PK or if we need to recreate
        # Simple check: if customer_id is not in columns, or if we want to ensure PK structure
        # Let's check if customer_id exists first
        needs_migration = "customer_id" not in columns

        if not needs_migration:
            # Check if PK is composite. This is harder to check quickly, but if we just added column via ALTER,
            # it's NOT in PK. So if we haven't done full migration, we assume we need to recreate if it was just
            # ALTERed or if it's old schema.
            # Actually, simply dropping and recreating if schema doesn't match desired definition is safer for dev.
            # But preserving data is needed.
            # Let's look at primary keys.
            pk_cols = [row["name"] for row in rows if row["pk"] > 0]
            if "customer_id" in pk_cols and "key" in pk_cols:
                return

        # Recreate table
        conn.execute("ALTER TABLE settings RENAME TO settings_old")

        conn.execute("""
            CREATE TABLE settings (
                key TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (customer_id, key)
            )
        """)

        # Migrate old global settings to a default customer (e.g., 'system' or keep in admin_settings?)
        # For now, let's just migrate them as 'global_legacy' or similar, or specific admin if we knew ID.
        # Or just drop them if they were dev data.
        # Let's preserve as "admin" if possible, or just ignore. "settings" was mainly used for LLM profiles?
        # If LLM profiles were global, we might want to move them to admin_settings.
        # But set_llm_profiles writes to `settings`.
        # Let's just migrate with customer_id='system' for now.
        conn.execute("""
            INSERT INTO settings (key, customer_id, value, updated_at)
            SELECT key, 'system', value, updated_at FROM settings_old
        """)

        conn.execute("DROP TABLE settings_old")


def _migrate_usage_table() -> None:
    """Migrate usage table to support per-key tracking (requires recreate for PK change)."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(usage)")
        columns = {row["name"] for row in cursor.fetchall()}

        # Check if we need migration (if api_key_id missing)
        if "api_key_id" in columns:
            return

        # Migration needed: recreate table
        conn.execute("ALTER TABLE usage RENAME TO usage_old")

        conn.execute("""
            CREATE TABLE usage (
                customer_id TEXT NOT NULL,
                api_key_id TEXT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                calls_used INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (customer_id, period_start, api_key_id),
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        """)

        # Migrate data (aggregate old usage -> null api_key_id)
        conn.execute("""
            INSERT INTO usage (customer_id, period_start, period_end, calls_used, api_key_id)
            SELECT customer_id, period_start, period_end, calls_used, NULL
            FROM usage_old
        """)

        conn.execute("DROP TABLE usage_old")


# API Key Management Functions


@dataclass
class ApiKey:
    id: str
    customer_id: str
    key_hash: str
    name: str
    created_at: str
    last_used_at: Optional[str]
    is_active: bool


def create_api_key(customer_id: str, name: str, key_hash: str) -> ApiKey:
    """Create a new API Key record."""
    init_db()
    key_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO api_keys (id, customer_id, key_hash, name, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key_id, customer_id, key_hash, name, timestamp, True),
        )

    return ApiKey(
        id=key_id,
        customer_id=customer_id,
        key_hash=key_hash,
        name=name,
        created_at=timestamp,
        last_used_at=None,
        is_active=True,
    )


def list_api_keys(customer_id: str) -> List[ApiKey]:
    """List all API keys for a customer."""
    init_db()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE customer_id = ? ORDER BY created_at DESC",
            (customer_id,),
        ).fetchall()

    return [ApiKey(**dict(row)) for row in rows]


def get_api_key_by_hash(key_hash: str) -> Optional[ApiKey]:
    """Get API key by hash."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()

    if not row:
        return None
    return ApiKey(**dict(row))


def update_api_key_usage(key_hash: str) -> None:
    """Update last_used_at for an API key."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
            (timestamp, key_hash),
        )


def delete_api_key(key_id: str, customer_id: str) -> bool:
    """Delete (or disable) an API key."""
    # Hard delete for now, or soft delete? Let's just delete row or set inactive.
    # Task says "max 10 active", so maybe delete is better for UX, or soft delete ("revoked").
    # Let's delete for simplicity as per common pattern, or soft delete if we want history.
    # I'll modify to hard delete for now to free up slots.
    with get_db() as conn:
        conn.execute(
            "DELETE FROM api_keys WHERE id = ? AND customer_id = ?",
            (key_id, customer_id),
        )
        return conn.total_changes > 0


@dataclass
class UserCanonicalMapping:
    id: int
    customer_id: str
    source_token: str
    target_token: str
    created_at: str
    updated_at: str


def create_user_canonical_mapping(
    customer_id: str, source_token: str, target_token: str
) -> UserCanonicalMapping:
    """Create or update a user-specific canonical mapping."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized_source = source_token.strip().lower()

    with get_db() as conn:
        # Upsert logic for user mappings
        cursor = conn.execute(
            """
            INSERT INTO user_canonical_mappings (customer_id, source_token, target_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(customer_id, source_token) DO UPDATE SET
                target_token = excluded.target_token,
                updated_at = excluded.updated_at
            RETURNING id, customer_id, source_token, target_token, created_at, updated_at
            """,
            (customer_id, normalized_source, target_token, timestamp, timestamp),
        )
        row = cursor.fetchone()

    return UserCanonicalMapping(
        id=row["id"],
        customer_id=row["customer_id"],
        source_token=row["source_token"],
        target_token=row["target_token"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_user_canonical_mappings(customer_id: str) -> List[UserCanonicalMapping]:
    """List all canonical mappings for a customer."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, customer_id, source_token, target_token, created_at, updated_at
            FROM user_canonical_mappings
            WHERE customer_id = ?
            ORDER BY created_at DESC
            """,
            (customer_id,),
        )
        return [
            UserCanonicalMapping(
                id=row["id"],
                customer_id=row["customer_id"],
                source_token=row["source_token"],
                target_token=row["target_token"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in cursor.fetchall()
        ]


def delete_user_canonical_mapping(customer_id: str, mapping_id: int) -> bool:
    """Delete a user canonical mapping."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM user_canonical_mappings WHERE id = ? AND customer_id = ?",
            (mapping_id, customer_id),
        )
        return cursor.rowcount > 0


def list_disabled_ootb_mappings(customer_id: str) -> List[str]:
    init_db()
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT mapping_source_token FROM user_ootb_mapping_disabled WHERE customer_id = ?",
                (customer_id,),
            )
            return [row["mapping_source_token"].lower() for row in cursor.fetchall()]
    except Exception:
        return []


def get_combined_canonical_mappings(
    customer_id: Optional[str] = None,
) -> Dict[str, str]:
    """Get combined mappings (Global + User). User overrides global."""
    # 1. Get global
    global_cache = get_canonical_mappings_cache().get_all()
    mappings = dict(global_cache)

    if not customer_id:
        return mappings

    disabled = set(list_disabled_ootb_mappings(customer_id))
    for token in disabled:
        if token in mappings:
            del mappings[token]

    # 2. Get user mappings (normalized keys so overrides work case-insensitively)
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT source_token, target_token FROM user_canonical_mappings WHERE customer_id = ?",
            (customer_id,),
        )
        for row in cursor.fetchall():
            mappings[row["source_token"].strip().lower()] = row["target_token"]

    return mappings


def toggle_ootb_mapping(customer_id: str, source_token: str, enabled: bool) -> None:
    """Enable or disable an OOTB mapping for a user."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    source = source_token.lower()

    with get_db() as conn:
        if enabled:
            # Remove from disabled table
            conn.execute(
                "DELETE FROM user_ootb_mapping_disabled WHERE customer_id = ? AND mapping_source_token = ?",
                (customer_id, source),
            )
        else:
            # Add to disabled table
            conn.execute(
                """
                INSERT INTO user_ootb_mapping_disabled (customer_id, mapping_source_token, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (customer_id, source, timestamp),
            )


def list_subscription_plans(
    include_inactive: bool = False,
    include_non_public: bool = True,
) -> List[SubscriptionPlan]:
    """List all subscription plans."""
    init_db()
    with get_db() as conn:
        query = "SELECT * FROM subscription_plans"
        filters: List[str] = []
        if not include_inactive:
            filters.append("is_active = 1")
        if not include_non_public:
            filters.append("is_public = 1")
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY monthly_quota ASC"

        cursor = conn.execute(query)
        return [
            SubscriptionPlan(
                id=row["id"],
                name=row["name"],
                description=row["description"] if "description" in row.keys() else None,
                stripe_price_id=row["stripe_price_id"],
                monthly_price_cents=row["monthly_price_cents"],
                annual_price_cents=row["annual_price_cents"],
                monthly_quota=row["monthly_quota"],
                rate_limit_rpm=row["rate_limit_rpm"],
                concurrent_optimization_jobs=row["concurrent_optimization_jobs"],
                batch_size_limit=row["batch_size_limit"],
                optimization_history_retention_days=row[
                    "optimization_history_retention_days"
                ],
                telemetry_retention_days=row["telemetry_retention_days"],
                audit_log_retention_days=row["audit_log_retention_days"],
                custom_canonical_mappings_limit=row["custom_canonical_mappings_limit"],
                max_api_keys=row["max_api_keys"],
                features=json.loads(row["features"]) if row["features"] else [],
                is_active=bool(row["is_active"]),
                is_public=(
                    bool(row["is_public"]) if "is_public" in row.keys() else True
                ),
                plan_term=row["plan_term"] if "plan_term" in row.keys() else "monthly",
                monthly_discount_percent=(
                    row["monthly_discount_percent"]
                    if "monthly_discount_percent" in row.keys()
                    else 0
                ),
                yearly_discount_percent=(
                    row["yearly_discount_percent"]
                    if "yearly_discount_percent" in row.keys()
                    else 0
                ),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in cursor.fetchall()
        ]


def get_subscription_plan_by_id(
    plan_id: str,
    include_inactive: bool = True,
    include_non_public: bool = True,
) -> Optional[SubscriptionPlan]:
    """Get a subscription plan by ID."""
    init_db()
    with get_db() as conn:
        query = "SELECT * FROM subscription_plans WHERE id = ?"
        params: List[Any] = [plan_id]
        if not include_inactive:
            query += " AND is_active = 1"
        if not include_non_public:
            query += " AND is_public = 1"
        row = conn.execute(query, params).fetchone()

    if not row:
        return None

    return SubscriptionPlan(
        id=row["id"],
        name=row["name"],
        description=row["description"] if "description" in row.keys() else None,
        stripe_price_id=row["stripe_price_id"],
        monthly_price_cents=row["monthly_price_cents"],
        annual_price_cents=row["annual_price_cents"],
        monthly_quota=row["monthly_quota"],
        rate_limit_rpm=row["rate_limit_rpm"],
        concurrent_optimization_jobs=row["concurrent_optimization_jobs"],
        batch_size_limit=row["batch_size_limit"],
        optimization_history_retention_days=row["optimization_history_retention_days"],
        telemetry_retention_days=row["telemetry_retention_days"],
        audit_log_retention_days=row["audit_log_retention_days"],
        custom_canonical_mappings_limit=row["custom_canonical_mappings_limit"],
        max_api_keys=row["max_api_keys"],
        features=json.loads(row["features"]) if row["features"] else [],
        is_active=bool(row["is_active"]),
        is_public=bool(row["is_public"]) if "is_public" in row.keys() else True,
        plan_term=row["plan_term"] if "plan_term" in row.keys() else "monthly",
        monthly_discount_percent=(
            row["monthly_discount_percent"]
            if "monthly_discount_percent" in row.keys()
            else 0
        ),
        yearly_discount_percent=(
            row["yearly_discount_percent"]
            if "yearly_discount_percent" in row.keys()
            else 0
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def plan_requires_payment(plan: SubscriptionPlan) -> bool:
    return (plan.monthly_price_cents or 0) > 0 or (plan.annual_price_cents or 0) > 0


def plan_requires_sales_contact(plan: SubscriptionPlan) -> bool:
    return (plan.monthly_price_cents or 0) < 0 or (plan.annual_price_cents or 0) < 0


def create_subscription_plan(
    name: str,
    monthly_price_cents: int,
    monthly_quota: int,
    id: Optional[str] = None,
    description: Optional[str] = None,
    annual_price_cents: Optional[int] = None,
    rate_limit_rpm: int = 1000,
    concurrent_optimization_jobs: int = 5,
    batch_size_limit: int = 1000,
    optimization_history_retention_days: int = 365,
    telemetry_retention_days: int = 365,
    audit_log_retention_days: int = 365,
    custom_canonical_mappings_limit: int = 1000,
    stripe_price_id: Optional[str] = None,
    max_api_keys: int = 10,
    features: List[str] = None,
    is_active: bool = True,
    is_public: bool = True,
    plan_term: str = "monthly",
    monthly_discount_percent: int = 0,
    yearly_discount_percent: int = 0,
) -> SubscriptionPlan:
    """Create or update a subscription plan."""
    init_db()
    if not id:
        id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    features_json = json.dumps(features or [])

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO subscription_plans (
                id, name, description, stripe_price_id, monthly_price_cents, annual_price_cents, monthly_quota,
                rate_limit_rpm, concurrent_optimization_jobs, batch_size_limit, optimization_history_retention_days,
                telemetry_retention_days, audit_log_retention_days, custom_canonical_mappings_limit,
                max_api_keys, features, is_active, is_public, plan_term, monthly_discount_percent,
                yearly_discount_percent, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                stripe_price_id = excluded.stripe_price_id,
                monthly_price_cents = excluded.monthly_price_cents,
                annual_price_cents = excluded.annual_price_cents,
                monthly_quota = excluded.monthly_quota,
                rate_limit_rpm = excluded.rate_limit_rpm,
                concurrent_optimization_jobs = excluded.concurrent_optimization_jobs,
                batch_size_limit = excluded.batch_size_limit,
                optimization_history_retention_days = excluded.optimization_history_retention_days,
                telemetry_retention_days = excluded.telemetry_retention_days,
                audit_log_retention_days = excluded.audit_log_retention_days,
                custom_canonical_mappings_limit = excluded.custom_canonical_mappings_limit,
                max_api_keys = excluded.max_api_keys,
                features = excluded.features,
                is_active = excluded.is_active,
                is_public = excluded.is_public,
                plan_term = excluded.plan_term,
                monthly_discount_percent = excluded.monthly_discount_percent,
                yearly_discount_percent = excluded.yearly_discount_percent,
                updated_at = excluded.updated_at
            """,
            (
                id,
                name,
                description,
                stripe_price_id,
                monthly_price_cents,
                annual_price_cents,
                monthly_quota,
                rate_limit_rpm,
                concurrent_optimization_jobs,
                batch_size_limit,
                optimization_history_retention_days,
                telemetry_retention_days,
                audit_log_retention_days,
                custom_canonical_mappings_limit,
                max_api_keys,
                features_json,
                int(is_active),
                int(is_public),
                plan_term,
                monthly_discount_percent,
                yearly_discount_percent,
                timestamp,
                timestamp,
            ),
        )

    return SubscriptionPlan(
        id=id,
        name=name,
        description=description,
        stripe_price_id=stripe_price_id,
        monthly_price_cents=monthly_price_cents,
        annual_price_cents=annual_price_cents,
        monthly_quota=monthly_quota,
        rate_limit_rpm=rate_limit_rpm,
        concurrent_optimization_jobs=concurrent_optimization_jobs,
        batch_size_limit=batch_size_limit,
        optimization_history_retention_days=optimization_history_retention_days,
        telemetry_retention_days=telemetry_retention_days,
        audit_log_retention_days=audit_log_retention_days,
        custom_canonical_mappings_limit=custom_canonical_mappings_limit,
        max_api_keys=max_api_keys,
        features=features or [],
        is_active=is_active,
        is_public=is_public,
        plan_term=plan_term,
        monthly_discount_percent=monthly_discount_percent,
        yearly_discount_percent=yearly_discount_percent,
        created_at=timestamp,
        updated_at=timestamp,
    )


def delete_subscription_plan(plan_id: str) -> bool:
    """Delete a subscription plan if it has no subscriptions.

    Returns True if deleted, False if plan has subscriptions or doesn't exist.
    """
    init_db()
    with get_db() as conn:
        # Check if any customers are subscribed to this plan
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM customers WHERE subscription_tier = ?",
            (plan_id,),
        )
        count = cursor.fetchone()["count"]

        if count > 0:
            return False  # Plan has subscriptions, cannot delete

        # Delete the plan
        cursor = conn.execute(
            "DELETE FROM subscription_plans WHERE id = ?",
            (plan_id,),
        )

        return cursor.rowcount > 0


def _migrate_subscription_plans() -> None:
    """Seed default subscription plans if they don't exist."""
    init_db()
    with get_db() as conn:
        cursor = conn.execute("PRAGMA table_info(subscription_plans)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "description" not in columns:
            conn.execute("ALTER TABLE subscription_plans ADD COLUMN description TEXT")
        if "rate_limit_rpm" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN rate_limit_rpm INTEGER NOT NULL DEFAULT 1000"
            )
        if "concurrent_optimization_jobs" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN concurrent_optimization_jobs INTEGER NOT NULL DEFAULT 5"
            )
        if "batch_size_limit" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN batch_size_limit INTEGER NOT NULL DEFAULT 1000"
            )
        if "optimization_history_retention_days" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans "
                "ADD COLUMN optimization_history_retention_days INTEGER NOT NULL DEFAULT 365"
            )
        if "telemetry_retention_days" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN telemetry_retention_days INTEGER NOT NULL DEFAULT 365"
            )
        if "audit_log_retention_days" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN audit_log_retention_days INTEGER NOT NULL DEFAULT 365"
            )
        if "custom_canonical_mappings_limit" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans "
                "ADD COLUMN custom_canonical_mappings_limit INTEGER NOT NULL DEFAULT 1000"
            )
        if "monthly_price_cents" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN monthly_price_cents INTEGER NOT NULL DEFAULT 0"
            )
        if "annual_price_cents" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN annual_price_cents INTEGER"
            )
        if "plan_term" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN plan_term TEXT NOT NULL DEFAULT 'monthly'"
            )
        if "monthly_discount_percent" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN monthly_discount_percent INTEGER NOT NULL DEFAULT 0"
            )
        if "yearly_discount_percent" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN yearly_discount_percent INTEGER NOT NULL DEFAULT 0"
            )
        if "is_public" not in columns:
            conn.execute(
                "ALTER TABLE subscription_plans ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 1"
            )
    plans = [
        (
            "Trial Plan",
            "Basic tier for new users to explore the platform.",
            None,
            0,
            None,
            200,
            1000,
            2,
            ["Basic tier for new users"],
        ),
        (
            "Beginner Plan",
            "Perfect for individual developers and small projects.",
            None,
            500,
            None,
            1000,
            1000,
            5,
            ["Beginner tier"],
        ),
        (
            "Pro Plan",
            "For professional developers and growing teams.",
            "price_pro_placeholder",
            1000,
            10000,
            5000,
            1000,
            10,
            ["Professional tier"],
        ),
        (
            "Pro Plus Plan",
            "Advanced features for high-volume needs.",
            None,
            2500,
            None,
            15000,
            1000,
            15,
            ["Professional Plus tier"],
        ),
        (
            "Enterprise Plan",
            "Custom solutions for large organizations.",
            "price_ent_placeholder",
            -1,
            None,
            -1,
            1000,
            50,
            ["Enterprise tier", "Unlimited quota", "Contact sales"],
        ),
    ]

    for (
        name,
        description,
        price_id,
        monthly_price_cents,
        annual_price_cents,
        quota,
        rpm,
        keys,
        features,
    ) in plans:
        # Check if plan exists by name to avoid duplication
        # and ensure we update the existing record instead of creating a new one
        existing_id = None
        with get_db() as conn:
            row = conn.execute(
                "SELECT id FROM subscription_plans WHERE name = ?", (name,)
            ).fetchone()
            if row:
                existing_id = row["id"]

        create_subscription_plan(
            id=existing_id,
            name=name,
            description=description,
            monthly_price_cents=monthly_price_cents,
            annual_price_cents=annual_price_cents,
            monthly_quota=quota,
            rate_limit_rpm=rpm,
            stripe_price_id=price_id,
            max_api_keys=keys,
            features=features,
        )

    # Attempt to cleanup legacy 'free' plan if it exists and has no subscribers
    try:
        delete_subscription_plan("free")
    except Exception:
        pass


def set_admin_setting(key: str, value: Any) -> None:
    """Set a global admin setting."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    raw_value = json.dumps(value)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO admin_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, raw_value, timestamp),
        )


def get_admin_setting(key: str, default: Any = None) -> Any:
    """Get a global admin setting."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM admin_settings WHERE key = ?",
            (key,),
        ).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return default


def _parse_admin_setting_int(raw_value: Any, fallback: int = 0) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return value


def get_canonical_mappings_cache_version() -> int:
    """Get the persistent canonical mappings cache version."""
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM admin_settings WHERE key = ?",
            (_ADMIN_SETTING_CANONICAL_CACHE_VERSION,),
        ).fetchone()
        if not row:
            timestamp = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO admin_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (_ADMIN_SETTING_CANONICAL_CACHE_VERSION, json.dumps(0), timestamp),
            )
            return 0
    try:
        raw_value = json.loads(row["value"])
    except (TypeError, ValueError):
        raw_value = 0
    return _parse_admin_setting_int(raw_value, fallback=0)


def increment_canonical_mappings_cache_version() -> int:
    """Increment and return the persistent canonical mappings cache version."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM admin_settings WHERE key = ?",
            (_ADMIN_SETTING_CANONICAL_CACHE_VERSION,),
        ).fetchone()
        if row:
            try:
                raw_value = json.loads(row["value"])
            except (TypeError, ValueError):
                raw_value = 0
            current_value = _parse_admin_setting_int(raw_value, fallback=0)
        else:
            current_value = 0
        new_value = current_value + 1
        conn.execute(
            """
            INSERT INTO admin_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (_ADMIN_SETTING_CANONICAL_CACHE_VERSION, json.dumps(new_value), timestamp),
        )
    return new_value


def try_acquire_admin_setting_lock(
    key: str,
    new_value: Any,
    lock_state: str = "running",
    stale_before_epoch: Optional[int] = None,
) -> Tuple[bool, Any]:
    """Atomically set a setting unless it is already locked in the desired state."""
    init_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    raw_value = json.dumps(new_value)
    current_value = None
    params = [key, raw_value, timestamp, lock_state]
    stale_clause = ""
    if stale_before_epoch is not None:
        stale_clause = (
            " OR COALESCE(json_extract(admin_settings.value, '$.started_at_epoch'), 0)"
            " < ?"
        )
        params.append(stale_before_epoch)

    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            f"""
            INSERT INTO admin_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            WHERE json_extract(admin_settings.value, '$.state') IS NULL
                OR json_extract(admin_settings.value, '$.state') != ?
                {stale_clause}
            """,
            params,
        )
        if cursor.rowcount > 0:
            return True, new_value

        row = conn.execute(
            "SELECT value FROM admin_settings WHERE key = ?",
            (key,),
        ).fetchone()
        if row:
            try:
                current_value = json.loads(row["value"])
            except (TypeError, ValueError):
                current_value = None

    return False, current_value
