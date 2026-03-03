import importlib
import sys
import types

import pytest


def _install_fastapi_stubs() -> None:
    if "fastapi" in sys.modules:
        return

    fastapi_module = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args)

    class Response:
        def __init__(self, *args, **kwargs):
            return None

    class BackgroundTasks:
        def __init__(self, *args, **kwargs):
            return None

    class Depends:
        def __init__(self, *args, **kwargs):
            return None

    class Security:
        def __init__(self, *args, **kwargs):
            return None

    class _RouteMixin:
        def _decorator(self, *args, **kwargs):
            def _inner(func):
                return func

            return _inner

        get = post = put = delete = patch = options = head = _decorator

    class APIRouter(_RouteMixin):
        def __init__(self, *args, **kwargs):
            return None

        def include_router(self, *args, **kwargs):
            return None

    class FastAPI(APIRouter):
        def add_middleware(self, *args, **kwargs):
            return None

        def on_event(self, *args, **kwargs):
            def _inner(func):
                return func

            return _inner

        def middleware(self, *args, **kwargs):
            def _inner(func):
                return func

            return _inner

        def exception_handler(self, *args, **kwargs):
            def _inner(func):
                return func

            return _inner

        def mount(self, *args, **kwargs):
            return None

    fastapi_module.APIRouter = APIRouter
    fastapi_module.BackgroundTasks = BackgroundTasks
    fastapi_module.Depends = Depends
    fastapi_module.FastAPI = FastAPI
    fastapi_module.HTTPException = HTTPException
    fastapi_module.Security = Security
    fastapi_module.Response = Response
    fastapi_module.status = types.SimpleNamespace(
        HTTP_400_BAD_REQUEST=400,
        HTTP_401_UNAUTHORIZED=401,
        HTTP_402_PAYMENT_REQUIRED=402,
        HTTP_403_FORBIDDEN=403,
        HTTP_404_NOT_FOUND=404,
    )

    sys.modules["fastapi"] = fastapi_module

    concurrency_module = types.ModuleType("fastapi.concurrency")
    concurrency_module.run_in_threadpool = lambda *args, **kwargs: None
    sys.modules["fastapi.concurrency"] = concurrency_module

    middleware_module = types.ModuleType("fastapi.middleware")
    sys.modules["fastapi.middleware"] = middleware_module
    gzip_module = types.ModuleType("fastapi.middleware.gzip")
    gzip_module.GZipMiddleware = type("GZipMiddleware", (), {})
    sys.modules["fastapi.middleware.gzip"] = gzip_module

    responses_module = types.ModuleType("fastapi.responses")
    responses_module.ORJSONResponse = type("ORJSONResponse", (), {})
    sys.modules["fastapi.responses"] = responses_module

    staticfiles_module = types.ModuleType("fastapi.staticfiles")
    staticfiles_module.StaticFiles = type("StaticFiles", (), {})
    sys.modules["fastapi.staticfiles"] = staticfiles_module

    security_module = types.ModuleType("fastapi.security")
    security_module.APIKeyHeader = type("APIKeyHeader", (), {})
    security_module.OAuth2PasswordBearer = type("OAuth2PasswordBearer", (), {})
    sys.modules["fastapi.security"] = security_module


def _install_starlette_stubs() -> None:
    if "starlette" in sys.modules:
        return

    starlette_module = types.ModuleType("starlette")
    middleware_module = types.ModuleType("starlette.middleware")
    cors_module = types.ModuleType("starlette.middleware.cors")
    responses_module = types.ModuleType("starlette.responses")
    cors_module.CORSMiddleware = type("CORSMiddleware", (), {})
    responses_module.Response = type("Response", (), {})
    responses_module.JSONResponse = type("JSONResponse", (), {})

    sys.modules["starlette"] = starlette_module
    sys.modules["starlette.middleware"] = middleware_module
    sys.modules["starlette.middleware.cors"] = cors_module
    sys.modules["starlette.responses"] = responses_module


def _install_dotenv_stub() -> None:
    if "dotenv" not in sys.modules:
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv_module


def _install_stripe_stub() -> None:
    if "stripe" in sys.modules:
        return

    stripe_module = types.ModuleType("stripe")
    stripe_module.api_key = ""

    class _Customer:
        @staticmethod
        def create(*args, **kwargs):
            return types.SimpleNamespace(id="customer")

    class _CheckoutSession:
        @staticmethod
        def create(*args, **kwargs):
            return types.SimpleNamespace(url="session")

    class _BillingPortalSession:
        @staticmethod
        def create(*args, **kwargs):
            return types.SimpleNamespace(url="portal")

    class _SubscriptionItem:
        @staticmethod
        def create_usage_record(*args, **kwargs):
            return None

    class _Subscription:
        @staticmethod
        def retrieve(*args, **kwargs):
            return {"items": {"data": [{"id": "item"}]}}

        @staticmethod
        def modify(*args, **kwargs):
            return {}

        @staticmethod
        def delete(*args, **kwargs):
            return {}

    stripe_module.Customer = _Customer
    stripe_module.checkout = types.SimpleNamespace(Session=_CheckoutSession)
    stripe_module.billing_portal = types.SimpleNamespace(Session=_BillingPortalSession)
    stripe_module.SubscriptionItem = _SubscriptionItem
    stripe_module.Subscription = _Subscription

    sys.modules["stripe"] = stripe_module


def _install_pydantic_stub() -> None:
    if "pydantic" in sys.modules:
        return

    pydantic_module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            return None

    class ConfigDict(dict):
        pass

    def Field(default=None, **kwargs):
        return default

    pydantic_module.BaseModel = BaseModel
    pydantic_module.ConfigDict = ConfigDict
    pydantic_module.Field = Field
    sys.modules["pydantic"] = pydantic_module


def _install_internal_stubs() -> None:
    def _module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    auth_module = _module("auth")
    auth_module.get_current_customer = lambda *args, **kwargs: None
    auth_module.require_admin = lambda *args, **kwargs: None
    auth_module.track_usage = lambda *args, **kwargs: None

    database_module = _module("database")
    database_module.Customer = type("Customer", (), {})
    database_module.aggregate_history_stats = lambda *args, **kwargs: {}
    database_module.bulk_create_canonical_mappings = lambda *args, **kwargs: []
    database_module.create_batch_job = lambda *args, **kwargs: None
    database_module.create_canonical_mapping = lambda *args, **kwargs: None
    database_module.create_customer = lambda *args, **kwargs: None
    database_module.customer_scope = lambda *args, **kwargs: None
    database_module.delete_canonical_mappings = lambda *args, **kwargs: None
    database_module.get_canonical_mappings_cache = (
        lambda *args, **kwargs: types.SimpleNamespace(warm_up=lambda: None)
    )
    database_module.get_customer_by_id = lambda *args, **kwargs: None
    database_module.get_optimization_history_record = lambda *args, **kwargs: None
    database_module.get_llm_profiles = lambda *args, **kwargs: []
    database_module.get_subscription_plan_by_id = lambda *args, **kwargs: None
    database_module.get_usage = lambda *args, **kwargs: {}
    database_module.init_db = lambda *args, **kwargs: None
    database_module.is_telemetry_enabled = lambda *args, **kwargs: False
    database_module.list_batch_jobs = lambda *args, **kwargs: []
    database_module.list_canonical_mappings = lambda *args, **kwargs: []
    database_module.list_recent_history = lambda *args, **kwargs: []
    database_module.list_recent_telemetry = lambda *args, **kwargs: []
    database_module.set_llm_profiles = lambda *args, **kwargs: None
    database_module.update_batch_job = lambda *args, **kwargs: None
    database_module.update_canonical_mapping = lambda *args, **kwargs: None
    database_module.record_optimization_history = lambda *args, **kwargs: None
    database_module.get_admin_setting = lambda *args, **kwargs: None

    models_module = _module("models")
    canonical_module = _module("models.canonical_mapping")
    canonical_module.CanonicalMappingBulkCreate = type(
        "CanonicalMappingBulkCreate", (), {}
    )
    canonical_module.CanonicalMappingBulkDelete = type(
        "CanonicalMappingBulkDelete", (), {}
    )
    canonical_module.CanonicalMappingCreate = type("CanonicalMappingCreate", (), {})
    canonical_module.CanonicalMappingDeleteResponse = type(
        "CanonicalMappingDeleteResponse", (), {}
    )
    canonical_module.CanonicalMappingListResponse = type(
        "CanonicalMappingListResponse", (), {}
    )
    canonical_module.CanonicalMappingResponse = type("CanonicalMappingResponse", (), {})
    canonical_module.CanonicalMappingUpdate = type("CanonicalMappingUpdate", (), {})
    models_module.canonical_mapping = canonical_module

    optimization_module = _module("models.optimization")
    optimization_module.OptimizationBatchResponse = type(
        "OptimizationBatchResponse", (), {}
    )
    optimization_module.OptimizationRequest = type("OptimizationRequest", (), {})
    optimization_module.OptimizationResponse = type("OptimizationResponse", (), {})
    optimization_module.OptimizationStats = type("OptimizationStats", (), {})
    models_module.optimization = optimization_module

    routers_module = _module("routers")
    for name in [
        "admin_routes",
        "api_key_routes",
        "auth_routes",
        "billing_routes",
        "mapping_routes",
        "subscription_routes",
        "usage_routes",
        "webhook_routes",
    ]:
        module = _module(f"routers.{name}")
        module.router = object()
        if name == "admin_routes":
            module.bump_model_cache_validation_version = lambda *args, **kwargs: 0
            module.get_model_cache_validation_version = lambda *args, **kwargs: 0
            module.refresh_model_cache_snapshot = lambda *args, **kwargs: {}
        setattr(routers_module, name, module)

    services_module = _module("services")
    billing_module = _module("services.billing")
    billing_module.create_checkout_session = lambda *args, **kwargs: None
    billing_module.create_stripe_customer = lambda *args, **kwargs: None
    services_module.billing = billing_module

    llm_proxy_module = _module("services.llm_proxy")
    llm_proxy_module.LLMProviderError = type("LLMProviderError", (Exception,), {})
    llm_proxy_module.LLMResult = type("LLMResult", (), {})
    llm_proxy_module.call_llm = lambda *args, **kwargs: None
    llm_proxy_module.get_llm_providers = lambda *args, **kwargs: []
    services_module.llm_proxy = llm_proxy_module

    model_cache_module = _module("services.model_cache_manager")
    model_cache_module.ensure_models_cached = lambda *args, **kwargs: ([], [])
    model_cache_module.ensure_spacy_model_cached = lambda *args, **kwargs: False
    model_cache_module.get_spacy_cache_status = (
        lambda *args, **kwargs: {"cached_ok": False}
    )
    model_cache_module.resolve_hf_home = lambda *args, **kwargs: ""
    services_module.model_cache_manager = model_cache_module

    optimizer_module = _module("services.optimizer")
    optimizer_module.optimizer = types.SimpleNamespace(
        warm_up=lambda: None, model_status=lambda: {}
    )
    config_module = _module("services.optimizer.config")
    config_module.DEFAULT_CANONICAL_MAP = {}
    optimizer_module.config = config_module
    config_utils_module = _module("services.optimizer.config_utils")
    config_utils_module.sanitize_canonical_map = lambda *args, **kwargs: {}
    model_capabilities_module = _module("services.optimizer.model_capabilities")
    model_capabilities_module.build_model_readiness = lambda *args, **kwargs: {}
    model_capabilities_module.build_not_used_warnings = lambda *args, **kwargs: []
    model_capabilities_module.build_not_ready_warnings = lambda *args, **kwargs: []
    model_capabilities_module.entropy_backend_ready = lambda *args, **kwargs: False
    model_capabilities_module.hard_required_ready_for_mode = (
        lambda *args, **kwargs: True
    )
    model_capabilities_module.model_lookup_from_status = lambda *args, **kwargs: {}
    pipeline_config_module = _module("services.optimizer.pipeline_config")
    pipeline_config_module.resolve_optimization_config = lambda *args, **kwargs: {
        "disabled_passes": []
    }
    protect_module = _module("services.optimizer.protect")
    protect_module.ProtectTagError = type("ProtectTagError", (Exception,), {})
    router_module = _module("services.optimizer.router")
    router_module.ContentProfile = type("ContentProfile", (), {})
    router_module.SmartContext = type("SmartContext", (), {})
    router_module.get_profile = lambda *args, **kwargs: {}
    router_module.get_profile_for_text = lambda *args, **kwargs: None
    router_module.merge_disabled_passes = lambda *args, **kwargs: set()
    router_module.resolve_smart_context = lambda *args, **kwargs: None
    optimizer_module.config_utils = config_utils_module
    optimizer_module.model_capabilities = model_capabilities_module
    optimizer_module.pipeline_config = pipeline_config_module
    optimizer_module.protect = protect_module
    optimizer_module.router = router_module
    services_module.optimizer = optimizer_module

    quota_module = _module("services.quota_manager")
    quota_module.quota_manager = object()
    services_module.quota_manager = quota_module

    telemetry_module = _module("services.telemetry_control")
    telemetry_module.set_enabled = lambda *args, **kwargs: None
    services_module.telemetry_control = telemetry_module

    logging_control_module = _module("services.logging_control")
    logging_control_module.set_level = lambda *args, **kwargs: None
    services_module.logging_control = logging_control_module


def test_run_model_preparation_worker_ignores_non_dict_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ensure_models_cached(
        hf_home, models_to_sync, refresh_mode="download_missing"
    ):
        assert refresh_mode == "download_missing"
        return [], []

    def fake_model_status():
        return {"model-a": {"loaded": True}, "__warmup_epoch": 1}

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    def unexpected_exception(*args, **kwargs):
        raise AssertionError("logger.exception called unexpectedly")

    _install_fastapi_stubs()
    _install_starlette_stubs()
    _install_dotenv_stub()
    _install_stripe_stub()
    _install_pydantic_stub()
    _install_internal_stubs()

    sys.modules.setdefault("bcrypt", types.ModuleType("bcrypt"))
    if "jose" not in sys.modules:
        jose_module = types.ModuleType("jose")
        jwt_module = types.SimpleNamespace(
            encode=lambda *args, **kwargs: "",
            decode=lambda *args, **kwargs: {},
        )
        jose_module.JWTError = type("JWTError", (Exception,), {})
        jose_module.jwt = jwt_module
        sys.modules["jose"] = jose_module
    if "passlib" not in sys.modules:
        passlib_module = types.ModuleType("passlib")
        context_module = types.ModuleType("passlib.context")

        class DummyCryptContext:
            def __init__(self, *args, **kwargs):
                return None

            def verify(self, *args, **kwargs):
                return True

            def hash(self, *args, **kwargs):
                return "hash"

        context_module.CryptContext = DummyCryptContext
        passlib_module.context = context_module
        sys.modules["passlib"] = passlib_module
        sys.modules["passlib.context"] = context_module
    server = importlib.import_module("server")

    monkeypatch.setattr(server, "ensure_models_cached", fake_ensure_models_cached)
    monkeypatch.setattr(server.optimizer, "warm_up", lambda: None)
    monkeypatch.setattr(server.optimizer, "model_status", fake_model_status)
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(server.logger, "exception", unexpected_exception)

    server._run_model_preparation_worker(
        hf_home="/tmp/hf",
        missing_models=["model-a"],
        prewarm_enabled=True,
    )


def test_run_model_preparation_worker_refreshes_snapshot_and_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ensure_models_cached(
        hf_home, models_to_sync, refresh_mode="download_missing"
    ):
        return models_to_sync, []

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    _install_fastapi_stubs()
    _install_starlette_stubs()
    _install_dotenv_stub()
    _install_stripe_stub()
    _install_pydantic_stub()
    _install_internal_stubs()

    server = importlib.import_module("server")

    marker = {"snapshot": 0, "bump": 0, "invalidate": 0}

    monkeypatch.setattr(server, "ensure_models_cached", fake_ensure_models_cached)
    monkeypatch.setattr(server.optimizer, "warm_up", lambda: None)
    monkeypatch.setattr(server.optimizer, "model_status", lambda: {})
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        server,
        "refresh_model_cache_snapshot",
        lambda: marker.__setitem__("snapshot", marker["snapshot"] + 1),
    )
    monkeypatch.setattr(
        server,
        "bump_model_cache_validation_version",
        lambda: marker.__setitem__("bump", marker["bump"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_invalidate_model_availability_cache",
        lambda: marker.__setitem__("invalidate", marker["invalidate"] + 1),
    )

    server._run_model_preparation_worker(
        hf_home="/tmp/hf",
        missing_models=["model-a"],
        prewarm_enabled=False,
    )

    assert marker["snapshot"] == 1
    assert marker["bump"] == 1
    assert marker["invalidate"] == 1


def test_run_model_preparation_worker_syncs_spacy_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {"sync_models": None, "spacy": 0}

    def fake_ensure_models_cached(
        hf_home, models_to_sync, refresh_mode="download_missing"
    ):
        seen["sync_models"] = list(models_to_sync)
        return [], []

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    _install_fastapi_stubs()
    _install_starlette_stubs()
    _install_dotenv_stub()
    _install_stripe_stub()
    _install_pydantic_stub()
    _install_internal_stubs()

    server = importlib.import_module("server")
    model_cache_module = sys.modules["services.model_cache_manager"]
    setattr(
        model_cache_module,
        "ensure_spacy_model_cached",
        lambda model_name, allow_downloads=False: seen.__setitem__(
            "spacy", seen["spacy"] + 1
        )
        or True,
    )

    monkeypatch.setattr(server, "ensure_models_cached", fake_ensure_models_cached)
    monkeypatch.setattr(server.optimizer, "warm_up", lambda: None)
    monkeypatch.setattr(server.optimizer, "model_status", lambda: {})
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)

    server._run_model_preparation_worker(
        hf_home="/tmp/hf",
        missing_models=[],
        prewarm_enabled=False,
        sync_spacy=True,
    )

    assert seen["sync_models"] == []
    assert seen["spacy"] == 1


def test_run_model_preparation_worker_prewarms_when_cache_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"warmup": 0}

    def failing_ensure_models_cached(
        hf_home, models_to_sync, refresh_mode="download_missing"
    ):
        raise RuntimeError("simulated cache sync failure")

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    _install_fastapi_stubs()
    _install_starlette_stubs()
    _install_dotenv_stub()
    _install_stripe_stub()
    _install_pydantic_stub()
    _install_internal_stubs()

    server = importlib.import_module("server")

    monkeypatch.setattr(server, "ensure_models_cached", failing_ensure_models_cached)
    monkeypatch.setattr(
        server.optimizer,
        "warm_up",
        lambda: calls.__setitem__("warmup", calls["warmup"] + 1),
    )
    monkeypatch.setattr(server.optimizer, "model_status", lambda: {})
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)

    server._run_model_preparation_worker(
        hf_home="/tmp/hf",
        missing_models=["model-a"],
        prewarm_enabled=True,
    )

    assert calls["warmup"] == 1


def test_startup_prepares_spacy_before_first_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order = []

    _install_fastapi_stubs()
    _install_starlette_stubs()
    _install_dotenv_stub()
    _install_stripe_stub()
    _install_pydantic_stub()
    _install_internal_stubs()

    server = importlib.import_module("server")
    model_cache_module = sys.modules["services.model_cache_manager"]

    monkeypatch.setattr(
        server,
        "_detect_hf_volume",
        lambda: {
            "path": "/tmp/hf",
            "exists": True,
            "is_mounted": True,
            "writable": True,
            "size_bytes": 0,
        },
    )
    monkeypatch.setattr(
        server,
        "_detect_cached_models",
        lambda _path: {"available": ["coreference"], "missing": []},
    )
    monkeypatch.setattr(server, "refresh_model_cache_snapshot", lambda: None)
    monkeypatch.setattr(server, "get_admin_setting", lambda key, default=None: True)
    monkeypatch.setattr(server, "_env_truthy", lambda _k, _d=None: True)
    monkeypatch.setattr(server, "_run_model_preparation_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_MEMORY_GUARD_ACTIVE", False)
    monkeypatch.setattr(server, "_TOTAL_SYSTEM_MEMORY_BYTES", None)

    setattr(
        model_cache_module,
        "get_spacy_cache_status",
        lambda _name: {"cached_ok": False},
    )

    def _fake_ensure_spacy(_name, allow_downloads=False):
        call_order.append("ensure_spacy")
        return True

    setattr(
        model_cache_module,
        "ensure_spacy_model_cached",
        _fake_ensure_spacy,
    )

    monkeypatch.setattr(
        server.optimizer,
        "warm_up",
        lambda: call_order.append("warm_up"),
    )

    server._run_startup_model_tasks_safe()

    assert call_order == ["ensure_spacy", "warm_up"]
