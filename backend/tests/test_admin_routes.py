import sys
from types import SimpleNamespace

import auth_utils
import pytest
from database import create_customer, create_subscription_plan, init_db, update_customer
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def login_as_admin():
    response = client.post(
        "/api/auth/login",
        data={"username": "admin_test@example.com", "password": "adminpass"},
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    # Create admin user
    admin_email = "admin_test@example.com"
    pwd_hash = auth_utils.get_password_hash("adminpass")

    from database import get_customer_by_email

    existing = get_customer_by_email(admin_email)
    if existing:
        admin = existing
    else:
        admin = create_customer(name="Admin User", email=admin_email)

    update_customer(
        admin.id,
        password_hash=pwd_hash,
        role="admin",
        is_active=True,
        subscription_status="active",
    )
    return admin


def test_admin_list_users():
    # Login as admin
    login_data = {"username": "admin_test@example.com", "password": "adminpass"}
    response = client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # List users
    response = client.get("/api/admin/users", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert len(data["users"]) >= 1


def test_admin_access_denied_for_customer():
    # Create standard customer
    import uuid

    suffix = str(uuid.uuid4())[:8]
    c_email = f"customer_test_{suffix}@example.com"

    c_pwd = auth_utils.get_password_hash("custpass")
    cust = create_customer(name="Customer", email=c_email)
    update_customer(cust.id, password_hash=c_pwd)

    # Login
    login_data = {"username": c_email, "password": "custpass"}
    response = client.post("/api/auth/login", data=login_data)
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Try to access admin route
    response = client.get("/api/admin/users", headers=headers)
    assert response.status_code == 403


def test_admin_create_customer_requires_plan(setup_db):
    headers = login_as_admin()
    payload = {
        "name": "Planless Customer",
        "email": "planless@example.com",
        "role": "customer",
    }
    response = client.post(
        "/api/admin/users?password=secret123", json=payload, headers=headers
    )
    assert response.status_code == 400
    assert (
        response.json().get("detail")
        == "Customers must have a subscription plan selected"
    )


def test_admin_create_customer_with_plan(setup_db):
    import uuid

    plan = create_subscription_plan(
        name="Test Plan",
        monthly_price_cents=0,
        monthly_quota=10,
        rate_limit_rpm=10,
        max_api_keys=5,
        features=["test"],
        is_active=True,
        plan_term="monthly",
    )

    headers = login_as_admin()
    suffix = str(uuid.uuid4())[:8]
    payload = {
        "name": "Paying Customer",
        "email": f"paying_{suffix}@example.com",
        "role": "customer",
        "subscription_tier": plan.id,
    }
    response = client.post(
        "/api/admin/users?password=secret123", json=payload, headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["subscription_tier"] == plan.id


def test_admin_can_assign_non_public_plan(setup_db):
    import uuid

    plan = create_subscription_plan(
        name="Enterprise Private Plan",
        monthly_price_cents=0,
        monthly_quota=100000,
        rate_limit_rpm=500,
        max_api_keys=25,
        features=["enterprise", "private"],
        is_active=True,
        is_public=False,
        plan_term="monthly",
    )

    headers = login_as_admin()
    suffix = str(uuid.uuid4())[:8]
    payload = {
        "name": "Enterprise Customer",
        "email": f"enterprise_{suffix}@example.com",
        "role": "customer",
        "subscription_tier": plan.id,
    }
    response = client.post(
        "/api/admin/users?password=secret123", json=payload, headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["subscription_tier"] == plan.id


def test_admin_create_customer_rejects_unknown_plan(setup_db):
    import uuid

    headers = login_as_admin()
    suffix = str(uuid.uuid4())[:8]
    payload = {
        "name": "Unknown Plan Customer",
        "email": f"unknown_plan_{suffix}@example.com",
        "role": "customer",
        "subscription_tier": f"missing_plan_{suffix}",
    }
    response = client.post(
        "/api/admin/users?password=secret123", json=payload, headers=headers
    )
    assert response.status_code == 400
    assert response.json().get("detail") == "Selected subscription plan does not exist"


def test_admin_update_customer_requires_plan(setup_db):
    import uuid

    suffix = str(uuid.uuid4())[:8]
    email = f"update_customer_{suffix}@example.com"
    customer = create_customer(name="Updater", email=email)
    update_customer(customer.id, password_hash=auth_utils.get_password_hash("custpass"))

    headers = login_as_admin()
    response = client.put(
        f"/api/admin/users/{customer.id}",
        json={"subscription_tier": ""},
        headers=headers,
    )
    assert response.status_code == 400
    assert (
        response.json().get("detail")
        == "Customers must have a subscription plan selected"
    )


def test_admin_list_plans_supports_public_only_filter(setup_db):
    import uuid

    public_id = f"public_filter_{uuid.uuid4().hex[:8]}"
    hidden_id = f"hidden_filter_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=public_id,
        name="Public Filter Plan",
        monthly_price_cents=0,
        monthly_quota=100,
        is_active=True,
        is_public=True,
    )
    create_subscription_plan(
        id=hidden_id,
        name="Hidden Filter Plan",
        monthly_price_cents=0,
        monthly_quota=100,
        is_active=True,
        is_public=False,
    )

    headers = login_as_admin()
    all_response = client.get("/api/admin/plans", headers=headers)
    assert all_response.status_code == 200
    all_ids = {plan["id"] for plan in all_response.json()}
    assert public_id in all_ids
    assert hidden_id in all_ids

    public_only_response = client.get(
        "/api/admin/plans?public_only=true", headers=headers
    )
    assert public_only_response.status_code == 200
    public_only_ids = {plan["id"] for plan in public_only_response.json()}
    assert public_id in public_only_ids
    assert hidden_id not in public_only_ids


def test_normalize_live_loaded_status_treats_zero_warmup_epoch_as_warmup_run():
    from routers.admin_routes import _normalize_live_loaded_status

    live_status = {
        "__warmup_epoch": 0,
        "semantic_guard": {"loaded": False},
    }

    loaded_ok, loaded_reason = _normalize_live_loaded_status(
        live_status, "semantic_guard"
    )

    assert loaded_ok is False
    assert loaded_reason == "load_failed"


def test_capability_lookup_uses_snapshot_when_live_status_missing():
    from routers.admin_routes import _capability_lookup_from_statuses

    snapshot_status = {
        "semantic_guard": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        "coreference": {"loaded_ok": False, "loaded_reason": "load_failed"},
        "semantic_rank": {"loaded_ok": None, "loaded_reason": "not_warmed"},
    }

    lookup = _capability_lookup_from_statuses(snapshot_status, live_status=None)

    assert lookup["semantic_guard"] is True
    assert lookup["coreference"] is False
    assert "semantic_rank" not in lookup


def test_capability_lookup_merges_live_with_snapshot_fallback():
    from routers.admin_routes import _capability_lookup_from_statuses

    snapshot_status = {
        "semantic_guard": {"loaded_ok": True},
        "coreference": {"loaded_ok": True},
    }
    live_status = {
        "__warmup_epoch": 123,
        "semantic_guard": {"loaded": False},
        # coreference missing in live status should fall back to snapshot
    }

    lookup = _capability_lookup_from_statuses(snapshot_status, live_status=live_status)

    assert lookup["semantic_guard"] is False
    assert lookup["coreference"] is True


def test_capability_lookup_ignores_non_authoritative_live_false_without_warmup():
    from routers.admin_routes import _capability_lookup_from_statuses

    snapshot_status = {
        "semantic_guard": {"loaded_ok": True},
        "coreference": {"loaded_ok": True},
    }
    live_status = {
        "semantic_guard": {"loaded": False},
        # without warm-up epoch this should not override snapshot readiness
    }

    lookup = _capability_lookup_from_statuses(snapshot_status, live_status=live_status)

    assert lookup["semantic_guard"] is True
    assert lookup["coreference"] is True


def _patch_admin_models_state(
    monkeypatch,
    snapshot,
    live_status,
    *,
    spacy_cache_status=None,
    probe_model_readiness=None,
):
    from routers import admin_routes
    import services.model_cache_manager as model_cache_manager

    monkeypatch.setattr(admin_routes, "_get_model_cache_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        admin_routes,
        "list_model_inventory",
        lambda: [
            {
                "model_type": "semantic_guard",
                "model_name": "guard-model",
                "component": "semantic_guard",
                "library_type": "transformers",
                "usage": "guard",
                "expected_files": "[]",
            }
        ],
    )
    monkeypatch.setattr(
        admin_routes,
        "list_capabilities_for_model",
        lambda _model_type: {
            "intended_features": [],
            "required_mode_gates": [],
            "required_profile_gates": [],
            "hard_required": False,
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(
            optimizer=SimpleNamespace(
                model_status=lambda: live_status,
                _spacy_model_name="en_core_web_sm",
                probe_model_readiness=(
                    probe_model_readiness
                    if probe_model_readiness is not None
                    else (lambda _model_type: {"loaded": True})
                ),
            )
        ),
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        (
            (lambda *args, **kwargs: spacy_cache_status)
            if spacy_cache_status is not None
            else (lambda *args, **kwargs: {"cached_ok": True, "cached_reason": "cached_ok"})
        ),
    )


def test_admin_models_snapshot_does_not_report_loaded_before_warmup(monkeypatch):
    headers = login_as_admin()
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {
                    "cached_ok": True,
                    "cached_reason": "cached_ok",
                },
                "spacy": {
                    "cached_ok": True,
                    "cached_reason": "cached_ok",
                },
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
            "spacy": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        },
    }

    _patch_admin_models_state(monkeypatch, snapshot=snapshot, live_status={})

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    models = {item["model_type"]: item for item in response.json()["models"]}

    assert models["semantic_guard"]["loaded_ok"] is None
    assert models["semantic_guard"]["loaded_reason"] == "not_warmed"
    assert models["spacy"]["loaded_ok"] is None
    assert models["spacy"]["loaded_reason"] == "not_warmed"


def test_admin_models_partial_live_status_only_overrides_probed_model(monkeypatch):
    headers = login_as_admin()
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {"cached_ok": True, "cached_reason": "cached_ok"},
                "spacy": {"cached_ok": True, "cached_reason": "cached_ok"},
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
            "spacy": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        },
    }
    live_status = {
        "__warmup_epoch": 456,
        "semantic_guard": {"loaded": False},
        # spacy omitted on purpose; should keep snapshot values
    }

    _patch_admin_models_state(monkeypatch, snapshot=snapshot, live_status=live_status)

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    models = {item["model_type"]: item for item in response.json()["models"]}

    assert models["semantic_guard"]["loaded_ok"] is False
    assert models["semantic_guard"]["loaded_reason"] == "load_failed"
    assert models["spacy"]["loaded_ok"] is True
    assert models["spacy"]["loaded_reason"] == "loaded_ok"


def test_admin_models_explicit_live_loaded_true_overrides_snapshot_without_warmup(
    monkeypatch,
):
    headers = login_as_admin()
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {"cached_ok": True, "cached_reason": "cached_ok"},
                "spacy": {"cached_ok": True, "cached_reason": "cached_ok"},
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": False, "loaded_reason": "load_failed"},
            "spacy": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        },
    }
    live_status = {
        "semantic_guard": {"loaded": True},
    }

    _patch_admin_models_state(monkeypatch, snapshot=snapshot, live_status=live_status)

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    models = {item["model_type"]: item for item in response.json()["models"]}

    assert models["semantic_guard"]["loaded_ok"] is True
    assert models["semantic_guard"]["loaded_reason"] == "loaded_ok"
    assert models["spacy"]["loaded_ok"] is None
    assert models["spacy"]["loaded_reason"] == "not_warmed"


def test_admin_models_full_warmup_status_overrides_snapshot(monkeypatch):
    headers = login_as_admin()
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {"cached_ok": True, "cached_reason": "cached_ok"},
                "spacy": {"cached_ok": True, "cached_reason": "cached_ok"},
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
            "spacy": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        },
    }
    live_status = {
        "__warmup_epoch": 123,
        "semantic_guard": {"loaded": False},
        "spacy": {"loaded": False},
    }

    _patch_admin_models_state(monkeypatch, snapshot=snapshot, live_status=live_status)

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    models = {item["model_type"]: item for item in response.json()["models"]}

    assert models["semantic_guard"]["loaded_ok"] is False
    assert models["semantic_guard"]["loaded_reason"] == "load_failed"
    assert models["spacy"]["loaded_ok"] is False
    assert models["spacy"]["loaded_reason"] == "load_failed"


def test_admin_models_returns_snapshot_warnings_and_cached_error_detail(monkeypatch):
    headers = login_as_admin()
    auth_warning = (
        "semantic_guard: HuggingFace access denied for org/private-model. "
        "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) with access to this repo in backend env/.env."
    )
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {
                    "cached_ok": False,
                    "cached_reason": "auth_failed",
                    "cached_error_detail": (
                        "HuggingFace access denied for org/private-model. "
                        "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) with access to this repo in backend env/.env."
                    ),
                },
                "spacy": {
                    "cached_ok": True,
                    "cached_reason": "cached_ok",
                },
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": False, "loaded_reason": "not_warmed"},
            "spacy": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
        },
        "warnings": [auth_warning],
    }

    _patch_admin_models_state(monkeypatch, snapshot=snapshot, live_status={})

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert auth_warning in payload["warnings"]
    models = {item["model_type"]: item for item in payload["models"]}
    assert models["semantic_guard"]["cached_reason"] == "auth_failed"
    assert "HF_TOKEN" in (models["semantic_guard"]["cached_error_detail"] or "")


def test_admin_models_spacy_cache_status_uses_live_check_over_stale_snapshot(
    monkeypatch,
):
    headers = login_as_admin()
    snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "stats": {
            "models": {
                "semantic_guard": {"cached_ok": True, "cached_reason": "cached_ok"},
                "spacy": {"cached_ok": False, "cached_reason": "cache_missing"},
            },
            "total_size_bytes": 0,
            "total_size_formatted": "0 B",
        },
        "model_status": {
            "semantic_guard": {"loaded_ok": True, "loaded_reason": "loaded_ok"},
            "spacy": {"loaded_ok": False, "loaded_reason": "load_failed"},
        },
    }

    _patch_admin_models_state(
        monkeypatch,
        snapshot=snapshot,
        live_status={"__warmup_epoch": 456, "spacy": {"loaded": True}},
        spacy_cache_status={
            "cached_ok": True,
            "cached_reason": "cached_ok",
            "size_bytes": 123,
            "size_formatted": "123 B",
            "model_name": "en_core_web_sm",
        },
    )

    response = client.get("/api/admin/models", headers=headers)
    assert response.status_code == 200
    models = {item["model_type"]: item for item in response.json()["models"]}

    assert models["spacy"]["cached_ok"] is True
    assert models["spacy"]["cached_reason"] == "cached_ok"
    assert models["spacy"]["loaded_ok"] is True
    assert models["spacy"]["loaded_reason"] == "loaded_ok"


def test_admin_global_refresh_resets_download_failures(monkeypatch):
    from routers import admin_routes

    headers = login_as_admin()
    reset_calls = []
    thread_starts = []

    monkeypatch.setattr(admin_routes, "_try_start_model_refresh", lambda state: state)
    monkeypatch.setattr(
        admin_routes,
        "reset_model_download_issues",
        lambda model_types=None: reset_calls.append(model_types) or 2,
    )
    monkeypatch.setattr(
        admin_routes,
        "bump_model_cache_validation_version",
        lambda: 0,
    )

    class DummyThread:
        def __init__(self, target, args, daemon):
            self.args = args

        def start(self):
            thread_starts.append(self.args)

    monkeypatch.setattr(admin_routes, "Thread", DummyThread)

    response = client.post(
        "/api/admin/models/refresh?mode=download_missing", headers=headers
    )

    assert response.status_code == 200
    assert reset_calls == [None]
    assert len(thread_starts) == 1


def test_admin_model_refresh_resets_failure_state_for_target_model(monkeypatch):
    from routers import admin_routes

    headers = login_as_admin()
    reset_calls = []
    thread_starts = []

    monkeypatch.setattr(admin_routes, "_try_start_model_refresh", lambda state: state)
    monkeypatch.setattr(
        admin_routes,
        "reset_model_download_issues",
        lambda model_types=None: reset_calls.append(model_types) or 1,
    )
    monkeypatch.setattr(
        admin_routes,
        "bump_model_cache_validation_version",
        lambda: 0,
    )
    monkeypatch.setattr(
        admin_routes,
        "get_model_inventory_item",
        lambda model_type: {"model_type": model_type},
    )

    class DummyThread:
        def __init__(self, target, args, daemon):
            self.args = args

        def start(self):
            thread_starts.append(self.args)

    monkeypatch.setattr(admin_routes, "Thread", DummyThread)

    response = client.post(
        "/api/admin/models/semantic_guard/refresh?mode=download_missing",
        headers=headers,
    )

    assert response.status_code == 200
    assert reset_calls == [["semantic_guard"]]
    assert len(thread_starts) == 1


def test_run_model_refresh_marks_failed_when_target_model_still_missing(monkeypatch):
    import services.model_cache_manager as model_cache_manager
    from routers import admin_routes

    captured_state = {}

    monkeypatch.setattr(
        admin_routes,
        "_get_model_refresh_state",
        lambda: {
            "started_at": "2026-02-13T00:00:00+00:00",
            "started_at_epoch": 1,
            "target_models": ["semantic_guard"],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(admin_routes, "bump_model_cache_validation_version", lambda: 0)
    monkeypatch.setattr(
        admin_routes,
        "refresh_model_cache_snapshot",
        lambda validator=None, model_status=None: {
            "stats": {"models": {}},
            "model_status": {},
            "missing_models": ["semantic_guard"],
            "available_models": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "model_lookup_from_status",
        lambda status: {"semantic_guard": False},
    )
    monkeypatch.setattr(
        admin_routes,
        "build_model_readiness",
        lambda readiness: {
            "semantic_guard": {"hard_required": True, "intended_usage_ready": False}
        },
    )

    monkeypatch.setattr(
        model_cache_manager, "get_model_configs", lambda: {"semantic_guard": {}}
    )
    monkeypatch.setattr(
        admin_routes,
        "ensure_models_cached",
        lambda *args, **kwargs: ([], ["semantic_guard"]),
    )
    monkeypatch.setattr(
        model_cache_manager, "ensure_spacy_model_cached", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        lambda *args, **kwargs: {"cached_ok": True, "cached_reason": "cached_ok"},
    )

    class DummyValidator:
        def __init__(self, hf_home):
            self.hf_home = hf_home
            self.configs = {}
            self._size_cache = {}
            self._validation_cache = {}

        def get_missing_models(self, model_types):
            return ["semantic_guard"]

    monkeypatch.setattr(admin_routes, "ModelCacheValidator", DummyValidator)

    fake_optimizer = SimpleNamespace(
        _spacy_model_name="en_core_web_md",
        refresh_model_configs=lambda: None,
        warm_up=lambda: None,
        model_status=lambda: {"semantic_guard": {"loaded": False}, "__warmup_epoch": 1},
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(optimizer=fake_optimizer),
    )

    admin_routes._run_model_refresh(
        hf_home="hf-home",
        refresh_mode="download_missing",
        model_types=["semantic_guard"],
    )

    assert captured_state["state"] == "failed"
    assert "semantic_guard" in str(captured_state["error"] or "")
    assert captured_state["target_models"] == ["semantic_guard"]


def test_run_model_refresh_targeted_completes_when_only_other_models_missing(
    monkeypatch,
):
    import services.model_cache_manager as model_cache_manager
    from routers import admin_routes

    captured_state = {}

    monkeypatch.setattr(
        admin_routes,
        "_get_model_refresh_state",
        lambda: {
            "started_at": "2026-02-13T00:00:00+00:00",
            "started_at_epoch": 1,
            "target_models": ["semantic_guard"],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(admin_routes, "bump_model_cache_validation_version", lambda: 0)
    monkeypatch.setattr(
        admin_routes,
        "refresh_model_cache_snapshot",
        lambda validator=None, model_status=None: {
            "stats": {"models": {}},
            "model_status": {},
            "missing_models": ["entropy"],
            "available_models": ["semantic_guard"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "model_lookup_from_status",
        lambda status: {"semantic_guard": True},
    )
    monkeypatch.setattr(admin_routes, "build_model_readiness", lambda readiness: {})

    monkeypatch.setattr(
        model_cache_manager, "get_model_configs", lambda: {"semantic_guard": {}}
    )
    monkeypatch.setattr(
        admin_routes,
        "ensure_models_cached",
        lambda *args, **kwargs: (["semantic_guard"], []),
    )
    monkeypatch.setattr(
        model_cache_manager, "ensure_spacy_model_cached", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        lambda *args, **kwargs: {"cached_ok": True, "cached_reason": "cached_ok"},
    )

    class DummyValidator:
        def __init__(self, hf_home):
            self.hf_home = hf_home
            self.configs = {}
            self._size_cache = {}
            self._validation_cache = {}

        def get_missing_models(self, model_types):
            return []

    monkeypatch.setattr(admin_routes, "ModelCacheValidator", DummyValidator)

    fake_optimizer = SimpleNamespace(
        _spacy_model_name="en_core_web_md",
        refresh_model_configs=lambda: None,
        warm_up=lambda: None,
        model_status=lambda: {"semantic_guard": {"loaded": True}, "__warmup_epoch": 1},
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(optimizer=fake_optimizer),
    )

    admin_routes._run_model_refresh(
        hf_home="hf-home",
        refresh_mode="download_missing",
        model_types=["semantic_guard"],
    )

    assert captured_state["state"] == "completed"
    assert captured_state["error"] is None
    assert captured_state["target_models"] == ["semantic_guard"]


def test_run_model_refresh_targeted_fails_when_target_model_not_ready_after_warmup(
    monkeypatch,
):
    import services.model_cache_manager as model_cache_manager
    from routers import admin_routes

    captured_state = {}

    monkeypatch.setattr(
        admin_routes,
        "_get_model_refresh_state",
        lambda: {
            "started_at": "2026-02-13T00:00:00+00:00",
            "started_at_epoch": 1,
            "target_models": ["semantic_guard"],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(admin_routes, "bump_model_cache_validation_version", lambda: 0)
    monkeypatch.setattr(
        admin_routes,
        "refresh_model_cache_snapshot",
        lambda validator=None, model_status=None: {
            "stats": {"models": {}},
            "model_status": {
                "semantic_guard": {"intended_usage_ready": False, "hard_required": True}
            },
            "missing_models": [],
            "available_models": ["semantic_guard"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "model_lookup_from_status",
        lambda status: {"semantic_guard": False},
    )
    monkeypatch.setattr(
        admin_routes,
        "build_model_readiness",
        lambda readiness: {
            "semantic_guard": {"hard_required": True, "intended_usage_ready": False}
        },
    )

    monkeypatch.setattr(
        model_cache_manager, "get_model_configs", lambda: {"semantic_guard": {}}
    )
    monkeypatch.setattr(
        admin_routes,
        "ensure_models_cached",
        lambda *args, **kwargs: (["semantic_guard"], []),
    )
    monkeypatch.setattr(
        model_cache_manager, "ensure_spacy_model_cached", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        lambda *args, **kwargs: {"cached_ok": True, "cached_reason": "cached_ok"},
    )

    class DummyValidator:
        def __init__(self, hf_home):
            self.hf_home = hf_home
            self.configs = {}
            self._size_cache = {}
            self._validation_cache = {}

        def get_missing_models(self, model_types):
            return []

    monkeypatch.setattr(admin_routes, "ModelCacheValidator", DummyValidator)

    fake_optimizer = SimpleNamespace(
        _spacy_model_name="en_core_web_md",
        refresh_model_configs=lambda: None,
        warm_up=lambda: None,
        model_status=lambda: {"semantic_guard": {"loaded": False}, "__warmup_epoch": 1},
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(optimizer=fake_optimizer),
    )

    admin_routes._run_model_refresh(
        hf_home="hf-home",
        refresh_mode="download_missing",
        model_types=["semantic_guard"],
    )

    assert captured_state["state"] == "failed"
    assert "not-ready model(s) after warm-up" in str(captured_state["error"] or "")
    assert captured_state["target_models"] == ["semantic_guard"]


def test_run_model_refresh_spacy_target_fails_when_spacy_cache_missing(monkeypatch):
    import services.model_cache_manager as model_cache_manager
    from routers import admin_routes

    captured_state = {}

    monkeypatch.setattr(
        admin_routes,
        "_get_model_refresh_state",
        lambda: {
            "started_at": "2026-02-13T00:00:00+00:00",
            "started_at_epoch": 1,
            "target_models": ["spacy"],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(admin_routes, "bump_model_cache_validation_version", lambda: 0)
    monkeypatch.setattr(
        admin_routes,
        "refresh_model_cache_snapshot",
        lambda validator=None, model_status=None: {
            "stats": {"models": {}},
            "model_status": {},
            "missing_models": [],
            "available_models": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(admin_routes, "model_lookup_from_status", lambda status: {})
    monkeypatch.setattr(admin_routes, "build_model_readiness", lambda readiness: {})

    monkeypatch.setattr(
        model_cache_manager,
        "get_model_configs",
        lambda: {"semantic_guard": {"model_name": "test-semantic"}},
    )
    monkeypatch.setattr(
        model_cache_manager, "ensure_spacy_model_cached", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        lambda *args, **kwargs: {"cached_ok": False, "cached_reason": "cache_missing"},
    )

    class DummyValidator:
        def __init__(self, hf_home):
            self.hf_home = hf_home
            self.configs = {}
            self._size_cache = {}
            self._validation_cache = {}

        def get_missing_models(self, model_types):
            return []

    monkeypatch.setattr(admin_routes, "ModelCacheValidator", DummyValidator)

    fake_optimizer = SimpleNamespace(
        _spacy_model_name="en_core_web_sm",
        refresh_model_configs=lambda: None,
        warm_up=lambda: None,
        model_status=lambda: {"spacy": {"loaded": False}, "__warmup_epoch": 1},
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(optimizer=fake_optimizer),
    )

    admin_routes._run_model_refresh(
        hf_home="hf-home",
        refresh_mode="download_missing",
        model_types=["spacy"],
    )

    assert captured_state["state"] == "failed"
    assert "spacy" in list(captured_state.get("missing_models") or [])
    assert "spacy" in str(captured_state.get("error") or "")


def test_run_model_refresh_spacy_target_completes_when_spacy_cache_ready(monkeypatch):
    import services.model_cache_manager as model_cache_manager
    from routers import admin_routes

    captured_state = {}

    monkeypatch.setattr(
        admin_routes,
        "_get_model_refresh_state",
        lambda: {
            "started_at": "2026-02-13T00:00:00+00:00",
            "started_at_epoch": 1,
            "target_models": ["spacy"],
        },
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: captured_state.update(state),
    )
    monkeypatch.setattr(admin_routes, "bump_model_cache_validation_version", lambda: 0)
    monkeypatch.setattr(
        admin_routes,
        "refresh_model_cache_snapshot",
        lambda validator=None, model_status=None: {
            "stats": {"models": {}},
            "model_status": {},
            "missing_models": ["semantic_guard"],
            "available_models": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(admin_routes, "model_lookup_from_status", lambda status: {})
    monkeypatch.setattr(admin_routes, "build_model_readiness", lambda readiness: {})

    monkeypatch.setattr(
        model_cache_manager,
        "get_model_configs",
        lambda: {"semantic_guard": {"model_name": "test-semantic"}},
    )
    monkeypatch.setattr(
        model_cache_manager, "ensure_spacy_model_cached", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        model_cache_manager,
        "get_spacy_cache_status",
        lambda *args, **kwargs: {"cached_ok": True, "cached_reason": "cached_ok"},
    )

    class DummyValidator:
        def __init__(self, hf_home):
            self.hf_home = hf_home
            self.configs = {}
            self._size_cache = {}
            self._validation_cache = {}

        def get_missing_models(self, model_types):
            return ["semantic_guard"]

    monkeypatch.setattr(admin_routes, "ModelCacheValidator", DummyValidator)

    fake_optimizer = SimpleNamespace(
        _spacy_model_name="en_core_web_sm",
        refresh_model_configs=lambda: None,
        warm_up=lambda: None,
        model_status=lambda: {"spacy": {"loaded": True}, "__warmup_epoch": 1},
    )
    monkeypatch.setitem(
        sys.modules,
        "services.optimizer.core",
        SimpleNamespace(optimizer=fake_optimizer),
    )

    admin_routes._run_model_refresh(
        hf_home="hf-home",
        refresh_mode="download_missing",
        model_types=["spacy"],
    )

    assert captured_state["state"] == "completed"
    assert "spacy" in list(captured_state.get("available_models") or [])
    assert "spacy" not in list(captured_state.get("missing_models") or [])


def test_admin_refresh_status_excludes_effective_offline_mode():
    headers = login_as_admin()

    response = client.get("/api/admin/models/refresh", headers=headers)

    assert response.status_code == 200
    assert "effective_offline_mode" not in response.json()


def test_get_model_refresh_state_resets_stale_running_state(monkeypatch):
    from routers import admin_routes

    stale_running = {
        "state": "running",
        "started_at": "2026-02-13T00:00:00+00:00",
        "started_at_epoch": 10,
        "finished_at": None,
        "error": None,
        "mode": "download_missing",
        "available_models": [],
        "missing_models": [],
        "target_models": [],
        "warnings": [],
    }
    persisted = {}

    monkeypatch.setattr(
        admin_routes,
        "get_admin_setting",
        lambda key, default: stale_running,
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: persisted.update(state),
    )
    monkeypatch.setattr(
        admin_routes.time,
        "time",
        lambda: 10 + admin_routes._MODEL_REFRESH_TTL_SECONDS + 1,
    )

    state = admin_routes._get_model_refresh_state()

    assert state["state"] == "idle"
    assert state["started_at"] is None
    assert persisted.get("state") == "idle"


def test_get_model_refresh_state_resets_running_state_with_invalid_epoch(monkeypatch):
    from routers import admin_routes

    invalid_epoch_running = {
        "state": "running",
        "started_at": "2026-02-14T00:00:00+00:00",
        "started_at_epoch": "invalid",
        "finished_at": None,
        "error": None,
        "mode": "download_missing",
        "available_models": [],
        "missing_models": [],
        "target_models": [],
        "warnings": [],
    }
    persisted = {}

    monkeypatch.setattr(
        admin_routes,
        "get_admin_setting",
        lambda key, default: invalid_epoch_running,
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_model_refresh_state",
        lambda state: persisted.update(state),
    )

    state = admin_routes._get_model_refresh_state()

    assert state["state"] == "idle"
    assert state["started_at"] is None
    assert persisted.get("state") == "idle"


def test_admin_refresh_rejects_validate_only_mode():
    headers = login_as_admin()

    response = client.post(
        "/api/admin/models/refresh?mode=validate_only", headers=headers
    )

    assert response.status_code == 400
    assert "Invalid refresh mode" in response.json()["detail"]


def test_admin_model_refresh_rejects_validate_only_mode():
    headers = login_as_admin()

    response = client.post(
        "/api/admin/models/semantic_guard/refresh?mode=validate_only",
        headers=headers,
    )

    assert response.status_code == 400
    assert "Invalid refresh mode" in response.json()["detail"]


def test_admin_settings_rejects_model_refresh_offline_mode_flag():
    headers = login_as_admin()

    update_response = client.patch(
        "/api/admin/settings",
        json={"model_refresh_offline_mode": True},
        headers=headers,
    )
    assert update_response.status_code == 400
    assert (
        "Cannot update model_refresh_offline_mode" in update_response.json()["detail"]
    )


def test_admin_settings_update_accepts_admin_ui_payload():
    headers = login_as_admin()
    update_payload = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 2525,
        "smtp_user": "ops@example.com",
        "smtp_from_email": "noreply@example.com",
        "smtp_password": "smtp-secret",
        "stripe_publishable_key": "pk_test_123",
        "stripe_secret_key": "sk_test_456",
        "log_level": "debug",
        "telemetry_enabled": True,
        "access_token_expire_minutes": 45,
        "refresh_token_expire_days": 14,
        "history_enabled": False,
        "optimizer_prewarm_models": False,
        "cors_origins": "https://app.example.com, http://localhost:5173",
    }
    update_response = client.patch(
        "/api/admin/settings",
        json=update_payload,
        headers=headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "success"

    get_response = client.get("/api/admin/settings", headers=headers)
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["smtp_host"] == "smtp.example.com"
    assert body["smtp_port"] == 2525
    assert body["smtp_user"] == "ops@example.com"
    assert body["smtp_from_email"] == "noreply@example.com"
    assert body["smtp_password_set"] is True
    assert body["stripe_publishable_key"] == "pk_test_123"
    assert body["stripe_secret_key_set"] is True
    assert body["log_level"] == "DEBUG"
    assert body["telemetry_enabled"] is True
    assert body["access_token_expire_minutes"] == 45
    assert body["refresh_token_expire_days"] == 14
    assert body["history_enabled"] is False
    assert body["optimizer_prewarm_models"] is False
    assert body["cors_origins"] == "https://app.example.com, http://localhost:5173"


@pytest.mark.parametrize(
    "payload, detail_fragment",
    [
        ({"smtp_port": 70000}, "smtp_port must be between 1 and 65535"),
        ({"access_token_expire_minutes": 0}, "access_token_expire_minutes must be >= 1"),
        ({"refresh_token_expire_days": "bad"}, "refresh_token_expire_days must be an integer"),
        ({"telemetry_enabled": "true"}, "telemetry_enabled must be a boolean"),
        ({"log_level": "trace"}, "log_level must be one of"),
    ],
)
def test_admin_settings_rejects_invalid_values(payload, detail_fragment):
    headers = login_as_admin()
    response = client.patch("/api/admin/settings", json=payload, headers=headers)
    assert response.status_code == 400
    assert detail_fragment in response.json()["detail"]
