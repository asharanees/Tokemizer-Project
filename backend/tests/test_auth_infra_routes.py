from fastapi.testclient import TestClient

import routers.auth_routes as auth_routes
from server import app

client = TestClient(app)


def test_get_ec2_status_endpoint(monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "invoke_ec2_control",
        lambda action: {
            "ok": True,
            "action": action,
            "instance_state": "running",
            "details": {"state": "running"},
        },
    )

    response = client.get("/api/auth/infrastructure/ec2/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "status"
    assert payload["instance_state"] == "running"


def test_post_ec2_start_endpoint(monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "invoke_ec2_control",
        lambda action: {
            "ok": True,
            "action": action,
            "instance_state": "pending",
            "details": {"state": "pending"},
        },
    )

    response = client.post("/api/auth/infrastructure/ec2/start")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "start"


def test_post_ec2_stop_endpoint(monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "invoke_ec2_control",
        lambda action: {
            "ok": True,
            "action": action,
            "instance_state": "stopping",
            "details": {"state": "stopping"},
        },
    )

    response = client.post("/api/auth/infrastructure/ec2/stop")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "stop"
