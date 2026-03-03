import pytest
from database import init_db
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_register_flow():
    # 1. Register
    import uuid

    random_suffix = str(uuid.uuid4())[:8]
    email = f"test_auth_{random_suffix}@example.com"
    plans_response = client.get("/api/auth/plans")
    assert plans_response.status_code == 200
    plans = plans_response.json()
    assert plans
    selected_plan_id = plans[0]["id"]

    reg_data = {
        "email": email,
        "password": "securepassword123",
        "name": "Test User",
        "plan_id": selected_plan_id,
    }
    response = client.post("/api/auth/register", json=reg_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == reg_data["email"]
    assert data["user"]["role"] == "customer"

    # 2. Login
    login_data = {"username": email, "password": "securepassword123"}
    response = client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data

    # 3. Access Protected Endpoint (e.g. /me)
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200
    me_data = response.json()
    assert me_data["email"] == reg_data["email"]


def test_login_invalid_credentials():
    login_data = {"username": "wrong@example.com", "password": "wrongpassword"}
    response = client.post("/api/auth/login", data=login_data)
    assert response.status_code == 401


def test_protected_endpoint_no_auth():
    response = client.get("/api/auth/me")
    assert response.status_code == 401
