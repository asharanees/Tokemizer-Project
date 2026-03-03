from datetime import timedelta
import uuid

import auth_utils
import pytest
from database import create_customer, create_subscription_plan, init_db, update_customer
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_paid_enterprise_plan_allows_active_subscription_access():
    plan_id = f"enterprise_paid_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=plan_id,
        name="Enterprise Paid",
        monthly_price_cents=25000,
        annual_price_cents=250000,
        monthly_quota=100000,
    )

    email = f"enterprise_{uuid.uuid4().hex[:8]}@example.com"
    subscription_suffix = uuid.uuid4().hex[:8]
    customer = create_customer(name="Enterprise User", email=email)
    update_customer(
        customer.id,
        password_hash=auth_utils.get_password_hash("password123"),
        subscription_tier=plan_id,
        subscription_status="active",
        stripe_subscription_id=f"sub_active_{subscription_suffix}",
        stripe_subscription_item_id=f"item_active_{subscription_suffix}",
    )

    access_token = auth_utils.create_access_token(
        data={"sub": email, "id": customer.id, "role": customer.role}
    )
    response = client.put(
        "/api/auth/profile",
        json={"name": "Enterprise User Updated"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200


def test_refresh_token_restores_access_after_expired_access_token():
    email = f"refresh_{uuid.uuid4().hex[:8]}@example.com"
    customer = create_customer(name="Refresh User", email=email)
    update_customer(
        customer.id,
        password_hash=auth_utils.get_password_hash("password123"),
        subscription_status="active",
    )

    expired_access_token = auth_utils.create_access_token(
        data={"sub": email, "id": customer.id, "role": customer.role},
        expires_delta=timedelta(seconds=-1),
    )
    refresh_token = auth_utils.create_refresh_token(
        data={"sub": email, "id": customer.id}
    )

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {expired_access_token}"},
    )
    assert response.status_code == 401

    refresh_response = client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert refresh_response.status_code == 200
    refreshed_data = refresh_response.json()
    assert "access_token" in refreshed_data

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {refreshed_data['access_token']}"},
    )
    assert me_response.status_code == 200


def test_register_rejects_contact_sales_plan_with_negative_price():
    plan_id = f"enterprise_contact_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=plan_id,
        name="Enterprise Contact",
        monthly_price_cents=-1,
        annual_price_cents=None,
        monthly_quota=-1,
    )

    response = client.post(
        "/api/auth/register",
        json={
            "email": f"contact_register_{uuid.uuid4().hex[:8]}@example.com",
            "password": "password123",
            "name": "Contact Sales User",
            "plan_id": plan_id,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Selected plan requires contacting sales"


def test_upgrade_rejects_contact_sales_plan_with_negative_price():
    base_plan_id = f"starter_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=base_plan_id,
        name="Starter",
        monthly_price_cents=0,
        annual_price_cents=None,
        monthly_quota=1000,
    )
    contact_plan_id = f"enterprise_contact_upgrade_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=contact_plan_id,
        name="Enterprise Contact Upgrade",
        monthly_price_cents=-1,
        annual_price_cents=None,
        monthly_quota=-1,
    )

    email = f"upgrade_contact_{uuid.uuid4().hex[:8]}@example.com"
    customer = create_customer(name="Upgrade Contact User", email=email)
    update_customer(
        customer.id,
        password_hash=auth_utils.get_password_hash("password123"),
        subscription_tier=base_plan_id,
        subscription_status="active",
    )

    access_token = auth_utils.create_access_token(
        data={"sub": email, "id": customer.id, "role": customer.role}
    )
    response = client.post(
        "/api/subscription/upgrade",
        json={"plan_id": contact_plan_id},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Selected plan requires contacting sales"


def test_register_rejects_active_non_public_plan():
    hidden_plan_id = f"enterprise_hidden_register_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=hidden_plan_id,
        name="Enterprise Hidden Register",
        monthly_price_cents=0,
        annual_price_cents=None,
        monthly_quota=5000,
        is_active=True,
        is_public=False,
    )

    plans_response = client.get("/api/auth/plans")
    assert plans_response.status_code == 200
    assert hidden_plan_id not in {plan["id"] for plan in plans_response.json()}

    response = client.post(
        "/api/auth/register",
        json={
            "email": f"hidden_register_{uuid.uuid4().hex[:8]}@example.com",
            "password": "password123",
            "name": "Hidden Register User",
            "plan_id": hidden_plan_id,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == f"Invalid plan: {hidden_plan_id}"


def test_upgrade_rejects_active_non_public_plan():
    base_plan_id = f"starter_public_{uuid.uuid4().hex[:8]}"
    hidden_plan_id = f"enterprise_hidden_upgrade_{uuid.uuid4().hex[:8]}"
    create_subscription_plan(
        id=base_plan_id,
        name="Starter Public",
        monthly_price_cents=0,
        annual_price_cents=None,
        monthly_quota=1000,
        is_active=True,
        is_public=True,
    )
    create_subscription_plan(
        id=hidden_plan_id,
        name="Enterprise Hidden Upgrade",
        monthly_price_cents=0,
        annual_price_cents=None,
        monthly_quota=20000,
        is_active=True,
        is_public=False,
    )

    email = f"hidden_upgrade_{uuid.uuid4().hex[:8]}@example.com"
    customer = create_customer(name="Hidden Upgrade User", email=email)
    update_customer(
        customer.id,
        password_hash=auth_utils.get_password_hash("password123"),
        subscription_tier=base_plan_id,
        subscription_status="active",
    )

    access_token = auth_utils.create_access_token(
        data={"sub": email, "id": customer.id, "role": customer.role}
    )
    response = client.post(
        "/api/subscription/upgrade",
        json={"plan_id": hidden_plan_id},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid plan"
