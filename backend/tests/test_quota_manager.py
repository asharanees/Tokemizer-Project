import uuid

import auth_utils
from database import create_customer, create_subscription_plan, init_db, update_customer
from services.quota_manager import quota_manager


def test_quota_manager_applies_updated_plan_limit_without_restart():
    init_db()
    plan_id = f"dynamic_quota_{uuid.uuid4().hex[:8]}"

    create_subscription_plan(
        id=plan_id,
        name="Dynamic Quota Plan",
        monthly_price_cents=0,
        monthly_quota=2,
        is_active=True,
    )

    email = f"dynamic_quota_{uuid.uuid4().hex[:8]}@example.com"
    customer = create_customer(name="Dynamic Quota User", email=email)
    update_customer(
        customer.id,
        password_hash=auth_utils.get_password_hash("password123"),
        subscription_tier=plan_id,
        subscription_status="active",
    )

    allowed_before, remaining_before, total_before = quota_manager.check_quota(customer.id)
    assert allowed_before is True
    assert remaining_before == 2
    assert total_before == 2

    create_subscription_plan(
        id=plan_id,
        name="Dynamic Quota Plan",
        monthly_price_cents=0,
        monthly_quota=7,
        is_active=True,
    )

    allowed_after, remaining_after, total_after = quota_manager.check_quota(customer.id)
    assert allowed_after is True
    assert remaining_after == 7
    assert total_after == 7
