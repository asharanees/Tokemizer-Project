import logging
from typing import List, Optional

from auth import get_current_active_customer_unrestricted
from database import (Customer, get_subscription_plan_by_id, list_subscription_plans, plan_requires_payment,
                      plan_requires_sales_contact,
                      update_customer)
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from services.email import send_plan_change_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subscription", tags=["subscription"])


class SubscriptionUpgradeRequest(BaseModel):
    plan_id: str


class PlanResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    monthly_price_cents: int
    annual_price_cents: Optional[int] = None
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
    is_public: bool = True
    plan_term: str = "monthly"
    monthly_discount_percent: int = 0
    yearly_discount_percent: int = 0


@router.get("/plans", response_model=List[PlanResponse])
async def list_public_plans():
    """List active and public catalog plans for pricing pages."""
    plans = list_subscription_plans(include_inactive=False, include_non_public=False)
    plans.sort(
        key=lambda plan: (
            1 if plan_requires_sales_contact(plan) else 0,
            plan.monthly_price_cents if plan.monthly_price_cents >= 0 else 0,
            plan.monthly_quota if plan.monthly_quota >= 0 else 0,
        )
    )
    return [
        PlanResponse(
            id=p.id,
            name=p.name,
            description=p.description,
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


@router.post("/upgrade")
async def upgrade_subscription(
    payload: SubscriptionUpgradeRequest,
    customer: Customer = Depends(get_current_active_customer_unrestricted),
):
    all_plans = list_subscription_plans(include_inactive=True, include_non_public=True)
    plan = get_subscription_plan_by_id(
        payload.plan_id,
        include_inactive=False,
        include_non_public=False,
    )
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid plan"
        )

    if plan_requires_sales_contact(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected plan requires contacting sales",
        )

    if plan_requires_payment(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected plan requires payment",
        )

    if customer.stripe_subscription_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Manage paid subscriptions in the billing portal",
        )

    updated_user = update_customer(
        customer.id,
        subscription_status="active",
        subscription_tier=plan.id,
        stripe_subscription_id=None,
        stripe_subscription_item_id=None,
    )

    if not updated_user:
        raise HTTPException(status_code=500, detail="Failed to update subscription")

    old_plan_lookup = {p.id: p for p in all_plans}
    old_plan = old_plan_lookup.get(customer.subscription_tier)
    old_plan_name = (
        old_plan.name if old_plan else (customer.subscription_tier or "previous plan")
    )
    if updated_user:
        try:
            send_plan_change_email(
                to_email=customer.email,
                customer_name=customer.name or customer.email,
                old_plan_name=old_plan_name,
                plan=plan,
            )
        except Exception:
            logger.exception("Failed to send plan change email to %s", customer.email)

    return {
        "status": "success",
        "subscription_status": updated_user.subscription_status,
        "subscription_tier": updated_user.subscription_tier,
    }
