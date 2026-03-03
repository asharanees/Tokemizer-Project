from typing import Optional

from auth import get_current_active_customer_unrestricted
from database import (Customer, get_subscription_plan_by_id, plan_requires_payment,
                      plan_requires_sales_contact,
                      update_customer)
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from services.billing import (create_billing_portal_session,
                              create_checkout_session, create_stripe_customer)

router = APIRouter(prefix="/api/billing", tags=["billing"])


class CheckoutSessionRequest(BaseModel):
    plan_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


def _resolve_base_url(request: Request) -> str:
    origin = request.headers.get("origin")
    if origin:
        return origin.rstrip("/")
    return str(request.base_url).rstrip("/")


@router.post("/checkout-session")
async def create_checkout_session_endpoint(
    payload: CheckoutSessionRequest,
    request: Request,
    customer: Customer = Depends(get_current_active_customer_unrestricted),
):
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

    if not plan_requires_payment(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected plan does not require payment",
        )

    if not plan.stripe_price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected plan is missing Stripe pricing configuration",
        )

    stripe_customer_id = customer.stripe_customer_id
    if not stripe_customer_id:
        if not customer.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer email is required for Stripe checkout",
            )
        stripe_customer_id = create_stripe_customer(
            customer.email,
            customer.name or customer.email,
        )
        update_customer(customer.id, stripe_customer_id=stripe_customer_id)

    base_url = _resolve_base_url(request)
    success_url = payload.success_url or f"{base_url}/subscription?checkout=success"
    cancel_url = payload.cancel_url or f"{base_url}/subscription?checkout=cancel"

    checkout_url = create_checkout_session(
        stripe_customer_id,
        success_url,
        cancel_url,
        plan.stripe_price_id,
    )

    return {"url": checkout_url}


@router.post("/portal-session")
async def create_portal_session_endpoint(
    request: Request,
    customer: Customer = Depends(get_current_active_customer_unrestricted),
):
    if not customer.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer is missing Stripe configuration",
        )

    base_url = _resolve_base_url(request)
    return_url = f"{base_url}/subscription"
    portal_url = create_billing_portal_session(
        customer.stripe_customer_id,
        return_url,
    )

    return {"url": portal_url}
