import logging
import os
from typing import Optional

import stripe
from database import (Customer, SubscriptionPlan, get_customer_by_stripe_id,
                      list_subscription_plans, update_customer_subscription)
from fastapi import APIRouter, Header, HTTPException, Request
from services.billing import (resolve_subscription_tier,
                              sync_subscription_from_stripe)
from services.email import send_plan_change_email

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")


@router.post("/stripe")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Handle Stripe webhooks."""
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        handle_checkout_session(session)
    elif event["type"] == "invoice.paid":
        invoice = event["data"]["object"]
        handle_invoice_paid(invoice)
    elif event["type"] == "customer.subscription.updated":
        subscription = event["data"]["object"]
        handle_subscription_updated(subscription)
    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        handle_subscription_deleted(subscription)
    else:
        logger.info(f"Unhandled event type {event['type']}")

    return {"status": "success"}


def handle_checkout_session(session):
    stripe_customer_id = session.get("customer")
    subscription_id = session.get("subscription")
    plan_lookup = _plan_lookup()
    customer = (
        get_customer_by_stripe_id(stripe_customer_id) if stripe_customer_id else None
    )
    old_plan_name = _plan_name_from_lookup(
        plan_lookup, customer.subscription_tier if customer else None
    )
    if subscription_id:
        sub = stripe.Subscription.retrieve(subscription_id)
        tier, item_id = resolve_subscription_tier(sub)
        updated = update_customer_subscription(
            stripe_customer_id,
            status=sub.status,
            tier=tier,
            stripe_subscription_id=subscription_id,
            stripe_subscription_item_id=item_id,
        )
        if updated:
            _notify_plan_change(customer, old_plan_name, tier, plan_lookup)


def handle_subscription_updated(subscription):
    stripe_customer_id = subscription.get("customer")
    plan_lookup = _plan_lookup()
    customer = (
        get_customer_by_stripe_id(stripe_customer_id) if stripe_customer_id else None
    )
    old_plan_name = _plan_name_from_lookup(
        plan_lookup, customer.subscription_tier if customer else None
    )
    tier, item_id = resolve_subscription_tier(subscription)
    updated = update_customer_subscription(
        stripe_customer_id,
        status=subscription.status,
        tier=tier,
        stripe_subscription_id=subscription.id,
        stripe_subscription_item_id=item_id,
    )
    if updated:
        _notify_plan_change(customer, old_plan_name, tier, plan_lookup)


def handle_subscription_deleted(subscription):
    stripe_customer_id = subscription.get("customer")
    plan_lookup = _plan_lookup()
    customer = (
        get_customer_by_stripe_id(stripe_customer_id) if stripe_customer_id else None
    )
    old_plan_name = _plan_name_from_lookup(
        plan_lookup, customer.subscription_tier if customer else None
    )
    updated = update_customer_subscription(
        stripe_customer_id,
        status="canceled",
        tier="free",
        stripe_subscription_id=None,
        stripe_subscription_item_id=None,
    )
    if updated:
        _notify_plan_change(customer, old_plan_name, "free", plan_lookup)


def handle_invoice_paid(invoice):
    stripe_customer_id = invoice.get("customer")
    if stripe_customer_id:
        sync_subscription_from_stripe(stripe_customer_id)


def _plan_lookup() -> dict[str, SubscriptionPlan]:
    return {plan.id: plan for plan in list_subscription_plans(include_inactive=True)}


def _plan_name_from_lookup(
    plan_lookup: dict[str, SubscriptionPlan], tier: Optional[str]
) -> str:
    plan = plan_lookup.get(tier)
    return plan.name if plan else (tier or "previous plan")


def _notify_plan_change(
    customer: Optional[Customer],
    old_plan_name: str,
    new_tier: Optional[str],
    plan_lookup: dict[str, SubscriptionPlan],
) -> None:
    if not customer or not new_tier:
        return
    if not customer.email:
        return
    new_plan = plan_lookup.get(new_tier)
    if not new_plan:
        return
    try:
        send_plan_change_email(
            to_email=customer.email,
            customer_name=customer.name or customer.email,
            old_plan_name=old_plan_name,
            plan=new_plan,
        )
    except Exception:
        logger.exception("Failed to send plan change email to %s", customer.email)
