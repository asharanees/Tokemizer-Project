import os
from typing import Any, Dict, Optional, Tuple

import stripe
from database import get_plan_by_stripe_price_id, update_customer_subscription

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")


def create_stripe_customer(email: str, name: str) -> str:
    """Create a new customer in Stripe and return the customer ID."""
    customer = stripe.Customer.create(
        email=email,
        name=name,
    )
    return customer.id


def create_checkout_session(
    customer_id: str, success_url: str, cancel_url: str, price_id: str
) -> str:
    """Create a Stripe Checkout session for a subscription."""
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[
            {
                "price": price_id,
                "quantity": 1,
            }
        ],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
    )
    return session.url


def report_usage(subscription_item_id: str, quantity: int):
    """Report usage to Stripe for metered billing."""
    stripe.SubscriptionItem.create_usage_record(
        subscription_item_id,
        quantity=quantity,
        timestamp="now",
        action="increment",
    )


def create_billing_portal_session(customer_id: str, return_url: str) -> str:
    """Create a Stripe Billing Portal session."""
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url


def update_stripe_subscription(subscription_id: str, price_id: str) -> Dict[str, Any]:
    subscription = stripe.Subscription.retrieve(subscription_id)
    items = subscription.get("items", {}).get("data", [])
    if not items:
        raise ValueError("Subscription has no items")
    item_id = items[0]["id"]
    return stripe.Subscription.modify(
        subscription_id,
        cancel_at_period_end=False,
        items=[{"id": item_id, "price": price_id}],
    )


def cancel_stripe_subscription(
    subscription_id: str, at_period_end: bool = True
) -> Dict[str, Any]:
    if at_period_end:
        return stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True,
        )
    return stripe.Subscription.delete(subscription_id)


def sync_subscription_from_stripe(stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    subscriptions = stripe.Subscription.list(
        customer=stripe_customer_id,
        status="all",
        limit=1,
    )
    data = subscriptions.get("data", [])
    if not data:
        update_customer_subscription(
            stripe_customer_id,
            status="inactive",
            tier="free",
            stripe_subscription_id=None,
            stripe_subscription_item_id=None,
        )
        return None
    subscription = data[0]
    tier, item_id = resolve_subscription_tier(subscription)
    update_customer_subscription(
        stripe_customer_id,
        status=subscription.get("status", "inactive"),
        tier=tier,
        stripe_subscription_id=subscription.get("id"),
        stripe_subscription_item_id=item_id,
    )
    return subscription


def resolve_subscription_tier(
    subscription: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    items = subscription.get("items", {}).get("data", [])
    if not items:
        return None, None
    first_item = items[0]
    price = first_item.get("price", {})
    price_id = price.get("id") if isinstance(price, dict) else price
    plan = get_plan_by_stripe_price_id(price_id) if price_id else None
    tier = plan.id if plan else None
    return tier, first_item.get("id")
