import hashlib
from datetime import datetime, timezone
from typing import Optional

import auth_utils
from database import (Customer, get_customer_by_api_key_hash,
                      get_customer_by_email, get_subscription_plan_by_id,
                      plan_requires_payment)
from fastapi import Depends, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError
from services.billing import report_usage
from services.quota_manager import quota_manager
from services.rate_limiter import api_rate_limiter

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# Removed hardcoded SUBSCRIPTION_TIERS in favor of database-driven plans


def hash_api_key(api_key: str) -> str:
    """Hash the API key for secure storage and lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_current_customer(
    response: Response,
    api_key: Optional[str] = Security(API_KEY_HEADER),
    token: Optional[str] = Security(OAUTH2_SCHEME),
) -> Customer:
    """
    Validate authentication via API Key or JWT Token.
    Returns the authenticated customer context.
    """

    # 1. Try API Key Authentication
    if api_key:
        key_hash = hash_api_key(api_key)
        customer = get_customer_by_api_key_hash(key_hash)
        if customer:
            if _subscription_requires_payment(customer):
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Subscription is inactive. Please update your billing information.",
                )
            # Enforce limits
            _check_rate_limit(customer, response)
            _check_quota(customer, response)
            return customer
        else:
            # Invalid API key provided
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

    # 2. Try JWT Token Authentication
    if token:
        try:
            payload = auth_utils.decode_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            customer = get_customer_by_email(email)
            if customer is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if _subscription_requires_payment(customer):
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Subscription is inactive. Please update your billing information.",
                )

            _check_rate_limit(customer, response)
            _check_quota(customer, response)
            return customer

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 3. No credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide X-API-Key header or Bearer token.",
    )


async def get_current_active_customer(
    current_customer: Customer = Depends(get_current_customer),
) -> Customer:
    """Verify user is active."""
    if not current_customer.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_customer


async def get_current_customer_unrestricted(
    api_key: Optional[str] = Security(API_KEY_HEADER),
    token: Optional[str] = Security(OAUTH2_SCHEME),
) -> Customer:
    if api_key:
        key_hash = hash_api_key(api_key)
        customer = get_customer_by_api_key_hash(key_hash)
        if customer:
            return customer
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    if token:
        try:
            payload = auth_utils.decode_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            customer = get_customer_by_email(email)
            if customer is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return customer

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide X-API-Key header or Bearer token.",
    )


async def get_current_active_customer_unrestricted(
    current_customer: Customer = Depends(get_current_customer_unrestricted),
) -> Customer:
    if not current_customer.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_customer


async def require_admin(
    current_customer: Customer = Depends(get_current_active_customer_unrestricted),
) -> Customer:
    """Verify user has admin role."""
    if current_customer.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_customer


def _check_rate_limit(customer: Customer, response: Optional[Response] = None):
    """Enforce rate limits based on customer identity."""
    # Admins bypass rate limits
    if customer.role == "admin":
        return
    plan = get_subscription_plan_by_id(
        customer.subscription_tier,
        include_inactive=True,
        include_non_public=True,
    )
    rpm_limit = plan.rate_limit_rpm if plan else 1000

    allowed, info = api_rate_limiter.check_rate_limit(
        customer.id, max_requests=rpm_limit
    )

    if response:
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset_at"])

    if not allowed:
        retry_after = info["retry_after"]
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _subscription_requires_payment(customer: Customer) -> bool:
    if customer.role == "admin":
        return False
    plan = get_subscription_plan_by_id(
        customer.subscription_tier,
        include_inactive=True,
        include_non_public=True,
    )
    if not plan or not plan_requires_payment(plan):
        return False
    if not customer.stripe_subscription_id:
        return True
    subscription_status = (customer.subscription_status or "").lower()
    return subscription_status == "canceled"


def _check_quota(customer: Customer, response: Optional[Response] = None):
    """Check if the customer has exceeded their monthly quota."""
    if customer.role == "admin":
        return
    allowed, remaining, total = quota_manager.check_quota(customer.id)
    used = max(total - remaining, 0)

    if response:
        response.headers["X-Quota-Limit"] = str(total)
        response.headers["X-Quota-Remaining"] = str(remaining)
        response.headers["X-Quota-Used"] = str(used)
        now = datetime.now(timezone.utc)
        if now.month == 12:
            reset_at = now.replace(
                year=now.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            reset_at = now.replace(
                month=now.month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        response.headers["X-Quota-Reset"] = reset_at.isoformat()

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Monthly usage quota of {total} token calls reached.",
        )


def track_usage(customer: Customer, count: int = 1):
    """Increment usage counter for the customer."""
    if count <= 0:
        return

    # Don't track usage for admins if desired, but usually good to track anyway
    # Let's track everyone for metrics

    api_key_id = getattr(customer, "api_key_id", None)
    quota_manager.increment_usage(customer.id, api_key_id, count=count)

    # Optional: Report to Stripe for metered billing
    if customer.stripe_subscription_item_id:
        try:
            report_usage(customer.stripe_subscription_item_id, count)
        except Exception as e:
            # We don't want to fail the request if reporting fails,
            # but we should log it for reconciliation.
            import logging

            logging.error(f"Stripe usage reporting failed for {customer.id}: {e}")
