from datetime import datetime, timezone

from auth import get_current_customer
from database import (Customer, get_usage, list_usage_breakdown,
                      list_usage_history)
from fastapi import APIRouter, Depends
from services.quota_manager import quota_manager

router = APIRouter(prefix="/api/usage", tags=["usage"])


@router.get("")
async def get_current_usage(customer: Customer = Depends(get_current_customer)):
    """Get current usage and quota status."""
    now = datetime.now(timezone.utc)
    period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()

    usage_record = get_usage(customer.id, period_start)
    _, remaining, total = quota_manager.check_quota(customer.id)
    breakdown = list_usage_breakdown(customer.id, period_start)

    return {
        "calls_used": usage_record.calls_used if usage_record else 0,
        "quota_limit": total,
        "remaining": remaining,
        "period_start": period_start,
        "subscription_tier": customer.subscription_tier,
        "subscription_status": customer.subscription_status,
        "breakdown": [
            {
                "source": entry.source,
                "calls": entry.calls_used,
                "api_key_id": entry.api_key_id,
                "name": entry.api_key_name,
            }
            for entry in breakdown
        ],
    }


@router.get("/history")
async def get_usage_history(
    limit: int = 6, customer: Customer = Depends(get_current_customer)
):
    """Get usage history by period."""
    history = list_usage_history(customer.id, limit=limit)
    return {
        "history": [
            {
                "period_start": row.period_start,
                "period_end": row.period_end,
                "calls_used": row.calls_used,
            }
            for row in history
        ]
    }
