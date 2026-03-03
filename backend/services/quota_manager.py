from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Dict, Optional, Tuple

from database import get_customer_by_id, get_subscription_plan_by_id, get_usage, increment_usage


class QuotaManager:
    """Manages monthly token quotas and enforcement."""

    def __init__(self):
        self._cache: Dict[str, "QuotaState"] = {}
        self._cache_ttl_seconds = 60
        self._pending: Dict[Tuple[str, str, str, Optional[str]], int] = {}
        self._lock = Lock()
        self._flush_interval_seconds = 5
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _resolve_quota_limit(self, customer_id: str) -> int:
        customer = get_customer_by_id(customer_id)
        if not customer:
            return 0
        if customer.quota_override is not None:
            return customer.quota_override + int(customer.quota_overage_bonus or 0)
        plan = get_subscription_plan_by_id(
            customer.subscription_tier,
            include_inactive=True,
            include_non_public=True,
        )
        base_quota = plan.monthly_quota if plan else 1000
        return base_quota + int(customer.quota_overage_bonus or 0)

    def _get_period_bounds(self, now: datetime) -> Tuple[str, str]:
        period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        if now.month == 12:
            period_end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            period_end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
        return period_start.isoformat(), period_end.isoformat()

    def _get_pending_usage(self, customer_id: str, period_start: str) -> int:
        with self._lock:
            return sum(
                count
                for (cid, start, _, _), count in self._pending.items()
                if cid == customer_id and start == period_start
            )

    def check_quota(self, customer_id: str) -> Tuple[bool, int, int]:
        """Check if customer has remaining monthly quota.
        Returns: (allowed, remaining, total_quota)
        """
        customer = get_customer_by_id(customer_id)
        if not customer:
            return False, 0, 0

        if customer.role == "admin":
            return True, 1000000000, 1000000000

        now = datetime.now(timezone.utc)
        period_start, period_end = self._get_period_bounds(now)
        max_quota = self._resolve_quota_limit(customer_id)

        cached = self._cache.get(customer_id)
        if cached and now.timestamp() - cached.cached_at <= self._cache_ttl_seconds:
            cached.limit = max_quota
            if max_quota < 0:
                return True, 1000000000, max_quota
            remaining = max(0, max_quota - cached.used)
            return remaining > 0, remaining, max_quota

        usage_record = get_usage(customer_id, period_start)
        current_usage = usage_record.calls_used if usage_record else 0
        current_usage += self._get_pending_usage(customer_id, period_start)

        with self._lock:
            self._cache[customer_id] = QuotaState(
                limit=max_quota,
                used=current_usage,
                period_start=period_start,
                period_end=period_end,
                cached_at=now.timestamp(),
            )

        remaining = max_quota - current_usage
        if max_quota < 0:
            # Unlimited quota
            return True, 1000000000, max_quota

        remaining = max(0, max_quota - current_usage)
        return remaining > 0, remaining, max_quota

    def increment_usage(
        self, customer_id: str, api_key_id: Optional[str], count: int = 1
    ) -> None:
        if count <= 0:
            return
        now = datetime.now(timezone.utc)
        period_start, period_end = self._get_period_bounds(now)
        key = (customer_id, period_start, period_end, api_key_id)
        with self._lock:
            self._pending[key] = self._pending.get(key, 0) + count
            cached = self._cache.get(customer_id)
            if cached and cached.period_start == period_start:
                cached.used += count

    def flush_pending(self) -> None:
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
        if not pending:
            return
        for (
            customer_id,
            period_start,
            period_end,
            api_key_id,
        ), count in pending.items():
            increment_usage(customer_id, period_start, period_end, count, api_key_id)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._flush_interval_seconds)
            self.flush_pending()


@dataclass
class QuotaState:
    limit: int
    used: int
    period_start: str
    period_end: str
    cached_at: float


quota_manager = QuotaManager()
