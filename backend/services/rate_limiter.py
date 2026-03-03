from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict, Optional, Tuple

from services.redis_client import get_redis_client


class RateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, window_seconds: int = 60, max_requests: int = 60):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._redis = get_redis_client()
        self._local: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def check_rate_limit(
        self, key: str, max_requests: Optional[int] = None
    ) -> Tuple[bool, Dict[str, int]]:
        limit = max_requests if max_requests is not None else self.max_requests
        now = time.time()
        window_start = now - self.window_seconds
        reset_at = int(now + self.window_seconds)

        if self._redis is not None:
            try:
                redis_key = f"rate:{key}"
                pipe = self._redis.pipeline()
                pipe.zremrangebyscore(redis_key, 0, window_start)
                pipe.zadd(redis_key, {str(now): now})
                pipe.zcard(redis_key)
                pipe.zrange(redis_key, 0, 0, withscores=True)
                pipe.expire(redis_key, int(self.window_seconds * 2))
                _, _, count, oldest, _ = pipe.execute()
                remaining = max(0, limit - count)
                if count > limit:
                    if oldest:
                        oldest_ts = float(oldest[0][1])
                        reset_at = int(oldest_ts + self.window_seconds)
                    retry_after = max(0, int(reset_at - now))
                    return False, {
                        "remaining": 0,
                        "limit": limit,
                        "reset_at": reset_at,
                        "retry_after": retry_after,
                    }
                return True, {
                    "remaining": remaining,
                    "limit": limit,
                    "reset_at": reset_at,
                    "retry_after": 0,
                }
            except Exception:
                self._redis = None

        with self._lock:
            bucket = self._local.get(key)
            if bucket is None:
                bucket = deque()
                self._local[key] = bucket
            while bucket and bucket[0] <= window_start:
                bucket.popleft()
            if len(bucket) >= limit:
                oldest_ts = bucket[0] if bucket else now
                reset_at = int(oldest_ts + self.window_seconds)
                retry_after = max(0, int(reset_at - now))
                return False, {
                    "remaining": 0,
                    "limit": limit,
                    "reset_at": reset_at,
                    "retry_after": retry_after,
                }
            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return True, {
                "remaining": remaining,
                "limit": limit,
                "reset_at": int(now + self.window_seconds),
                "retry_after": 0,
            }


# Global rate limiter instance for API requests
api_rate_limiter = RateLimiter(window_seconds=60, max_requests=60)
