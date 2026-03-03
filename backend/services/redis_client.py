from __future__ import annotations

import os
from typing import Optional

import redis

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    return _redis_client
