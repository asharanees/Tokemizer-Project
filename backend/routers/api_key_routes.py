import secrets
from typing import List, Optional

from auth import get_current_customer, hash_api_key
from database import (Customer, create_api_key, delete_api_key, list_api_keys,
                      get_subscription_plan_by_id)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/keys", tags=["api_keys"])


class ApiKeyCreate(BaseModel):
    name: str


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: Optional[str]
    is_active: bool


class ApiKeyCreatedResponse(ApiKeyResponse):
    key: str  # Only returned on creation


@router.get("", response_model=List[ApiKeyResponse])
def get_my_api_keys(customer: Customer = Depends(get_current_customer)):
    keys = list_api_keys(customer.id)
    # Don't return full hash, maybe just masked/prefix?
    # ApiKey model (database) has key_hash.
    # Response model keys are derived.
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": "tok_..."
            + k.id[:4],  # Just a dummy prefix indicator or derive from something?
            # Actually, we don't store the prefix/clear key. We only store hash.
            # So we can't show the key again.
            # We can show a partial ID or something.
            # Let's say we don't return the key itself except on creation.
            "created_at": k.created_at,
            "last_used_at": k.last_used_at,
            "is_active": k.is_active,
        }
        for k in keys
    ]


@router.post("", response_model=ApiKeyCreatedResponse)
def create_new_api_key(
    data: ApiKeyCreate, customer: Customer = Depends(get_current_customer)
):
    if customer.role != "admin":
        current_keys = list_api_keys(customer.id)
        plan = get_subscription_plan_by_id(
            customer.subscription_tier,
            include_inactive=True,
            include_non_public=True,
        )
        plan_limit = plan.max_api_keys if plan else None
        if plan_limit is None:
            max_keys_allowed = 10
        elif plan_limit < 0:
            max_keys_allowed = None
        else:
            max_keys_allowed = plan_limit

        if max_keys_allowed is not None and len(current_keys) >= max_keys_allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum limit of {max_keys_allowed} API keys reached.",
            )

    # Generate key
    raw_key = f"tok_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(raw_key)

    new_key_record = create_api_key(customer.id, data.name, key_hash)

    return {
        "id": new_key_record.id,
        "name": new_key_record.name,
        "prefix": raw_key[:7] + "...",
        "created_at": new_key_record.created_at,
        "last_used_at": None,
        "is_active": True,
        "key": raw_key,
    }


@router.delete("/{key_id}")
def revoke_api_key(key_id: str, customer: Customer = Depends(get_current_customer)):
    success = delete_api_key(key_id, customer.id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "success"}
