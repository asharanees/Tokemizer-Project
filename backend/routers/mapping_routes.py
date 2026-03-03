from typing import List

from auth import get_current_customer
from database import (
    Customer,
    create_user_canonical_mapping,
    get_subscription_plan_by_id,
    delete_user_canonical_mapping,
    increment_canonical_mappings_cache_version,
    list_disabled_ootb_mappings,
    list_user_canonical_mappings,
    toggle_ootb_mapping,
)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.cache_invalidation import invalidate_canonical_mappings_cache

router = APIRouter(prefix="/api/v1/mappings", tags=["mappings"])


def _enforce_custom_mapping_limit(customer: Customer) -> None:
    plan = get_subscription_plan_by_id(
        customer.subscription_tier,
        include_inactive=True,
        include_non_public=True,
    )
    mapping_limit = int(getattr(plan, "custom_canonical_mappings_limit", 1000) or 1000)
    current = list_user_canonical_mappings(customer.id)
    if len(current) >= mapping_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Custom canonical mappings limit reached ({mapping_limit})",
        )


class MappingCreate(BaseModel):
    source_token: str
    target_token: str


class MappingUpdate(BaseModel):
    source_token: str
    target_token: str


class MappingResponse(BaseModel):
    id: int
    source_token: str
    target_token: str
    created_at: str
    updated_at: str


class DisabledOotbResponse(BaseModel):
    tokens: List[str]


@router.get("", response_model=List[MappingResponse])
async def get_my_mappings(customer: Customer = Depends(get_current_customer)):
    """List customer's custom mappings."""
    mappings = list_user_canonical_mappings(customer.id)
    return [
        MappingResponse(
            id=m.id,
            source_token=m.source_token,
            target_token=m.target_token,
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in mappings
    ]


@router.post("", response_model=MappingResponse)
async def add_mapping(
    data: MappingCreate, customer: Customer = Depends(get_current_customer)
):
    """Create or update a customer mapping (upsert by source_token)."""
    existing = list_user_canonical_mappings(customer.id)
    exists_for_source = any(m.source_token.lower() == data.source_token.strip().lower() for m in existing)
    if not exists_for_source:
        _enforce_custom_mapping_limit(customer)
    m = create_user_canonical_mapping(customer.id, data.source_token, data.target_token)
    increment_canonical_mappings_cache_version()
    invalidate_canonical_mappings_cache(increment_db_version=False)
    return MappingResponse(
        id=m.id,
        source_token=m.source_token,
        target_token=m.target_token,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


@router.put("/{mapping_id}", response_model=MappingResponse)
async def update_mapping(
    mapping_id: int,
    data: MappingUpdate,
    customer: Customer = Depends(get_current_customer),
):
    """Update a customer mapping by ID.

    If the source_token changes, the old record is deleted and a new one is created
    to maintain proper uniqueness constraints.
    """
    # Get current mappings to check if this ID exists
    current_mappings = list_user_canonical_mappings(customer.id)
    existing = next((m for m in current_mappings if m.id == mapping_id), None)

    if not existing:
        raise HTTPException(status_code=404, detail="Mapping not found")

    # If source_token changed, delete old record first
    # This ensures no duplicate/orphan records remain
    if existing.source_token.lower() != data.source_token.strip().lower():
        delete_user_canonical_mapping(customer.id, mapping_id)

    # Create/upsert with new values
    existing = list_user_canonical_mappings(customer.id)
    exists_for_source = any(m.source_token.lower() == data.source_token.strip().lower() for m in existing)
    if not exists_for_source:
        _enforce_custom_mapping_limit(customer)
    m = create_user_canonical_mapping(customer.id, data.source_token, data.target_token)
    increment_canonical_mappings_cache_version()
    invalidate_canonical_mappings_cache(increment_db_version=False)
    return MappingResponse(
        id=m.id,
        source_token=m.source_token,
        target_token=m.target_token,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


@router.delete("/{mapping_id}")
async def remove_mapping(
    mapping_id: int, customer: Customer = Depends(get_current_customer)
):
    """Delete a customer mapping."""
    success = delete_user_canonical_mapping(customer.id, mapping_id)
    if not success:
        raise HTTPException(status_code=404, detail="Mapping not found")
    increment_canonical_mappings_cache_version()
    invalidate_canonical_mappings_cache(increment_db_version=False)
    return {"status": "success"}


@router.post("/toggle-ootb")
async def toggle_default_mapping(
    source_token: str, enabled: bool, customer: Customer = Depends(get_current_customer)
):
    """Enable or disable a global/OOTB mapping for this user."""
    toggle_ootb_mapping(customer.id, source_token, enabled)
    increment_canonical_mappings_cache_version()
    invalidate_canonical_mappings_cache(increment_db_version=False)
    return {"status": "success"}


@router.get("/disabled-ootb", response_model=DisabledOotbResponse)
async def get_disabled_ootb_mappings(
    customer: Customer = Depends(get_current_customer),
):
    tokens = list_disabled_ootb_mappings(customer.id)
    return DisabledOotbResponse(tokens=tokens)
