from typing import List

from pydantic import BaseModel, ConfigDict, Field


class CanonicalMappingResponse(BaseModel):
    """Response model for a single canonical mapping."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., description="Unique identifier")
    source_token: str = Field(..., description="Source token (long form)")
    target_token: str = Field(..., description="Target token (abbreviated form)")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")


class CanonicalMappingCreate(BaseModel):
    """Request model for creating a single canonical mapping."""

    model_config = ConfigDict(extra="forbid")

    source_token: str = Field(..., min_length=1, description="Source token (long form)")
    target_token: str = Field(
        ..., min_length=1, description="Target token (abbreviated form)"
    )


class CanonicalMappingBulkCreate(BaseModel):
    """Request model for bulk creating canonical mappings."""

    model_config = ConfigDict(extra="forbid")

    mappings: List[CanonicalMappingCreate] = Field(
        ..., min_length=1, description="List of mappings to create"
    )


class CanonicalMappingUpdate(BaseModel):
    """Request model for updating a canonical mapping."""

    model_config = ConfigDict(extra="forbid")

    source_token: str = Field(..., min_length=1, description="Source token (long form)")
    target_token: str = Field(
        ..., min_length=1, description="Target token (abbreviated form)"
    )


class CanonicalMappingListResponse(BaseModel):
    """Response model for listing canonical mappings."""

    model_config = ConfigDict(extra="forbid")

    mappings: List[CanonicalMappingResponse] = Field(
        ..., description="List of canonical mappings"
    )
    total: int = Field(..., ge=0, description="Total number of mappings")
    offset: int = Field(..., ge=0, description="Offset used for pagination")
    limit: int = Field(..., ge=1, description="Limit used for pagination")


class CanonicalMappingBulkDelete(BaseModel):
    """Request model for bulk deleting canonical mappings."""

    model_config = ConfigDict(extra="forbid")

    ids: List[int] = Field(
        ..., min_length=1, description="List of mapping IDs to delete"
    )


class CanonicalMappingDeleteResponse(BaseModel):
    """Response model for delete operation."""

    model_config = ConfigDict(extra="forbid")

    deleted_count: int = Field(..., ge=0, description="Number of mappings deleted")
