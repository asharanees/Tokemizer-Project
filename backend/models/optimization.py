from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SegmentSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(..., ge=0, description="Start character index (inclusive)")
    end: int = Field(..., ge=1, description="End character index (exclusive)")
    label: Optional[str] = Field(
        None, description="Optional label used for segment weighting"
    )
    weight: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Segment weight (0-1). Higher preserves more.",
    )

    @model_validator(mode="after")
    def ensure_range_valid(self) -> "SegmentSpan":
        if self.end <= self.start:
            raise ValueError("segment_spans end must be greater than start")
        return self


class OptimizationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "summary": "Single prompt (balanced)",
                    "value": {
                        "prompt": "Summarize onboarding process",
                        "optimization_mode": "balanced",
                    },
                },
                {
                    "summary": "Batch with balanced mode",
                    "value": {
                        "prompts": ["Prompt A", "Prompt B"],
                        "optimization_mode": "balanced",
                    },
                },
                {
                    "summary": "Protect a critical span",
                    "value": {
                        "prompt": "Investigate account ABC123, compare with ABC124.",
                        "optimization_mode": "balanced",
                        "segment_spans": [{"start": 12, "end": 27, "weight": 1.0}],
                    },
                },
                {
                    "summary": "Query-aware compression (RAG)",
                    "value": {
                        "prompt": "Long retrieval context goes here...",
                        "optimization_mode": "balanced",
                        "query": "What are the SLOs and escalation paths?",
                    },
                },
            ]
        },
    )

    prompt: Optional[str] = Field(None, description="Single prompt to optimize")
    prompts: Optional[List[str]] = Field(
        None,
        description="Batch of prompts to optimize synchronously",
    )
    name: Optional[str] = Field(None, description="Friendly batch name for dashboards")
    optimization_mode: Literal["conservative", "balanced", "maximum"] = Field(
        "balanced",
        description=(
            "Optimization mode. 'conservative' (~70% faster; fewer passes), 'balanced' (~40% faster), "
            "'maximum' (all passes including entropy pruning, example compression, and history summarization)."
        ),
    )
    segment_spans: Optional[List[SegmentSpan]] = Field(
        None,
        description="Optional segment weights for preserving critical spans",
    )
    query: Optional[str] = Field(
        None,
        description="Optional query hint for query-aware compression (RAG only)",
    )
    custom_canonicals: Optional[Dict[str, str]] = Field(
        None,
        description="Optional per-request canonical mappings (long form -> short form)",
    )

    @field_validator("optimization_mode", mode="before")
    @classmethod
    def normalize_optimization_mode(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower()
        aliases = {
            "max": "maximum",
            "maximal": "maximum",
            "safe": "conservative",
            "normal": "balanced",
        }
        return aliases.get(normalized, normalized)

    @model_validator(mode="after")
    def ensure_prompts_present(self) -> "OptimizationRequest":
        if self.query is not None:
            cleaned_query = self.query.strip()
            self.query = cleaned_query if cleaned_query else None

        single = (self.prompt or "").strip() if isinstance(self.prompt, str) else ""
        batch = self.prompts or []

        if single and batch:
            raise ValueError("Provide either prompt or prompts, not both")

        if batch:
            cleaned = [item.strip() for item in batch if item and item.strip()]
            if len(cleaned) != len(batch):
                raise ValueError("Prompts within batch cannot be empty")
            if not cleaned:
                raise ValueError("Prompts must contain at least one non-empty entry")
            self.prompts = cleaned
            self.prompt = None
            return self

        if single:
            self.prompt = single
            self.prompts = None
            return self

        raise ValueError("Either prompt or prompts must be supplied")

    @model_validator(mode="after")
    def ensure_custom_canonicals_valid(self) -> "OptimizationRequest":
        if not self.custom_canonicals:
            return self
        for key, value in self.custom_canonicals.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("custom_canonicals keys and values must be strings")
            if not key.strip() or not value.strip():
                raise ValueError(
                    "custom_canonicals cannot contain empty keys or values"
                )
        return self

class OptimizationStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Character-level stats
    original_chars: int = Field(..., ge=0, description="Original character count")
    optimized_chars: int = Field(..., ge=0, description="Optimized character count")
    compression_percentage: float = Field(
        ..., description="Percentage of character compression achieved"
    )

    # Token-level stats
    original_tokens: int = Field(..., ge=0, description="Original token count")
    optimized_tokens: int = Field(..., ge=0, description="Optimized token count")
    token_savings: int = Field(..., ge=0, description="Tokens saved after optimization")

    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    fast_path: bool = Field(
        ..., description="Indicates whether the fast-path optimization was applied"
    )
    content_profile: str = Field(
        ..., description="Detected content profile name used for smart selection"
    )
    smart_context_description: str = Field(
        ..., description="Summary of smart selection context for this prompt"
    )
    semantic_similarity: Optional[float] = Field(
        None, description="Similarity score used by the semantic guard"
    )
    semantic_similarity_source: Optional[str] = Field(
        None, description="Source used to compute semantic similarity"
    )
    deduplication: Optional[Dict[str, int]] = Field(
        None, description="Counts of duplicates removed by type"
    )

    # Optional technique-specific counters
    toon_conversions: Optional[int] = Field(
        None, ge=0, description="Number of TOON JSON conversions applied (if enabled)"
    )
    toon_bytes_saved: Optional[int] = Field(
        None, ge=0, description="Bytes saved by TOON conversion (if enabled)"
    )
    embedding_reuse_count: Optional[int] = Field(
        None, ge=0, description="Number of embedding cache reuses during optimization"
    )
    embedding_calls_saved: Optional[int] = Field(
        None,
        ge=0,
        description="Number of embedding model calls avoided due to cache reuse",
    )
    embedding_wall_clock_savings_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated wall-clock time saved from embedding cache reuse",
    )
    section_ranking_selected_sections: Optional[List[int]] = Field(
        None,
        description="Indices of the sections retained by the ranking pre-pass (if applied)",
    )
    section_ranking_trigger: Optional[str] = Field(
        None,
        description=(
            "Reason section ranking was triggered (e.g., 'size', 'redundancy', 'size_and_redundancy')"
        ),
    )
    maximum_prepass_selected_sentences: Optional[List[int]] = Field(
        None,
        description="Sentence indices retained by the maximum-mode budgeted pre-pass (if applied)",
    )
    maximum_prepass_target_tokens: Optional[int] = Field(
        None,
        description="Token budget targeted by the maximum-mode pre-pass (if applied)",
    )

    # Maximum-mode prepass policy details (useful for debugging/observability)
    maximum_prepass_policy_source: Optional[str] = Field(
        None, description="Resolved maximum prepass policy source (e.g., auto/forced/off)"
    )
    maximum_prepass_policy_enabled: Optional[bool] = Field(
        None, description="Whether the maximum prepass policy resolved to enabled"
    )
    maximum_prepass_policy_mode: Optional[str] = Field(
        None, description="Policy mode string used to resolve maximum prepass behavior"
    )
    maximum_prepass_policy_enabled_override: Optional[bool] = Field(
        None, description="Whether policy enablement was forced via env override"
    )
    maximum_prepass_policy_minimum_tokens: Optional[int] = Field(
        None, ge=0, description="Effective minimum token threshold for maximum prepass"
    )
    maximum_prepass_policy_budget_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Effective keep ratio for maximum prepass",
    )
    maximum_prepass_policy_adaptive_budget_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Adaptive keep ratio resolved for maximum prepass",
    )
    maximum_prepass_policy_adaptive_redundancy_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Redundancy signal used to adapt maximum prepass keep ratio",
    )
    maximum_prepass_policy_adaptive_constraint_density: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Constraint-density signal used for maximum prepass adaptation",
    )
    maximum_prepass_policy_max_sentences: Optional[int] = Field(
        None, ge=0, description="Effective sentence cap for maximum prepass"
    )
    profiling_ms: Optional[Dict[str, float]] = Field(
        None, description="Profiling information for pipeline stages (if enabled)"
    )


class RouterInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: str
    profile: str


class OptimizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimized_output: str = Field(..., description="The optimized prompt")
    stats: OptimizationStats = Field(..., description="Optimization statistics")
    router: Optional[RouterInfo] = Field(
        None, description="Smart Router classification details (if available)"
    )
    techniques_applied: Optional[List[str]] = Field(
        None, description="Optimization techniques executed"
    )
    warnings: Optional[List[str]] = Field(
        None, description="Warnings about disabled techniques or configuration issues"
    )


class OptimizationBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_items: int
    avg_compression: float
    total_processing_time_ms: float
    throughput_prompts_per_second: float


class OptimizationBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_job_id: str = Field(..., description="Batch job ID")
    results: List[OptimizationResponse] = Field(
        ..., description="Batch of optimization results"
    )
    summary: OptimizationBatchSummary = Field(..., description="Batch summary metrics")
