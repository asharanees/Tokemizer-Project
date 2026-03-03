"""Central contract for optimizer model capabilities and readiness."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

ModelCapability = Dict[str, Any]


MODEL_CAPABILITIES: Dict[str, ModelCapability] = {
    "semantic_guard": {
        "intended_features": ["semantic safeguard similarity checks"],
        "required_mode_gates": ["conservative", "balanced", "maximum"],
        "required_profile_gates": [],
        "hard_required": True,
        "notes": "Required by strict mode gates across conservative/balanced/maximum.",
    },
    "semantic_rank": {
        "intended_features": [
            "query-aware compression",
            "semantic section ranking",
        ],
        "required_mode_gates": ["balanced", "maximum"],
        "required_profile_gates": [
            "general_prose",
            "markdown",
            "technical_doc",
            "heavy_document",
        ],
        "hard_required": False,
        "notes": "Applied when ranking/query-aware paths are selected.",
    },
    "coreference": {
        "intended_features": ["compress_coreferences"],
        "required_mode_gates": ["balanced", "maximum"],
        "required_profile_gates": [],
        "hard_required": False,
        "notes": "Optional compression enhancer.",
    },
    "spacy": {
        "intended_features": [
            "semantic deduplication",
            "linguistic passes",
        ],
        "required_mode_gates": ["balanced", "maximum"],
        "required_profile_gates": [
            "general_prose",
            "markdown",
            "technical_doc",
            "heavy_document",
            "dialogue",
            "short",
        ],
        "hard_required": False,
        "notes": "Optional NLP pipeline used for semantic deduplication and linguistic passes.",
    },
    "entropy": {
        "intended_features": ["prune_low_entropy"],
        "required_mode_gates": ["maximum"],
        "required_profile_gates": [],
        "hard_required": False,
        "notes": "Optional teacher quality guard; maximum mode can fall back to entropy_fast.",
    },
    "entropy_fast": {
        "intended_features": ["prune_low_entropy"],
        "required_mode_gates": ["conservative", "balanced", "maximum"],
        "required_profile_gates": [],
        "hard_required": True,
        "notes": "Primary required entropy backend across strict modes.",
    },
    "token_classifier": {
        "intended_features": ["maximum-mode classifier fast path"],
        "required_mode_gates": ["maximum"],
        "required_profile_gates": [],
        "hard_required": False,
        "notes": "Optional maximum-mode speed/quality enhancer.",
    },
}


def list_capabilities_for_model(model_type: str) -> ModelCapability:
    return MODEL_CAPABILITIES.get(
        model_type,
        {
            "intended_features": [],
            "required_mode_gates": [],
            "required_profile_gates": [],
            "hard_required": False,
            "notes": "",
        },
    )


def entropy_backend_ready(
    model_lookup: Dict[str, bool], optimization_mode: Optional[str] = None
) -> bool:
    _ = optimization_mode
    return bool(model_lookup.get("entropy_fast"))


def build_model_readiness(model_lookup: Dict[str, bool]) -> Dict[str, Dict[str, Any]]:
    readiness: Dict[str, Dict[str, Any]] = {}
    for model_type, capability in MODEL_CAPABILITIES.items():
        loaded = bool(model_lookup.get(model_type))
        ready = loaded
        reason = "loaded" if loaded else "not_loaded"
        readiness[model_type] = {
            "intended_usage_ready": ready,
            "intended_usage_reason": reason,
            "hard_required": bool(capability.get("hard_required")),
            "intended_features": list(capability.get("intended_features") or []),
            "required_mode_gates": list(capability.get("required_mode_gates") or []),
            "required_profile_gates": list(
                capability.get("required_profile_gates") or []
            ),
        }
    return readiness


def build_not_ready_warnings(
    model_lookup: Dict[str, bool],
    optimization_mode: str,
    *,
    query_present: bool,
    profile_name: Optional[str] = None,
    segment_spans_present: bool = False,
    disabled_passes: Optional[Sequence[str]] = None,
) -> List[str]:
    disabled = set(disabled_passes or [])
    warnings: List[str] = []

    def _capability_profile_enabled(model_type: str) -> bool:
        profile_gates = set(
            MODEL_CAPABILITIES.get(model_type, {}).get("required_profile_gates") or []
        )
        if not profile_gates:
            return True
        if profile_name is None:
            # Unknown profile: conservative warning behavior.
            return True
        return profile_name in profile_gates

    def _capability_path_enabled(model_type: str) -> bool:
        mode_gates = set(
            MODEL_CAPABILITIES.get(model_type, {}).get("required_mode_gates") or []
        )
        mode_enabled = not mode_gates or optimization_mode in mode_gates
        return mode_enabled and _capability_profile_enabled(model_type)

    semantic_rank_path_enabled = (
        _capability_path_enabled("semantic_rank")
        and query_present
        and not segment_spans_present
    )
    semantic_guard_path_enabled = _capability_path_enabled("semantic_guard")
    token_classifier_path_enabled = (
        _capability_path_enabled("token_classifier") and not segment_spans_present
    )

    if semantic_rank_path_enabled and not model_lookup.get("semantic_rank"):
        warnings.append(
            "semantic_rank not ready; query-aware compression and semantic section ranking skipped"
        )

    coreference_enabled = "compress_coreferences" not in disabled
    if (
        _capability_path_enabled("coreference")
        and coreference_enabled
        and not model_lookup.get("coreference")
    ):
        warnings.append("coreference not ready; compress_coreferences skipped")

    if _capability_path_enabled("spacy") and not model_lookup.get("spacy"):
        warnings.append(
            "spacy not ready; semantic deduplication and linguistic passes skipped"
        )

    if semantic_guard_path_enabled and not model_lookup.get("semantic_guard"):
        warnings.append(
            "semantic_guard not ready; strict mode requires semantic_guard for this path"
        )

    if not entropy_backend_ready(model_lookup, optimization_mode):
        warnings.append(
            "entropy_fast backend not ready; strict mode requires entropy_fast for prune_low_entropy"
        )
    elif optimization_mode == "maximum" and not model_lookup.get("entropy"):
        warnings.append(
            "entropy teacher not ready; maximum mode uses entropy_fast fallback (quality guard reduced)"
        )

    if token_classifier_path_enabled and not model_lookup.get("token_classifier"):
        warnings.append(
            "token_classifier not ready; strict mode requires token_classifier for this path"
        )

    return warnings


def build_not_used_warnings(
    model_lookup: Dict[str, bool],
    optimization_mode: str,
    techniques_applied: Optional[Sequence[str]],
    *,
    query_present: bool,
    profile_name: Optional[str] = None,
    segment_spans_present: bool = False,
    disabled_passes: Optional[Sequence[str]] = None,
    semantic_guard_enabled: bool = True,
    token_classifier_post_enabled: bool = True,
) -> List[str]:
    disabled = set(disabled_passes or [])
    applied = set(techniques_applied or [])
    warnings: List[str] = []

    def _capability_profile_enabled(model_type: str) -> bool:
        profile_gates = set(
            MODEL_CAPABILITIES.get(model_type, {}).get("required_profile_gates") or []
        )
        if not profile_gates:
            return True
        if profile_name is None:
            return True
        return profile_name in profile_gates

    def _capability_path_enabled(model_type: str) -> bool:
        mode_gates = set(
            MODEL_CAPABILITIES.get(model_type, {}).get("required_mode_gates") or []
        )
        mode_enabled = not mode_gates or optimization_mode in mode_gates
        return mode_enabled and _capability_profile_enabled(model_type)

    semantic_rank_path_enabled = (
        _capability_path_enabled("semantic_rank")
        and query_present
        and not segment_spans_present
    )
    if semantic_rank_path_enabled and model_lookup.get("semantic_rank"):
        has_ranking = any(
            technique.startswith("Context Section Ranking") for technique in applied
        )
        if not has_ranking:
            warnings.append(
                "semantic_rank was ready but not exercised for this prompt "
                "(ranking path produced no applicable changes)"
            )

    coreference_enabled = "compress_coreferences" not in disabled
    if (
        _capability_path_enabled("coreference")
        and coreference_enabled
        and model_lookup.get("coreference")
        and "Coreference Compression" not in applied
    ):
        warnings.append("coreference was ready but not exercised for this prompt")

    spacy_used = (
        "Content Deduplication" in applied
        or "Paragraph Semantic Deduplication" in applied
    )
    if (
        _capability_path_enabled("spacy")
        and model_lookup.get("spacy")
        and not spacy_used
    ):
        warnings.append("spacy was ready but not exercised for this prompt")

    if (
        _capability_path_enabled("entropy")
        and entropy_backend_ready(model_lookup, optimization_mode)
        and "Entropy Pruning" not in applied
    ):
        warnings.append("entropy backend was ready but not exercised for this prompt")

    if (
        _capability_path_enabled("token_classifier")
        and not segment_spans_present
        and semantic_guard_enabled
        and token_classifier_post_enabled
        and model_lookup.get("token_classifier")
        and "Token Classification Compression (Post)" not in applied
    ):
        warnings.append("token_classifier was ready but not exercised for this prompt")

    if (
        _capability_path_enabled("semantic_guard")
        and model_lookup.get("semantic_guard")
        and not semantic_guard_enabled
    ):
        warnings.append(
            "semantic_guard model is ready but semantic_guard_enabled=false, so semantic guard is not used"
        )

    return warnings


def model_lookup_from_status(
    model_status: Optional[Dict[str, Any]] = None,
    cached_availability: Optional[Dict[str, bool]] = None,
    model_types: Optional[Sequence[str]] = None,
) -> Dict[str, bool]:
    lookup: Dict[str, bool] = {}
    selected_types = list(model_types or MODEL_CAPABILITIES.keys())
    for model_type in selected_types:
        loaded = False
        if model_status:
            loaded = bool(model_status.get(model_type, {}).get("loaded"))
        elif cached_availability is not None:
            loaded = bool(cached_availability.get(model_type))
        lookup[model_type] = loaded
    return lookup
