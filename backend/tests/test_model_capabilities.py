from services.optimizer.model_capabilities import (
    build_model_readiness,
    build_not_ready_warnings,
    build_not_used_warnings,
)


def test_build_model_readiness_includes_spacy_capability_contract() -> None:
    readiness = build_model_readiness({"spacy": False})

    spacy = readiness["spacy"]
    assert spacy["intended_usage_ready"] is False
    assert spacy["intended_usage_reason"] == "not_loaded"
    assert spacy["hard_required"] is False
    assert spacy["intended_features"] == [
        "semantic deduplication",
        "linguistic passes",
    ]
    assert spacy["required_mode_gates"] == ["balanced", "maximum"]
    assert spacy["required_profile_gates"] == [
        "general_prose",
        "markdown",
        "technical_doc",
        "heavy_document",
        "dialogue",
        "short",
    ]


def test_build_not_ready_warnings_suppresses_coreference_when_disabled() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": False, "semantic_rank": True, "spacy": True},
        "maximum",
        query_present=False,
        disabled_passes=["compress_coreferences"],
    )

    assert "coreference not ready; compress_coreferences skipped" not in warnings


def test_build_not_ready_warnings_coreference_warns_when_enabled() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": False, "semantic_rank": True, "spacy": True},
        "maximum",
        query_present=False,
        disabled_passes=[],
    )

    assert "coreference not ready; compress_coreferences skipped" in warnings


def test_build_not_ready_warnings_semantic_rank_suppressed_for_non_ranked_profile() -> (
    None
):
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": False, "spacy": True},
        "maximum",
        query_present=True,
        profile_name="code",
        segment_spans_present=False,
        disabled_passes=[],
    )

    assert not any("semantic_rank not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_semantic_rank_warns_for_ranked_profile() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": False, "spacy": True},
        "maximum",
        query_present=True,
        profile_name="general_prose",
        segment_spans_present=False,
        disabled_passes=[],
    )

    assert any("semantic_rank not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_semantic_rank_warns_when_profile_unknown() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": False, "spacy": True},
        "maximum",
        query_present=True,
        profile_name=None,
        segment_spans_present=False,
        disabled_passes=[],
    )

    assert any("semantic_rank not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_spacy_warns_for_gated_mode_and_profile() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": True, "spacy": False},
        "maximum",
        query_present=False,
        profile_name="dialogue",
        disabled_passes=[],
    )

    assert any("spacy not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_spacy_warns_when_profile_unknown() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": True, "spacy": False},
        "maximum",
        query_present=False,
        profile_name=None,
        disabled_passes=[],
    )

    assert any("spacy not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_spacy_suppressed_for_non_gated_profile() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": True, "spacy": False},
        "maximum",
        query_present=False,
        profile_name="json",
        disabled_passes=[],
    )

    assert not any("spacy not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_spacy_suppressed_for_non_gated_mode() -> None:
    warnings = build_not_ready_warnings(
        {"coreference": True, "semantic_rank": True, "spacy": False},
        "conservative",
        query_present=False,
        profile_name="general_prose",
        disabled_passes=[],
    )

    assert not any("spacy not ready" in warning for warning in warnings)


def test_build_not_ready_warnings_include_hard_required_models() -> None:
    warnings = build_not_ready_warnings(
        {
            "semantic_guard": False,
            "entropy": False,
            "entropy_fast": False,
            "token_classifier": False,
        },
        "maximum",
        query_present=False,
        profile_name="general_prose",
        segment_spans_present=False,
        disabled_passes=[],
    )

    assert any("semantic_guard not ready" in warning for warning in warnings)
    assert any("entropy_fast backend not ready" in warning for warning in warnings)
    assert any("token_classifier not ready" in warning for warning in warnings)


def test_build_not_used_warnings_detect_ready_models_not_exercised() -> None:
    warnings = build_not_used_warnings(
        {
            "semantic_rank": True,
            "coreference": True,
            "spacy": True,
            "entropy": True,
            "entropy_fast": True,
            "token_classifier": True,
            "semantic_guard": True,
        },
        "maximum",
        techniques_applied=[],
        query_present=True,
        profile_name="general_prose",
        segment_spans_present=False,
        disabled_passes=[],
    )

    assert any(
        "semantic_rank was ready but not exercised" in warning for warning in warnings
    )
    assert any(
        "coreference was ready but not exercised" in warning for warning in warnings
    )
    assert any("spacy was ready but not exercised" in warning for warning in warnings)
    assert any(
        "entropy backend was ready but not exercised" in warning for warning in warnings
    )
    assert any(
        "token_classifier was ready but not exercised" in warning
        for warning in warnings
    )


def test_build_not_used_warnings_suppresses_disabled_or_off_paths() -> None:
    warnings = build_not_used_warnings(
        {
            "semantic_rank": True,
            "coreference": True,
            "spacy": True,
            "entropy": True,
            "entropy_fast": False,
            "token_classifier": True,
            "semantic_guard": True,
        },
        "maximum",
        techniques_applied=["Entropy Pruning"],
        query_present=False,
        profile_name="json",
        segment_spans_present=True,
        disabled_passes=["compress_coreferences"],
        semantic_guard_enabled=False,
    )

    assert not any(
        "semantic_rank was ready but not exercised" in warning for warning in warnings
    )
    assert not any(
        "coreference was ready but not exercised" in warning for warning in warnings
    )
    assert not any(
        "spacy was ready but not exercised" in warning for warning in warnings
    )
    assert not any(
        "token_classifier was ready but not exercised" in warning
        for warning in warnings
    )
    assert any(
        "semantic_guard model is ready but semantic_guard_enabled=false" in warning
        for warning in warnings
    )
