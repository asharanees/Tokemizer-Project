"""Pipeline configuration for optimization passes and presets."""

OPTIMIZATION_MODES = {
    "conservative": {
        "description": "Conservative - Fast processing with reduced passes (~70% faster, ~35% savings)",
        "enable_toon_conversion": False,
        "disabled_passes": [
            "alias_json_keys",
            "compress_coreferences",
            "compress_examples",
            "dedup_normalized_sentences",
            "summarize_history",
            "remove_articles",
            "apply_symbolic_replacements",
            "apply_macro_dictionary",
        ],
        "pass_toggles": {
            "alias_references": False,
            "compress_field_labels": False,
            "compress_parentheticals": False,
            "hoist_constraints": False,
        },
    },
    "balanced": {
        "description": "Balanced - Enhanced optimization with most passes (~40% faster, ~45% savings)",
        "enable_toon_conversion": False,
        "disabled_passes": [
            "alias_json_keys",
            "compress_examples",
            "summarize_history",
            "remove_articles",
            "apply_symbolic_replacements",
            "apply_macro_dictionary",
        ],
        "pass_toggles": {
            "alias_references": True,
            "compress_field_labels": True,
            "compress_parentheticals": True,
            "hoist_constraints": True,
        },
    },
    "maximum": {
        "description": "Maximum - All passes including advanced techniques (baseline speed, 50-70% savings)",
        "enable_toon_conversion": True,
        "disabled_passes": [],
        "pass_toggles": {
            "alias_references": True,
            "compress_field_labels": True,
            "compress_parentheticals": True,
            "hoist_constraints": True,
        },
    },
}


def resolve_optimization_config(optimization_mode: str) -> dict:
    """
    Resolve optimization configuration from optimization mode.

    Args:
        optimization_mode: Optimization mode ('conservative', 'balanced', 'maximum')

    Returns:
        Configuration dict for the requested optimization mode
    """
    return OPTIMIZATION_MODES.get(optimization_mode, OPTIMIZATION_MODES["maximum"])
