"""
Configuration constants and dictionaries for prompt optimization.

This module centralizes all configuration data, patterns, and mappings used
throughout the optimizer package. Extracting these constants makes the core
orchestration logic cleaner and allows for easier configuration management.
"""

import os
import re

from .config_utils import get_env_bool, get_env_float, get_env_int

_AUTOTUNE_PROFILE_DEFAULT = "balanced"
_AUTOTUNE_PROFILES = {"safe", "balanced", "aggressive"}
_autotune_profile_env = (
    os.environ.get("PROMPT_OPTIMIZER_AUTOTUNE_PROFILE", _AUTOTUNE_PROFILE_DEFAULT)
    or _AUTOTUNE_PROFILE_DEFAULT
)
_autotune_profile_normalized = _autotune_profile_env.strip().lower()
PROMPT_OPTIMIZER_AUTOTUNE_PROFILE = (
    _autotune_profile_normalized
    if _autotune_profile_normalized in _AUTOTUNE_PROFILES
    else _AUTOTUNE_PROFILE_DEFAULT
)

AUTOTUNE_PROFILE_PRESETS = {
    "safe": {
        "maximum_prepass": {
            "policy": "conservative",
            "min_tokens": 4200,
            "budget_ratio": 0.8,
            "max_sentences": 140,
        },
        "multi_candidate": {
            "max_candidates": 1,
            "guard_floor": 0.92,
        },
        "token_classifier_post": {
            "enabled": False,
            "min_confidence": 0.68,
            "min_keep_ratio": 0.84,
        },
    },
    "balanced": {
        "maximum_prepass": {
            "policy": "auto",
            "min_tokens": 3500,
            "budget_ratio": 0.72,
            "max_sentences": 120,
        },
        "multi_candidate": {
            "max_candidates": 2,
            "guard_floor": 0.9,
        },
        "token_classifier_post": {
            "enabled": False,
            "min_confidence": 0.6,
            "min_keep_ratio": 0.75,
        },
    },
    "aggressive": {
        "maximum_prepass": {
            "policy": "aggressive",
            "min_tokens": 2800,
            "budget_ratio": 0.62,
            "max_sentences": 100,
        },
        "multi_candidate": {
            "max_candidates": 3,
            "guard_floor": 0.88,
        },
        "token_classifier_post": {
            "enabled": True,
            "min_confidence": 0.54,
            "min_keep_ratio": 0.68,
        },
    },
}

OUTPUT_GUIDANCE_BY_STYLE = {
    "direct": {
        "code": "[Output code block only]",
        "math": "[Show key formula then answer]",
        "reasoning": "[Give brief reasoning then answer]",
        "table": "[Use compact table]",
        "json": "[Respond with minimal JSON]",
        "list": "[Output concise list]",
        "comparison": "[Provide concise comparison]",
        "summary": "[Be concise]",
        "explanation": "[Be direct and clear]",
    },
    "detailed": {
        "code": "[Provide well-commented code]",
        "math": "[Show detailed steps and result]",
        "reasoning": "[Explain reasoning clearly]",
        "table": "[Provide detailed table]",
        "json": "[Return structured JSON]",
        "list": "[Provide comprehensive list]",
        "comparison": "[Provide detailed comparison]",
        "summary": "[Provide thorough summary]",
        "explanation": "[Provide clear explanation]",
    },
    "structured": {
        "code": "[Return structured code block]",
        "math": "[List steps then result]",
        "reasoning": "[Use numbered reasoning]",
        "table": "[Format as markdown table]",
        "json": "[Return valid JSON]",
        "list": "[Use numbered list]",
        "comparison": "[Use table for comparison]",
        "summary": "[Summarize key points]",
        "explanation": "[Use bullet explanation]",
    },
    "balanced": {
        "code": "[Return code snippet]",
        "math": "[Show key steps and answer]",
        "reasoning": "[Provide concise reasoning]",
        "table": "[Use clear table]",
        "json": "[Respond with compact JSON]",
        "list": "[Give concise list]",
        "comparison": "[Offer brief comparison]",
        "summary": "[Summarize briefly]",
        "explanation": "[Explain clearly]",
    },
}

# Semantic safeguard defaults
SEMANTIC_GUARD_ENABLED = True
SEMANTIC_GUARD_THRESHOLD = 0.82
SEMANTIC_GUARD_PER_PASS_ENABLED = True
# Fallback model if database configuration is missing (Model_Inventory table is primary source)
# This constant is now primarily used as a default/fallback.
# The active model is determined dynamically by the Model_Inventory table in the database.
SEMANTIC_GUARD_MODEL = "BAAI/bge-small-en-v1.5"
SEMANTIC_GUARD_MAX_PROMPT_TOKENS = 2048
SEMANTIC_GUARD_LATENCY_GUARD_MS = (
    600  # Maximum allowed optimization latency (per run) before guardrails fail.
)
SEMANTIC_GUARD_TOKEN_SAVINGS_BASELINE = (
    20  # Target minimum average tokens saved per optimization run.
)
ONNX_USE_INT8 = get_env_bool("PROMPT_OPTIMIZER_ONNX_INT8", False)
SHARED_SEMANTIC_EMBEDDINGS = get_env_bool(
    "PROMPT_OPTIMIZER_SHARED_SEMANTIC_EMBEDDINGS", False
)
TELEMETRY_BASELINE_WINDOW_DAYS = (
    30  # Telemetry sampling window (days) when deriving baseline metrics.
)
TOKEN_CLASSIFIER_COMBINED_CLASSIFIER_WEIGHT = get_env_float(
    "PROMPT_OPTIMIZER_COMBINED_CLASSIFIER_WEIGHT", 0.5
)
TOKEN_CLASSIFIER_COMBINED_ENTROPY_WEIGHT = get_env_float(
    "PROMPT_OPTIMIZER_COMBINED_ENTROPY_WEIGHT", 0.5
)
TOKEN_CLASSIFIER_COMBINED_KEEP_THRESHOLD = get_env_float(
    "PROMPT_OPTIMIZER_COMBINED_KEEP_THRESHOLD", 0.5
)
TOKEN_CLASSIFIER_COMBINED_ENTROPY_HIGH_THRESHOLD = get_env_float(
    "PROMPT_OPTIMIZER_COMBINED_ENTROPY_HIGH_THRESHOLD", 3.0
)
_AUTOTUNE_POST_PRESET = AUTOTUNE_PROFILE_PRESETS[PROMPT_OPTIMIZER_AUTOTUNE_PROFILE][
    "token_classifier_post"
]

TOKEN_CLASSIFIER_POST_PASS_ENABLED = get_env_bool(
    "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_PASS", _AUTOTUNE_POST_PRESET["enabled"]
)
TOKEN_CLASSIFIER_POST_MIN_CONFIDENCE = get_env_float(
    "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_MIN_CONFIDENCE",
    _AUTOTUNE_POST_PRESET["min_confidence"],
)
TOKEN_CLASSIFIER_POST_MIN_KEEP_RATIO = get_env_float(
    "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_MIN_KEEP_RATIO",
    _AUTOTUNE_POST_PRESET["min_keep_ratio"],
)
TOKEN_CLASSIFIER_AGGRESSIVE_KEEP_RATIO_MULTIPLIER = get_env_float(
    "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_AGGRESSIVE_KEEP_RATIO_MULTIPLIER", 0.85
)
TOKEN_CLASSIFIER_AGGRESSIVE_CONFIDENCE_MULTIPLIER = get_env_float(
    "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_AGGRESSIVE_CONFIDENCE_MULTIPLIER", 0.85
)


MAXIMUM_PREPASS_ENABLED = get_env_bool(
    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED", False
)

_AUTOTUNE_MAX_PREPASS_PRESET = AUTOTUNE_PROFILE_PRESETS[
    PROMPT_OPTIMIZER_AUTOTUNE_PROFILE
]["maximum_prepass"]

_MAXIMUM_PREPASS_POLICY_DEFAULT = _AUTOTUNE_MAX_PREPASS_PRESET["policy"]
_MAXIMUM_PREPASS_POLICY_VALUES = {"off", "auto", "conservative", "aggressive"}
_maximum_prepass_policy_env = (
    os.environ.get(
        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY",
        _MAXIMUM_PREPASS_POLICY_DEFAULT,
    )
    or _MAXIMUM_PREPASS_POLICY_DEFAULT
)
_maximum_prepass_policy_normalized = _maximum_prepass_policy_env.strip().lower()
PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY = (
    _maximum_prepass_policy_normalized
    if _maximum_prepass_policy_normalized in _MAXIMUM_PREPASS_POLICY_VALUES
    else _MAXIMUM_PREPASS_POLICY_DEFAULT
)

MAXIMUM_PREPASS_MIN_TOKENS = get_env_int(
    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS",
    _AUTOTUNE_MAX_PREPASS_PRESET["min_tokens"],
)
MAXIMUM_PREPASS_BUDGET_RATIO = get_env_float(
    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO",
    _AUTOTUNE_MAX_PREPASS_PRESET["budget_ratio"],
)
MAXIMUM_PREPASS_MAX_SENTENCES = get_env_int(
    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES",
    _AUTOTUNE_MAX_PREPASS_PRESET["max_sentences"],
)

_AUTOTUNE_MC_PRESET = AUTOTUNE_PROFILE_PRESETS[PROMPT_OPTIMIZER_AUTOTUNE_PROFILE][
    "multi_candidate"
]

MULTI_CANDIDATE_PASS_SETTINGS = {
    "compress_examples": {
        "max_candidates": get_env_int(
            "PROMPT_OPTIMIZER_MC_COMPRESS_EXAMPLES_MAX",
            _AUTOTUNE_MC_PRESET["max_candidates"],
        ),
        "min_guard_threshold": get_env_float(
            "PROMPT_OPTIMIZER_MC_COMPRESS_EXAMPLES_MIN_GUARD",
            max(0.88, _AUTOTUNE_MC_PRESET["guard_floor"]),
        ),
        "aggressive_summary_max_length": get_env_int(
            "PROMPT_OPTIMIZER_MC_COMPRESS_EXAMPLES_AGGRESSIVE_MAX_LEN", 120
        ),
    },
    "summarize_history": {
        "max_candidates": get_env_int(
            "PROMPT_OPTIMIZER_MC_SUMMARIZE_HISTORY_MAX",
            _AUTOTUNE_MC_PRESET["max_candidates"],
        ),
        "min_guard_threshold": get_env_float(
            "PROMPT_OPTIMIZER_MC_SUMMARIZE_HISTORY_MIN_GUARD",
            max(0.9, _AUTOTUNE_MC_PRESET["guard_floor"]),
        ),
        "aggressive_keep_ratio_modifier": get_env_float(
            "PROMPT_OPTIMIZER_MC_SUMMARIZE_HISTORY_AGGRESSIVE_MODIFIER", 0.75
        ),
    },
    "prune_low_entropy": {
        "max_candidates": get_env_int(
            "PROMPT_OPTIMIZER_MC_PRUNE_ENTROPY_MAX",
            _AUTOTUNE_MC_PRESET["max_candidates"],
        ),
        "min_guard_threshold": get_env_float(
            "PROMPT_OPTIMIZER_MC_PRUNE_ENTROPY_MIN_GUARD",
            max(0.9, _AUTOTUNE_MC_PRESET["guard_floor"]),
        ),
        "aggressive_ratio_multiplier": get_env_float(
            "PROMPT_OPTIMIZER_MC_PRUNE_ENTROPY_RATIO_MULTIPLIER", 1.25
        ),
        "aggressive_max_ratio_multiplier": get_env_float(
            "PROMPT_OPTIMIZER_MC_PRUNE_ENTROPY_MAX_RATIO_MULTIPLIER", 1.15
        ),
    },
    "token_classifier": {
        "max_candidates": get_env_int(
            "PROMPT_OPTIMIZER_MC_TOKEN_CLASSIFIER_MAX",
            _AUTOTUNE_MC_PRESET["max_candidates"],
        ),
        "min_guard_threshold": get_env_float(
            "PROMPT_OPTIMIZER_MC_TOKEN_CLASSIFIER_MIN_GUARD",
            max(0.9, _AUTOTUNE_MC_PRESET["guard_floor"]),
        ),
    },
}

ADJUNCT_DISCOURSE_MARKERS = (
    "as a reminder",
    "in order to",
    "it is important to note that",
    "please note",
    "as a note",
    "to be clear",
    "to clarify",
    "for reference",
    "as a heads up",
)
ADJUNCT_ALLOWED_DEPS = ("advcl", "prep", "npadvmod", "advmod", "discourse")
ADJUNCT_NEGATION_TOKENS = ("not", "n't", "no", "never")
ADJUNCT_CONDITION_TOKENS = (
    "if",
    "unless",
    "when",
    "provided",
    "assuming",
    "in case",
)
ADJUNCT_MODAL_TOKENS = (
    "must",
    "should",
    "shall",
    "may",
    "might",
    "could",
    "would",
    "can",
    "will",
    "need",
)

ENTITY_ALIAS_MIN_OCCURRENCES = 3
ENTITY_ALIAS_MIN_CHARS = 12
ENTITY_ALIAS_MAX_ALIASES = 6
ENTITY_ALIAS_PREFIX = "E"
ENTITY_ALIAS_LABELS = ("ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC")

PLACEHOLDER_PREFIXES = {
    "code_blocks": "CODE",
    "urls": "URL",
    "quotes": "QUOTE",
    "numbers": "NUM",
    "citations": "CIT",
    "protected": "PROTECT",
    "forced": "FORCE",
    "json_tokens": "JSONTOK",
    "json_literals": "JSONLIT",
    "toon_blocks": "TOON",
}

PLACEHOLDER_PATTERN = re.compile(r"__\w+_\d+__")

FORCE_PRESERVE_PATTERNS: tuple[str, ...] = ()
FORCE_PRESERVE_DIGITS: bool = False

# Canonical request lead-ins used across instruction simplification rules
INSTRUCTION_REQUEST_PREFIXES = {
    "can": r"\bcan you(?: please)?\s+",
    "could": r"\bcould you(?: please)?\s+",
    "would": r"\bwould you(?: please)?\s+",
    "will": r"\bwill you(?: please)?\s+",
}

SYNONYM_SHORTENINGS = {
    "approximately": "about",
    "utilize": "use",
    "utilizes": "uses",
    "utilized": "used",
    "utilizing": "using",
    "assistance": "help",
    "assisting": "helping",
    "assisted": "helped",
    "assists": "helps",
    "individuals": "people",
    "numerous": "many",
    "purchase": "buy",
    "purchases": "buys",
    "purchased": "bought",
    "purchasing": "buying",
    "requirement": "need",
    "requirements": "needs",
    "additionally": "also",
    "subsequently": "then",
    "consequently": "so",
    "nevertheless": "still",
    "furthermore": "also",
    "simultaneously": "together",
    "immediately": "now",
    "previously": "before",
    "currently": "now",
    "commence": "start",
    "commenced": "started",
    "commencing": "starting",
    "terminate": "end",
    "terminated": "ended",
    "terminating": "ending",
    "obtain": "get",
    "obtained": "got",
    "facilitate": "help",
    "demonstrates": "shows",
    "demonstrate": "show",
    "demonstrated": "showed",
    "implement": "apply",
    "implemented": "applied",
    "implementing": "applying",
    "modification": "change",
    "modifications": "changes",
    "methodology": "method",
    "methodologies": "methods",
    "prioritize": "rank",
    "prioritized": "ranked",
    "establish": "set",
    "established": "set",
    "leverage": "use",
    "leveraged": "used",
    "leveraging": "using",
    "accomplish": "do",
    "accomplished": "did",
    "sufficient": "enough",
    "regarding": "about",
    "concerning": "about",
    "substantial": "large",
    "fundamental": "basic",
    "preliminary": "early",
    "supplementary": "extra",
    "additional": "more",
    "alternative": "other",
    "alternatives": "others",
    "complete": "finish",
    "completed": "finished",
    "completes": "finishes",
    "completing": "finishing",
    "construct": "build",
    "constructed": "built",
    "constructs": "builds",
    "constructing": "building",
    "determine": "find",
    "determined": "found",
    "determines": "finds",
    "determining": "finding",
    "discontinue": "stop",
    "discontinued": "stopped",
    "discontinues": "stops",
    "discontinuing": "stopping",
    "eliminate": "remove",
    "eliminated": "removed",
    "eliminates": "removes",
    "eliminating": "removing",
    "equivalent": "equal",
    "identical": "same",
    "initial": "first",
    "objective": "goal",
    "objectives": "goals",
    "previous": "prior",
    "primary": "main",
}

UNIT_ABBREVIATIONS = {
    "meter": "m",
    "metre": "m",
    "centimeter": "cm",
    "centimetre": "cm",
    "millimeter": "mm",
    "millimetre": "mm",
    "kilometer": "km",
    "kilometre": "km",
    "gram": "g",
    "kilogram": "kg",
    "milligram": "mg",
    "microgram": "µg",
    "tonne": "t",
    "ton": "t",
    "liter": "L",
    "litre": "L",
    "milliliter": "mL",
    "millilitre": "mL",
    "second": "s",
    "minute": "min",
    "hour": "h",
    "day": "d",
    "week": "wk",
    "month": "mo",
    "year": "yr",
    "foot": "ft",
    "inch": "in",
    "yard": "yd",
    "mile": "mi",
    "percent": "%",
    "degree celsius": "°C",
    "celsius": "°C",
    "degree fahrenheit": "°F",
    "fahrenheit": "°F",
    "kelvin": "K",
    "cup": "c",
    "tablespoon": "tbsp",
    "teaspoon": "tsp",
    "ounce": "oz",
    "pound": "lb",
    "gallon": "gal",
    "kilobyte": "KB",
    "megabyte": "MB",
    "gigabyte": "GB",
    "terabyte": "TB",
    "petabyte": "PB",
    "kilobit": "Kb",
    "megabit": "Mb",
    "gigabit": "Gb",
    "hertz": "Hz",
    "kilohertz": "kHz",
    "megahertz": "MHz",
    "gigahertz": "GHz",
    "joule": "J",
    "kilojoule": "kJ",
    "watt": "W",
    "kilowatt": "kW",
    "megawatt": "MW",
    "ampere": "A",
    "milliampere": "mA",
    "volt": "V",
    "millivolt": "mV",
    "kilovolt": "kV",
    "ohm": "Ω",
    "newton": "N",
    "pascal": "Pa",
    "bar": "bar",
    "atmosphere": "atm",
    "kilometers per hour": "km/h",
    "kilometer per hour": "km/h",
    "miles per hour": "mph",
    "mile per hour": "mph",
    "meters per second": "m/s",
    "meter per second": "m/s",
    "feet per second": "ft/s",
    "foot per second": "ft/s",
    "revolutions per minute": "RPM",
}

SYMBOLIC_REPLACEMENTS = {
    "and": "&",
    "percent": "%",
    "equals": "=",
    "plus": "+",
    "minus": "-",
    "number": "#",
    "at": "@",
}

CURRENCY_SYMBOLS = {
    "usd": "$",
    "dollar": "$",
    "eur": "€",
    "euro": "€",
    "gbp": "£",
    "pound": "£",
    "jpy": "¥",
    "yen": "¥",
    "cny": "¥",
    "aud": "$",
    "cad": "$",
    "inr": "₹",
}

# Consecutive duplicate patterns - remove repeated words/phrases
CONSECUTIVE_DUPLICATE_PATTERNS = [
    # "please, please, please" -> "please"
    (r"\b(please)(?:[,\s]+\1)+\b", r"\1"),
    # "thank you so much, thank you so much" -> "thank you so much"
    (r"(thank you(?: so much)?)(?:[,.\s]+\1)+", r"\1"),
    # "really, really" -> "really" (any word repeated with comma/space)
    (r"\b(\w{4,})(?:[,\s]+\1)+\b", r"\1"),
    # "I really, really appreciate it" variations
    (r"\b(really)(?:[,\s]+\1)+\b", r"\1"),
    # "I really I really" -> "I"
    (r"\b(I really)(?:\s+\1)+\b", r"\1"),
    # "and and" -> "and"
    (r"\b(and)(?:\s+\1)+\b", r"\1"),
    # Generic two-word phrase repetition "X Y X Y X Y" -> "X Y"
    (r"\b(\w+\s+\w+)(?:\s+\1)+\b", r"\1"),
    # HuggingFace-style identifiers repeated with whitespace: "org/model org/model" -> "org/model"
    (
        r"\b([A-Za-z0-9][A-Za-z0-9_.-]{0,63}(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}){1,3})(?:[,\s]+\1)+\b",
        r"\1",
    ),
    # HuggingFace-style identifiers accidentally concatenated: "org/modelorg/model" -> "org/model"
    (
        r"([A-Za-z0-9][A-Za-z0-9_.-]{0,63}(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}){1,3})(?:\1)+",
        r"\1",
    ),
]

# Paradoxical phrase collapse patterns - consolidate contradictory statements
PARADOXICAL_PHRASE_PATTERNS = [
    # "very important, but also not important" -> "importance TBD"
    (r"(?:a )?very important(?: thing)?,? but also not important", "importance varies"),
    # "urgent, but also not urgent" -> "priority flexible"
    (r"urgent,? but also not urgent", "priority flexible"),
    # "simple, but also extremely complex" -> "complexity varies"
    (r"simple,? but also (?:extremely )?complex", "complexity varies"),
    # "do X perfectly, but also don't spend time" -> "do X efficiently"
    (
        r"do it perfectly,? but also (?:do not|don't) spend any time on it",
        "do it efficiently",
    ),
    # "be very detailed, but also keep it extremely short"
    (
        r"be (?:very )?detailed,? but also keep it (?:extremely )?short",
        "be concise yet complete",
    ),
    # "include everything, but also do not include anything unnecessary"
    (
        r"include everything,? but also (?:do not|don't) include anything unnecessary",
        "include only essentials",
    ),
    # Generic: "X, but also not X" pattern
    (r"(\w+),? but also not \1", r"\1 as appropriate"),
    # "read carefully, and also do not read too carefully"
    (
        r"read carefully,? (?:and )?also (?:do not|don't) read too carefully",
        "read normally",
    ),
    # "cite sources, but also do not cite sources"
    (r"cite sources,? but also (?:do not|don't) cite sources", "cite if needed"),
    # "redo it, but also do not redo it"
    (r"redo it,? but also (?:do not|don't) redo it", "iterate as needed"),
]

# Repeated phrase consolidation - collapse identical phrases separated by connectors
REPEATED_PHRASE_CONSOLIDATION = [
    # "X, X, X" or "X, X, and X" patterns for kindly do the needful
    (
        r"((?:please )?kindly do the needful(?: and thank you so much)?)(?:[,\s]+\1)+",
        r"\1",
    ),
    # Trailing "and and and" cleanup
    (r"\s+and\s+and(?:\s+and)+\s*$", " and"),
    # "thank you, thank you" patterns
    (r"(thank you)(?:[,\s]+\1)+", r"\1"),
    # Summary/plan/explanation repetition
    (
        r"(?:a )?full (plan|explanation|summary|outline)(?:[,\s]+(?:and )?(?:a )?(?:also )?(?:a )?(?:full |short )?"
        r"\1)+",
        r"complete \1",
    ),
    # "summary of the summary" -> "brief summary"
    (r"(?:a )?summary of the summary", "brief summary"),
    # "plan for the plan" -> "meta-plan"
    (r"(?:a )?plan for the plan", "meta-plan"),
    # "explanation of the explanation" -> "clarification"
    (r"(?:an )?explanation of the explanation", "clarification"),
    # "I really appreciate it, I really appreciate it" -> "I really appreciate it"
    (r"(I (?:really )?appreciate it)(?:[,\s]+\1)+", r"\1"),
]

# Politeness phrases to strip (instruction simplification)
POLITENESS_PATTERNS = [
    r"^\s*(?:hello|hi|hey)(?:[,.!]*\s+)",
    r"\b(?:please|kindly)(?:[,.]*\s+|\s+)",
    *INSTRUCTION_REQUEST_PREFIXES.values(),
    r"\bmay I\s+",
    (
        r"\b(?:and|also)\s+(?:thank you|thanks|appreciate it|appreciate that)"
        r"(?:\s+(?:so much|in advance|a lot|very much|again))*\b(?:[,.!]*\s*)?"
    ),
    (
        r"\b(thank you|thanks|appreciate it|appreciate that)"
        r"(?:\s+(?:so much|in advance|a lot|very much|again))*\b(?:[,.!]*\s*)?"
    ),
    r"\b(thank you|thanks)\s+for\s+[^\n.?!]+(?:[,.!]*\s*)?",
    r"\bthanks\s+in\s+advance\b(?:[,.!]*\s*)?",
    r"\bI (?:really\s+)?appreciate\s+(?:your|the)\s+[^\n.?!]+(?:[,.!]*\s*)?",
    r"\bI (?:would\s+)?appreciate\s+(?:it|that)?\s+if\s+you\s+",
    r"\bI (?:would\s+)?appreciate\s+(?:it|that)?\b(?:[,.!]*\s*)?",
    r"\b(?:and\s+)?(?:I\s+am|I'm)\s+(?:truly\s+|very\s+)?grateful\b(?:[,.!]*\s*)?",
    r"\b(?:and\s+)?(?:truly\s+|very\s+)?grateful\b(?:[,.!]*\s*)?",
    r"\b(?:and\s+)?sorry\b(?:[,.!]*\s*)?",
    r"\b(if you (?:do not|don\'t) mind|if you (?:could|would))\b(?:[,.!]*\s*)?",
    r"\bI (?:really\s+)*(?:would\s+)?(like|appreciate|be grateful|love)\b",
    r"\b(I\s+)?(?:really[,.]*\s+)+appreciate it\b(?:[,.!]*\s*)?",
    r"\bwould you mind\b",
    r"\bif (it\'s|it is) (not too much|possible)(?:\s+trouble)?\b(?:[,.!]*\s*)?",
    r"\b(and\s+)?only if (it\'s|it is) convenient(?: for you)?\b(?:[,.!]*\s*)?",
    r"\bI hope this helps\b",
    r"\bI hope this is clear\b",
    r"\bif you have any questions\b",
    r"\bdoes this make sense\b",
    r"\blet me know if\b",
    r"\bfeel free to\b",
    r"\bdo not hesitate to\b",
    r"\bI appreciate your\b",
    r"\bI would be grateful if\b",
    r"\bat your earliest convenience\b",
    r"\bplease(?:[,\s]+please)+\b",
    r"\bdo the needful\b",
    r"\bplease and thank you\b",
    # New aggressive patterns
    r"\bbecause I really,? really appreciate it\b",
    r"\bif you do not mind,? and if it (?:is not|isn't) too much trouble\b",
    r"\bI really,? really appreciate it\b",
]

# Verbose instruction patterns (instruction simplification)
VERBOSE_PATTERNS = [
    (r"provide me with a(?:\s+[a-z]+)*\s+(list|summary) of", r"\1"),
    (r"give me a(?:\s+[a-z]+)*\s+(list|summary) of", r"\1"),
    (r"show me a(?:\s+[a-z]+)*\s+(list|summary) of", r"\1"),
]

_REQUEST_VERB_REPLACEMENTS = {
    "provide": "provide",
    "give": "give",
    "explain": "explain",
    "summarize": "summarize",
}

VERBOSE_PATTERNS.extend(
    (f"{prefix}{verb}", replacement)
    for prefix in INSTRUCTION_REQUEST_PREFIXES.values()
    for verb, replacement in _REQUEST_VERB_REPLACEMENTS.items()
)

VERBOSE_PATTERNS.extend(
    [
        (r"I need you to", ""),
        (r"I want you to", ""),
        (r"I would like (you )?to", ""),
        (r"make sure (to|that)", ""),
        (r"the following text", "text"),
        (r"the text below", "text"),
        (r"the above text", "text"),
        (r"in order to", "to"),
        (r"due to the fact that", "because"),
        (r"at this point in time", "now"),
        (r"for the purpose of", "for"),
        (r"with regard to", "regarding"),
        (r"with reference to", "regarding"),
        (r"as a result of", "from"),
        (r"in spite of the fact that", "although"),
        (r"until such time as", "until"),
        (r"it is important to mention that", "note:"),
        (r"it is worth pointing out that", "note:"),
        (r"a large number of", "many"),
        (r"a small number of", "few"),
        (r"the vast majority of", "most"),
        (r"a significant portion of", "much of"),
        (r"in the process of", "currently"),
        (r"is able to", "can"),
        (r"are able to", "can"),
        (r"has the ability to", "can"),
        (r"have the ability to", "can"),
        (r"is capable of", "can"),
        (r"are capable of", "can"),
        (r"in close proximity to", "near"),
        (r"in the vicinity of", "near"),
        (r"with the exception of", "except"),
        (r"on a regular basis", "regularly"),
        (r"on a daily basis", "daily"),
        (r"at the present moment", "now"),
        (r"during the course of", "during"),
        (r"subsequent to", "after"),
        (r"prior to", "before"),
        (r"in addition to", "besides"),
    ]
)

# Instruction format simplification rules
INSTRUCTION_FORMAT_PATTERNS = [
    (
        r"(summarize|list|explain|describe|analyze|show|demonstrate)\s+the\s+following\s+(\w+):?",
        r"\1 \2:",
    ),
    *(
        (f"{prefix}(?P<verb>\\w+)", r"\g<verb>")
        for prefix in INSTRUCTION_REQUEST_PREFIXES.values()
    ),
    (r"I need a (\w+) of", r"\1"),
    (r"I want a (\w+) of", r"\1"),
    (r"(show|demonstrate)\s+the\s+following\s+(\w+):?", r"\1 \2:"),
]

# Common word shortenings (entity canonicalization)
CANONICALIZATIONS = {
    "artificial intelligence": "AI",
    "machine learning": "ML",
    "natural language processing": "NLP",
    "large language model": "LLM",
    "application programming interface": "API",
    "user interface": "UI",
    "user experience": "UX",
    "information": "info",
    "application": "app",
    "configuration": "config",
    "database": "db",
    "documentation": "docs",
    "repository": "repo",
    "environment": "env",
    "function": "func",
    "reference": "ref",
    "authentication": "auth",
    "authorization": "authz",
    "implementation": "impl",
    "specification": "spec",
    "initialize": "init",
    "development": "dev",
    "production": "prod",
    "maximum": "max",
    "minimum": "min",
    "average": "avg",
    "number": "num",
    "string": "str",
    "boolean": "bool",
    "integer": "int",
    "continuous integration": "CI",
    "continuous deployment": "CD",
    "continuous integration/continuous deployment": "CI/CD",
    "software development kit": "SDK",
    "integrated development environment": "IDE",
    "graphical user interface": "GUI",
    "command line interface": "CLI",
    "structured query language": "SQL",
    "javascript object notation": "JSON",
    "extensible markup language": "XML",
    "hypertext markup language": "HTML",
    "cascading style sheets": "CSS",
    "uniform resource locator": "URL",
    "uniform resource identifier": "URI",
    "domain name system": "DNS",
    "hypertext transfer protocol": "HTTP",
    "secure hypertext transfer protocol": "HTTPS",
    "javascript": "JS",
    "typescript": "TS",
    "implement": "impl",
    "file transfer protocol": "FTP",
    "transmission control protocol": "TCP",
    "internet protocol": "IP",
    "rest application programming interface": "REST API",
    "representational state transfer": "REST",
    "create read update delete": "CRUD",
    "single sign-on": "SSO",
    "two-factor authentication": "2FA",
    "multi-factor authentication": "MFA",
    "object relational mapping": "ORM",
    "content delivery network": "CDN",
    "virtual private network": "VPN",
    "operating system": "OS",
    "central processing unit": "CPU",
    "graphics processing unit": "GPU",
    "random access memory": "RAM",
    "read only memory": "ROM",
    "solid state drive": "SSD",
    "hard disk drive": "HDD",
    "terms of service": "ToS",
    "end user license agreement": "EULA",
    "service level agreement": "SLA",
    "key performance indicator": "KPI",
    "return on investment": "ROI",
    "frequently asked questions": "FAQ",
    "as soon as possible": "ASAP",
    "for example": "e.g.",
    "for your information": "FYI",
    "please respond": "RSVP",
}

# Extra-short canonicals reserved for technical profiles only.
# These may be confusing in general prose but are common in technical prompts.
TECHNICAL_CANONICALIZATIONS = {
    "function": "fn",
    "parameter": "param",
    "parameters": "params",
    "variable": "var",
    "variables": "vars",
}

FLUFF_CANONICALIZATIONS = {
    "please note that": "",
    "keep in mind that": "",
    "as a reminder": "",
}

CONTEXTUAL_CANONICALIZATIONS = {
    "in the context of": "in",
    "from the perspective of": "from",
    "for the purposes of": "for",
    "with respect to": "about",
}

PROMPT_SPECIFIC_CANONICALIZATIONS = {
    "please make sure": "ensure",
    "make sure to": "ensure",
    "make sure that": "ensure",
    "be sure to": "ensure",
}

SMART_DEFAULT_CANONICALIZATIONS = {
    "for example": "e.g.",
    "for instance": "e.g.",
    "that is": "i.e.",
}

# Units that can safely be rendered without a space (e.g., "5 km" -> "5km").
# Intentionally excludes ambiguous forms like "m".
COMPACT_UNIT_ABBREVIATIONS = {
    "%",
    "°C",
    "°F",
    "K",
    "km",
    "cm",
    "mm",
    "mi",
    "ft",
    "in",
    "yd",
    "kg",
    "g",
    "mg",
    "µg",
    "L",
    "mL",
    "s",
    "min",
    "h",
    "d",
    "wk",
    "mo",
    "yr",
    "KB",
    "MB",
    "GB",
    "TB",
    "PB",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "J",
    "kJ",
    "W",
    "kW",
    "MW",
    "A",
    "mA",
    "V",
    "mV",
    "kV",
    "Ω",
    "N",
    "Pa",
    "bar",
    "atm",
    "km/h",
    "mph",
    "m/s",
    "ft/s",
}

# Filler words to remove (redundancy elimination)
FILLER_WORDS = [
    "actually",
    "basically",
    "really",
    "very",
    "quite",
    "just",
    "simply",
    "literally",
    "essentially",
    "particularly",
    "definitely",
    "absolutely",
    "totally",
    "completely",
    "extremely",
    "incredibly",
    "somewhat",
    "rather",
    "relatively",
    "fairly",
    "slightly",
    "indeed",
    "truly",
    "honestly",
    "frankly",
    "overall",
    "generally",
    "typically",
    "usually",
    "perhaps",
    "possibly",
    "presumably",
    "seemingly",
    "apparently",
    "politely",
    "respectfully",
]

# Multi-word filler phrases to remove (low-risk discourse markers).
MULTIWORD_FILLER_PHRASES = [
    "you know",
    "i mean",
    "you see",
]

# Redundant phrase patterns
REDUNDANT_PHRASES = [
    (r"in the event that", "if"),
    (r"add together", "add"),
    (r"final outcome", "outcome"),
    (r"the fact that", "that"),
    (r"first and foremost", "first"),
    (r"as a matter of fact", "actually"),
    (r"at the present time", "now"),
    (r"in the near future", "soon"),
    (r"at the end of the day", "ultimately"),
    (r"it is important to note that", ""),
    (r"it should be mentioned that", ""),
    (r"it is worth noting that", ""),
    (
        r"What (?:is|are) (.+?) and (?:how|why) (?:does|do) (?:it|they) work\?",
        r"Explain \1",
    ),
    (
        r"Can you (?:tell|explain|describe|share) (?:me|us)? (?:about|regarding) (.+?)\?",
        r"Explain \1",
    ),
    (r"What are (?:some|a few) examples of (.+?)\?", r"Examples of \1"),
    (r"How would you (?:describe|explain|define) (.+?)\?", r"Describe \1"),
    (r"Could you (?:please )?(?:provide|give) (.+?)\?", r"Provide \1"),
    (r"(?:As|Like) (?:an AI|a language model|an assistant),?\s*", ""),
    (r"I (?:am|will be) (?:happy|glad|pleased) to (?:help|assist)\.\s*", ""),
    (r"(?:So )?(?:you're asking|you want to know|your question is about)\s+", ""),
    (r"Based on (?:the|your) (?:question|query|request),?\s*", ""),
    (r"(?:Given|Considering) (?:the|this) context,?\s*", ""),
    (r"(?:Let me|Allow me to|I'll) (?:explain|describe|clarify)\.?\s*", ""),
    (
        r"Here(?:'s| is) (?:the|my|an?) (?:answer|response|explanation):?\s*",
        "",
    ),
]

CONTRACTION_MAP = {
    r"\bdo not\b": "don't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bshould not\b": "shouldn't",
    r"\bwould not\b": "wouldn't",
    r"\bcould not\b": "couldn't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bhave not\b": "haven't",
    r"\bhas not\b": "hasn't",
    r"\bhad not\b": "hadn't",
    r"\bare not\b": "aren't",
    r"\bis not\b": "isn't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bi am\b": "I'm",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\byou are\b": "you're",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bwhat is\b": "what's",
    r"\bwho is\b": "who's",
    r"\bwhere is\b": "where's",
    r"\bI will\b": "I'll",
    r"\bthey will\b": "they'll",
    r"\bwe will\b": "we'll",
    r"\byou will\b": "you'll",
    r"\bit will\b": "it'll",
    r"\blet us\b": "let's",
    r"\bI have\b": "I've",
    r"\bthey have\b": "they've",
    r"\bwe have\b": "we've",
    r"\byou have\b": "you've",
    r"\bI would\b": "I'd",
    r"\bthey would\b": "they'd",
    r"\bwe would\b": "we'd",
    r"\byou would\b": "you'd",
}

BOILERPLATE_PATTERNS = [
    (r"(?:Please note that|Note:|Disclaimer:)\s*[^.]+?\.", "[Note]"),
    (r"(?:Copyright|©|All rights reserved)[^.]*\.", "[Copyright]"),
    (r"(?:We appreciate|Thank you for|We value)\s+your\s+[^.]+?\.", "[Acknowledgment]"),
    (r"By (?:using|accessing|continuing)[^.]+?terms[^.]*?\.", "[Terms accepted]"),
    (r"(?:With that being said|Having said that|That being said),?\s*", ""),
    (r"(?:^|(?<=[.!?\n]))\s*(?:Moving on|To continue|Next|Now)\b,?\s*", ""),
]

# Important patterns for discourse weighting
IMPORTANT_PATTERNS = [
    r"\bdo not\b",
    r"\bdon\'t\b",
    r"\bnever\b",
    r"\balways\b",
    r"\bmust\b",
    r"\brequired\b",
    r"\bcritical\b",
    r"\bimportant\b",
    r"\bwarning\b",
    r"\berror\b",
    r"\bfailure\b",
    r"\bsuccess\b",
]

ROLE_WEIGHTS = {
    "system": 3.0,
    "developer": 2.5,
    "user": 2.0,
    "assistant": 1.0,
    "tool": 1.5,
}

DISCOURSE_LABEL_WEIGHTS = {
    "instruction": 0.9,
    "constraint": 0.85,
    "example": 0.6,
    "background": 0.4,
}

DISCOURSE_DEFAULT_WEIGHT = 0.6

SEGMENT_WEIGHT_HIGH = 0.85
SEGMENT_WEIGHT_MODERATE = 0.7

DIRECTIVE_KEYWORDS = [
    "do not",
    "never",
    "always",
    "must",
    "required",
    "critical",
    "important",
    "warning",
    "error",
    "failure",
    "success",
    "avoid",
    "prevent",
    "should",
    "need to",
    "ensure",
    "remember",
    "follow",
    "enforce",
    "constraint",
    "policy",
    "safety",
    "security",
    "limit",
    "without",
    "include",
    "exclude",
    "format",
    "json",
]

# Fast regex-based unit normalization (Technique 1)
FAST_UNIT_PATTERNS = [
    (r"(\d+)\s*(?:kilometers?|km)\b", r"\1km"),
    (r"(\d+)\s*(?:meters?|m)\b", r"\1m"),
    (r"(\d+)\s*(?:miles?|mi)\b", r"\1mi"),
    (r"(\d+)\s*(?:centimeters?|cm)\b", r"\1cm"),
    (r"(\d+)\s*(?:millimeters?|mm)\b", r"\1mm"),
    (r"(\d+)\s*(?:dollars?|USD?)\b", r"$\1"),
    (r"(\d+)\s*(?:euros?|EUR?)\b", r"€\1"),
    (r"(\d+)\s*(?:pounds?|GBP|£)\b", r"£\1"),
    (r"(\d+)\s*(?:kilograms?|kg)\b", r"\1kg"),
    (r"(\d+)\s*(?:grams?|g)\b", r"\1g"),
    (r"(\d+)\s*(?:pounds?|lb)\b", r"\1lb"),
    (r"(\d+)\s*(?:ounces?|oz)\b", r"\1oz"),
    (r"(\d+)\s*(?:seconds?|sec)\b", r"\1s"),
    (r"(\d+)\s*(?:minutes?|min)\b", r"\1min"),
    (r"(\d+)\s*(?:hours?|hrs?)\b", r"\1h"),
    (r"(\d+)\s*(?:percent|%)\b", r"\1%"),
]

# Aggressive punctuation compression (Technique 2)
AGGRESSIVE_PUNCTUATION_PATTERNS = [
    (r"\.\s*\.\s*\.", "."),
    (r"\s*([,:;])\s+", r"\1 "),
    (r"['`']\s+", " "),
]

# Additional Redundant Phrases (Technique 3)
TELEGRAM_REDUNDANCY_PATTERNS = [
    (r"\bis a\b", "is"),
    (r"\bare able to\b", "can"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bwith regard to\b", "about"),
    (r"\bdue to the fact that\b", "because"),
    (r"\badd together\b", "add"),
    (r"\bfinal outcome\b", "outcome"),
]

# Prepend telegram patterns to ensure they override less-aggressive defaults
REDUNDANT_PHRASES = TELEGRAM_REDUNDANCY_PATTERNS + REDUNDANT_PHRASES
