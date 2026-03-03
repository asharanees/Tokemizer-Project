"""Content-aware optimization routing for Tokemizer.

This module implements the Smart Router, which automatically detects content types
and applies appropriate optimization profiles to maximize token reduction while
preserving content integrity.

The router operates in O(1) average time by analyzing only a sample of the input,
ensuring zero impact on overall latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

# =============================================================================
# Smart Context & Technical Parameter Selection
# =============================================================================


@dataclass
class SmartContext:
    """Recommended technical parameters derived from content analysis."""

    enable_frequency_learning: bool
    use_discourse_weighting: bool
    chunking_mode: str  # 'fixed', 'structured', 'semantic', 'off'
    section_ranking_enabled: bool
    preserve_digits: bool
    description: str = ""


@dataclass(frozen=True)
class ContentProfile:
    """Optimization profile for a specific content type.

    Attributes:
        name: Human-readable profile identifier
        disabled_passes: Set of pass names to skip for this content type
        threshold_modifiers: Multipliers for various thresholds (e.g., semantic_guard)
        smart_defaults: Default technical settings for this content type
        description: Brief explanation of why this profile exists
    """

    name: str
    disabled_passes: FrozenSet[str] = field(default_factory=frozenset)
    threshold_modifiers: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    # Default technical parameters for this content type
    smart_defaults: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def get_threshold_modifier(self, key: str, default: float = 1.0) -> float:
        """Get threshold modifier by key."""
        for k, v in self.threshold_modifiers:
            if k == key:
                return v
        return default


def resolve_smart_context(
    text: str,
    profile: Optional[ContentProfile] = None,
    current_token_count: int = 0,
) -> SmartContext:
    """Derive smart technical parameters from content type and length.

    This function implements the "Auto-Select" logic for technical parameters,
    intelligently inhibiting risky behaviors (like frequency learning on code)
    and enabling helpful ones (like section ranking for massive docs).

    Args:
        text: Input prompt text
        profile: Resolved content profile (optional, classified if None)
        current_token_count: Token count of input (optional optimization)

    Returns:
        SmartContext object with recommended settings
    """
    if profile is None:
        _, profile = get_profile_for_text(text)

    text_len = len(text)
    token_count = current_token_count if current_token_count > 0 else text_len // 4

    defaults = profile.smart_defaults  # Base defaults from profile

    # 1. Frequency Learning Logic
    # Enable only for prose-like content of sufficient length
    # NEVER enable for code/json as it corrupts syntax
    enable_freq = False
    if defaults.get("allow_frequency_learning", True):
        if text_len > 2000:  # Only worth it for longer text
            enable_freq = True

    # 2. Discourse Weighting Logic
    # Enable for prose/dialogue, disable for structure-heavy formats
    use_discourse = defaults.get("use_discourse_weighting", True)

    # 3. Chunking Mode Logic
    chunking_mode = defaults.get("chunking_mode", "semantic")

    # 4. Section Ranking Logic
    # Only enable for massive documents where structure search helps
    section_ranking = False
    if token_count > 10000:  # Massive document threshold
        if profile.name in ("heavy_document", "technical_doc", "markdown"):
            section_ranking = True

    preserve_digits = defaults.get("preserve_digits", False)
    if not preserve_digits and profile.name == "general_prose":
        sample = text
        if len(sample) > 4000:
            midpoint = len(sample) // 2
            sample = (
                sample[:2000] + sample[midpoint - 500 : midpoint + 500] + sample[-2000:]
            )
        if (
            _NUMERIC_TICKET_PATTERN.search(sample)
            or _NUMERIC_VERSION_PATTERN.search(sample)
            or _NUMERIC_ID_PATTERN.search(sample)
        ):
            preserve_digits = True

    return SmartContext(
        enable_frequency_learning=enable_freq,
        use_discourse_weighting=use_discourse,
        chunking_mode=chunking_mode,
        section_ranking_enabled=section_ranking,
        preserve_digits=preserve_digits,
        description=f"Auto-derived for {profile.name} ({token_count} tokens)",
    )


# =============================================================================
# Content Detection Patterns (Pre-compiled for speed)
# =============================================================================

# Code detection - common programming keywords and patterns
_CODE_KEYWORDS = re.compile(
    r"\b(?:"
    r"def\s+\w+\s*\(|"  # Python function
    r"function\s+\w+\s*\(|"  # JavaScript function
    r"class\s+\w+|"  # Class definition
    r"import\s+[\w.]+|"  # Import statement
    r"from\s+\w+\s+import|"  # Python from import
    r"const\s+\w+\s*=|"  # JavaScript const
    r"let\s+\w+\s*=|"  # JavaScript let
    r"var\s+\w+\s*=|"  # JavaScript var
    r"public\s+(?:static\s+)?(?:void|int|String)|"  # Java
    r"fn\s+\w+\s*\(|"  # Rust
    r"func\s+\w+\s*\(|"  # Go
    r"async\s+(?:def|function)|"  # Async functions
    r"=>|"  # Arrow functions
    r"if\s*\([^)]+\)\s*{|"  # C-style if
    r"for\s*\([^)]+\)\s*{|"  # C-style for
    r"while\s*\([^)]+\)\s*{"  # C-style while
    r")\b",
    re.MULTILINE,
)

# Additional code indicators (syntax elements)
_CODE_SYNTAX = re.compile(
    r"(?:"
    r'\breturn\s+[\w\[\]{}"\']+|'  # Return statement
    r"===|!==|&&|\|\||"  # JavaScript operators
    r"\w+\s*\+=\s*\w+|"  # Compound assignment
    r"\w+\.\w+\(|"  # Method call
    r"\[\s*\d+\s*\]|"  # Array index
    r"#include\s*<|"  # C/C++ include
    r"@\w+\s*\("  # Decorator/annotation
    r")",
    re.MULTILINE,
)

# JSON detection - starts with { or [ and ends with } or ]
_JSON_START = re.compile(r"^\s*[\[{]")
_JSON_END = re.compile(r"[\]}]\s*$")

# Dialogue/Chat detection - role prefixes at line starts
_DIALOGUE_ROLES = re.compile(
    r"^(?:User|Assistant|Human|AI|System|Bot|Agent|Customer|Support|Admin|"
    r"Interviewer|Interviewee|Speaker\s*\d*|Person\s*\d*|Q|A)\s*:",
    re.MULTILINE | re.IGNORECASE,
)

# Markdown detection - headers, lists, code blocks
_MARKDOWN_PATTERNS = re.compile(
    r"(?:"
    r"^#{1,6}\s+\w+|"  # Headers
    r"^\s*[-*+]\s+\w+|"  # Unordered lists
    r"^\s*\d+\.\s+\w+|"  # Ordered lists
    r"^(?:```|~~~)\w*\s*$|"  # Code block markers
    r"\[.+?\]\(.+?\)|"  # Links
    r"\*\*[^*]+\*\*|"  # Bold
    r"__[^_]+__|"  # Bold alt
    r"\*[^*]+\*|"  # Italic
    r"_[^_]+_"  # Italic alt
    r")",
    re.MULTILINE,
)

# Technical/API documentation patterns
_TECHNICAL_DOC_PATTERNS = re.compile(
    r"(?:"
    r"\bAPI\s+(?:endpoint|reference|documentation)|"
    r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+/|"
    r"\bHTTP/\d|"
    r"\bContent-Type:|"
    r"\bAuthorization:|"
    r"\b(?:request|response)\s+body|"
    r"\bJSON\s+schema|"
    r"\bparameters?:|"
    r"\breturns?:"
    r")",
    re.IGNORECASE,
)

_NUMERIC_TICKET_PATTERN = re.compile(r"\b(?:[A-Z]{2,}-\d{2,}|#\d{3,})\b")
_NUMERIC_VERSION_PATTERN = re.compile(r"\b(?:v?\d+\.\d+(?:\.\d+){0,2})\b")
_NUMERIC_ID_PATTERN = re.compile(r"\b(?:[A-Z]{2,}\d{2,}|\d{2,}[A-Z]{2,})\b")


# =============================================================================
# Content Profiles
# =============================================================================

PROFILES: Dict[str, ContentProfile] = {
    "code": ContentProfile(
        name="code",
        disabled_passes=frozenset(
            {
                "clean_instruction_noise",
                "shorten_synonyms",  # May corrupt identifiers
                "compress_clauses",  # Comments are important in code
                "remove_fillers",  # May break syntax
                "apply_contractions",  # Don't modify string literals
                "collapse_paradoxical_phrases",  # Not applicable to code
                "consolidate_repeated_phrases",  # Intentional repetition in code
                "alias_json_keys",  # Preserve exact structure
                "apply_macro_dictionary",  # Avoid aliasing in code
                "normalize_whitespace",  # Preserve code indentation
                "final_whitespace",  # Preserve trailing newlines/indentation
                "compress_punctuation",  # Preserve code punctuation exactly
                "prune_low_entropy",  # Avoid removing syntactic elements
            }
        ),
        threshold_modifiers=(
            ("semantic_guard", 0.985),
            ("dedup_similarity", 0.97),
            ("entropy_budget", 0.4),
            ("summarize_threshold", 1.1),
        ),
        smart_defaults={
            "allow_frequency_learning": False,  # Never rename vars
            "use_discourse_weighting": False,  # Code is flat structure
            "chunking_mode": "fixed",
            "preserve_digits": True,
            "classifier_min_confidence": 0.2,
            "classifier_min_keep_ratio": 0.9,
        },
        description="Optimized for source code - preserves identifiers and syntax",
    ),
    "json": ContentProfile(
        name="json",
        disabled_passes=frozenset(
            {
                "shorten_synonyms",  # JSON keys/values must be preserved
                "compress_clauses",  # Structure matters
                "clean_instruction_noise",  # No instructions in JSON
                "apply_contractions",  # Don't modify strings
                "remove_fillers",  # Not applicable
                "normalize_whitespace",  # Preserve whitespace inside JSON strings
                "final_whitespace",  # Preserve whitespace inside JSON strings
                "compress_punctuation",  # Preserve punctuation inside JSON strings
                "alias_json_keys",  # Preserve exact structure
                "apply_macro_dictionary",  # Avoid aliasing inside JSON
            }
        ),
        threshold_modifiers=(
            ("semantic_guard", 0.99),
            ("dedup_similarity", 0.98),
            ("entropy_budget", 0.35),
            ("summarize_threshold", 1.2),
        ),
        smart_defaults={
            "allow_frequency_learning": False,
            "use_discourse_weighting": False,
            "chunking_mode": "fixed",
            "preserve_digits": True,
            "classifier_min_confidence": 0.2,
            "classifier_min_keep_ratio": 0.95,
        },
        description="Optimized for JSON - maximum structure preservation",
    ),
    "dialogue": ContentProfile(
        name="dialogue",
        disabled_passes=frozenset(),  # All passes safe for dialogue
        threshold_modifiers=(
            ("summarize_threshold", 0.65),
            ("entropy_budget", 1.35),
            ("dedup_similarity", 0.92),
        ),
        smart_defaults={
            "allow_frequency_learning": True,
            "use_discourse_weighting": True,
            "chunking_mode": "structured",
            "preserve_digits": False,
            "classifier_min_confidence": 0.45,
            "classifier_min_keep_ratio": 0.6,
        },
        description="Optimized for conversations - enables aggressive summarization",
    ),
    "markdown": ContentProfile(
        name="markdown",
        disabled_passes=frozenset(
            {
                "compress_clauses",  # Preserve document structure
            }
        ),
        threshold_modifiers=(
            ("semantic_guard", 0.93),
            ("summarize_threshold", 0.85),
            ("entropy_budget", 1.1),
        ),
        smart_defaults={
            "allow_frequency_learning": True,
            "use_discourse_weighting": True,
            "chunking_mode": "structured",
            "preserve_digits": False,
            "classifier_min_confidence": 0.5,
            "classifier_min_keep_ratio": 0.7,
        },
        description="Optimized for markdown documents - preserves formatting",
    ),
    "technical_doc": ContentProfile(
        name="technical_doc",
        disabled_passes=frozenset(
            {
                "shorten_synonyms",  # Technical terms matter
                "compress_clauses",  # Preserve specifications
            }
        ),
        threshold_modifiers=(
            ("semantic_guard", 0.96),
            ("query_budget", 1.05),
            ("summarize_threshold", 0.9),
            ("entropy_budget", 1.05),
        ),
        smart_defaults={
            "allow_frequency_learning": True,
            "use_discourse_weighting": True,
            "chunking_mode": "structured",
            "preserve_digits": True,  # Versions/ports matter
            "classifier_min_confidence": 0.5,
            "classifier_min_keep_ratio": 0.75,
        },
        description="Optimized for API/technical docs - preserves terminology",
    ),
    "heavy_document": ContentProfile(
        name="heavy_document",
        disabled_passes=frozenset(
            {
                "clean_instruction_noise",  # Large docs may have mixed content
            }
        ),
        threshold_modifiers=(
            ("entropy_budget", 1.5),  # More aggressive pruning
            ("chunk_overlap", 0.15),  # More overlap for context
            ("summarize_threshold", 0.6),  # Aggressive summarization
            ("query_budget", 1.15),
        ),
        smart_defaults={
            "allow_frequency_learning": True,
            "use_discourse_weighting": True,
            "chunking_mode": "semantic",
            "preserve_digits": False,
            "classifier_min_confidence": 0.4,
            "classifier_min_keep_ratio": 0.6,
        },
        description="Optimized for very large documents (>10k tokens)",
    ),
    "short": ContentProfile(
        name="short",
        disabled_passes=frozenset(
            {
                "compress_repeated_fragments",  # Unlikely in short text
                "prune_low_entropy",  # Not needed
                "summarize_history",  # Not applicable
                "compress_examples",  # Not applicable
            }
        ),
        threshold_modifiers=(),
        smart_defaults={
            "allow_frequency_learning": False,  # Too short to learn
            "use_discourse_weighting": True,
            "chunking_mode": "off",
            "preserve_digits": False,
            "classifier_min_confidence": 0.55,
            "classifier_min_keep_ratio": 0.75,
        },
        description="Optimized for short prompts - skips heavy passes",
    ),
    "general_prose": ContentProfile(
        name="general_prose",
        disabled_passes=frozenset(),  # All passes enabled
        threshold_modifiers=(
            ("query_budget", 0.9),
            ("summarize_threshold", 0.85),
            ("entropy_budget", 1.15),
        ),
        smart_defaults={
            "allow_frequency_learning": True,
            "use_discourse_weighting": True,
            "chunking_mode": "semantic",
            "preserve_digits": False,
            "classifier_min_confidence": 0.45,
            "classifier_min_keep_ratio": 0.6,
        },
        description="Default profile for general text - full optimization",
    ),
}


# Default profile when no specific match
DEFAULT_PROFILE = PROFILES["general_prose"]

# =============================================================================
# Content Classification
# =============================================================================


def classify_content(text: str, *, sample_size: int = 2000) -> str:
    """Classify content type for optimization routing.

    This function analyzes the input text to determine its type (code, JSON,
    dialogue, etc.) and returns a profile name. It operates in O(1) average
    time by only analyzing a sample of the text.

    Args:
        text: Input text to classify
        sample_size: Maximum characters to analyze (default 2000)

    Returns:
        Content type string matching a key in PROFILES

    Examples:
        >>> classify_content("def hello(): return 'world'")
        'code'
        >>> classify_content('{"key": "value"}')
        'json'
        >>> classify_content("User: Hi\\nAssistant: Hello!")
        'dialogue'
    """
    if not text:
        return "general_prose"

    text_len = len(text)

    # Sample for analysis (beginning + middle + end for large texts)
    if text_len <= sample_size:
        sample = text
    else:
        chunk_size = sample_size // 3
        sample = (
            text[:chunk_size]
            + text[text_len // 2 - chunk_size // 2 : text_len // 2 + chunk_size // 2]
            + text[-chunk_size:]
        )

    stripped = text.strip()

    # JSON detection (check full text boundaries)
    if _JSON_START.match(stripped) and _JSON_END.search(stripped):
        # Additional validation: check for JSON structure indicators
        if '":' in sample or '",' in sample or '": ' in sample:
            return "json"

    # Very short non-structured text gets special handling
    if text_len < 100:
        return "short"

    # Code detection - look for programming patterns
    code_keyword_matches = len(_CODE_KEYWORDS.findall(sample))
    code_syntax_matches = len(_CODE_SYNTAX.findall(sample))
    total_code_indicators = code_keyword_matches + code_syntax_matches

    if total_code_indicators >= 3:
        return "code"
    if code_keyword_matches >= 2:
        return "code"

    # Dialogue detection
    dialogue_matches = len(_DIALOGUE_ROLES.findall(sample))
    if dialogue_matches >= 2:
        return "dialogue"

    # Technical documentation
    if _TECHNICAL_DOC_PATTERNS.search(sample):
        return "technical_doc"

    # Markdown detection
    markdown_matches = len(_MARKDOWN_PATTERNS.findall(sample))
    if markdown_matches >= 5:
        return "markdown"

    # Heavy document detection (by length)
    if text_len > 15000:
        return "heavy_document"

    return "general_prose"


def get_profile(content_type: str) -> ContentProfile:
    """Get optimization profile for a content type.

    Args:
        content_type: Content type string (e.g., 'code', 'json')

    Returns:
        ContentProfile instance for the given type, or default profile
    """
    return PROFILES.get(content_type, DEFAULT_PROFILE)


def get_profile_for_text(text: str) -> Tuple[str, ContentProfile]:
    """Convenience function to classify and get profile in one call.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (content_type, ContentProfile)
    """
    content_type = classify_content(text)
    return content_type, get_profile(content_type)


# =============================================================================
# Integration Helpers
# =============================================================================


def merge_disabled_passes(
    mode_disabled: Set[str],
    profile: ContentProfile,
) -> Set[str]:
    """Merge mode-based and profile-based disabled passes.

    Args:
        mode_disabled: Passes disabled by optimization mode (conservative/balanced/maximum)
        profile: Content profile with additional disabled passes

    Returns:
        Combined set of all disabled passes
    """
    return mode_disabled | set(profile.disabled_passes)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContentProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "classify_content",
    "get_profile",
    "get_profile_for_text",
    "merge_disabled_passes",
]
