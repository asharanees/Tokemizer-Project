from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple

from services.model_cache_manager import get_model_configs
from services.telemetry_control import is_enabled as telemetry_is_enabled

try:  # Optional dependency for vector operations and summarization
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore
    logging.warning("numpy not available - advanced summarization disabled")

from ..discourse import DiscourseAnalyzer
from ..repetition import RepetitionDetector
from . import adjunct as _adjunct
from . import chunking as _chunking
from . import config
from . import entity_aliasing as _entity_aliasing
from . import entropy as _entropy
from . import history as _history
from . import lexical as _lexical
from . import max_prepass as _max_prepass
from . import metrics as _metrics
from . import preservation as _preservation
from . import section_ranking as _section_ranking
from . import structural as _structural
from . import telemetry as _telemetry
from . import token_classifier as _token_classifier
from .config_utils import get_env_float, load_phrase_dictionary, sanitize_canonical_map
from .coref_utils import build_coref_alias, select_coref_pronoun
from .glossary import GlossaryCollector
from .lsh import MinHashSignature as MinHash
from .lsh import SentenceLSHIndex
from .pipeline_config import resolve_optimization_config
from .placeholders import span_overlaps_placeholder
from .profiling import PipelineProfiler
from .router import (
    ContentProfile,
    SmartContext,
    classify_content,
    get_profile,
    merge_disabled_passes,
    resolve_smart_context,
)
from .segment_weights import analyze_segment_spans
from .tiktoken_init import init_tiktoken
from .trie_replacer import TrieReplacer, apply_phrase_dictionary, trie_canonicalize

spacy = None  # type: ignore
spacy_coref = None  # type: ignore

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - strict mode requires tokenizer support")


logger = logging.getLogger(__name__)

_SPACY_IMPORT_LOCK = threading.Lock()
_SPACY_COREF_IMPORT_LOCK = threading.Lock()
_COREF_NLP_SINGLETON_LOCK = threading.Lock()
_COREF_NLP_SINGLETON = None


def _import_spacy():
    global spacy
    if spacy is not None:
        return spacy
    with _SPACY_IMPORT_LOCK:
        if spacy is not None:
            return spacy
        try:  # pragma: no cover - optional dependency
            import spacy as _spacy  # type: ignore
        except ImportError:
            return None
        spacy = _spacy
        return spacy


def _import_spacy_coref():
    global spacy_coref
    if spacy_coref is not None:
        return spacy_coref
    with _SPACY_COREF_IMPORT_LOCK:
        if spacy_coref is not None:
            return spacy_coref
        try:  # pragma: no cover - optional dependency
            import spacy_coref as _spacy_coref  # type: ignore
        except ImportError:
            return None
        spacy_coref = _spacy_coref
        return spacy_coref


_DEFAULT_JSON_COMPRESSION_CONFIG = {"default": False, "overrides": {}}
_ENTROPY_BUDGET_RATIO = 0.08
_ENTROPY_CAP_FLOOR = 80
_ENTROPY_MAX_RATIO = 0.12
_ENTROPY_MIN_BUDGET = 20
_ENTROPY_MIN_LENGTH = 200
_ENTROPY_SAMPLE_MAX_CHARS = 1800
_ENTROPY_DENSE_THRESHOLD = 2.8
_ENTROPY_FLUFFY_THRESHOLD = 1.4
_ENTROPY_DENSE_MULTIPLIER = 1.35
_ENTROPY_FLUFFY_MULTIPLIER = 0.65
_ENTROPY_CONFIDENCE_FLOOR = 0.05
_BOUNDARY_PROTECT_HEAD_RATIO = 0.15
_BOUNDARY_PROTECT_TAIL_RATIO = 0.10
_BOUNDARY_PROTECT_MIN_TOKENS = 80
_FASTPATH_TOKEN_THRESHOLD = 1000  # Increased from 400 to cover 80%+ of typical prompts
_FASTPATH_MAX_NEWLINES = 4
_LSH_THRESHOLD = 0.8
_MINHASH_THRESHOLD = 0.45
_MINHASH_NUM_PERM = 128
_PROMPT_COST_PER_1K = 0.0
_REPEAT_MIN_OCCURRENCES = 2
_REPEAT_MIN_TOKENS = 20
_NEAR_DUP_LOOKBACK = 8
_MINHASH_PARAPHRASE_DEFAULTS = {
    "conservative": 0.80,
    "balanced": 0.75,
    "maximum": 0.70,
}
_PRECHUNK_SENTENCE_DEDUP_REMOVAL_THRESHOLD = {
    "conservative": 0.78,
    "balanced": 0.66,
    "maximum": 0.54,
}
_MINHASH_LIGHT_SCAN_THRESHOLD = 0.82
_MINHASH_PARAPHRASE_MIN_WORDS = 7
_MINHASH_TOKEN_OVERLAP_THRESHOLD = 0.5
_CHAR_NEAR_DUP_MAX_LEN = 240
_NEAR_DUP_SIMILARITY = 0.90
_SEMANTIC_CHUNK_SIMILARITY = 0.6
_LOW_GAIN_HEAVY_PASS_MAX_TOKENS = 2000
_LOW_GAIN_HEAVY_PASS_SAVINGS_RATIO = 0.01
_POST_CHUNK_DEDUP_MIN_TOKENS = 1200
_POST_CHUNK_DEDUP_MIN_DUPLICATES = 2
_VERBATIM_BLOCK_MIN_LENGTH = 100  # Min chars for verbatim block deduplication
_VERBATIM_SLIDING_WINDOW_TOKENS = (
    20  # Min tokens for sliding window duplicate detection
)
_PRESERVED_ALIAS_MIN_TOKENS = 80
_REDUNDANCY_SECTION_RANKING_RATIO_THRESHOLD = 0.2
_REDUNDANCY_SECTION_RANKING_TOKEN_MIN = 3500
_SHARED_TOKEN_CACHE_SIZE = 2048
_SHARED_TOKEN_CACHE_TTL_SECONDS = 300
_SPACY_MODEL_NAME = "en_core_web_sm"
_SPACY_MODEL_PATH_ENV = "PROMPT_OPTIMIZER_SPACY_MODEL_PATH"
_TOKEN_CACHE_MAX_SIZE = 1024
_TOKEN_BUDGET_UNCERTAINTY_THRESHOLD = 80
_TOKEN_BUDGET_TELEMETRY_SAMPLE_RATE = 0
_TOKEN_BUDGET_EXACT_PASS_COUNTING_ENV = "PROMPT_OPTIMIZER_EXACT_PASS_TOKEN_COUNTING"
_TOKEN_BUDGET_TELEMETRY_SAMPLE_RATE_ENV = "PROMPT_OPTIMIZER_PASS_TOKEN_SAMPLE_RATE"
_TOKEN_BUDGET_UNCERTAINTY_THRESHOLD_ENV = (
    "PROMPT_OPTIMIZER_PASS_TOKEN_UNCERTAINTY_THRESHOLD"
)
_COREF_MODEL_NAME = (
    "talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
)
_COREF_PIPE_NAME = "coref_minilm"
_COREF_PIPE_BY_MODEL_NAME = {
    _COREF_MODEL_NAME: _COREF_PIPE_NAME,
    _COREF_PIPE_NAME: _COREF_PIPE_NAME,
}
_PHRASE_DICTIONARY_ENV = "PROMPT_OPTIMIZER_PHRASE_DICTIONARY_PATH"
_TOKEN_CLASSIFIER_MODEL_ENV = "PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_MODEL"
_ALLOW_MODEL_ENV_OVERRIDE_ENV = "ALLOW_MODEL_ENV_OVERRIDE"
_TOKEN_ESTIMATE_BASE_PATTERN = re.compile(r"\b\w+\b|[.!?,;:\"'()\[\]{}]")
_TOKEN_ESTIMATE_DECIMAL_PATTERN = re.compile(r"\d+\.\d+")
_TOKEN_ESTIMATE_URL_PATTERN = re.compile(r"https?://\S+")
_TOKEN_ESTIMATE_EMAIL_PATTERN = re.compile(r"\S+@\S+")
_TOKEN_ESTIMATE_CODE_FENCE_PATTERN = re.compile(r"```")
_QUERY_AWARE_BUDGET_BY_MODE = {
    "conservative": 0.7,
    "balanced": 0.55,
    "maximum": 0.45,
}
_QUERY_AWARE_LONG_PROMPT_TOKENS = 4000
_QUERY_AWARE_LONG_PROMPT_MULTIPLIER = 1.05

_PREPIPELINE_SECTION_RANKING_BUDGET_FLOOR_MAXIMUM = 0.5
_PREPIPELINE_SECTION_RANKING_HARD_FLOOR_MAXIMUM = 0.35

_HEAVY_LATENCY_BUDGET_MS_BY_MODE: Dict[str, float] = {
    "conservative": 120.0,
    "balanced": 220.0,
    "maximum": 350.0,
}
_SHARED_TOKEN_CACHE: "OrderedDict[str, Tuple[List[int], float]]" = OrderedDict()
_SHARED_TOKEN_CACHE_LOCK = threading.Lock()


def _shared_token_cache_enabled() -> bool:
    return _SHARED_TOKEN_CACHE_SIZE > 0 and _SHARED_TOKEN_CACHE_TTL_SECONDS > 0


def _resolve_coref_pipe_name(model_name: str) -> str:
    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        return _COREF_PIPE_NAME
    return _COREF_PIPE_BY_MODEL_NAME.get(normalized_model_name, normalized_model_name)


def _get_shared_tokens(text: str) -> Optional[List[int]]:
    if not _shared_token_cache_enabled():
        return None
    now = time.monotonic()
    with _SHARED_TOKEN_CACHE_LOCK:
        cached = _SHARED_TOKEN_CACHE.get(text)
        if cached is None:
            return None
        tokens, expires_at = cached
        if expires_at <= now:
            _SHARED_TOKEN_CACHE.pop(text, None)
            return None
        refreshed_expires = now + _SHARED_TOKEN_CACHE_TTL_SECONDS
        _SHARED_TOKEN_CACHE[text] = (tokens, refreshed_expires)
        _SHARED_TOKEN_CACHE.move_to_end(text)
        return tokens


def _set_shared_tokens(text: str, tokens: List[int]) -> None:
    if not _shared_token_cache_enabled():
        return
    expires_at = time.monotonic() + _SHARED_TOKEN_CACHE_TTL_SECONDS
    with _SHARED_TOKEN_CACHE_LOCK:
        _SHARED_TOKEN_CACHE[text] = (tokens, expires_at)
        _SHARED_TOKEN_CACHE.move_to_end(text)
        if len(_SHARED_TOKEN_CACHE) > _SHARED_TOKEN_CACHE_SIZE:
            _SHARED_TOKEN_CACHE.popitem(last=False)


def _record_telemetry_flag(
    collector: Optional[_telemetry.OptimizationTelemetryCollector],
    name: str,
    value: Any,
) -> None:
    if collector is None:
        return
    try:
        collector.record_flag(name, value)
    except Exception:
        pass


_SEMANTIC_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\n+|(?<=[.!?])\s+)")


def _split_semantic_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SEMANTIC_SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def _split_semantic_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _split_semantic_sections(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?m)^(?=#{1,6}\s+|[A-Z][A-Za-z0-9 _-]{0,80}:)", text)
    sections = [part.strip() for part in parts if part and part.strip()]
    if len(sections) <= 1:
        return _split_semantic_paragraphs(text)
    return sections


def _build_semantic_plan(prompt: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    sentences = _split_semantic_sentences(prompt)
    paragraphs = _split_semantic_paragraphs(prompt)
    sections = _split_semantic_sections(prompt)
    return {
        "prompt": prompt,
        "units": {
            "sentences": sentences,
            "paragraphs": paragraphs,
            "sections": sections,
        },
        "embedding_vectors": {},
        "metrics": metrics,
        "created_at": time.perf_counter(),
    }


def _build_canonical_hint_pattern(
    canonical_map: Dict[str, str],
) -> Optional[re.Pattern]:
    if not canonical_map:
        return None

    tokens: List[str] = []
    for key in canonical_map:
        trimmed = key.strip()
        if not trimmed:
            continue
        token = trimmed.split()[0]
        if token:
            tokens.append(re.escape(token))

    if not tokens:
        return None

    return re.compile(
        "|".join(sorted(set(tokens), key=len, reverse=True)), re.IGNORECASE
    )


def _merge_canonical_maps(
    base_map: Dict[str, str], additions: Dict[str, str], allow_override: bool = False
) -> Dict[str, str]:
    """Merge canonical maps with optional override support.

    Args:
        base_map: The base mappings
        additions: Mappings to add
        allow_override: If True, additions override base on conflict. If False, skip.

    Returns:
        Merged dictionary
    """
    merged = dict(base_map)
    if not additions:
        return merged
    for key, value in additions.items():
        if not key or not value:
            continue
        existing_key = next((k for k in merged if k.lower() == key.lower()), None)
        if existing_key:
            if allow_override:
                # Override: remove existing and add new
                del merged[existing_key]
                merged[key] = value
            # else: skip (keep existing)
            continue
        merged[key] = value
    return merged


def _filter_disabled_canonicals(
    canonical_map: Dict[str, str], disabled_tokens: Collection[str]
) -> Dict[str, str]:
    if not canonical_map or not disabled_tokens:
        return canonical_map
    disabled_normalized = {
        token.strip().lower() for token in disabled_tokens if token and token.strip()
    }
    if not disabled_normalized:
        return canonical_map
    return {
        key: value
        for key, value in canonical_map.items()
        if key.strip().lower() not in disabled_normalized
    }


def _segment_allows_contextual_canon(segment: str) -> bool:
    if not segment:
        return False
    if "?" in segment:
        return False
    return _NEGATION_PATTERN.search(segment) is None


LIST_MARKER_PATTERN = re.compile(
    r"""
    ^(
        (?:[-*•+])(?:\s+|$) |
        (?:\(?\d+\)?[\.)]?)(?:\s+|$) |
        (?:[A-Za-z][\.)]|[ivxlcdm]+[\.)])(?:\s+|$)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_NORMALIZED_SENTENCE_SIGNATURE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
_ARABIC_CHAR_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

_EMOJI_REPEAT_PATTERN = re.compile(
    r"([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])\1+"
)

_WHITESPACE_MULTILINE_PATTERN = re.compile(r"\n\s*\n\s*\n+")
_WHITESPACE_INTERIOR_SPACE_PATTERN = re.compile(r"(?<=\S) {2,}")
_WHITESPACE_TRAILING_PATTERN = re.compile(r" +\n")
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|without|cannot|can't|don't|doesn't|won't|isn't|aren't|wasn't|weren't|"
    r"shouldn't|wouldn't|couldn't)\b",
    re.IGNORECASE,
)


_INSTRUCTION_TECHNIQUE_BY_CATEGORY = {
    "politeness": "Politeness Removal",
    "verbose": "Instruction Simplification",
    "redundant": "Redundancy Removal",
    "format": "Format Simplification",
    "filler": "Filler Word Removal",
}


_DIRECTIVE_KEYWORDS = [keyword.lower() for keyword in config.DIRECTIVE_KEYWORDS]
_DIRECTIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in config.IMPORTANT_PATTERNS
]
_CONSTRAINT_MARKER_PATTERN = re.compile(
    r"\b(?:must|must not|never|always|exactly|required|need to|has to|shall|do not|don't|cannot|can't|without)\b",
    re.IGNORECASE,
)
_INSTRUCTION_HEAD_TERMS = {
    "ensure",
    "include",
    "exclude",
    "follow",
    "output",
    "return",
    "format",
    "write",
    "provide",
    "use",
    "avoid",
    "keep",
    "list",
    "answer",
    "respond",
}
_TERMINAL_PUNCTUATION = {".", "!", "?"}

_FASTPATH_DISABLED_PASSES: Set[str] = {
    "deduplicate_content",
    "learn_frequency_abbreviations",
    "trim_adjunct_clauses",
    "alias_named_entities",
    "compress_coreferences",
    "compress_examples",
    "compress_repeated_fragments",
    "prune_low_entropy",
    "summarize_history",
}

_COREF_MAX_TOKENS = 512
_PRONOUN_MAX_DISTANCE = 200
_CONSTRAINT_NEGATION_TERMS = {
    "not",
    "never",
    "no",
    "don't",
    "do not",
    "must not",
    "should not",
    "cannot",
    "can't",
    "without",
}
_CONSTRAINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
    "we",
    "our",
}
_CONSTRAINT_HEAVY_PASS_NAMES = {
    "compress_examples",
    "summarize_history",
    "prune_low_entropy",
}


@dataclass
class _OptimizerThreadState:
    """State container used to isolate mutable attributes per worker thread."""

    techniques_applied: List[str] = field(default_factory=list)
    prefer_background_summary: bool = False
    skip_sentence_deduplication: bool = False
    skip_exact_deduplication: bool = False
    semantic_deduplication_enabled: bool = True
    optimization_mode: str = "balanced"
    token_cache: "OrderedDict[str, List[int]]" = field(default_factory=OrderedDict)
    toon_stats: Dict[str, int] = field(
        default_factory=lambda: {"conversions": 0, "bytes_saved": 0}
    )
    embedding_cache: Dict[Tuple[str, str, str], Any] = field(default_factory=dict)
    semantic_similarity_threshold_override: Optional[float] = None
    semantic_guard_threshold_override: Optional[float] = None
    minhash_paraphrase_threshold_override: Optional[float] = None
    dedup_counts: Dict[str, int] = field(
        default_factory=lambda: {"exact": 0, "near": 0, "semantic": 0}
    )
    semantic_plan: Optional[Dict[str, Any]] = None
    constraint_fingerprint: Optional[Dict[str, Any]] = None
    original_tokens: int = 0
    semantic_plan_metrics: Dict[str, float] = field(
        default_factory=lambda: {
            "embedding_reuse_count": 0.0,
            "embedding_calls_saved": 0.0,
            "embedding_wall_clock_savings_ms": 0.0,
        }
    )


@dataclass
class TokenBudgetTracker:
    """Track lightweight token estimates between stage boundaries."""

    estimated_tokens: int
    last_text_length: int
    sample_rate: int = 0
    uncertainty_threshold: int = 80
    uncertainty: float = 0.0
    pass_counter: int = 0

    def estimate_after_edit(self, before_text: str, after_text: str) -> int:
        before_len = max(len(before_text), 0)
        after_len = max(len(after_text), 0)
        if before_len <= 0:
            char_per_token = 4.0
        else:
            char_per_token = max(before_len / max(self.estimated_tokens, 1), 1.5)

        delta_chars = after_len - before_len
        delta_tokens = int(round(delta_chars / char_per_token))
        estimate = max(0, self.estimated_tokens + delta_tokens)

        self.uncertainty += abs(delta_tokens) * 0.15 + abs(delta_chars) * 0.01
        self.estimated_tokens = estimate
        self.last_text_length = after_len
        return estimate

    def should_sample_exact(self) -> bool:
        self.pass_counter += 1
        if self.sample_rate > 0 and self.pass_counter % self.sample_rate == 0:
            return True
        return self.uncertainty >= self.uncertainty_threshold

    def calibrate(self, text: str, exact_tokens: int) -> None:
        self.estimated_tokens = max(0, exact_tokens)
        self.last_text_length = len(text)
        self.uncertainty = 0.0


class PromptOptimizer:
    """
    Advanced prompt optimization service with context preservation.

    Token optimization approach:
    - Reduces token usage without sacrificing quality
    - Preserves critical information (code, numbers, quotes)
    - Applies multiple optimization passes
    - Handles large prompts up to 500K tokens with chunking
    - Adaptive heuristics for optimal results
    """

    def __init__(self):
        """Initialize the optimizer with tokenizer"""
        self._state_local = threading.local()
        self._nlp_lock = threading.RLock()
        self._coref_lock = threading.RLock()
        # Initialize tiktoken encoder for token counting with offline support
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = init_tiktoken(tiktoken)
        else:
            logger.warning(
                "tiktoken library not available - strict mode requires exact tokenization"
            )
            self.tokenizer = None

        self.techniques_applied = []
        self.enable_semantic_deduplication = True
        self.semantic_similarity_threshold = self._get_env_float(
            "PROMPT_OPTIMIZER_SEMANTIC_SIMILARITY", 0.92
        )
        self.minhash_candidate_threshold = _MINHASH_THRESHOLD
        self.enable_lsh_deduplication = self._get_env_bool(
            "PROMPT_OPTIMIZER_LSH_ENABLED", True
        )
        self.lsh_similarity_threshold = _LSH_THRESHOLD
        self.semantic_guard_enabled = self._get_env_bool(
            "PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED",
            config.SEMANTIC_GUARD_ENABLED,
        )
        self.semantic_guard_threshold = get_env_float(
            "PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD",
            config.SEMANTIC_GUARD_THRESHOLD,
        )
        self.semantic_guard_per_pass_enabled = self._get_env_bool(
            "PROMPT_OPTIMIZER_SEMANTIC_GUARD_PER_PASS",
            config.SEMANTIC_GUARD_PER_PASS_ENABLED,
        )

        db_configs = get_model_configs()
        allow_model_env_override = self._get_env_bool(
            _ALLOW_MODEL_ENV_OVERRIDE_ENV,
            False,
        )
        self._resolve_model_names(db_configs, allow_model_env_override)
        self.semantic_guard_max_prompt_tokens = config.SEMANTIC_GUARD_MAX_PROMPT_TOKENS
        self.json_compression_config = dict(_DEFAULT_JSON_COMPRESSION_CONFIG)
        self.prompt_cost_per_1k = _PROMPT_COST_PER_1K
        self.chunk_size = 50000
        self.chunk_threshold = self.chunk_size
        self.max_chunk_workers = max(1, os.cpu_count() or 4)
        self._chunk_executor: Optional[ThreadPoolExecutor] = None
        self.default_chunking_mode = "fixed"
        self._last_dedup_short_circuit = False
        self.semantic_chunk_similarity = _SEMANTIC_CHUNK_SIMILARITY
        self._minhash_num_perm = _MINHASH_NUM_PERM
        self.repeat_min_tokens = _REPEAT_MIN_TOKENS
        self.repeat_min_occurrences = _REPEAT_MIN_OCCURRENCES
        self.near_dup_similarity = _NEAR_DUP_SIMILARITY
        self.fastpath_token_threshold = _FASTPATH_TOKEN_THRESHOLD
        self.dedup_phrase_length = self._get_env_int(
            "PROMPT_OPTIMIZER_DEDUP_PHRASE_LENGTH", 5
        )
        self.prefer_shorter_duplicates = self._get_env_bool(
            "PROMPT_OPTIMIZER_PREFER_SHORTER_DUPLICATES", True
        )
        self._repetition_detector = RepetitionDetector(
            self.repeat_min_tokens, self.repeat_min_occurrences
        )

    def _resolve_model_names(
        self, db_configs: Dict[str, dict], allow_model_env_override: bool
    ) -> None:
        semantic_entry = db_configs.get("semantic_guard", {})
        semantic_inventory = semantic_entry.get("model_name")
        semantic_env = os.environ.get("PROMPT_OPTIMIZER_SEMANTIC_GUARD_MODEL")
        semantic_model = semantic_inventory
        if allow_model_env_override and semantic_env:
            semantic_model = semantic_env

        if not semantic_model:
            logger.error(
                "Semantic guard disabled: Model Inventory missing 'semantic_guard' entry."
                " Add the configuration before enabling semantic deduplication."
            )
            self.semantic_guard_model = None
            self.semantic_guard_enabled = False
        else:
            self.semantic_guard_model = semantic_model

        semantic_rank_entry = db_configs.get("semantic_rank", {})
        semantic_rank_inventory = semantic_rank_entry.get("model_name")
        semantic_rank_env = os.environ.get("PROMPT_OPTIMIZER_SEMANTIC_RANK_MODEL")
        semantic_rank_model = semantic_rank_inventory
        if allow_model_env_override and semantic_rank_env:
            semantic_rank_model = semantic_rank_env
        if not semantic_rank_model:
            logger.warning(
                "Semantic rank disabled: Model Inventory missing 'semantic_rank' entry."
                " Strict balanced/maximum requests require semantic_rank to be configured."
            )
            self.semantic_rank_model = None
        else:
            self.semantic_rank_model = semantic_rank_model

        token_entry = db_configs.get("token_classifier", {})
        token_inventory = token_entry.get("model_name")
        token_env = os.environ.get(_TOKEN_CLASSIFIER_MODEL_ENV)
        token_model = token_inventory
        if allow_model_env_override and token_env:
            token_model = token_env

        if not token_model:
            logger.warning(
                "Token classifier disabled: Model Inventory missing 'token_classifier' entry."
                " Add the configuration to restore token-aware compression."
            )
            self.token_classifier_model = None
        else:
            self.token_classifier_model = token_model

        coref_entry = db_configs.get("coreference", {})
        coref_inventory = coref_entry.get("model_name")
        if not coref_inventory:
            logger.warning(
                "Coreference model missing from Model Inventory; defaulting to '%s'.",
                _COREF_MODEL_NAME,
            )
        self._coref_model_name = coref_inventory or _COREF_MODEL_NAME
        self._coref_pipe_name = _resolve_coref_pipe_name(self._coref_model_name)

        self.semantic_chunk_model = self.semantic_guard_model
        self.entropy_prune_min_length = _ENTROPY_MIN_LENGTH
        self.entropy_prune_min_budget = _ENTROPY_MIN_BUDGET
        self.entropy_prune_ratio = min(0.5, max(0.0, _ENTROPY_BUDGET_RATIO))
        self.entropy_prune_max_ratio = min(
            0.5,
            max(self.entropy_prune_ratio, _ENTROPY_MAX_RATIO),
        )
        self.entropy_prune_cap_floor = max(
            self.entropy_prune_min_budget,
            _ENTROPY_CAP_FLOOR,
        )
        self.entropy_prune_min_confidence = min(
            1.0, max(0.0, _ENTROPY_CONFIDENCE_FLOOR)
        )
        self.chunk_overlap_ratio = 0.1
        self.summarize_keep_ratio_modifier = 1.0
        self._spacy_model_name = _SPACY_MODEL_NAME
        self._spacy_model_path = (
            str(os.environ.get(_SPACY_MODEL_PATH_ENV) or "").strip() or None
        )
        self._nlp = None
        self._nlp_load_failed = False
        self._nlp_disabled_pipes: List[str] = []
        self._nlp_pipe_names: List[str] = []
        self._linguistic_nlp = None
        self._linguistic_nlp_load_failed = False
        self._linguistic_nlp_pipe_names: List[str] = []
        self._canonical_patterns_cache_key: Optional[int] = None
        self._canonical_replacer_cache: Optional[TrieReplacer] = None
        self._canonical_map_cache: Dict[str, str] = {}
        self._canonical_hint_pattern: Optional[re.Pattern] = None
        self._technical_canonical_patterns_cache_key: Optional[int] = None
        self._technical_canonical_replacer_cache: Optional[TrieReplacer] = None
        self._technical_canonical_map_cache: Dict[str, str] = {}
        self._technical_canonical_hint_pattern: Optional[re.Pattern] = None
        self._discourse_analyzer = DiscourseAnalyzer(config.DIRECTIVE_KEYWORDS)
        self._coref_nlp = None
        self._coref_load_failed = False
        self._profiling_enabled = False
        self.token_cache_max_size = max(0, _TOKEN_CACHE_MAX_SIZE)
        self.exact_pass_token_counting = self._get_env_bool(
            _TOKEN_BUDGET_EXACT_PASS_COUNTING_ENV,
            False,
        )
        self.token_budget_telemetry_sample_rate = max(
            0,
            self._get_env_int(
                _TOKEN_BUDGET_TELEMETRY_SAMPLE_RATE_ENV,
                _TOKEN_BUDGET_TELEMETRY_SAMPLE_RATE,
            ),
        )
        self.token_budget_uncertainty_threshold = max(
            1,
            self._get_env_int(
                _TOKEN_BUDGET_UNCERTAINTY_THRESHOLD_ENV,
                _TOKEN_BUDGET_UNCERTAINTY_THRESHOLD,
            ),
        )

        section_mode_env = (
            os.environ.get("PROMPT_OPTIMIZER_SECTION_RANKING_MODE", "off") or "off"
        )
        self.section_ranking_mode = _section_ranking.normalize_ranking_mode(
            section_mode_env
        )
        token_budget_env = self._get_env_int(
            "PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET", 0
        )
        self.section_ranking_token_budget = (
            token_budget_env if token_budget_env > 0 else None
        )

        self.maximum_prepass_policy = config.PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY
        self.maximum_prepass_min_tokens = max(1, config.MAXIMUM_PREPASS_MIN_TOKENS)
        self.maximum_prepass_budget_ratio = min(
            0.95, max(0.2, config.MAXIMUM_PREPASS_BUDGET_RATIO)
        )
        self.maximum_prepass_max_sentences = max(
            8, config.MAXIMUM_PREPASS_MAX_SENTENCES
        )
        self.autotune_profile = config.PROMPT_OPTIMIZER_AUTOTUNE_PROFILE
        logger.info(
            (
                "Optimizer autotune profile resolved: %s "
                "(max_prepass_policy=%s min_tokens=%s budget_ratio=%.2f "
                "max_sentences=%s token_classifier_post=%s)"
            ),
            self.autotune_profile,
            self.maximum_prepass_policy,
            self.maximum_prepass_min_tokens,
            self.maximum_prepass_budget_ratio,
            self.maximum_prepass_max_sentences,
            config.TOKEN_CLASSIFIER_POST_PASS_ENABLED,
        )

        phrase_dictionary_path = os.environ.get(_PHRASE_DICTIONARY_ENV)
        self.phrase_dictionary = load_phrase_dictionary(phrase_dictionary_path)

        default_token_classifier = db_configs.get("token_classifier", {}).get(
            "model_name"
        )
        if allow_model_env_override:
            env_token_classifier = os.environ.get(_TOKEN_CLASSIFIER_MODEL_ENV)
            self.token_classifier_model = (
                env_token_classifier
                if env_token_classifier
                else default_token_classifier
            )
        else:
            self.token_classifier_model = default_token_classifier
        self._model_load_status: Dict[str, Any] = {}

    def refresh_model_configs(self) -> None:
        db_configs = get_model_configs()
        allow_model_env_override = self._get_env_bool(
            _ALLOW_MODEL_ENV_OVERRIDE_ENV,
            False,
        )
        self._resolve_model_names(db_configs, allow_model_env_override)
        self._spacy_model_path = (
            str(os.environ.get(_SPACY_MODEL_PATH_ENV) or "").strip() or None
        )

        _entropy.reset_entropy_model()
        _token_classifier.reset_token_classifier()
        _metrics.reset_encoder_cache()
        self._nlp_load_failed = False
        self._nlp = None
        self._nlp_disabled_pipes = []
        self._nlp_pipe_names = []
        self._linguistic_nlp_load_failed = False
        self._linguistic_nlp = None
        self._linguistic_nlp_pipe_names = []
        self._coref_load_failed = False
        self._coref_nlp = None

    @staticmethod
    def _find_spacy_model_subdir(base_path: str) -> Optional[str]:
        if not os.path.isdir(base_path):
            return None
        try:
            entries = sorted(os.listdir(base_path))
        except OSError:
            return None

        for entry in entries:
            candidate = os.path.join(base_path, entry)
            if not os.path.isdir(candidate):
                continue
            if os.path.isfile(os.path.join(candidate, "config.cfg")):
                return candidate
        return None

    def _resolve_spacy_model_target(self) -> str:
        model_name = self._spacy_model_name
        configured_paths: List[str] = []
        if self._spacy_model_path:
            configured_paths.append(self._spacy_model_path)

        spacy_home = str(os.environ.get("SPACY_HOME") or "").strip()
        if spacy_home:
            configured_paths.append(os.path.join(spacy_home, model_name))

        for configured_path in list(dict.fromkeys(configured_paths)):
            if not os.path.isdir(configured_path):
                logger.warning(
                    "spaCy model path %s not found; falling back to %s",
                    configured_path,
                    model_name,
                )
                continue
            if os.path.isfile(os.path.join(configured_path, "config.cfg")):
                return configured_path

            nested_model_dir = self._find_spacy_model_subdir(configured_path)
            if nested_model_dir:
                return nested_model_dir

            logger.warning(
                "spaCy model path %s missing config.cfg; falling back to %s",
                configured_path,
                model_name,
            )
        return model_name

    # ------------------------------------------------------------------
    # Thread-local state management
    # ------------------------------------------------------------------
    def _get_state(self) -> _OptimizerThreadState:
        state = getattr(self._state_local, "value", None)
        if state is None:
            state = _OptimizerThreadState()
            self._state_local.value = state
        return state

    def _resolve_semantic_similarity_threshold(self) -> float:
        override = self._get_state().semantic_similarity_threshold_override
        return override if override is not None else self.semantic_similarity_threshold

    def _resolve_semantic_guard_threshold(self) -> float:
        override = self._get_state().semantic_guard_threshold_override
        if override is not None:
            return override
        return self.semantic_guard_threshold

    def _require_semantic_guard_similarity(
        self,
        before_text: str,
        after_text: str,
        *,
        embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    ) -> float:
        similarity = _metrics.score_similarity(
            before_text,
            after_text,
            self.semantic_guard_model,
            embedding_cache=embedding_cache,
        )
        if similarity is None:
            raise RuntimeError(
                "Semantic guard similarity scoring unavailable; strict mode forbids fallback."
            )
        return similarity

    def _per_pass_guard_enabled(self, optimization_mode: str) -> bool:
        state = self._get_state()
        fastpath_floor = max(self.fastpath_token_threshold or 0, 0)
        if fastpath_floor and state.original_tokens <= fastpath_floor:
            # Keep the final semantic guard only; skip per-pass guard for short prompts.
            return False
        return (
            optimization_mode == "maximum"
            and self.semantic_guard_enabled
            and self.semantic_guard_model is not None
            and self.semantic_guard_per_pass_enabled
        )

    def _apply_per_pass_semantic_guard(
        self,
        pass_name: str,
        before_text: str,
        after_text: str,
        *,
        guard_threshold: float,
        telemetry_collector: Optional[Any],
    ) -> Tuple[str, bool, Optional[float]]:
        if before_text == after_text:
            return after_text, False, None
        similarity = _metrics.score_similarity(
            before_text,
            after_text,
            self.semantic_guard_model,
            embedding_cache=self._get_state().embedding_cache,
        )
        if similarity is None:
            logger.warning(
                "Per-pass semantic guard similarity unavailable for '%s'; reverting pass output.",
                pass_name,
            )
            return before_text, True, None
        rollback = similarity < guard_threshold
        if rollback:
            _record_telemetry_flag(
                telemetry_collector, f"{pass_name}_guard_rollback", True
            )
        _record_telemetry_flag(
            telemetry_collector, f"{pass_name}_guard_similarity", similarity
        )
        return (before_text if rollback else after_text), rollback, similarity

    def _resolve_multi_candidate_settings(
        self, pass_name: str, optimization_mode: str
    ) -> Dict[str, Any]:
        settings = dict(config.MULTI_CANDIDATE_PASS_SETTINGS.get(pass_name, {}))
        if optimization_mode != "maximum":
            settings["max_candidates"] = 1
        else:
            # For short prompts, multi-candidate semantic selection creates
            # high fixed latency with negligible quality gains.
            state = self._get_state()
            fastpath_floor = max(self.fastpath_token_threshold or 0, 0)
            if fastpath_floor and state.original_tokens <= fastpath_floor:
                settings["max_candidates"] = 1
        max_candidates = settings.get("max_candidates", 1)
        try:
            max_candidates = int(max_candidates)
        except (TypeError, ValueError):
            max_candidates = 1
        settings["max_candidates"] = max(1, max_candidates)
        return settings

    def _resolve_maximum_prepass_policy(
        self,
        *,
        prompt_tokens: int,
        content_profile: ContentProfile,
        query_hint: Optional[str],
        redundancy_estimate: float,
        constraint_density: float = 0.0,
    ) -> Dict[str, Any]:
        min_tokens_default = max(1, self.maximum_prepass_min_tokens)
        budget_ratio_default = min(0.95, max(0.2, self.maximum_prepass_budget_ratio))
        max_sentences_default = max(8, self.maximum_prepass_max_sentences)

        policy = (self.maximum_prepass_policy or "auto").strip().lower()
        if policy not in {"off", "auto", "conservative", "aggressive"}:
            policy = "auto"

        profile_name = content_profile.name
        query_present = bool(query_hint)
        risk_aware_profile = profile_name in {
            "general_prose",
            "heavy_document",
            "technical_doc",
            "markdown",
            "dialogue",
        }

        if policy == "off":
            resolved: Dict[str, Any] = {
                "enabled": False,
                "minimum_tokens": min_tokens_default,
                "budget_ratio": budget_ratio_default,
                "max_sentences": max_sentences_default,
                "policy_source": "off",
                "policy_mode": "off",
                "enabled_override": False,
            }
        elif policy == "conservative":
            resolved = {
                "enabled": True,
                "minimum_tokens": int(max(64, round(min_tokens_default * 1.25))),
                "budget_ratio": min(0.95, max(0.2, budget_ratio_default * 1.1)),
                "max_sentences": max_sentences_default,
                "policy_source": "conservative",
                "policy_mode": "conservative",
                "enabled_override": False,
            }
        elif policy == "aggressive":
            resolved = {
                "enabled": True,
                "minimum_tokens": int(max(64, round(min_tokens_default * 0.7))),
                "budget_ratio": min(0.95, max(0.2, budget_ratio_default * 0.82)),
                "max_sentences": max_sentences_default,
                "policy_source": "aggressive",
                "policy_mode": "aggressive",
                "enabled_override": False,
            }
        else:
            large_prompt_threshold = int(max(512, min_tokens_default * 0.85))
            auto_enabled = (
                prompt_tokens >= large_prompt_threshold
                and risk_aware_profile
                and profile_name not in {"json", "code"}
            )
            if redundancy_estimate >= 0.28 and risk_aware_profile:
                auto_enabled = auto_enabled or prompt_tokens >= int(
                    max(512, min_tokens_default * 0.7)
                )
            if query_present and risk_aware_profile:
                auto_enabled = auto_enabled and prompt_tokens >= int(
                    max(512, min_tokens_default * 0.75)
                )

            resolved = {
                "enabled": auto_enabled,
                "minimum_tokens": min_tokens_default,
                "budget_ratio": budget_ratio_default,
                "budget_floor_ratio": 0.2,
                "budget_cap_ratio": 0.95,
                "max_sentences": max_sentences_default,
                "policy_source": "auto",
                "policy_mode": "auto",
                "enabled_override": False,
            }

            adaptive_ratio = budget_ratio_default
            if constraint_density >= 0.2:
                adaptive_ratio += 0.1
            elif constraint_density >= 0.1:
                adaptive_ratio += 0.05

            if redundancy_estimate >= 0.35 and constraint_density <= 0.18:
                adaptive_ratio -= 0.12
            elif redundancy_estimate >= 0.24 and constraint_density <= 0.12:
                adaptive_ratio -= 0.06

            adaptive_ratio = min(0.95, max(0.2, adaptive_ratio))
            resolved["budget_ratio"] = adaptive_ratio
            resolved["adaptive_budget_ratio"] = adaptive_ratio
            resolved["adaptive_redundancy_ratio"] = redundancy_estimate
            resolved["adaptive_constraint_density"] = constraint_density
            resolved["adaptive_applied"] = True

        enabled_env = os.environ.get("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED")
        if enabled_env is not None:
            resolved["enabled"] = self._get_env_bool(
                "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED",
                bool(resolved["enabled"]),
            )
            resolved["policy_source"] = "forced"
            resolved["enabled_override"] = True

        min_tokens_env = os.environ.get("PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS")
        if min_tokens_env not in (None, ""):
            resolved["minimum_tokens"] = max(
                1,
                self._get_env_int(
                    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS",
                    int(resolved["minimum_tokens"]),
                ),
            )

        budget_ratio_env = os.environ.get(
            "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO"
        )
        if budget_ratio_env not in (None, ""):
            resolved["budget_ratio"] = min(
                0.95,
                max(
                    0.2,
                    self._get_env_float(
                        "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO",
                        float(resolved["budget_ratio"]),
                    ),
                ),
            )
            resolved["budget_cap_ratio"] = min(
                float(resolved.get("budget_cap_ratio", 0.95)),
                float(resolved["budget_ratio"]),
            )

        max_sentences_env = os.environ.get(
            "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES"
        )
        if max_sentences_env not in (None, ""):
            resolved["max_sentences"] = max(
                8,
                self._get_env_int(
                    "PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES",
                    int(resolved["max_sentences"]),
                ),
            )

        fastpath_floor = max(self.fastpath_token_threshold or 0, 0)
        if bool(resolved.get("enabled", False)) and prompt_tokens < fastpath_floor:
            resolved["enabled"] = False
            resolved["policy_source"] = (
                "forced_small_prompt"
                if bool(resolved.get("enabled_override", False))
                else "auto_small_prompt"
            )

        return resolved

    def _select_semantic_candidate(
        self,
        original: str,
        candidates: Sequence[Tuple[str, str, Dict[str, Any]]],
        *,
        pass_name: str,
        guard_threshold: float,
        telemetry_collector: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not candidates:
            return original, {}

        original_tokens = self.count_tokens(original)
        best_label = "original"
        best_text = original
        best_meta: Dict[str, Any] = {}
        best_savings = -1
        scored_candidates: List[Dict[str, Any]] = []
        verification_failures: List[Dict[str, Any]] = []
        enforce_constraints = pass_name in _CONSTRAINT_HEAVY_PASS_NAMES
        source_fingerprint = self._get_state().constraint_fingerprint or {}

        for label, text, meta in candidates:
            candidate_tokens = self.count_tokens(text)
            savings = max(original_tokens - candidate_tokens, 0)
            if text == original:
                similarity = 1.0
            else:
                similarity = self._require_semantic_guard_similarity(
                    original,
                    text,
                    embedding_cache=self._get_state().embedding_cache,
                )

            scored_candidates.append(
                {
                    "label": label,
                    "similarity": similarity,
                    "tokens_saved": savings,
                }
            )

            if enforce_constraints and label != "original":
                fingerprint_ok, failures = self._verify_constraint_fingerprint(
                    source_fingerprint,
                    text,
                )
                if failures:
                    verification_failures.append(
                        {
                            "label": label,
                            "failures": failures,
                        }
                    )
                    scored_candidates[-1]["constraint_failures"] = failures
                if not fingerprint_ok:
                    scored_candidates[-1]["rejected_by_constraints"] = True
                    continue

            if similarity >= guard_threshold and savings > best_savings:
                best_label = label
                best_text = text
                best_meta = meta
                best_savings = savings

        _record_telemetry_flag(
            telemetry_collector,
            f"{pass_name}_candidate_count",
            len(candidates),
        )
        _record_telemetry_flag(
            telemetry_collector,
            f"{pass_name}_guard_threshold",
            guard_threshold,
        )
        _record_telemetry_flag(
            telemetry_collector,
            f"{pass_name}_candidate_scores",
            scored_candidates,
        )
        _record_telemetry_flag(
            telemetry_collector,
            f"{pass_name}_selected_candidate",
            best_label,
        )
        if enforce_constraints and verification_failures:
            _record_telemetry_flag(
                telemetry_collector,
                f"{pass_name}_constraint_verification_failures",
                verification_failures,
            )

        return best_text, best_meta

    @staticmethod
    def _normalize_constraint_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _action_tokens(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        filtered = [
            token
            for token in tokens
            if token not in _CONSTRAINT_STOPWORDS
            and token not in _CONSTRAINT_NEGATION_TERMS
            and len(token) > 1
        ]
        return filtered[:8]

    def _extract_constraint_fingerprint(self, text: str) -> Dict[str, Any]:
        normalized_text = self._normalize_constraint_text(text)
        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
            if segment and segment.strip()
        ]
        negations: List[Dict[str, Any]] = []
        numeric_constraints: List[Dict[str, Any]] = []
        must_directives: List[Dict[str, Any]] = []
        should_directives: List[Dict[str, Any]] = []
        do_not_rules: List[Dict[str, Any]] = []
        quoted_literals: List[str] = []

        for raw_segment in segments:
            segment = self._normalize_constraint_text(raw_segment)
            if not segment:
                continue

            has_negation = any(
                re.search(rf"\b{re.escape(term)}\b", segment)
                for term in _CONSTRAINT_NEGATION_TERMS
            )
            anchors = self._action_tokens(segment)
            if has_negation:
                negations.append({"text": segment, "anchors": anchors})

            if re.search(r"\d", segment):
                has_operator = bool(
                    re.search(
                        r"(?:<=|>=|<|>|=|at least|at most|more than|less than|no more than|no less than|exactly)",
                        segment,
                    )
                )
                numeric_constraints.append(
                    {
                        "text": segment,
                        "numbers": re.findall(r"\d+(?:\.\d+)?", segment),
                        "anchors": anchors,
                        "strict": has_operator,
                    }
                )

            if re.search(r"\b(?:must|required to|need to|has to|shall)\b", segment):
                must_directives.append({"text": segment, "anchors": anchors})
            elif re.search(r"\b(?:should|recommended to|ought to)\b", segment):
                should_directives.append({"text": segment, "anchors": anchors})

            if re.search(
                r"\b(?:do not|don't|must not|should not|never|cannot|can't)\b",
                segment,
            ):
                do_not_rules.append({"text": segment, "anchors": anchors})

        for match in re.finditer(
            r"(?:\"([^\"]{2,160})\"|'([^']{2,160})'|`([^`]{2,160})`)",
            text,
        ):
            literal = next((group for group in match.groups() if group), "").strip()
            if literal:
                quoted_literals.append(self._normalize_constraint_text(literal))

        return {
            "raw": normalized_text,
            "negations": negations,
            "numeric_constraints": numeric_constraints,
            "must_directives": must_directives,
            "should_directives": should_directives,
            "do_not_rules": do_not_rules,
            "quoted_literals": sorted(set(quoted_literals)),
        }

    def _verify_constraint_fingerprint(
        self,
        fingerprint: Dict[str, Any],
        candidate_text: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        if not fingerprint:
            return True, []

        candidate_norm = self._normalize_constraint_text(candidate_text)
        failures: List[Dict[str, Any]] = []

        def _anchors_present(anchors: Sequence[str], *, min_matches: int = 1) -> bool:
            if not anchors:
                return True
            window = anchors[:4]
            matches = sum(1 for anchor in window if anchor in candidate_norm)
            return matches >= min(min_matches, len(window))

        def _has_negated_action(anchors: Sequence[str]) -> bool:
            if not anchors:
                return any(
                    re.search(rf"\b{re.escape(term)}\b", candidate_norm)
                    for term in _CONSTRAINT_NEGATION_TERMS
                )
            for anchor in anchors[:4]:
                idx = candidate_norm.find(anchor)
                if idx < 0:
                    continue
                window = candidate_norm[max(0, idx - 80) : idx + 80]
                if any(
                    re.search(rf"\b{re.escape(term)}\b", window)
                    for term in _CONSTRAINT_NEGATION_TERMS
                ):
                    return True
            return False

        for literal in fingerprint.get("quoted_literals", []):
            if literal and literal not in candidate_norm:
                failures.append(
                    {
                        "category": "quoted_literal",
                        "severity": "mandatory",
                        "constraint": literal,
                        "reason": "missing_literal",
                    }
                )

        for numeric in fingerprint.get("numeric_constraints", []):
            numbers = numeric.get("numbers", [])
            anchors = numeric.get("anchors", [])
            numbers_present = all(number in candidate_norm for number in numbers)
            if not numbers_present or not _anchors_present(anchors, min_matches=1):
                failures.append(
                    {
                        "category": "numeric_constraint",
                        "severity": "mandatory",
                        "constraint": numeric.get("text", ""),
                        "reason": "missing_or_changed_numeric_constraint",
                    }
                )

        for directive in fingerprint.get("must_directives", []):
            anchors = directive.get("anchors", [])
            if not _anchors_present(anchors, min_matches=1):
                failures.append(
                    {
                        "category": "must_directive",
                        "severity": "mandatory",
                        "constraint": directive.get("text", ""),
                        "reason": "missing_must_directive",
                    }
                )

        for directive in fingerprint.get("should_directives", []):
            anchors = directive.get("anchors", [])
            if not _anchors_present(anchors, min_matches=1):
                failures.append(
                    {
                        "category": "should_directive",
                        "severity": "advisory",
                        "constraint": directive.get("text", ""),
                        "reason": "missing_should_directive",
                    }
                )

        for negation in fingerprint.get("negations", []):
            anchors = negation.get("anchors", [])
            if not _has_negated_action(anchors):
                failures.append(
                    {
                        "category": "negation",
                        "severity": "mandatory",
                        "constraint": negation.get("text", ""),
                        "reason": "polarity_violation",
                    }
                )

        for rule in fingerprint.get("do_not_rules", []):
            anchors = rule.get("anchors", [])
            if not _has_negated_action(anchors):
                failures.append(
                    {
                        "category": "do_not_rule",
                        "severity": "mandatory",
                        "constraint": rule.get("text", ""),
                        "reason": "polarity_violation",
                    }
                )

        mandatory_failed = any(
            failure.get("severity") == "mandatory" for failure in failures
        )
        return not mandatory_failed, failures

    def _normalize_minhash_paraphrase_threshold_override(
        self, threshold: Optional[float]
    ) -> Optional[float]:
        if threshold is None:
            return None
        try:
            resolved = float(threshold)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(resolved, 1.0))

    def _resolve_minhash_paraphrase_threshold(self) -> float:
        override = self._get_state().minhash_paraphrase_threshold_override
        if override is not None:
            return max(0.0, min(float(override), 1.0))

        mode = self._get_state().optimization_mode
        default_value = _MINHASH_PARAPHRASE_DEFAULTS.get(mode, 0.85)
        if mode == "maximum" and not self.semantic_guard_enabled:
            # Safer default when semantic guard is unavailable.
            default_value = max(default_value, _MINHASH_PARAPHRASE_DEFAULTS["balanced"])
        return default_value

    def _track_dedup_counts(self, kind: str, count: int) -> None:
        if count <= 0:
            return
        counts = self._get_state().dedup_counts
        counts[kind] = counts.get(kind, 0) + int(count)

    @staticmethod
    def _compute_idf_weights(token_sets: Sequence[Set[str]]) -> Dict[str, float]:
        if not token_sets:
            return {}
        df: Dict[str, int] = defaultdict(int)
        for tokens in token_sets:
            for token in tokens:
                df[token] += 1
        total = len(token_sets)
        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log((1 + total) / (1 + freq)) + 1.0
        return idf

    @staticmethod
    def _weighted_jaccard(
        tokens_a: Set[str], tokens_b: Set[str], weights: Dict[str, float]
    ) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        if not union:
            return 0.0
        numerator = sum(weights.get(token, 1.0) for token in intersection)
        denominator = sum(weights.get(token, 1.0) for token in union)
        return numerator / denominator if denominator else 0.0

    def _normalize_semantic_threshold_override(
        self, semantic_threshold: Optional[float]
    ) -> Optional[float]:
        if semantic_threshold is None:
            return None
        resolved = max(0.0, min(float(semantic_threshold), 1.0))
        min_threshold = max(
            self.semantic_similarity_threshold, self.semantic_guard_threshold
        )
        if resolved < min_threshold:
            raise ValueError(
                f"semantic_threshold must be >= {min_threshold:.2f} to preserve meaning"
            )
        return resolved

    def _get_chunk_executor(
        self, max_workers: Optional[int] = None
    ) -> ThreadPoolExecutor:
        cpu_limit = os.cpu_count() or 4
        configured_limit = min(self.max_chunk_workers, cpu_limit)
        desired_workers = max(1, min(max_workers or configured_limit, configured_limit))
        executor = getattr(self, "_chunk_executor", None)
        if executor is None or getattr(executor, "_shutdown", False):
            self._chunk_executor = ThreadPoolExecutor(
                max_workers=desired_workers, thread_name_prefix="optimizer-chunk"
            )
        return self._chunk_executor

    def _get_token_cache(self) -> "OrderedDict[str, List[int]]":
        return self._get_state().token_cache

    def _init_token_cache(
        self,
        seed: Optional[Dict[str, List[int]]] = None,
    ) -> "OrderedDict[str, List[int]]":
        if isinstance(seed, OrderedDict):
            return seed
        cache: "OrderedDict[str, List[int]]" = OrderedDict()
        if seed:
            for key, value in seed.items():
                cache[key] = value
        return cache

    def _encode_cached(self, text: str) -> Optional[List[int]]:
        if not text or self.tokenizer is None:
            return None

        cache = self._get_token_cache()
        cached = cache.get(text)
        if cached is not None:
            cache.move_to_end(text)
            return cached

        shared_cached = _get_shared_tokens(text)
        if shared_cached is not None:
            if self.token_cache_max_size > 0:
                cache[text] = shared_cached
                cache.move_to_end(text)
                if len(cache) > self.token_cache_max_size:
                    cache.popitem(last=False)
            return shared_cached

        try:
            tokens = self.tokenizer.encode(text)
            if self.token_cache_max_size <= 0:
                _set_shared_tokens(text, tokens)
                return tokens
            cache[text] = tokens
            cache.move_to_end(text)
            if len(cache) > self.token_cache_max_size:
                cache.popitem(last=False)
            _set_shared_tokens(text, tokens)
            return tokens
        except Exception as exc:
            logger.debug("Tokenization failed for caching: %s", exc)
            return None

    def tokenize(self, text: str) -> Optional[List[int]]:
        """Return cached tokens for *text*, computing them if needed."""
        return self._encode_cached(text)

    def _run_pipeline_threadsafe(
        self,
        prompt: str,
        mode: str,
        optimization_mode: str,
        enable_frequency_learning: bool,
        force_preserve_digits: Optional[bool],
        *,
        profiler: Optional[PipelineProfiler],
        telemetry_collector: Optional[Any],
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
        use_discourse_weighting: bool = True,
        json_policy: Optional[Dict[str, Any]] = None,
        token_cache: Optional[Dict[str, List[int]]] = None,
        embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        customer_id: Optional[str] = None,
        custom_canonicals: Optional[Dict[str, str]] = None,
        force_disabled_passes: Optional[Collection[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        previous_state = getattr(self._state_local, "value", None)
        current_state = _OptimizerThreadState(
            token_cache=self._init_token_cache(token_cache),
            embedding_cache=embedding_cache or {},
        )
        self._state_local.value = current_state

        try:
            optimized = self._optimize_pipeline(
                prompt,
                mode,
                optimization_mode,
                enable_frequency_learning,
                force_preserve_digits,
                profiler=profiler,
                telemetry_collector=telemetry_collector,
                segment_spans=segment_spans,
                use_discourse_weighting=use_discourse_weighting,
                json_policy=json_policy,
                force_disabled_passes=force_disabled_passes,
                content_type=content_type,
                content_profile=content_profile,
                customer_id=customer_id,
                custom_canonicals=custom_canonicals,
            )
        finally:
            snapshot = {
                "techniques_applied": list(current_state.techniques_applied),
                "prefer_background_summary": current_state.prefer_background_summary,
                "skip_sentence_deduplication": current_state.skip_sentence_deduplication,
                "toon_stats": dict(current_state.toon_stats),
            }
            if previous_state is None:
                if hasattr(self._state_local, "value"):
                    delattr(self._state_local, "value")
            else:
                self._state_local.value = previous_state

        return optimized, snapshot

    def _merge_pipeline_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return

        state = self._get_state()
        for technique in snapshot.get("techniques_applied", []):
            self._record_technique(technique)

        if "prefer_background_summary" in snapshot:
            state.prefer_background_summary = snapshot["prefer_background_summary"]

        if "skip_sentence_deduplication" in snapshot:
            state.skip_sentence_deduplication = snapshot["skip_sentence_deduplication"]

        toon_stats = snapshot.get("toon_stats")
        if isinstance(toon_stats, dict):
            state.toon_stats = {
                "conversions": int(state.toon_stats.get("conversions", 0))
                + int(toon_stats.get("conversions", 0)),
                "bytes_saved": int(state.toon_stats.get("bytes_saved", 0))
                + int(toon_stats.get("bytes_saved", 0)),
            }

    @property
    def techniques_applied(self) -> List[str]:
        return self._get_state().techniques_applied

    @techniques_applied.setter
    def techniques_applied(self, value: List[str]) -> None:
        self._get_state().techniques_applied = value

    def _record_technique(self, technique: str) -> None:
        """Append a technique only once to avoid duplicates."""
        if technique not in self.techniques_applied:
            self.techniques_applied.append(technique)

    @property
    def _prefer_background_summary(self) -> bool:
        return self._get_state().prefer_background_summary

    @_prefer_background_summary.setter
    def _prefer_background_summary(self, value: bool) -> None:
        self._get_state().prefer_background_summary = value

    @property
    def _skip_sentence_deduplication(self) -> bool:
        return self._get_state().skip_sentence_deduplication

    @_skip_sentence_deduplication.setter
    def _skip_sentence_deduplication(self, value: bool) -> None:
        self._get_state().skip_sentence_deduplication = value

    def _build_token_budget_tracker(self, text: str) -> TokenBudgetTracker:
        return TokenBudgetTracker(
            estimated_tokens=self.count_tokens(text),
            last_text_length=len(text),
            sample_rate=self.token_budget_telemetry_sample_rate,
            uncertainty_threshold=self.token_budget_uncertainty_threshold,
        )

    def _tokens_for_stage_decision(self, text: str) -> int:
        """Exact counting for stage boundaries (chunking/guards/final stats)."""
        return self.count_tokens(text)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using strict tokenizer-backed encoding."""
        if not text:
            return 0

        tokens = self._encode_cached(text)
        if tokens is None:
            raise RuntimeError(
                "Exact token counting is unavailable; strict mode requires tokenizer support."
            )
        return len(tokens)

    def _estimate_costs(
        self,
        original_tokens: int,
        optimized_tokens: int,
    ) -> Tuple[float, float, float]:
        prompt_rate = max(self.prompt_cost_per_1k, 0.0)

        cost_before = (original_tokens / 1000.0) * prompt_rate
        cost_after = (optimized_tokens / 1000.0) * prompt_rate
        cost_saved = max(cost_before - cost_after, 0.0)
        return cost_before, cost_after, cost_saved

    def _record_history(
        self,
        *,
        mode: str,
        prompt: str,
        optimized: str,
        stats: Dict[str, Any],
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        """Record optimization history using async batch writer."""
        try:
            from database import record_optimization_history

            original_tokens = stats.get("original_tokens", 0)
            optimized_tokens = stats.get("optimized_tokens", 0)
            cost_before, cost_after, cost_saved = self._estimate_costs(
                original_tokens,
                optimized_tokens,
            )

            compression_percentage = stats.get("compression_percentage")
            semantic_similarity = stats.get("semantic_similarity")
            if semantic_similarity is None:
                semantic_similarity = self._compute_semantic_similarity(
                    prompt, optimized
                )

            techniques = list(dict.fromkeys(self.techniques_applied))

            record_id = record_optimization_history(
                mode=mode,
                raw_prompt=prompt,
                optimized_prompt=optimized,
                raw_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                processing_time_ms=stats.get("processing_time_ms") or 0.0,
                estimated_cost_before=cost_before,
                estimated_cost_after=cost_after,
                estimated_cost_saved=cost_saved,
                compression_percentage=compression_percentage,
                semantic_similarity=semantic_similarity,
                techniques_applied=techniques,
            )

            # Submit telemetry asynchronously if available
            if telemetry_collector:
                # Set the optimization_id to match the history record
                if record_id:
                    telemetry_collector._optimization_id = record_id
                _telemetry.submit_optimization_telemetry(telemetry_collector)

        except Exception as exc:
            logger.debug("Failed to record optimization history: %s", exc)

    # ------------------------------------------------------------------
    # Segment weighting helpers
    # ------------------------------------------------------------------
    def _analyze_segment_spans(
        self,
        text: str,
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        return analyze_segment_spans(text, self._discourse_analyzer, segment_spans)

    def _resolve_json_compression_config(self) -> Dict[str, Any]:
        base_default = self.json_compression_config.get("default", False)
        base_overrides = dict(self.json_compression_config.get("overrides", {}))
        base_minify = self.json_compression_config.get("minify", False)

        return {
            "default": base_default,
            "overrides": base_overrides,
            "minify": base_minify,
        }

    def _get_env_float(self, variable: str, default: float) -> float:
        """Read a float configuration value from the environment safely."""
        raw_value = os.environ.get(variable)
        if raw_value in (None, ""):
            return default

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid %s value '%s'; using default %.3f",
                variable,
                raw_value,
                default,
            )
            return default

    def _get_env_int(self, variable: str, default: int) -> int:
        """Read an integer configuration value from the environment safely."""
        raw_value = os.environ.get(variable)
        if raw_value in (None, ""):
            return default

        try:
            return int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid %s value '%s'; using default %d", variable, raw_value, default
            )
            return default

    def _get_env_bool(self, variable: str, default: bool) -> bool:
        """Read a boolean configuration value from the environment safely."""
        raw_value = os.environ.get(variable)
        if raw_value in (None, ""):
            return default

        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

        logger.warning(
            "Invalid %s value '%s'; using default %s", variable, raw_value, default
        )
        return default

    def _optimize_with_chunking(
        self,
        prompt: str,
        mode: str,
        optimization_mode: str,
        *,
        chunk_size: Optional[int] = None,
        strategy: Optional[str] = None,
        overlap_ratio: Optional[float] = None,
        enable_frequency_learning: bool = False,
        use_discourse_weighting: bool = True,
        force_preserve_digits: Optional[bool] = None,
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
        chat_metadata: Optional[Dict[str, Any]] = None,
        profiler: Optional[PipelineProfiler] = None,
        telemetry_collector: Optional[Any] = None,
        json_policy: Optional[Dict[str, Any]] = None,
        token_cache: Optional[Dict[str, List[int]]] = None,
        enable_toon_conversion: bool = False,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
        semantic_plan: Optional[Dict[str, Any]] = None,
        custom_canonicals: Optional[Dict[str, str]] = None,
        force_disabled_passes: Optional[Collection[str]] = None,
        customer_id: Optional[str] = None,
    ) -> Tuple[str, List[_chunking.ChunkSpec]]:
        resolved_size = chunk_size or self.chunk_size or 1
        resolved_strategy = strategy or self.default_chunking_mode
        return _chunking.optimize_with_chunking(
            self,
            prompt,
            mode,
            optimization_mode,
            resolved_size,
            strategy=resolved_strategy,
            overlap_ratio=overlap_ratio,
            enable_frequency_learning=enable_frequency_learning,
            use_discourse_weighting=use_discourse_weighting,
            force_preserve_digits=force_preserve_digits,
            segment_spans=segment_spans,
            chat_metadata=chat_metadata,
            semantic_model=self.semantic_chunk_model,
            similarity_threshold=self.semantic_chunk_similarity,
            profiler=profiler,
            telemetry_collector=telemetry_collector,
            json_policy=json_policy,
            token_cache=token_cache,
            enable_toon_conversion=enable_toon_conversion,
            content_type=content_type,
            content_profile=content_profile,
            embedding_cache=embedding_cache,
            semantic_plan=semantic_plan,
            custom_canonicals=custom_canonicals,
            force_disabled_passes=force_disabled_passes,
            customer_id=customer_id,
        )

    def _optimize_with_token_classifier(
        self,
        prompt: str,
        *,
        force_preserve_digits: bool,
        json_policy: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        min_confidence_override: Optional[float] = None,
        min_keep_ratio_override: Optional[float] = None,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        if not prompt.strip():
            return prompt, False, {"keep_ratio": 1.0, "decisions": 0, "removals": 0}

        result, preserved = _preservation.extract_and_preserve(
            self,
            prompt,
            force_digits=force_preserve_digits,
            json_policy=json_policy,
            enable_toon_conversion=False,
            enable_alias_json_keys=False,
        )
        protected_ranges = _entropy._placeholder_ranges(result)
        defaults = content_profile.smart_defaults if content_profile else {}
        min_confidence = defaults.get("classifier_min_confidence", 0.45)
        min_keep_ratio = defaults.get("classifier_min_keep_ratio", 0.6)
        if min_confidence_override is not None:
            min_confidence = min_confidence_override
        if min_keep_ratio_override is not None:
            min_keep_ratio = min_keep_ratio_override
        compressed, applied, metadata = (
            _token_classifier.compress_with_token_classifier(
                result,
                protected_ranges=protected_ranges,
                model_name=self.token_classifier_model,
                min_confidence=min_confidence,
                min_keep_ratio=min_keep_ratio,
                content_profile_name=(
                    content_profile.name if content_profile else None
                ),
            )
        )
        shadow_metadata = _token_classifier.evaluate_shadow_classifier(
            result,
            protected_ranges=protected_ranges,
            min_confidence=min_confidence,
            min_keep_ratio=min_keep_ratio,
        )
        if shadow_metadata:
            metadata["shadow"] = shadow_metadata
        if not applied or compressed == result:
            return prompt, False, metadata

        restored = _preservation.restore(self, compressed, preserved)
        restored, _ = self._normalize_text(
            restored,
            normalize_whitespace=True,
            compress_punctuation=False,
        )
        metadata.update(
            {
                "content_type": content_type or "",
                "min_confidence": min_confidence,
                "min_keep_ratio": min_keep_ratio,
            }
        )
        return restored, True, metadata

    def _apply_pre_chunk_normalized_sentence_dedup(
        self,
        text: str,
        *,
        force_preserve_digits: Optional[bool],
        json_policy: Optional[Dict[str, Any]],
        enable_toon_conversion: bool,
    ) -> Tuple[str, bool]:
        if not text.strip():
            return text, False

        preserved_text, preserved = _preservation.extract_and_preserve(
            self,
            text,
            force_digits=force_preserve_digits,
            json_policy=json_policy,
            enable_toon_conversion=enable_toon_conversion,
            enable_alias_json_keys=False,
        )

        if self._needs_whitespace_normalization(preserved_text):
            normalized, _ = self._normalize_text(
                preserved_text,
                normalize_whitespace=True,
                compress_punctuation=False,
            )
        else:
            normalized = preserved_text

        deduped, dedup_applied = self._deduplicate_normalized_sentences(normalized)
        restored = _preservation.restore(self, deduped, preserved)
        return restored, dedup_applied

    @staticmethod
    def _split_text_with_separators(
        text: str, pattern: re.Pattern
    ) -> List[Tuple[str, str, int, int]]:
        parts: List[Tuple[str, str, int, int]] = []
        cursor = 0
        for match in pattern.finditer(text):
            segment = text[cursor : match.start()]
            separator = text[match.start() : match.end()]
            parts.append((segment, separator, cursor, match.start()))
            cursor = match.end()
        parts.append((text[cursor:], "", cursor, len(text)))
        return parts

    def _split_post_chunk_units(self, text: str) -> List[Tuple[str, str, int, int]]:
        paragraph_pattern = re.compile(r"\n\s*\n+")
        if paragraph_pattern.search(text):
            return self._split_text_with_separators(text, paragraph_pattern)
        return self._split_text_with_separators(text, _SENTENCE_SPLIT_PATTERN)

    def _apply_post_chunk_dedup(
        self,
        text: str,
        *,
        content_profile: Optional[ContentProfile],
        preserved: Optional[Dict[str, Any]],
        telemetry_collector: Optional[Any],
    ) -> str:
        if not text.strip():
            return text
        if content_profile and content_profile.name in {"code", "json"}:
            logger.debug("Skipping post-chunk dedup (code/json profile)")
            return text

        tokens_before = self.count_tokens(text)
        if tokens_before < _POST_CHUNK_DEDUP_MIN_TOKENS:
            return text

        placeholder_ranges = _preservation.get_placeholder_ranges(self, text, preserved)
        parts = self._split_post_chunk_units(text)
        if len(parts) <= 1:
            return text

        pass_start = time.perf_counter() if telemetry_collector else None
        seen_signatures: Dict[str, int] = {}
        kept_parts: List[Tuple[str, str]] = []
        kept_minhashes: List[Optional[MinHash]] = []
        kept_token_sets: List[Optional[Set[str]]] = []
        duplicate_count = 0
        threshold = max(
            self._resolve_minhash_paraphrase_threshold(), _MINHASH_THRESHOLD
        )
        lsh_index = SentenceLSHIndex(
            threshold=threshold, num_perm=self._minhash_num_perm
        )

        def _overlaps_placeholder(start: int, end: int) -> bool:
            for p_start, p_end in placeholder_ranges:
                if start < p_end and end > p_start:
                    return True
            return False

        for segment, separator, start, end in parts:
            stripped = segment.strip()
            if not stripped:
                kept_parts.append((segment, separator))
                kept_minhashes.append(None)
                kept_token_sets.append(None)
                continue

            if _overlaps_placeholder(start, end):
                kept_parts.append((segment, separator))
                kept_minhashes.append(None)
                kept_token_sets.append(None)
                continue

            signature = self._normalized_sentence_signature(stripped)
            if signature in seen_signatures:
                duplicate_count += 1
                continue

            tokens = re.findall(r"\w+", stripped.lower())
            token_set = set(tokens) if tokens else None
            minhash_signature = None
            is_duplicate = False
            if len(tokens) >= _MINHASH_PARAPHRASE_MIN_WORDS and MinHash is not None:
                minhash_signature = lsh_index.create_signature(stripped, shingle_size=2)
                if minhash_signature is not None:
                    matches = lsh_index.query_similar(minhash_signature)
                    for match_id in matches:
                        match_idx = int(match_id)
                        if match_idx >= len(kept_minhashes):
                            continue
                        previous = kept_minhashes[match_idx]
                        previous_tokens = (
                            kept_token_sets[match_idx]
                            if match_idx < len(kept_token_sets)
                            else None
                        )
                        if (
                            previous
                            and minhash_signature.jaccard(previous) >= threshold
                        ):
                            if token_set and previous_tokens:
                                overlap = len(token_set & previous_tokens)
                                union = len(token_set | previous_tokens)
                                if union:
                                    token_overlap = overlap / union
                                    if token_overlap < _MINHASH_TOKEN_OVERLAP_THRESHOLD:
                                        continue
                            is_duplicate = True
                            break

            if is_duplicate:
                duplicate_count += 1
                continue

            seen_signatures[signature] = len(kept_parts)
            kept_parts.append((segment, separator))
            kept_minhashes.append(minhash_signature)
            kept_token_sets.append(token_set)
            if minhash_signature is not None:
                lsh_index.add_sentence(str(len(kept_minhashes) - 1), minhash_signature)

        if duplicate_count < _POST_CHUNK_DEDUP_MIN_DUPLICATES:
            return text

        deduped = "".join(segment + separator for segment, separator in kept_parts)
        if deduped == text:
            return text

        tokens_after = self.count_tokens(deduped)
        if tokens_after >= tokens_before:
            return text

        if telemetry_collector and pass_start is not None:
            pass_duration = (time.perf_counter() - pass_start) * 1000.0
            try:
                telemetry_collector.record_pass(
                    "post_chunk_dedup",
                    pass_duration,
                    tokens_before,
                    tokens_after,
                )
            except Exception:
                pass

        self._record_technique("Post-Chunk Deduplication")

        return deduped

    def _apply_paragraph_semantic_dedup(
        self,
        text: str,
        *,
        query_hint: Optional[str],
        preserved: Optional[Dict[str, Any]],
    ) -> Tuple[str, bool, int]:
        if not text.strip():
            return text, False, 0

        paragraphs = _split_semantic_paragraphs(text)
        if len(paragraphs) < 3:
            return text, False, 0

        placeholder_ranges = _preservation.get_placeholder_ranges(self, text, preserved)
        if placeholder_ranges:
            return text, False, 0

        threshold = max(self._resolve_minhash_paraphrase_threshold(), 0.84)
        lsh_index = SentenceLSHIndex(
            threshold=threshold, num_perm=self._minhash_num_perm
        )
        query_tokens = set(re.findall(r"\w+", (query_hint or "").lower()))

        selected_indices: List[int] = []
        selected_sigs: List[Optional[MinHash]] = []
        selected_token_sets: List[Set[str]] = []
        removed = 0

        for index, paragraph in enumerate(paragraphs):
            para = paragraph.strip()
            if not para:
                continue
            tokens = re.findall(r"\w+", para.lower())
            token_set = set(tokens)
            signature: Optional[MinHash] = None
            if len(tokens) >= _MINHASH_PARAPHRASE_MIN_WORDS and MinHash is not None:
                signature = lsh_index.create_signature(para, shingle_size=3)

            def _score(i: int) -> Tuple[float, int]:
                candidate_tokens = selected_token_sets[i]
                overlap = len(candidate_tokens & query_tokens) if query_tokens else 0
                length = len(paragraphs[selected_indices[i]])
                return float(overlap), -length

            duplicate_of: Optional[int] = None
            if signature is not None:
                for match_id in lsh_index.query_similar(signature):
                    match_idx = int(match_id)
                    if match_idx >= len(selected_indices):
                        continue
                    prev_sig = selected_sigs[match_idx]
                    prev_tokens = selected_token_sets[match_idx]
                    if prev_sig and signature.jaccard(prev_sig) >= threshold:
                        duplicate_of = match_idx
                        break
                    if token_set and prev_tokens:
                        overlap = len(token_set & prev_tokens)
                        union = len(token_set | prev_tokens)
                        if union and (overlap / union) >= 0.8:
                            duplicate_of = match_idx
                            break

            if duplicate_of is None and token_set:
                for candidate_idx, prev_tokens in enumerate(selected_token_sets):
                    if not prev_tokens:
                        continue
                    overlap = len(token_set & prev_tokens)
                    union = len(token_set | prev_tokens)
                    if union and (overlap / union) >= 0.8:
                        duplicate_of = candidate_idx
                        break

            if duplicate_of is None:
                selected_indices.append(index)
                selected_sigs.append(signature)
                selected_token_sets.append(token_set)
                if signature is not None:
                    lsh_index.add_sentence(str(len(selected_indices) - 1), signature)
                continue

            existing_score = _score(duplicate_of)
            candidate_overlap = len(token_set & query_tokens) if query_tokens else 0
            candidate_score = (float(candidate_overlap), -len(para))
            if candidate_score > existing_score:
                selected_indices[duplicate_of] = index
                selected_sigs[duplicate_of] = signature
                selected_token_sets[duplicate_of] = token_set
            removed += 1

        if removed <= 0:
            return text, False, 0

        deduped = "\n\n".join(
            paragraphs[i] for i in sorted(set(selected_indices))
        ).strip()
        if not deduped or deduped == text.strip():
            return text, False, 0
        return deduped, True, removed

    def optimize(
        self,
        prompt: str,
        mode: str = "basic",
        optimization_mode: str = "balanced",
        *,
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        semantic_threshold: Optional[float] = None,
        minhash_paraphrase_threshold: Optional[float] = None,
        chat_metadata: Optional[Dict[str, Any]] = None,
        background_tasks: Optional[Any] = None,
        skip_db_write: bool = False,
        customer_id: Optional[str] = None,
        custom_canonicals: Optional[Dict[str, str]] = None,
        force_disabled_passes: Optional[Collection[str]] = None,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        smart_context: Optional[SmartContext] = None,
    ) -> Dict:
        """
        Main optimization entry point.

        Args:
            prompt: The input prompt to optimize
            mode: 'basic' for rule-based optimization, 'advanced' for future LLM-based optimization
            optimization_mode: Controls optimization intensity (conservative/balanced/maximum)
            background_tasks: Optional FastAPI BackgroundTasks for async DB writes
            skip_db_write: If True, skips database write (for cached results)
            segment_spans: Optional weighted spans to preserve or de-emphasize
            query: Optional query hint for RAG-aware compression
            custom_canonicals: Optional per-request canonicalization mappings
            force_disabled_passes: Optional list of pipeline passes to disable
            minhash_paraphrase_threshold: Optional MinHash LSH paraphrase threshold override

        Returns:
            Dictionary with optimized_output, stats, and                              metadata
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Required dependency 'tiktoken' is unavailable. "
                "Strict mode requires exact tokenizer support."
            )
        if optimization_mode in {"balanced", "maximum"} and np is None:
            raise RuntimeError(
                "Required dependency 'numpy' is unavailable. "
                "Strict mode requires numpy-backed semantic components."
            )

        start_time = time.time()
        debug_max_stage_timing = (
            optimization_mode == "maximum" and logger.isEnabledFor(logging.DEBUG)
        )

        def _log_max_stage(
            stage_name: str,
            stage_started: Optional[float],
            *,
            before_tokens: Optional[int] = None,
            after_tokens: Optional[int] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not debug_max_stage_timing or stage_started is None:
                return
            elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
            payload: Dict[str, Any] = {
                "stage": stage_name,
                "elapsed_ms": round(elapsed_ms, 2),
            }
            if before_tokens is not None:
                payload["before_tokens"] = before_tokens
            if after_tokens is not None:
                payload["after_tokens"] = after_tokens
                if before_tokens is not None:
                    payload["token_delta"] = before_tokens - after_tokens
            if extra:
                payload.update(extra)
            logger.debug("Maximum stage timing: %s", payload)

        semantic_threshold_override = self._normalize_semantic_threshold_override(
            semantic_threshold
        )
        minhash_paraphrase_override = (
            self._normalize_minhash_paraphrase_threshold_override(
                minhash_paraphrase_threshold
            )
        )
        self.techniques_applied = []
        self._skip_sentence_deduplication = False
        profiler = PipelineProfiler(self._profiling_enabled)
        resolved_preserve_digits = False

        # Initialize telemetry collector
        telemetry_collector = None
        if telemetry_is_enabled():
            try:
                telemetry_collector = _telemetry.OptimizationTelemetryCollector(
                    optimization_id=str(uuid.uuid4()), enabled=True
                )
            except Exception:
                telemetry_collector = None

        try:
            request_token_cache: "OrderedDict[str, List[int]]" = OrderedDict()
            previous_state = getattr(self._state_local, "value", None)
            if previous_state is not None:
                request_token_cache.update(previous_state.token_cache)
            request_embedding_cache: Dict[Tuple[str, str, str], Any] = {}
            state_metrics = {
                "embedding_reuse_count": 0.0,
                "embedding_calls_saved": 0.0,
                "embedding_wall_clock_savings_ms": 0.0,
            }
            self._state_local.value = _OptimizerThreadState(
                token_cache=request_token_cache,
                optimization_mode=optimization_mode,
                embedding_cache=request_embedding_cache,
                semantic_similarity_threshold_override=semantic_threshold_override,
                semantic_guard_threshold_override=semantic_threshold_override,
                minhash_paraphrase_threshold_override=minhash_paraphrase_override,
                dedup_counts={"exact": 0, "near": 0, "semantic": 0},
                semantic_plan_metrics=state_metrics,
            )
            source_prompt = (
                chat_metadata.get("original_text", prompt) if chat_metadata else prompt
            )
            state = self._get_state()
            state.semantic_plan = _build_semantic_plan(
                source_prompt,
                state.semantic_plan_metrics,
            )
            state.constraint_fingerprint = self._extract_constraint_fingerprint(
                source_prompt
            )
            original_length = len(source_prompt)
            original_tokens = self._tokens_for_stage_decision(source_prompt)
            state.original_tokens = original_tokens
            content_type = (
                content_type
                or (content_profile.name if content_profile is not None else None)
                or classify_content(source_prompt)
            )
            if content_profile is None:
                content_profile = get_profile(content_type)
            if smart_context is None:
                smart_context = resolve_smart_context(
                    source_prompt,
                    content_profile,
                    current_token_count=original_tokens,
                )
            resolved_frequency_learning = smart_context.enable_frequency_learning
            resolved_discourse_weighting = smart_context.use_discourse_weighting
            resolved_chunking_mode = smart_context.chunking_mode
            resolved_preserve_digits = smart_context.preserve_digits
            ranking_override: Optional[Dict[str, Any]] = None
            redundancy_ratio = 0.0
            redundancy_triggered = False
            if smart_context.section_ranking_enabled and self.section_ranking_mode in {
                "",
                "off",
            }:
                ranking_override = {
                    "mode": (
                        "gzip"
                        if content_profile.name in {"code", "markdown", "technical_doc"}
                        else "bm25"
                    )
                }
                if self.section_ranking_token_budget is not None:
                    ranking_override["token_budget"] = self.section_ranking_token_budget
            chunk_size = max(self.chunk_size or 50000, 1)
            chunk_threshold = self.chunk_threshold or chunk_size
            chunk_threshold = max(chunk_threshold, chunk_size)

            working_prompt = prompt
            query_hint = (
                query.strip() if isinstance(query, str) and query.strip() else None
            )
            stage_query_started = (
                time.perf_counter() if debug_max_stage_timing else None
            )
            query_before_tokens = (
                self._tokens_for_stage_decision(working_prompt)
                if debug_max_stage_timing
                else None
            )
            if query_hint and not segment_spans:
                if content_profile.name in {
                    "general_prose",
                    "markdown",
                    "technical_doc",
                    "heavy_document",
                }:
                    budget_ratio = _QUERY_AWARE_BUDGET_BY_MODE.get(
                        optimization_mode, _QUERY_AWARE_BUDGET_BY_MODE["balanced"]
                    )
                    budget_modifier = content_profile.get_threshold_modifier(
                        "query_budget", 1.0
                    )
                    budget_ratio = min(0.9, max(0.2, budget_ratio * budget_modifier))
                    if original_tokens >= _QUERY_AWARE_LONG_PROMPT_TOKENS:
                        budget_ratio = min(
                            0.95,
                            max(
                                0.2, budget_ratio * _QUERY_AWARE_LONG_PROMPT_MULTIPLIER
                            ),
                        )
                    compressed, applied, _metadata = (
                        _section_ranking.query_aware_compress(
                            prompt=working_prompt,
                            query=query_hint,
                            model_name=self.semantic_rank_model,
                            budget_ratio=budget_ratio,
                            count_tokens=self.count_tokens,
                            embedding_cache=request_embedding_cache,
                            semantic_plan=self._get_state().semantic_plan,
                        )
                    )
                    if applied:
                        working_prompt = compressed
                        self._record_technique("Query-Aware Compression")
            if debug_max_stage_timing:
                _log_max_stage(
                    "query_aware_compression",
                    stage_query_started,
                    before_tokens=query_before_tokens,
                    after_tokens=self._tokens_for_stage_decision(working_prompt),
                    extra={
                        "query_present": bool(query_hint),
                        "segment_spans_present": bool(segment_spans),
                    },
                )

            maximum_prepass_metadata: Dict[str, Any] = {}
            maximum_prepass_applied = False
            maximum_prepass_policy: Dict[str, Any] = {
                "enabled": False,
                "minimum_tokens": self.maximum_prepass_min_tokens,
                "budget_ratio": self.maximum_prepass_budget_ratio,
                "max_sentences": self.maximum_prepass_max_sentences,
                "policy_source": "disabled_mode",
                "policy_mode": self.maximum_prepass_policy,
                "enabled_override": False,
            }
            stage_prepass_started = (
                time.perf_counter() if debug_max_stage_timing else None
            )
            prepass_before_tokens = (
                self._tokens_for_stage_decision(working_prompt)
                if debug_max_stage_timing
                else None
            )
            if optimization_mode == "maximum":
                prepass_redundancy_estimate = self._estimate_sentence_redundancy_ratio(
                    working_prompt
                )
                sentence_spans = _max_prepass._split_sentence_spans(working_prompt)
                sentence_count = max(len(sentence_spans), 1)
                prepass_constraint_density = (
                    _max_prepass._constraint_hit_count(working_prompt) / sentence_count
                )
                maximum_prepass_policy = self._resolve_maximum_prepass_policy(
                    prompt_tokens=self._tokens_for_stage_decision(working_prompt),
                    content_profile=content_profile,
                    query_hint=query_hint,
                    redundancy_estimate=prepass_redundancy_estimate,
                    constraint_density=prepass_constraint_density,
                )
                protected_ranges: List[Tuple[int, int]] = []
                if segment_spans:
                    protected_ranges.extend(
                        [
                            (span.start, span.end)
                            for span in segment_spans
                            if span.end > span.start
                        ]
                    )
                prepass_config = _max_prepass.BudgetedPrepassConfig(
                    enabled=bool(maximum_prepass_policy.get("enabled", False)),
                    minimum_tokens=int(
                        maximum_prepass_policy.get(
                            "minimum_tokens", self.maximum_prepass_min_tokens
                        )
                    ),
                    budget_ratio=float(
                        maximum_prepass_policy.get(
                            "budget_ratio", self.maximum_prepass_budget_ratio
                        )
                    ),
                    max_sentences=int(
                        maximum_prepass_policy.get(
                            "max_sentences", self.maximum_prepass_max_sentences
                        )
                    ),
                    budget_floor_ratio=float(
                        maximum_prepass_policy.get("budget_floor_ratio", 0.2)
                    ),
                    budget_cap_ratio=float(
                        maximum_prepass_policy.get("budget_cap_ratio", 0.95)
                    ),
                    adaptive_budgeting=bool(
                        maximum_prepass_policy.get("policy_mode", "auto") == "auto"
                    ),
                )
                prepass_output, prepass_applied, prepass_metadata = (
                    _max_prepass.budgeted_sentence_span_prepass(
                        prompt=working_prompt,
                        query=query_hint,
                        count_tokens=self.count_tokens,
                        protected_ranges=protected_ranges,
                        config=prepass_config,
                    )
                )
                if prepass_applied:
                    working_prompt = prepass_output
                    maximum_prepass_applied = True
                    maximum_prepass_metadata = prepass_metadata
                    self._record_technique("Maximum Mode Budgeted Pre-Pass")
            if debug_max_stage_timing:
                _log_max_stage(
                    "maximum_prepass",
                    stage_prepass_started,
                    before_tokens=prepass_before_tokens,
                    after_tokens=self._tokens_for_stage_decision(working_prompt),
                    extra={
                        "applied": maximum_prepass_applied,
                        "policy": maximum_prepass_policy.get("policy_mode"),
                    },
                )

            ranking_metadata: Dict[str, Any] = {}
            ranking_applied = False
            stage_ranking_started = (
                time.perf_counter() if debug_max_stage_timing else None
            )
            ranking_before_tokens = (
                self._tokens_for_stage_decision(working_prompt)
                if debug_max_stage_timing
                else None
            )
            working_prompt_tokens: Optional[Sequence[int]] = None
            initial_working_tokens = self._tokens_for_stage_decision(working_prompt)
            redundancy_ratio = self._estimate_sentence_redundancy_ratio(working_prompt)
            redundancy_triggered = (
                redundancy_ratio >= _REDUNDANCY_SECTION_RANKING_RATIO_THRESHOLD
                and initial_working_tokens >= _REDUNDANCY_SECTION_RANKING_TOKEN_MIN
            )
            if (
                redundancy_triggered
                and self.section_ranking_mode in {"", "off"}
                and ranking_override is None
            ):
                ranking_override = {
                    "mode": (
                        "gzip"
                        if content_profile.name in {"code", "markdown", "technical_doc"}
                        else "bm25"
                    )
                }
                if self.section_ranking_token_budget is not None:
                    ranking_override["token_budget"] = self.section_ranking_token_budget
            if telemetry_collector:
                _record_telemetry_flag(
                    telemetry_collector,
                    "section_ranking_redundancy_ratio",
                    redundancy_ratio,
                )
            tokens_exceed_threshold = initial_working_tokens > chunk_threshold
            should_rank_sections = tokens_exceed_threshold or redundancy_triggered
            ranking_trigger_reason: Optional[str] = None
            if (
                should_rank_sections
                and optimization_mode == "maximum"
                and query_hint
                and not segment_spans
                and original_tokens > 0
            ):
                prepipeline_ratio = initial_working_tokens / max(original_tokens, 1)
                if (
                    prepipeline_ratio
                    <= _PREPIPELINE_SECTION_RANKING_BUDGET_FLOOR_MAXIMUM
                ):
                    should_rank_sections = False
                    if telemetry_collector:
                        _record_telemetry_flag(
                            telemetry_collector,
                            "section_ranking_skipped_reason",
                            "prepipeline_budget_floor",
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "section_ranking_prepipeline_ratio",
                            round(prepipeline_ratio, 4),
                        )
            if should_rank_sections:
                if tokens_exceed_threshold and redundancy_triggered:
                    ranking_trigger_reason = "size_and_redundancy"
                elif redundancy_triggered:
                    ranking_trigger_reason = "redundancy"
                else:
                    ranking_trigger_reason = "size"
                ranking_config = _section_ranking.resolve_section_ranking(
                    default_mode=self.section_ranking_mode,
                    default_token_budget=self.section_ranking_token_budget,
                    override=ranking_override,
                )
                if ranking_config.enabled():
                    working_prompt_tokens = self.tokenize(working_prompt)
                    ranking_token_count = (
                        len(working_prompt_tokens)
                        if working_prompt_tokens is not None
                        else initial_working_tokens
                    )
                    if ranking_token_count > 0:
                        ranked_prompt, applied, metadata = (
                            _section_ranking.apply_section_ranking(
                                optimizer=self,
                                prompt=working_prompt,
                                ranking=ranking_config,
                                chunking_mode=resolved_chunking_mode,
                                chat_metadata=chat_metadata,
                                default_chunking_mode=self.default_chunking_mode,
                                chunk_size=self.chunk_size or 0,
                                semantic_model=self.semantic_chunk_model,
                                semantic_rank_model=self.semantic_rank_model,
                                semantic_similarity=self.semantic_chunk_similarity,
                                count_tokens=self.count_tokens,
                                content_profile=content_profile,
                                prompt_tokens=working_prompt_tokens,
                                embedding_cache=request_embedding_cache,
                                semantic_plan=self._get_state().semantic_plan,
                            )
                        )
                        if (
                            applied
                            and optimization_mode == "maximum"
                            and query_hint
                            and not segment_spans
                            and original_tokens > 0
                        ):
                            ranked_token_count = self._tokens_for_stage_decision(
                                ranked_prompt
                            )
                            ranked_ratio = ranked_token_count / max(original_tokens, 1)
                            if (
                                ranked_ratio
                                < _PREPIPELINE_SECTION_RANKING_HARD_FLOOR_MAXIMUM
                            ):
                                applied = False
                                metadata = dict(metadata or {})
                                metadata["rollback_reason"] = "prepipeline_hard_floor"
                                metadata["ranked_ratio"] = round(ranked_ratio, 4)
                                if telemetry_collector:
                                    _record_telemetry_flag(
                                        telemetry_collector,
                                        "section_ranking_rollback_reason",
                                        "prepipeline_hard_floor",
                                    )
                                    _record_telemetry_flag(
                                        telemetry_collector,
                                        "section_ranking_ranked_ratio",
                                        round(ranked_ratio, 4),
                                    )
                        if applied:
                            working_prompt = ranked_prompt
                            working_prompt_tokens = self.tokenize(working_prompt)
                            ranking_metadata = metadata
                            if ranking_trigger_reason:
                                ranking_metadata["trigger"] = ranking_trigger_reason
                            ranking_applied = True
                            mode_label = (
                                ranking_config.mode.upper()
                                if ranking_config.mode not in {"", "off"}
                                else "FILTER"
                            )
                            technique = f"Context Section Ranking ({mode_label})"
                            self._record_technique(technique)
            if debug_max_stage_timing:
                _log_max_stage(
                    "section_ranking",
                    stage_ranking_started,
                    before_tokens=ranking_before_tokens,
                    after_tokens=self._tokens_for_stage_decision(working_prompt),
                    extra={
                        "applied": ranking_applied,
                        "trigger": ranking_trigger_reason,
                    },
                )

            overlap_ratio = self.chunk_overlap_ratio
            overlap_modifier = content_profile.get_threshold_modifier(
                "chunk_overlap", 1.0
            )
            if overlap_modifier != 1.0:
                overlap_ratio = min(0.5, max(0.0, overlap_ratio * overlap_modifier))

            json_policy = self._resolve_json_compression_config()
            if content_profile.name == "json":
                json_policy["minify"] = True
            mode_config = resolve_optimization_config(optimization_mode)
            enable_toon_conversion = bool(
                mode_config.get("enable_toon_conversion", False)
            )
            resolved_disabled = merge_disabled_passes(
                set(mode_config.get("disabled_passes", [])),
                content_profile,
            )
            if force_disabled_passes:
                resolved_disabled.update(force_disabled_passes)

            def should_skip_pass(pass_name: str) -> bool:
                """Check if a pass should be skipped based on configuration."""
                return pass_name in resolved_disabled

            state = self._get_state()
            state.semantic_deduplication_enabled = (
                self.enable_semantic_deduplication
                and "semantic_deduplication" not in resolved_disabled
            )
            state.skip_exact_deduplication = (
                "deduplicate_content_exact" in resolved_disabled
            )
            if (
                content_profile.name not in {"code", "json"}
                and not should_skip_pass("dedup_normalized_sentences")
                and not resolved_frequency_learning
            ):
                before_text = working_prompt
                working_prompt, dedup_applied = (
                    self._apply_pre_chunk_normalized_sentence_dedup(
                        working_prompt,
                        force_preserve_digits=resolved_preserve_digits,
                        json_policy=json_policy,
                        enable_toon_conversion=enable_toon_conversion,
                    )
                )
                if dedup_applied and working_prompt != before_text:
                    self._record_technique("Normalized Sentence Deduplication")
                    self._record_technique("Content Deduplication")

            working_tokens = self._tokens_for_stage_decision(working_prompt)
            working_prompt_tokens = self.tokenize(working_prompt)

            comparative_threshold = max(self.chunk_threshold or 1, 1)
            relative_ratio = original_tokens / comparative_threshold
            self._prefer_background_summary = relative_ratio < 1.3

            resolved_strategy = _chunking.resolve_strategy(
                resolved_chunking_mode or self.default_chunking_mode
            )
            should_chunk = (
                resolved_strategy != "off" and working_tokens > chunk_threshold
            )
            optimized: Optional[str] = None
            fastpath_applied = False
            classifier_applied = False
            classifier_metadata: Dict[str, Any] = {}
            fastpath_threshold = max(self.fastpath_token_threshold or 0, 0)
            stage_classifier_started = (
                time.perf_counter() if debug_max_stage_timing else None
            )
            classifier_before_tokens = (
                self._tokens_for_stage_decision(working_prompt)
                if debug_max_stage_timing
                else None
            )

            if (
                optimization_mode == "maximum"
                and not segment_spans
                and (fastpath_threshold <= 0 or working_tokens > fastpath_threshold)
            ):
                settings = self._resolve_multi_candidate_settings(
                    "token_classifier", optimization_mode
                )
                if (
                    settings.get("max_candidates", 1) > 1
                    and self.semantic_guard_enabled
                ):
                    defaults = content_profile.smart_defaults if content_profile else {}
                    base_min_conf = defaults.get("classifier_min_confidence", 0.45)
                    base_min_keep = defaults.get("classifier_min_keep_ratio", 0.6)
                    base_text, base_applied, base_meta = (
                        self._optimize_with_token_classifier(
                            working_prompt,
                            force_preserve_digits=resolved_preserve_digits,
                            json_policy=json_policy,
                            content_type=content_type,
                            content_profile=content_profile,
                        )
                    )
                    candidates: List[Tuple[str, str, Dict[str, Any]]] = [
                        ("original", working_prompt, {"applied": False})
                    ]
                    if base_applied:
                        candidates.append(("default", base_text, base_meta))

                    if settings.get("max_candidates", 1) > 1:
                        aggressive_keep = max(
                            0.1,
                            min(
                                0.95,
                                base_min_keep
                                * config.TOKEN_CLASSIFIER_AGGRESSIVE_KEEP_RATIO_MULTIPLIER,
                            ),
                        )
                        aggressive_conf = max(
                            0.05,
                            min(
                                0.95,
                                base_min_conf
                                * config.TOKEN_CLASSIFIER_AGGRESSIVE_CONFIDENCE_MULTIPLIER,
                            ),
                        )
                        aggressive_text, aggressive_applied, aggressive_meta = (
                            self._optimize_with_token_classifier(
                                working_prompt,
                                force_preserve_digits=resolved_preserve_digits,
                                json_policy=json_policy,
                                content_type=content_type,
                                content_profile=content_profile,
                                min_confidence_override=aggressive_conf,
                                min_keep_ratio_override=aggressive_keep,
                            )
                        )
                        if aggressive_applied:
                            aggressive_meta.update(
                                {
                                    "min_confidence": aggressive_conf,
                                    "min_keep_ratio": aggressive_keep,
                                }
                            )
                            candidates.append(
                                ("aggressive", aggressive_text, aggressive_meta)
                            )

                    guard_threshold = max(
                        self._resolve_semantic_guard_threshold(),
                        settings.get("min_guard_threshold", 0.0) or 0.0,
                    )
                    optimized, classifier_metadata = self._select_semantic_candidate(
                        working_prompt,
                        candidates,
                        pass_name="token_classifier",
                        guard_threshold=guard_threshold,
                        telemetry_collector=telemetry_collector,
                    )
                    classifier_applied = optimized != working_prompt
                else:
                    (
                        optimized,
                        classifier_applied,
                        classifier_metadata,
                    ) = self._optimize_with_token_classifier(
                        working_prompt,
                        force_preserve_digits=resolved_preserve_digits,
                        json_policy=json_policy,
                        content_type=content_type,
                        content_profile=content_profile,
                    )
                if classifier_applied and self._per_pass_guard_enabled(
                    optimization_mode
                ):
                    optimized, guard_rollback, _similarity = (
                        self._apply_per_pass_semantic_guard(
                            "token_classifier",
                            working_prompt,
                            optimized,
                            guard_threshold=self._resolve_semantic_guard_threshold(),
                            telemetry_collector=telemetry_collector,
                        )
                    )
                    if guard_rollback:
                        classifier_applied = False
                        classifier_metadata = {}
                if classifier_applied:
                    self._record_technique("Token Classification Compression")
                _record_telemetry_flag(
                    telemetry_collector,
                    "token_classifier_applied",
                    classifier_applied,
                )
                _record_telemetry_flag(
                    telemetry_collector,
                    "token_classifier_content_type",
                    content_type,
                )
                if classifier_metadata:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_keep_ratio",
                        classifier_metadata.get("keep_ratio"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_combined_keep_ratio",
                        classifier_metadata.get("combined_keep_ratio"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_entropy_high_ratio",
                        classifier_metadata.get("entropy_high_ratio"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_entropy_high_threshold",
                        classifier_metadata.get("combined_entropy_high_threshold"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_combined_keep_threshold",
                        classifier_metadata.get("combined_keep_threshold"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_combined_classifier_weight",
                        classifier_metadata.get("combined_classifier_weight"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_combined_entropy_weight",
                        classifier_metadata.get("combined_entropy_weight"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_min_keep_ratio",
                        classifier_metadata.get("min_keep_ratio"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "token_classifier_min_confidence",
                        classifier_metadata.get("min_confidence"),
                    )
                    shadow_metadata = classifier_metadata.get("shadow")
                    if shadow_metadata:
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_model",
                            shadow_metadata.get("model_name"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_available",
                            shadow_metadata.get("available"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_applied",
                            shadow_metadata.get("applied"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_keep_ratio",
                            shadow_metadata.get("keep_ratio"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_combined_keep_ratio",
                            shadow_metadata.get("combined_keep_ratio"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_entropy_high_ratio",
                            shadow_metadata.get("entropy_high_ratio"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_entropy_high_threshold",
                            shadow_metadata.get("combined_entropy_high_threshold"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_combined_keep_threshold",
                            shadow_metadata.get("combined_keep_threshold"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_combined_classifier_weight",
                            shadow_metadata.get("combined_classifier_weight"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_combined_entropy_weight",
                            shadow_metadata.get("combined_entropy_weight"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_decisions",
                            shadow_metadata.get("decisions"),
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "token_classifier_shadow_removals",
                            shadow_metadata.get("removals"),
                        )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "semantic_guard_enabled",
                        self.semantic_guard_enabled,
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "semantic_guard_threshold",
                        self._resolve_semantic_guard_threshold(),
                    )

            # Treat token classifier as a pre-pass (do not short-circuit the rest of the pipeline).
            if classifier_applied and optimized is not None:
                working_prompt = optimized
                working_tokens = self._tokens_for_stage_decision(working_prompt)
                working_prompt_tokens = self.tokenize(working_prompt)
                should_chunk = (
                    resolved_strategy != "off" and working_tokens > chunk_threshold
                )
                optimized = None
            elif not classifier_applied:
                # Classifier no-op/unavailable should still continue through the normal pipeline.
                optimized = None
            if debug_max_stage_timing:
                _log_max_stage(
                    "token_classifier_prepass",
                    stage_classifier_started,
                    before_tokens=classifier_before_tokens,
                    after_tokens=self._tokens_for_stage_decision(working_prompt),
                    extra={
                        "applied": classifier_applied,
                        "skipped_by_fastpath_floor": bool(
                            fastpath_threshold > 0
                            and classifier_before_tokens is not None
                            and classifier_before_tokens <= fastpath_threshold
                        ),
                    },
                )

            stage_execution_started = (
                time.perf_counter() if debug_max_stage_timing else None
            )
            execution_path = "pipeline"
            if (
                not classifier_applied
                and fastpath_threshold
                and working_tokens <= fastpath_threshold
                and working_prompt.count("\n") <= _FASTPATH_MAX_NEWLINES
            ):
                if not should_chunk:
                    optimized = self._optimize_fastpath(
                        working_prompt,
                        mode,
                        optimization_mode,
                        resolved_preserve_digits,
                        use_discourse_weighting=resolved_discourse_weighting,
                        profiler=profiler,
                        telemetry_collector=telemetry_collector,
                        json_policy=json_policy,
                        segment_spans=segment_spans,
                        content_type=content_type,
                        content_profile=content_profile,
                        custom_canonicals=custom_canonicals,
                        force_disabled_passes=force_disabled_passes,
                        customer_id=customer_id,
                    )
                    fastpath_applied = True
                    execution_path = "fastpath"

            if should_chunk:
                logger.info(
                    "Using %s chunking strategy for prompt with %s tokens",
                    resolved_strategy,
                    working_tokens,
                )
                chunk_force_disabled_passes: Set[str] = set(force_disabled_passes or [])
                chunk_force_disabled_passes.add("alias_named_entities")
                optimized, chunk_specs = self._optimize_with_chunking(
                    working_prompt,
                    mode,
                    optimization_mode,
                    chunk_size=chunk_size,
                    strategy=resolved_strategy,
                    overlap_ratio=overlap_ratio,
                    enable_frequency_learning=resolved_frequency_learning,
                    use_discourse_weighting=resolved_discourse_weighting,
                    force_preserve_digits=resolved_preserve_digits,
                    segment_spans=segment_spans,
                    chat_metadata=chat_metadata,
                    profiler=profiler,
                    telemetry_collector=telemetry_collector,
                    json_policy=json_policy,
                    token_cache=request_token_cache,
                    enable_toon_conversion=enable_toon_conversion,
                    content_type=content_type,
                    content_profile=content_profile,
                    embedding_cache=request_embedding_cache,
                    semantic_plan=self._get_state().semantic_plan,
                    custom_canonicals=custom_canonicals,
                    force_disabled_passes=chunk_force_disabled_passes,
                    customer_id=customer_id,
                )
                strategy_label = chunk_specs[0].metadata.get(
                    "strategy", resolved_strategy
                )
                chunk_technique = f"Chunked Optimization ({strategy_label.title()})"
                self._record_technique(chunk_technique)
                execution_path = f"chunking:{resolved_strategy}"
            elif optimized is None:
                if mode.lower() in ["basic", "advanced"]:
                    optimized = self._optimize_pipeline(
                        working_prompt,
                        mode,
                        optimization_mode,
                        resolved_frequency_learning,
                        resolved_preserve_digits,
                        profiler=profiler,
                        telemetry_collector=telemetry_collector,
                        segment_spans=segment_spans,
                        use_discourse_weighting=resolved_discourse_weighting,
                        json_policy=json_policy,
                        force_disabled_passes=force_disabled_passes,
                        content_type=content_type,
                        content_profile=content_profile,
                        customer_id=customer_id,
                        custom_canonicals=custom_canonicals,
                        query_hint=query_hint,
                    )
                else:
                    raise ValueError(
                        f"Invalid mode: {mode}. Must be 'basic' or 'advanced'"
                    )
            if debug_max_stage_timing:
                _log_max_stage(
                    "final_execution_path",
                    stage_execution_started,
                    before_tokens=self._tokens_for_stage_decision(working_prompt),
                    after_tokens=(
                        self._tokens_for_stage_decision(optimized)
                        if isinstance(optimized, str)
                        else None
                    ),
                    extra={"path": execution_path},
                )

            if chat_metadata and not chat_metadata.get("skip_roles"):
                optimized = _history.restore_structured_chat(chat_metadata, optimized)

            # When section ranking was applied, compare against the ranked prompt
            # to avoid reverting legitimate ranking reductions
            # Semantic safeguard
            semantic_similarity: Optional[float] = None
            semantic_similarity_source: Optional[str] = None
            # Pre-compute baseline for semantic guard to avoid operator precedence issues
            baseline_prompt = working_prompt if ranking_applied else source_prompt
            guard_threshold = self._resolve_semantic_guard_threshold()
            if self.semantic_guard_enabled and not fastpath_applied:
                guard_limit = self.semantic_guard_max_prompt_tokens
                if guard_limit and guard_limit > 0 and original_tokens > guard_limit:
                    logger.debug(
                        "Semantic guard skipped: prompt tokens %s exceed encoder limit %s",
                        original_tokens,
                        guard_limit,
                    )
                else:
                    if optimized != baseline_prompt:
                        collapsed_baseline = (
                            _lexical._collapse_consecutive_duplicates_segment(
                                baseline_prompt
                            )
                            if baseline_prompt
                            else baseline_prompt
                        )
                        collapsed_similarity = (
                            self._lexical_similarity(collapsed_baseline, optimized)
                            if collapsed_baseline is not None
                            else None
                        )
                        if collapsed_baseline == optimized or (
                            collapsed_similarity is not None
                            and collapsed_similarity >= 0.98
                        ):
                            semantic_similarity = guard_threshold
                            semantic_similarity_source = "exact_repetition_collapse"
                        elif self._is_deduplication_only_transform(
                            baseline_prompt, optimized
                        ):
                            semantic_similarity = guard_threshold
                            semantic_similarity_source = "deduplication_collapse"
                        else:
                            semantic_similarity = _metrics.score_similarity(
                                baseline_prompt,
                                optimized,
                                self.semantic_guard_model,
                                embedding_cache=request_embedding_cache,
                            )
                            if semantic_similarity is None:
                                logger.warning(
                                    "Semantic guard similarity unavailable; reverting to baseline prompt."
                                )
                                optimized = baseline_prompt
                                semantic_similarity = 1.0
                                semantic_similarity_source = (
                                    "guard_unavailable_rollback"
                                )
                            else:
                                semantic_similarity_source = "model"
                    else:
                        logger.debug(
                            "Semantic guard skipped: optimized prompt unchanged"
                        )
            if semantic_similarity is not None and (
                semantic_similarity < guard_threshold
            ):
                optimized = baseline_prompt
                semantic_similarity_source = "guard_rollback"
                semantic_similarity = 1.0
                self.techniques_applied = ["Semantic Guard Rollback"]
            elif semantic_similarity is None:
                logger.debug(
                    "Semantic guard skipped due to unavailable similarity metric"
                )

            # Calculate stats
            optimized_length = len(optimized)
            optimized_tokens = self._tokens_for_stage_decision(optimized)
            chars_saved = max(original_length - optimized_length, 0)
            tokens_saved = max(original_tokens - optimized_tokens, 0)

            compression_percent = (
                (chars_saved / original_length * 100) if original_length > 0 else 0.0
            )

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "Optimization completed: %s -> %s tokens (%.1f%% compression) in %.1fms",
                original_tokens,
                optimized_tokens,
                (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0.0,
                processing_time,
            )

            stats: Dict[str, Any] = {
                "original_chars": original_length,
                "optimized_chars": optimized_length,
                "compression_percentage": round(compression_percent, 2),
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "token_savings": tokens_saved,
                "processing_time_ms": round(processing_time, 2),
                "fast_path": fastpath_applied,
                "content_profile": content_profile.name,
                "smart_context_description": smart_context.description,
            }
            toon_stats = self._get_state().toon_stats
            if toon_stats.get("conversions", 0) > 0:
                stats["toon_conversions"] = toon_stats["conversions"]
                stats["toon_bytes_saved"] = toon_stats["bytes_saved"]

            if semantic_similarity is not None:
                stats["semantic_similarity"] = float(semantic_similarity)
            if semantic_similarity_source:
                stats["semantic_similarity_source"] = semantic_similarity_source
            dedup_counts = self._get_state().dedup_counts
            if any(dedup_counts.values()):
                stats["deduplication"] = dict(dedup_counts)

            semantic_plan_metrics = self._get_state().semantic_plan_metrics
            embedding_reuse_count = int(
                semantic_plan_metrics.get("embedding_reuse_count", 0.0)
            )
            embedding_calls_saved = int(
                semantic_plan_metrics.get("embedding_calls_saved", 0.0)
            )
            embedding_wall_clock_savings_ms = float(
                semantic_plan_metrics.get("embedding_wall_clock_savings_ms", 0.0)
            )
            if embedding_reuse_count > 0:
                stats["embedding_reuse_count"] = embedding_reuse_count
            if embedding_calls_saved > 0:
                stats["embedding_calls_saved"] = embedding_calls_saved
            if embedding_wall_clock_savings_ms > 0:
                stats["embedding_wall_clock_savings_ms"] = round(
                    embedding_wall_clock_savings_ms,
                    3,
                )

            if telemetry_collector:
                _record_telemetry_flag(
                    telemetry_collector, "fast_path", fastpath_applied
                )
                if toon_stats.get("conversions", 0) > 0:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "toon_conversions",
                        toon_stats["conversions"],
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "toon_bytes_saved",
                        toon_stats["bytes_saved"],
                    )
                shared_reuse = _metrics.get_shared_embedding_reuse_count(
                    self._get_state().embedding_cache
                )
                if shared_reuse:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "shared_embedding_reuse_count",
                        shared_reuse,
                    )
                _record_telemetry_flag(
                    telemetry_collector,
                    "embedding_reuse_count",
                    embedding_reuse_count,
                )
                _record_telemetry_flag(
                    telemetry_collector,
                    "embedding_calls_saved",
                    embedding_calls_saved,
                )
                _record_telemetry_flag(
                    telemetry_collector,
                    "embedding_wall_clock_savings_ms",
                    round(embedding_wall_clock_savings_ms, 3),
                )

            if optimization_mode == "maximum":
                stats["maximum_prepass_policy_source"] = maximum_prepass_policy.get(
                    "policy_source", "auto"
                )
                stats["maximum_prepass_policy_enabled"] = bool(
                    maximum_prepass_policy.get("enabled", False)
                )
                stats["maximum_prepass_policy_mode"] = str(
                    maximum_prepass_policy.get(
                        "policy_mode", self.maximum_prepass_policy
                    )
                )
                stats["maximum_prepass_policy_enabled_override"] = bool(
                    maximum_prepass_policy.get("enabled_override", False)
                )
                stats["maximum_prepass_policy_minimum_tokens"] = int(
                    maximum_prepass_policy.get(
                        "minimum_tokens", self.maximum_prepass_min_tokens
                    )
                )
                stats["maximum_prepass_policy_budget_ratio"] = float(
                    maximum_prepass_policy.get(
                        "budget_ratio", self.maximum_prepass_budget_ratio
                    )
                )
                stats["maximum_prepass_policy_adaptive_budget_ratio"] = float(
                    maximum_prepass_policy.get(
                        "adaptive_budget_ratio",
                        maximum_prepass_policy.get(
                            "budget_ratio", self.maximum_prepass_budget_ratio
                        ),
                    )
                )
                stats["maximum_prepass_policy_adaptive_redundancy_ratio"] = float(
                    maximum_prepass_policy.get("adaptive_redundancy_ratio", 0.0)
                )
                stats["maximum_prepass_policy_adaptive_constraint_density"] = float(
                    maximum_prepass_policy.get("adaptive_constraint_density", 0.0)
                )
                stats["maximum_prepass_policy_max_sentences"] = int(
                    maximum_prepass_policy.get(
                        "max_sentences", self.maximum_prepass_max_sentences
                    )
                )
                if telemetry_collector:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_policy_source",
                        maximum_prepass_policy.get("policy_source", "auto"),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_policy_enabled",
                        bool(maximum_prepass_policy.get("enabled", False)),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_policy_adaptive_budget_ratio",
                        maximum_prepass_policy.get(
                            "adaptive_budget_ratio",
                            maximum_prepass_policy.get("budget_ratio", 0.0),
                        ),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_policy_adaptive_redundancy_ratio",
                        maximum_prepass_policy.get("adaptive_redundancy_ratio", 0.0),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_policy_adaptive_constraint_density",
                        maximum_prepass_policy.get("adaptive_constraint_density", 0.0),
                    )

            if ranking_applied:
                stats["section_ranking_selected_sections"] = ranking_metadata.get(
                    "selected_indices", []
                )
                if ranking_metadata.get("trigger"):
                    stats["section_ranking_trigger"] = ranking_metadata.get("trigger")
                logger.debug(
                    "Section ranking applied with selected sections %s",
                    ranking_metadata.get("selected_indices", []),
                )
                if telemetry_collector:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "section_ranking_trigger",
                        ranking_metadata.get("trigger", "size"),
                    )
                    if ranking_metadata.get("header_indices") is not None:
                        _record_telemetry_flag(
                            telemetry_collector,
                            "section_ranking_header_indices",
                            ranking_metadata.get("header_indices"),
                        )

            if maximum_prepass_applied:
                stats["maximum_prepass_selected_sentences"] = (
                    maximum_prepass_metadata.get("selected_indices", [])
                )
                stats["maximum_prepass_target_tokens"] = maximum_prepass_metadata.get(
                    "target_tokens", 0
                )
                stats["maximum_prepass_resolved_budget_ratio"] = (
                    maximum_prepass_metadata.get(
                        "resolved_budget_ratio",
                        maximum_prepass_policy.get(
                            "budget_ratio", self.maximum_prepass_budget_ratio
                        ),
                    )
                )
                stats["maximum_prepass_redundancy_ratio"] = (
                    maximum_prepass_metadata.get("redundancy_ratio", 0.0)
                )
                stats["maximum_prepass_constraint_density"] = (
                    maximum_prepass_metadata.get("constraint_density", 0.0)
                )
                if telemetry_collector:
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_selected_sentences",
                        maximum_prepass_metadata.get("selected_indices", []),
                    )
                    _record_telemetry_flag(
                        telemetry_collector,
                        "maximum_prepass_target_tokens",
                        maximum_prepass_metadata.get("target_tokens", 0),
                    )

            if profiler.enabled and profiler.records:
                stats["profiling_ms"] = profiler.snapshot()
                logger.debug("Pipeline profiling (ms): %s", stats["profiling_ms"])

            result = {
                "optimized_output": optimized,
                "stats": stats,
                "mode": mode.lower(),
                "techniques_applied": list(self.techniques_applied),
            }

            # Schedule DB write as background task (non-blocking)
            if not skip_db_write:
                if background_tasks:
                    background_tasks.add_task(
                        self._record_history,
                        mode=mode,
                        prompt=prompt,
                        optimized=optimized,
                        stats=stats,
                        telemetry_collector=telemetry_collector,
                    )
                else:
                    self._record_history(
                        mode=mode,
                        prompt=prompt,
                        optimized=optimized,
                        stats=stats,
                        telemetry_collector=telemetry_collector,
                    )

            return result
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise
        finally:
            self._prefer_background_summary = False
            self._skip_sentence_deduplication = False
            self._get_state().token_cache.clear()
            if "request_token_cache" in locals():
                request_token_cache.clear()

    def _optimize_fastpath(
        self,
        prompt: str,
        mode: str,
        optimization_mode: str,
        force_preserve_digits: Optional[bool],
        *,
        use_discourse_weighting: bool,
        profiler: PipelineProfiler,
        telemetry_collector: Optional[Any],
        json_policy: Optional[Dict[str, Any]],
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        custom_canonicals: Optional[Dict[str, str]] = None,
        force_disabled_passes: Optional[Collection[str]] = None,
        customer_id: Optional[str] = None,
    ) -> str:
        """Run a minimal lexical pipeline for small prompts."""
        resolved_disabled = set(_FASTPATH_DISABLED_PASSES)
        if force_disabled_passes:
            resolved_disabled.update(force_disabled_passes)

        optimized = self._optimize_pipeline(
            prompt,
            mode,
            optimization_mode,
            False,
            force_preserve_digits,
            profiler=profiler,
            telemetry_collector=telemetry_collector,
            segment_spans=segment_spans,
            use_discourse_weighting=use_discourse_weighting,
            json_policy=json_policy,
            force_disabled_passes=resolved_disabled,
            content_type=content_type,
            content_profile=content_profile,
            custom_canonicals=custom_canonicals,
            customer_id=customer_id,
        )

        self._record_technique("Fast Path Lexical Cleanup")

        _record_telemetry_flag(
            telemetry_collector,
            "fast_path",
            True,
        )

        return optimized

    def _optimize_pipeline(
        self,
        prompt: str,
        mode: str,
        optimization_mode: str = "balanced",
        enable_frequency_learning: bool = False,
        force_preserve_digits: Optional[bool] = None,
        *,
        profiler: Optional[PipelineProfiler] = None,
        telemetry_collector: Optional[Any] = None,
        segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
        use_discourse_weighting: bool = True,
        json_policy: Optional[Dict[str, Any]] = None,
        force_disabled_passes: Optional[Collection[str]] = None,
        content_type: Optional[str] = None,
        content_profile: Optional[ContentProfile] = None,
        customer_id: Optional[str] = None,
        custom_canonicals: Optional[Dict[str, str]] = None,
        query_hint: Optional[str] = None,
    ) -> str:
        """
        Execute comprehensive optimization pipeline.

        Applies multiple passes:
        1. Extract and preserve critical elements (code, numbers, URLs, quotes)
        2. Compress boilerplate sections
        3. Normalize whitespace
        4. Lexical sequence (lists, instruction cleanup, canonicalization, synonyms,
           contractions, number/unit normalization, numeric precision, clause compression)
        5. Learn abbreviations from repeated phrases
        6. Compress coreferences
        7. Compress repeated long fragments
        8. Deduplicate sentences and phrases
        9. Add output guidance if beneficial
        10. Compress examples (at maximum optimization level)
        11. Summarize history (at maximum optimization level)
        12. Entropy-guided pruning (at maximum optimization level)
        13. Final whitespace + punctuation normalization
        14. Restore preserved elements
        """
        result = prompt
        no_change_streak = 0
        initial_tokens = self._tokens_for_stage_decision(result)
        token_budget_tracker: Optional[TokenBudgetTracker] = None
        if telemetry_collector and not self.exact_pass_token_counting:
            token_budget_tracker = self._build_token_budget_tracker(result)
        glossary_collector = GlossaryCollector()

        def update_noop(before_text: str, after_text: str) -> None:
            nonlocal no_change_streak
            if after_text == before_text:
                no_change_streak += 1
            else:
                no_change_streak = 0

        def has_example_candidates(text: str) -> bool:
            if re.search(
                r"example\s+(?:\d+|__[^\s:]+__):",
                text,
                flags=re.IGNORECASE,
            ):
                return True
            return bool(
                re.search(
                    r"(?:input|output|expected|case)\s*[:=-]",
                    text,
                    flags=re.IGNORECASE,
                )
            )

        def has_history_candidates(text: str) -> bool:
            role_hits = len(_history.ROLE_PREFIX_PATTERN.findall(text))
            sentence_count = max(len(_split_semantic_sentences(text)), 1)
            return role_hits >= 4 and (role_hits / sentence_count) >= 0.2

        segment_weight_vector: Optional[List[float]] = None
        # Resolve optimization settings from optimization level
        level_config = resolve_optimization_config(optimization_mode)
        disabled_passes = level_config.get("disabled_passes", [])
        enable_toon_conversion = bool(level_config.get("enable_toon_conversion", False))
        resolved_disabled = set(disabled_passes)
        if force_disabled_passes:
            resolved_disabled.update(force_disabled_passes)

        resolved_content_type = (
            content_type
            or (content_profile.name if content_profile is not None else None)
            or classify_content(prompt)
        )
        resolved_profile = (
            content_profile
            if content_profile is not None
            else get_profile(resolved_content_type)
        )
        resolved_disabled = merge_disabled_passes(resolved_disabled, resolved_profile)

        def should_skip_pass(pass_name: str) -> bool:
            return pass_name in resolved_disabled

        pass_toggles = level_config.get("pass_toggles", {})
        for pass_name, enabled in pass_toggles.items():
            if not enabled:
                resolved_disabled.add(pass_name)
        state = self._get_state()
        state.semantic_deduplication_enabled = (
            self.enable_semantic_deduplication
            and "semantic_deduplication" not in resolved_disabled
        )
        state.skip_exact_deduplication = (
            "deduplicate_content_exact" in resolved_disabled
        )

        # Apply threshold modifiers from content profile
        state = self._get_state()
        original_semantic_guard = self._resolve_semantic_guard_threshold()
        original_entropy_ratio = self.entropy_prune_ratio
        original_entropy_max_ratio = self.entropy_prune_max_ratio
        original_near_dup_similarity = self.near_dup_similarity
        original_summarize_modifier = self.summarize_keep_ratio_modifier
        semantic_modifier = resolved_profile.get_threshold_modifier(
            "semantic_guard", 1.0
        )
        updated_semantic_guard = original_semantic_guard
        if semantic_modifier != 1.0:
            updated_semantic_guard = min(
                0.99, original_semantic_guard * semantic_modifier
            )
        if state.semantic_guard_threshold_override is not None:
            state.semantic_guard_threshold_override = updated_semantic_guard
        else:
            self.semantic_guard_threshold = updated_semantic_guard

        dedup_modifier = resolved_profile.get_threshold_modifier(
            "dedup_similarity", 1.0
        )
        if dedup_modifier != 1.0:
            self.near_dup_similarity = min(
                0.99, max(0.0, self.near_dup_similarity * dedup_modifier)
            )

        entropy_modifier = resolved_profile.get_threshold_modifier(
            "entropy_budget", 1.0
        )
        if entropy_modifier != 1.0:
            if entropy_modifier <= 1.0 or self.semantic_guard_enabled:
                updated_ratio = min(
                    0.5, max(0.0, self.entropy_prune_ratio * entropy_modifier)
                )
                self.entropy_prune_ratio = updated_ratio
                self.entropy_prune_max_ratio = min(
                    0.5, max(updated_ratio, self.entropy_prune_max_ratio)
                )

        summarize_modifier = resolved_profile.get_threshold_modifier(
            "summarize_threshold", 1.0
        )
        if summarize_modifier != 1.0:
            if summarize_modifier >= 1.0 or self.semantic_guard_enabled:
                self.summarize_keep_ratio_modifier = min(
                    1.25,
                    max(0.5, self.summarize_keep_ratio_modifier * summarize_modifier),
                )

        if resolved_content_type != "general_prose":
            logger.info(
                f"Smart Router: Detected '{resolved_content_type}' content, applying {resolved_profile.name} profile"
            )

        if resolved_disabled:
            logger.info(
                f"Optimization mode: {optimization_mode} - Disabled passes: {sorted(resolved_disabled)}"
            )
        else:
            logger.info(
                f"Optimization mode: {optimization_mode} - Running full pipeline"
            )

        enable_alias_json_keys = not should_skip_pass("alias_json_keys")

        # Telemetry recording helper
        def record_pass_telemetry(
            pass_name: str,
            duration_ms: float,
            tokens_before: int,
            tokens_after: int,
            *,
            estimated_tokens_after: Optional[int] = None,
            exact_tokens_after: Optional[int] = None,
            expected_utility: Optional[float] = None,
            actual_utility: Optional[float] = None,
            pass_skipped_reason: Optional[str] = None,
            content_profile: Optional[str] = None,
            optimization_mode: Optional[str] = None,
            token_bin: Optional[str] = None,
        ) -> None:
            """Helper to record telemetry for a pass without impacting performance."""
            if telemetry_collector:
                try:
                    telemetry_collector.record_pass(
                        pass_name,
                        duration_ms,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                        expected_utility=expected_utility,
                        actual_utility=actual_utility,
                        pass_skipped_reason=pass_skipped_reason,
                        content_profile=content_profile,
                        optimization_mode=optimization_mode,
                        token_bin=token_bin,
                    )
                except Exception:
                    pass  # Never let telemetry affect optimization

        def pass_tokens_before(text: str) -> int:
            if not telemetry_collector:
                return 0
            if self.exact_pass_token_counting or token_budget_tracker is None:
                return self.count_tokens(text)
            return token_budget_tracker.estimated_tokens

        def pass_tokens_after(
            before_text: str, after_text: str
        ) -> Tuple[int, Optional[int], Optional[int]]:
            if self.exact_pass_token_counting or token_budget_tracker is None:
                exact = self.count_tokens(after_text)
                return exact, exact, exact

            estimated = token_budget_tracker.estimate_after_edit(
                before_text, after_text
            )
            exact_sampled: Optional[int] = None
            if token_budget_tracker.should_sample_exact():
                exact_sampled = self.count_tokens(after_text)
                token_budget_tracker.calibrate(after_text, exact_sampled)
                estimated = token_budget_tracker.estimated_tokens
            return estimated, estimated, exact_sampled

        _record_telemetry_flag(
            telemetry_collector,
            "stage_order",
            "heuristics>lightweight_ml>heavy",
        )

        # Pass 0: Remove verbatim duplicate blocks BEFORE preservation
        # This must run FIRST because preservation creates unique placeholders
        # for each occurrence (e.g., __CIT_0__, __CIT_1__), making identical
        # paragraphs appear different if we wait until after preservation.
        skip_verbatim_for_learning = (
            enable_frequency_learning
            and self._estimate_sentence_redundancy_ratio(result) >= 0.8
        )
        if (
            not should_skip_pass("remove_verbatim_duplicates")
            and not skip_verbatim_for_learning
        ):
            before_text = result
            before = len(result)
            if telemetry_collector:
                tokens_before = pass_tokens_before(result)
                pass_start = time.perf_counter()
            with (
                profiler.step("remove_verbatim_duplicates")
                if profiler
                else nullcontext()
            ):
                result, removed_duplicates = self._remove_verbatim_duplicate_blocks(
                    result, preserved=None  # No preserved dict yet
                )
            if removed_duplicates > 0:
                self._track_dedup_counts("exact", removed_duplicates)
            if telemetry_collector:
                pass_duration = (time.perf_counter() - pass_start) * 1000.0
                tokens_after, estimated_tokens_after, exact_tokens_after = (
                    pass_tokens_after(before_text, result)
                )
                record_pass_telemetry(
                    "remove_verbatim_duplicates",
                    pass_duration,
                    tokens_before,
                    tokens_after,
                    estimated_tokens_after=estimated_tokens_after,
                    exact_tokens_after=exact_tokens_after,
                )
            if len(result) < before:
                self.techniques_applied.append("Verbatim Block Deduplication")
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: remove_verbatim_duplicates")

        # Pass 1: Preserve critical elements
        if not should_skip_pass("preserve_elements"):
            before_text = result
            with profiler.step("preserve_elements") if profiler else nullcontext():
                result, preserved = _preservation.extract_and_preserve(
                    self,
                    result,
                    force_digits=force_preserve_digits,
                    json_policy=json_policy,
                    enable_toon_conversion=enable_toon_conversion,
                    enable_alias_json_keys=enable_alias_json_keys,
                )
            update_noop(before_text, result)
            if (
                preserved["code_blocks"]
                or preserved["quotes"]
                or preserved["numbers"]
                or preserved["urls"]
                or preserved.get("protected")
                or preserved["citations"]
                or preserved["forced"]
            ):
                self.techniques_applied.append("Content Preservation")
            toon_stats = preserved.get(
                "toon_stats", {"conversions": 0, "bytes_saved": 0}
            )
            self._get_state().toon_stats = dict(toon_stats)
            if toon_stats.get("conversions", 0) > 0:
                self._record_technique("TOON Conversion")
        else:
            logger.debug("Skipping pass: preserve_elements")
            preserved = {
                "code_blocks": [],
                "quotes": [],
                "numbers": [],
                "urls": [],
                "citations": [],
                "protected": [],
                "forced": [],
                "json_tokens": [],
                "json_literals": [],
                "json_strings": [],
                "toon_blocks": [],
                "toon_stats": {"conversions": 0, "bytes_saved": 0},
            }
            self._get_state().toon_stats = {"conversions": 0, "bytes_saved": 0}

        # Pass 1b: Alias repeated references (URLs, citations)
        if not should_skip_pass("alias_references"):
            if resolved_profile.name in {"code", "json"} or preserved.get(
                "toon_blocks"
            ):
                logger.debug(
                    "Skipping pass: alias_references (profile or TOON blocks detected)"
                )
            else:
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                alias_applied = False
                legend_entries: Optional[List[Tuple[str, str]]] = None
                legend_savings = 0
                with profiler.step("alias_references") if profiler else nullcontext():
                    result, alias_applied, legend_entries, legend_savings = (
                        self._apply_reference_aliases(
                            result,
                            preserved,
                            token_counter=self.count_tokens,
                        )
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "alias_references",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if alias_applied:
                    self._record_technique("Reference Aliasing")
                    glossary_collector.add_entries(
                        "Refs",
                        legend_entries or [],
                        net_savings=legend_savings,
                    )
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: alias_references")

        # Pass 2: Compress boilerplate sections early
        if not should_skip_pass("compress_boilerplate"):
            before_text = result
            before = len(result)
            with profiler.step("compress_boilerplate") if profiler else nullcontext():
                result = self._compress_boilerplate(result, preserved)
            if len(result) < before:
                self.techniques_applied.append("Boilerplate Compression")
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_boilerplate")

        should_weight_segments = use_discourse_weighting or bool(segment_spans)
        # Recompute segment weights after preservation/boilerplate to account for injected placeholders
        if should_weight_segments:
            _, segment_weight_vector = self._analyze_segment_spans(
                result, segment_spans=segment_spans
            )

        # Pass 3: Normalize whitespace early
        if not should_skip_pass("normalize_whitespace"):
            if self._needs_whitespace_normalization(result):
                before_text = result
                before = len(result)
                # Only measure tokens and timing if telemetry is enabled
                if telemetry_collector:
                    tokens_before = pass_tokens_before(result)
                    pass_start = time.perf_counter()
                with (
                    profiler.step("normalize_whitespace") if profiler else nullcontext()
                ):
                    result, _ = self._normalize_text(
                        result,
                        normalize_whitespace=True,
                        compress_punctuation=False,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "normalize_whitespace",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if len(result) < before:
                    self.techniques_applied.append("Whitespace Compression")
                update_noop(before_text, result)
            else:
                logger.debug("Skipping pass: normalize_whitespace (clean)")

        else:
            logger.debug("Skipping pass: normalize_whitespace")

        label_pass_enabled = not should_skip_pass("compress_field_labels")
        if label_pass_enabled:
            if resolved_profile.name in {"code", "json"}:
                logger.debug("Skipping pass: compress_field_labels (code/json profile)")
            else:
                placeholder_ranges = _preservation.get_placeholder_ranges(
                    self, result, preserved
                )
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                label_changed = False
                legend_entries: Optional[List[Tuple[str, str]]] = None
                legend_savings = 0
                with (
                    profiler.step("compress_field_labels")
                    if profiler
                    else nullcontext()
                ):
                    result, label_changed, legend_entries, legend_savings = (
                        _lexical.compress_field_labels(
                            result,
                            placeholder_ranges=placeholder_ranges,
                            token_counter=self.count_tokens,
                        )
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "compress_field_labels",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if label_changed:
                    self._record_technique("Field Label Aliasing")
                    glossary_collector.add_entries(
                        "Labels",
                        legend_entries or [],
                        net_savings=legend_savings,
                    )
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_field_labels")

        if resolved_profile.name not in {"code", "json"} and not should_skip_pass(
            "deduplicate_exact_lines"
        ):
            before_text = result
            before_len = len(result)
            with (
                profiler.step("deduplicate_exact_lines") if profiler else nullcontext()
            ):
                result, removed_lines = self._deduplicate_exact_lines(result)
            if removed_lines > 0 and len(result) < before_len:
                self._track_dedup_counts("exact", removed_lines)
                self._record_technique("Exact Line Deduplication")
            update_noop(before_text, result)

        # Pass 3b: Enumerate shared line prefixes/suffixes
        if not should_skip_pass("compress_enumerated_prefix_suffix"):
            if resolved_profile.name in {"code", "json"}:
                logger.debug(
                    "Skipping pass: compress_enumerated_prefix_suffix (code/json profile)"
                )
            else:
                placeholder_ranges = _preservation.get_placeholder_ranges(
                    self, result, preserved
                )
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                with (
                    profiler.step("compress_enumerated_prefix_suffix")
                    if profiler
                    else nullcontext()
                ):
                    result = _structural.compress_enumerated_prefix_suffix(
                        result,
                        placeholder_ranges=placeholder_ranges,
                        token_counter=self.count_tokens,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "compress_enumerated_prefix_suffix",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if len(result) < len(before_text):
                    self._record_technique("Enumerated Prefix/Suffix Factoring")
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_enumerated_prefix_suffix")

        # Pass 3c: Factor repeated line prefixes/suffixes
        if not should_skip_pass("compress_repeated_prefix_suffix"):
            if resolved_profile.name in {"code", "json"}:
                logger.debug(
                    "Skipping pass: compress_repeated_prefix_suffix (code/json profile)"
                )
            else:
                placeholder_ranges = _preservation.get_placeholder_ranges(
                    self, result, preserved
                )
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                with (
                    profiler.step("compress_repeated_prefix_suffix")
                    if profiler
                    else nullcontext()
                ):
                    result = _structural.compress_repeated_prefix_suffix(
                        result,
                        placeholder_ranges=placeholder_ranges,
                        token_counter=self.count_tokens,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "compress_repeated_prefix_suffix",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if len(result) < len(before_text):
                    self._record_technique("Prefix/Suffix Factoring")
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_repeated_prefix_suffix")

        # Pass 4: Lexical sequence (lists, instruction cleanup, canonicalization, synonyms,
        # contractions, number/unit normalization, numeric precision, clause compression)
        list_enabled = not should_skip_pass("compress_lists")
        clean_enabled = not should_skip_pass("clean_instruction_noise")
        canonicalize_enabled = not should_skip_pass("canonicalize_entities")
        synonyms_enabled = not should_skip_pass("shorten_synonyms")
        contractions_enabled = not should_skip_pass("apply_contractions")
        numbers_enabled = not should_skip_pass("normalize_numbers_units")
        precision_enabled = not should_skip_pass("reduce_numeric_precision")
        clauses_enabled = not should_skip_pass("compress_clauses")
        symbolic_enabled = not should_skip_pass("apply_symbolic_replacements")
        articles_enabled = not should_skip_pass("remove_articles")
        consecutive_dup_enabled = not should_skip_pass(
            "collapse_consecutive_duplicates"
        )
        paradoxical_enabled = not should_skip_pass("collapse_paradoxical_phrases")
        consolidate_enabled = not should_skip_pass("consolidate_repeated_phrases")

        factoring_enabled = resolved_profile.name != "code"

        if result:
            if (
                consecutive_dup_enabled
                and not _lexical._CONSECUTIVE_DUP_DETECTOR.search(result)
            ):
                consecutive_dup_enabled = False
            if (
                paradoxical_enabled
                and not _lexical._PARADOXICAL_PHRASE_DETECTOR.search(result)
            ):
                paradoxical_enabled = False
            if consolidate_enabled and not _lexical._REPEATED_PHRASE_DETECTOR.search(
                result
            ):
                consolidate_enabled = False

        canonical_map: Dict[str, str] = {}
        tokens_before_lexical: Optional[int] = None
        tokens_after_lexical: Optional[int] = None

        if (
            list_enabled
            or clean_enabled
            or canonicalize_enabled
            or synonyms_enabled
            or contractions_enabled
            or numbers_enabled
            or precision_enabled
            or clauses_enabled
            or symbolic_enabled
            or articles_enabled
            or consecutive_dup_enabled
            or paradoxical_enabled
            or consolidate_enabled
        ):
            before_text = result
            tokens_before_lexical = self.count_tokens(result)
            if telemetry_collector:
                tokens_before = tokens_before_lexical
                pass_start = time.perf_counter()
            with profiler.step("lexical_transforms") if profiler else nullcontext():
                active_categories: Set[_lexical.InstructionCategory] = set()
                if clean_enabled:
                    active_categories = set(_lexical._INSTRUCTION_CATEGORY_ORDER)

                disabled_canonical_tokens: Set[str] = set()
                if canonicalize_enabled:
                    # Fetch mappings dynamically (cached by TrieReplacer internal cache based on dict content)
                    try:
                        from database import (
                            get_combined_canonical_mappings,
                            list_disabled_ootb_mappings,
                        )

                        canonical_map = get_combined_canonical_mappings(customer_id)
                        disabled_canonical_tokens = set(
                            list_disabled_ootb_mappings(customer_id)
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "Canonical mapping database access is unavailable; "
                            "strict mode requires database-backed mappings."
                        ) from exc
                contextual_canonical_map: Optional[Dict[str, str]] = None
                if canonical_map:
                    if resolved_content_type in {
                        "general_prose",
                        "dialogue",
                        "markdown",
                        "technical_doc",
                        "heavy_document",
                        "short",
                    }:
                        contextual_additions = {
                            **config.FLUFF_CANONICALIZATIONS,
                            **config.CONTEXTUAL_CANONICALIZATIONS,
                            **config.PROMPT_SPECIFIC_CANONICALIZATIONS,
                            **config.SMART_DEFAULT_CANONICALIZATIONS,
                        }
                        contextual_additions = _filter_disabled_canonicals(
                            contextual_additions, disabled_canonical_tokens
                        )
                        contextual_canonical_map = _merge_canonical_maps(
                            canonical_map,
                            contextual_additions,
                        )
                custom_canonicals_sanitized = sanitize_canonical_map(custom_canonicals)
                if custom_canonicals_sanitized:
                    canonical_map = (
                        _merge_canonical_maps(
                            canonical_map,
                            custom_canonicals_sanitized,
                            allow_override=True,
                        )
                        if canonical_map
                        else dict(custom_canonicals_sanitized)
                    )
                    if contextual_canonical_map:
                        contextual_canonical_map = _merge_canonical_maps(
                            contextual_canonical_map,
                            custom_canonicals_sanitized,
                            allow_override=True,
                        )

                has_placeholders = bool(config.PLACEHOLDER_PATTERN.search(result))
                if has_placeholders:
                    segments = config.PLACEHOLDER_PATTERN.split(result)
                    placeholders = config.PLACEHOLDER_PATTERN.findall(result)
                else:
                    segments = [result]
                    placeholders = []
                triggered_categories: Set[_lexical.InstructionCategory] = set()
                canonical_changed = False
                list_shortened = False
                clean_changed = False
                synonyms_changed = False
                contractions_shortened = False
                numbers_changed = False
                precision_shortened = False
                clauses_shortened = False
                symbolic_changed = False
                articles_removed = False
                consecutive_changed = False
                paradoxical_changed = False
                consolidate_changed = False

                def rebuild_text() -> str:
                    rebuilt: List[str] = []
                    for index, segment in enumerate(segments):
                        rebuilt.append(segment)
                        if index < len(placeholders):
                            rebuilt.append(placeholders[index])
                    return "".join(rebuilt)

                for index, segment in enumerate(segments):
                    if not segment:
                        continue

                    if consecutive_dup_enabled:
                        updated = _lexical._collapse_consecutive_duplicates_segment(
                            segment
                        )
                        if updated != segment:
                            consecutive_changed = True
                        segment = updated

                    if paradoxical_enabled:
                        updated = _lexical._collapse_paradoxical_phrases_segment(
                            segment
                        )
                        if updated != segment:
                            paradoxical_changed = True
                        segment = updated

                    if consolidate_enabled:
                        updated = _lexical._consolidate_repeated_phrases_segment(
                            segment
                        )
                        if updated != segment:
                            consolidate_changed = True
                        segment = updated

                    weight = _lexical._get_segment_weight(segment_weight_vector, index)
                    if list_enabled:
                        updated = _lexical._compress_list_segment(
                            segment,
                            weight=weight,
                            token_counter=self.count_tokens,
                            enable_prefix_suffix_factoring=factoring_enabled,
                        )
                        if updated != segment and len(updated) < len(segment):
                            list_shortened = True
                        segment = updated

                    if clean_enabled:
                        updated, segment_triggered = _lexical._apply_instruction_rules(
                            segment,
                            active_categories=active_categories,
                            weight=weight,
                        )
                        if segment_triggered:
                            triggered_categories.update(segment_triggered)
                        if updated != segment:
                            clean_changed = True
                        segment = updated
                        if consecutive_dup_enabled:
                            updated = _lexical._collapse_consecutive_duplicates_segment(
                                segment
                            )
                            if updated != segment:
                                consecutive_changed = True
                            segment = updated
                        if consolidate_enabled:
                            updated = _lexical._consolidate_repeated_phrases_segment(
                                segment
                            )
                            if updated != segment:
                                consolidate_changed = True
                            segment = updated

                    if canonicalize_enabled and canonical_map:
                        active_canonical_map = canonical_map
                        if (
                            contextual_canonical_map
                            and _segment_allows_contextual_canon(segment)
                        ):
                            active_canonical_map = contextual_canonical_map
                        updated = trie_canonicalize(segment, active_canonical_map)
                        if updated != segment:
                            canonical_changed = True
                        segment = updated

                    segments[index] = segment

                result = rebuild_text()

                if (
                    should_weight_segments
                    and (clean_enabled or canonicalize_enabled or synonyms_enabled)
                    and (
                        consecutive_changed
                        or paradoxical_changed
                        or consolidate_changed
                        or list_shortened
                        or clean_changed
                        or canonical_changed
                    )
                ):
                    _, segment_weight_vector = self._analyze_segment_spans(
                        result, segment_spans=segment_spans
                    )

                if (
                    synonyms_enabled
                    or contractions_enabled
                    or numbers_enabled
                    or precision_enabled
                    or clauses_enabled
                    or symbolic_enabled
                    or articles_enabled
                ):
                    token_cache: Dict[str, int] = {}
                    for index, segment in enumerate(segments):
                        if not segment:
                            continue

                        weight = _lexical._get_segment_weight(
                            segment_weight_vector, index
                        )
                        if synonyms_enabled:
                            updated = _lexical._shorten_synonyms_segment(
                                segment,
                                weight=weight,
                                token_counter=self.count_tokens,
                                token_cache=token_cache,
                            )
                            if updated != segment:
                                synonyms_changed = True
                            segment = updated

                        if contractions_enabled:
                            updated = _lexical._apply_contractions_segment(segment)
                            if updated != segment and len(updated) < len(segment):
                                contractions_shortened = True
                            segment = updated

                        if numbers_enabled:
                            updated = _lexical._normalize_numbers_and_units_segment(
                                segment
                            )
                            if updated != segment:
                                numbers_changed = True
                            segment = updated

                        if precision_enabled:
                            updated = _lexical._reduce_numeric_precision_segment(
                                segment
                            )
                            if updated != segment and len(updated) < len(segment):
                                precision_shortened = True
                            segment = updated

                        if clauses_enabled:
                            updated = _lexical._compress_clause_segment(
                                segment,
                                weight=weight,
                            )
                            if updated != segment and len(updated) < len(segment):
                                clauses_shortened = True
                            segment = updated

                        if symbolic_enabled and (
                            weight is None or weight < config.SEGMENT_WEIGHT_HIGH
                        ):
                            updated = _lexical._apply_symbolic_replacements_segment(
                                segment
                            )
                            if updated != segment:
                                symbolic_changed = True
                            segment = updated

                        if articles_enabled and (
                            weight is None or weight < config.SEGMENT_WEIGHT_HIGH
                        ):
                            updated = _lexical._remove_articles_segment(segment)
                            if updated != segment:
                                articles_removed = True
                            segment = updated

                        segments[index] = segment

                    result = rebuild_text()

            if telemetry_collector:
                pass_duration = (time.perf_counter() - pass_start) * 1000.0
                tokens_after, estimated_tokens_after, exact_tokens_after = (
                    pass_tokens_after(before_text, result)
                )
                record_pass_telemetry(
                    "lexical_transforms",
                    pass_duration,
                    tokens_before,
                    tokens_after,
                    estimated_tokens_after=estimated_tokens_after,
                    exact_tokens_after=exact_tokens_after,
                )
                tokens_after_lexical = tokens_after
            else:
                tokens_after_lexical = self.count_tokens(result)
            update_noop(before_text, result)

            if list_enabled and list_shortened:
                self.techniques_applied.append("List Compression")

            for category in _lexical._INSTRUCTION_CATEGORY_ORDER:
                if category not in triggered_categories:
                    continue
                technique = _INSTRUCTION_TECHNIQUE_BY_CATEGORY.get(category)
                if technique:
                    self._record_technique(technique)

            if canonicalize_enabled and canonical_changed:
                self._record_technique("Entity Canonicalization")

            if synonyms_enabled and synonyms_changed:
                self._record_technique("Synonym Shortening")

            if contractions_enabled and contractions_shortened:
                self.techniques_applied.append("Contraction Restoration")

            if numbers_enabled and numbers_changed:
                self.techniques_applied.append("Number & Unit Normalization")

            if precision_enabled and precision_shortened:
                self.techniques_applied.append("Numeric Precision Reduction")

            if clauses_enabled and clauses_shortened:
                self.techniques_applied.append("Clause Compression")

            if symbolic_enabled and symbolic_changed:
                self.techniques_applied.append("Symbolic Replacement")

            if articles_enabled and articles_removed:
                self.techniques_applied.append("Article Removal")

            if consecutive_dup_enabled and consecutive_changed:
                self._record_technique("Consecutive Duplicate Collapse")

            if paradoxical_enabled and paradoxical_changed:
                self._record_technique("Paradoxical Phrase Collapse")

            if consolidate_enabled and consolidate_changed:
                self._record_technique("Repeated Phrase Consolidation")
        else:
            logger.debug(
                "Skipping lexical sequence: compress_lists/clean_instruction_noise/"
                "canonicalize_entities/shorten_synonyms/apply_contractions/"
                "normalize_numbers_units/reduce_numeric_precision/compress_clauses/"
                "apply_symbolic_replacements/remove_articles"
            )

        if not list_enabled:
            logger.debug("Skipping pass: compress_lists")
        if not clean_enabled:
            logger.debug("Skipping pass: clean_instruction_noise")
        if not canonicalize_enabled:
            logger.debug("Skipping pass: canonicalize_entities")
        if not synonyms_enabled:
            logger.debug("Skipping pass: shorten_synonyms")
        if not contractions_enabled:
            logger.debug("Skipping pass: apply_contractions")
        if not numbers_enabled:
            logger.debug("Skipping pass: normalize_numbers_units")
        if not precision_enabled:
            logger.debug("Skipping pass: reduce_numeric_precision")
        if not clauses_enabled:
            logger.debug("Skipping pass: compress_clauses")
        if not symbolic_enabled:
            logger.debug("Skipping pass: apply_symbolic_replacements")
        if not articles_enabled:
            logger.debug("Skipping pass: remove_articles")

        # Pass 4a: Trim non-essential adjunct clauses with spaCy (if available)
        if not should_skip_pass("trim_adjunct_clauses"):
            if resolved_profile.name in {"code", "json"}:
                logger.debug("Skipping pass: trim_adjunct_clauses (code/json profile)")
            else:
                nlp_model = self._get_linguistic_nlp_model()
                if nlp_model is None:
                    logger.debug(
                        "Skipping pass: trim_adjunct_clauses (spaCy unavailable)"
                    )
                else:
                    placeholder_ranges = _preservation.get_placeholder_ranges(
                        self, result, preserved
                    )
                    before_text = result
                    tokens_before = (
                        pass_tokens_before(result) if telemetry_collector else None
                    )
                    pass_start = time.perf_counter() if telemetry_collector else None
                    removed_tokens = 0
                    with (
                        profiler.step("trim_adjunct_clauses")
                        if profiler
                        else nullcontext()
                    ):
                        allowlist_phrases = [
                            [token.lower() for token in phrase.split()]
                            for phrase in config.ADJUNCT_DISCOURSE_MARKERS
                        ]
                        updated, removed_tokens = _adjunct.trim_adjunct_clauses(
                            result,
                            nlp_model=nlp_model,
                            placeholder_ranges=placeholder_ranges,
                            allowlist_phrases=allowlist_phrases,
                            allowed_deps=config.ADJUNCT_ALLOWED_DEPS,
                            negation_tokens=config.ADJUNCT_NEGATION_TOKENS,
                            condition_tokens=config.ADJUNCT_CONDITION_TOKENS,
                            modal_tokens=config.ADJUNCT_MODAL_TOKENS,
                            token_counter=self.count_tokens,
                        )
                    rollback = False
                    similarity = None
                    if updated != result:
                        if self.semantic_guard_enabled:
                            similarity = self._require_semantic_guard_similarity(
                                before_text,
                                updated,
                                embedding_cache=self._get_state().embedding_cache,
                            )
                            guard_threshold = self._resolve_semantic_guard_threshold()
                            if similarity < guard_threshold:
                                updated = before_text
                                removed_tokens = 0
                                rollback = True
                        result = updated
                        if len(result) < len(before_text):
                            self._record_technique("Adjunct Clause Trimming")
                    if telemetry_collector:
                        pass_duration = (time.perf_counter() - pass_start) * 1000.0
                        tokens_after, estimated_tokens_after, exact_tokens_after = (
                            pass_tokens_after(before_text, result)
                        )
                        record_pass_telemetry(
                            "trim_adjunct_clauses",
                            pass_duration,
                            tokens_before,
                            tokens_after,
                            estimated_tokens_after=estimated_tokens_after,
                            exact_tokens_after=exact_tokens_after,
                        )
                        _record_telemetry_flag(
                            telemetry_collector,
                            "adjunct_trim_removed_tokens",
                            removed_tokens,
                        )
                        _record_telemetry_flag(
                            telemetry_collector, "adjunct_trim_rollback", rollback
                        )
                        if similarity is not None:
                            _record_telemetry_flag(
                                telemetry_collector,
                                "adjunct_trim_similarity",
                                similarity,
                            )
                    update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: trim_adjunct_clauses")

        parenthetical_enabled = not should_skip_pass("compress_parentheticals")
        if parenthetical_enabled:
            if resolved_profile.name in {"code", "json"}:
                logger.debug(
                    "Skipping pass: compress_parentheticals (code/json profile)"
                )
            else:
                placeholder_ranges = _preservation.get_placeholder_ranges(
                    self, result, preserved
                )
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                parenthetical_changed = False
                legend_entries: Optional[List[Tuple[str, str]]] = None
                legend_savings = 0
                with (
                    profiler.step("compress_parentheticals")
                    if profiler
                    else nullcontext()
                ):
                    (
                        result,
                        parenthetical_changed,
                        legend_entries,
                        legend_savings,
                    ) = _lexical.extract_parenthetical_glossary(
                        result,
                        placeholder_ranges=placeholder_ranges,
                        token_counter=self.count_tokens,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "compress_parentheticals",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if parenthetical_changed:
                    self._record_technique("Parenthetical Glossary")
                    glossary_collector.add_entries(
                        "Glossary",
                        legend_entries or [],
                        net_savings=legend_savings,
                    )
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_parentheticals")

        # Pass 4b: Offline phrase dictionary replacement
        phrase_dictionary = self.phrase_dictionary
        learned_phrase_dictionary = (
            self._load_learned_phrase_dictionary(customer_id) if customer_id else None
        )
        if learned_phrase_dictionary:
            phrase_dictionary = dict(phrase_dictionary) if phrase_dictionary else {}
            for phrase, alias in learned_phrase_dictionary.items():
                phrase_dictionary.setdefault(phrase, alias)
        if phrase_dictionary:
            before_text = result
            with profiler.step("phrase_dictionary") if profiler else nullcontext():
                result, applied, metadata = apply_phrase_dictionary(
                    result,
                    phrase_dictionary,
                    self.count_tokens,
                )
            if applied and learned_phrase_dictionary and metadata:
                used_phrases = metadata.get("used_phrases", [])
                learned_used = [
                    phrase
                    for phrase in used_phrases
                    if phrase in learned_phrase_dictionary
                ]
                if learned_used:
                    self._record_learned_phrase_usage(customer_id, learned_used)
            if applied and result != before_text:
                self._record_technique("Phrase Dictionary Compression")
            update_noop(before_text, result)
            if tokens_before_lexical is not None:
                tokens_after_lexical = self.count_tokens(result)

        if tokens_before_lexical is not None and tokens_after_lexical is not None:
            tokens_saved = tokens_before_lexical - tokens_after_lexical
            if tokens_before_lexical > 0 and tokens_saved >= 0:
                savings_ratio = tokens_saved / tokens_before_lexical
                if (
                    tokens_after_lexical <= _LOW_GAIN_HEAVY_PASS_MAX_TOKENS
                    and savings_ratio < _LOW_GAIN_HEAVY_PASS_SAVINGS_RATIO
                ):
                    resolved_disabled.update(
                        {
                            "compress_coreferences",
                            "deduplicate_content",
                        }
                    )
                    logger.debug(
                        "Skipping heavy passes due to low lexical savings: %.2f%% over %s tokens",
                        savings_ratio * 100.0,
                        tokens_after_lexical,
                    )

        # Pass 5: Learn new abbreviations from repeated phrases
        if enable_frequency_learning and not should_skip_pass(
            "learn_frequency_abbreviations"
        ):
            before_text = result
            before_length = len(result)
            with (
                profiler.step("learn_frequency_abbreviations")
                if profiler
                else nullcontext()
            ):
                (
                    result,
                    learned_map,
                    legend_entries,
                    legend_savings,
                ) = self._apply_frequency_abbreviations(
                    result, canonical_map, preserved
                )
            abbreviations_changed_text = learned_map and result != before_text
            if abbreviations_changed_text:
                self._skip_sentence_deduplication = True
            if learned_map and len(result) <= before_length:
                self.techniques_applied.append("Adaptive Abbreviation Learning")
                glossary_collector.add_entries(
                    "Abbrev",
                    legend_entries or [],
                    net_savings=legend_savings,
                )
                if abbreviations_changed_text:
                    guard_passed = True
                    if self.semantic_guard_enabled:
                        similarity = self._require_semantic_guard_similarity(
                            before_text,
                            result,
                            embedding_cache=self._get_state().embedding_cache,
                        )
                        guard_passed = (
                            similarity >= self._resolve_semantic_guard_threshold()
                        )
                    if guard_passed:
                        self._persist_learned_phrase_dictionary(
                            customer_id, learned_map
                        )
            update_noop(before_text, result)

        # Pass 5b: Build macro legend for repeated spans
        if not should_skip_pass("apply_macro_dictionary"):
            before_text = result
            with profiler.step("apply_macro_dictionary") if profiler else nullcontext():
                result, macro_map = _lexical.apply_macro_dictionary(
                    result,
                    token_counter=self.count_tokens,
                    placeholder_tokens=self._get_placeholder_tokens(preserved),
                )
            if macro_map and result != before_text:
                self.techniques_applied.append("Macro Dictionary Compression")
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: apply_macro_dictionary")

        if initial_tokens > 0:
            current_tokens = self.count_tokens(result)
            savings_ratio = (initial_tokens - current_tokens) / initial_tokens
        else:
            savings_ratio = 0.0
        if telemetry_collector:
            _record_telemetry_flag(
                telemetry_collector, "heavy_model_savings_ratio", savings_ratio
            )
            _record_telemetry_flag(telemetry_collector, "heavy_model_skipped", False)

        # Pass 6: Compress coreferences
        if not should_skip_pass("compress_coreferences"):
            before_text = result
            with profiler.step("compress_coreferences") if profiler else nullcontext():
                result = self._compress_coreferences(result, preserved)
            if self._per_pass_guard_enabled(optimization_mode):
                result, _rollback, _similarity = self._apply_per_pass_semantic_guard(
                    "compress_coreferences",
                    before_text,
                    result,
                    guard_threshold=self._resolve_semantic_guard_threshold(),
                    telemetry_collector=telemetry_collector,
                )
            if result != before_text:
                self.techniques_applied.append("Coreference Compression")
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_coreferences")

        # Pass 6b: Alias repeated named entities
        if not should_skip_pass("alias_named_entities"):
            if (
                enable_frequency_learning
                or resolved_profile.name in {"code", "json"}
                or preserved.get("toon_blocks")
            ):
                logger.debug(
                    "Skipping pass: alias_named_entities (profile or TOON blocks)"
                )
            else:
                nlp_model = self._get_linguistic_nlp_model()
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                placeholder_ranges = _preservation.get_placeholder_ranges(
                    self, result, preserved
                )
                (
                    updated,
                    applied,
                    legend_entries,
                    legend_savings,
                ) = _entity_aliasing.alias_named_entities(
                    result,
                    nlp_model=nlp_model,
                    placeholder_ranges=placeholder_ranges,
                    token_counter=self.count_tokens,
                    min_occurrences=config.ENTITY_ALIAS_MIN_OCCURRENCES,
                    min_chars=config.ENTITY_ALIAS_MIN_CHARS,
                    max_aliases=config.ENTITY_ALIAS_MAX_ALIASES,
                    alias_prefix=config.ENTITY_ALIAS_PREFIX,
                    allowed_labels=config.ENTITY_ALIAS_LABELS,
                    reserved_tokens=list(self._get_placeholder_tokens(preserved)),
                )
                if applied and legend_entries:
                    result = updated
                    glossary_collector.add_entries(
                        "Aliases",
                        legend_entries,
                        net_savings=legend_savings,
                    )
                    self._record_technique("Entity Aliasing")
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "alias_named_entities",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: alias_named_entities")

        # Pass 7: Compress repeated long fragments
        if not should_skip_pass("compress_repeated_fragments"):
            before_text = result
            before = len(result)
            with (
                profiler.step("compress_repeated_fragments")
                if profiler
                else nullcontext()
            ):
                result = self._compress_repeated_fragments(result, preserved)
            if len(result) < before:
                self.techniques_applied.append("Repeated Fragment Compression")
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: compress_repeated_fragments")

        dedup_enabled = not should_skip_pass("deduplicate_content")
        # Pass 8: Deduplicate sentences/phrases
        if dedup_enabled and not self._skip_sentence_deduplication:
            before_text = result
            before = len(result)
            # Only measure tokens and timing if telemetry is enabled
            if telemetry_collector:
                tokens_before = pass_tokens_before(result)
                pass_start = time.perf_counter()
            with profiler.step("deduplicate_content") if profiler else nullcontext():
                result = self._deduplicate_content(result)
            short_circuit = getattr(self, "_last_dedup_short_circuit", False)
            if telemetry_collector:
                pass_duration = (time.perf_counter() - pass_start) * 1000.0
                tokens_after, estimated_tokens_after, exact_tokens_after = (
                    pass_tokens_after(before_text, result)
                )
                record_pass_telemetry(
                    "deduplicate_content",
                    pass_duration,
                    tokens_before,
                    tokens_after,
                    estimated_tokens_after=estimated_tokens_after,
                    exact_tokens_after=exact_tokens_after,
                )
                _record_telemetry_flag(
                    telemetry_collector,
                    "deduplicate_content_short_circuit",
                    short_circuit,
                )
            if profiler:
                profiler.record_flag(
                    "deduplicate_content.short_circuit", 1.0 if short_circuit else 0.0
                )
            if len(result) < before:
                self.techniques_applied.append("Content Deduplication")
            update_noop(before_text, result)
        elif not dedup_enabled:
            logger.debug("Skipping pass: deduplicate_content")
        else:
            logger.debug(
                "Deferring pass: deduplicate_content due to pending abbreviation updates"
            )

        hoist_enabled = not should_skip_pass("hoist_constraints")
        if hoist_enabled:
            if resolved_profile.name in {"code", "json"}:
                logger.debug("Skipping pass: hoist_constraints (code/json profile)")
            else:
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                hoisted = False
                with profiler.step("hoist_constraints") if profiler else nullcontext():
                    result, hoisted = self._hoist_constraints(result)
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "hoist_constraints",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if hoisted:
                    self._record_technique("Constraint Hoisting")
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: hoist_constraints")

        paragraph_dedup_enabled = not should_skip_pass("paragraph_semantic_dedup")
        if paragraph_dedup_enabled:
            if resolved_profile.name in {"code", "json"}:
                logger.debug(
                    "Skipping pass: paragraph_semantic_dedup (code/json profile)"
                )
            elif self.count_tokens(result) <= 2000:
                logger.debug(
                    "Skipping pass: paragraph_semantic_dedup (below token threshold)"
                )
            else:
                before_text = result
                deduped_result, para_applied, para_removed = (
                    self._apply_paragraph_semantic_dedup(
                        result,
                        query_hint=query_hint,
                        preserved=preserved,
                    )
                )
                if para_applied:
                    result = deduped_result
                    self._record_technique("Paragraph Semantic Deduplication")
                    _record_telemetry_flag(
                        telemetry_collector,
                        "paragraph_semantic_dedup_removed",
                        para_removed,
                    )
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: paragraph_semantic_dedup")

        heavy_pass_names = (
            "compress_examples",
            "summarize_history",
            "prune_low_entropy",
            "semantic_candidate_selection",
        )
        heavy_latency_budget_ms = _HEAVY_LATENCY_BUDGET_MS_BY_MODE.get(
            optimization_mode,
            _HEAVY_LATENCY_BUDGET_MS_BY_MODE.get("balanced", 220.0),
        )
        heavy_tokens_now = self.count_tokens(result)
        token_bin = _telemetry.token_bin_for_count(heavy_tokens_now)
        utility_priors = _telemetry.get_pass_utility_priors(
            content_profile=resolved_profile.name,
            optimization_mode=optimization_mode,
            token_bin=token_bin,
        )
        heavy_expected_utilities: Dict[str, float] = {}
        heavy_expected_durations: Dict[str, float] = {}
        for pass_name in heavy_pass_names:
            prior = utility_priors.get(pass_name)
            if prior is not None:
                heavy_expected_utilities[pass_name] = prior.expected_utility
                heavy_expected_durations[pass_name] = prior.expected_duration_ms
            else:
                heavy_expected_utilities[pass_name] = 0.0
                heavy_expected_durations[pass_name] = 0.0

        remaining_heavy_budget_ms = heavy_latency_budget_ms

        def _record_skipped_heavy_pass(pass_name: str, reason: str) -> None:
            if not telemetry_collector:
                return
            current_tokens = self.count_tokens(result)
            record_pass_telemetry(
                pass_name,
                0.0,
                current_tokens,
                current_tokens,
                expected_utility=heavy_expected_utilities.get(pass_name),
                pass_skipped_reason=reason,
                content_profile=resolved_profile.name,
                optimization_mode=optimization_mode,
                token_bin=token_bin,
            )

        def _heavy_pass_precheck(
            pass_name: str, text: str
        ) -> Tuple[bool, Optional[str]]:
            if pass_name == "compress_examples":
                if not has_example_candidates(text):
                    return False, "precheck_no_candidates"
            elif pass_name == "summarize_history":
                if not has_history_candidates(text):
                    return False, "precheck_no_candidates"
            elif pass_name == "prune_low_entropy":
                if self.count_tokens(text) < max(self.entropy_prune_min_length, 256):
                    return False, "precheck_no_candidates"
                if self._entropy_prune_budget(text) <= 0:
                    return False, "precheck_no_candidates"
            return True, None

        def _should_execute_heavy_pass(pass_name: str) -> Tuple[bool, Optional[str]]:
            if should_skip_pass(pass_name):
                return False, "disabled"
            return True, None

        heavy_prechecked_passes: List[str] = []
        for heavy_name in (
            "compress_examples",
            "summarize_history",
            "prune_low_entropy",
        ):
            eligible, reason = _heavy_pass_precheck(heavy_name, result)
            if eligible:
                heavy_prechecked_passes.append(heavy_name)
            else:
                _record_skipped_heavy_pass(
                    heavy_name, reason or "precheck_no_candidates"
                )

        def _record_semantic_selection(
            *,
            tokens_before: int,
            tokens_after: int,
            semantic_start: Optional[float],
        ) -> None:
            nonlocal remaining_heavy_budget_ms
            if not telemetry_collector or semantic_start is None:
                return
            semantic_ms = (time.perf_counter() - semantic_start) * 1000.0
            semantic_utility = (
                ((tokens_before - tokens_after) / semantic_ms)
                if semantic_ms > 0
                else 0.0
            )
            record_pass_telemetry(
                "semantic_candidate_selection",
                semantic_ms,
                tokens_before,
                tokens_after,
                expected_utility=heavy_expected_utilities.get(
                    "semantic_candidate_selection"
                ),
                actual_utility=semantic_utility,
                content_profile=resolved_profile.name,
                optimization_mode=optimization_mode,
                token_bin=token_bin,
            )
            remaining_heavy_budget_ms = max(
                0.0, remaining_heavy_budget_ms - semantic_ms
            )

        def _run_heavy_pass(pass_name: str) -> None:
            nonlocal result, remaining_heavy_budget_ms
            before_text = result
            before_len = len(before_text)
            tokens_before = (
                pass_tokens_before(before_text) if telemetry_collector else 0
            )
            pass_start = time.perf_counter() if telemetry_collector else None

            if pass_name == "compress_examples":
                with profiler.step("compress_examples") if profiler else nullcontext():
                    settings = self._resolve_multi_candidate_settings(
                        "compress_examples", optimization_mode
                    )
                    max_candidates = settings.get("max_candidates", 1)
                    if max_candidates > 1 and self.semantic_guard_enabled:
                        run_semantic, semantic_skip_reason = _should_execute_heavy_pass(
                            "semantic_candidate_selection"
                        )
                        if run_semantic:
                            semantic_start = (
                                time.perf_counter() if telemetry_collector else None
                            )
                            candidates: List[Tuple[str, str, Dict[str, Any]]] = [
                                ("original", before_text, {})
                            ]
                            default_candidate = self._compress_examples(
                                before_text, preserved
                            )
                            if default_candidate != before_text:
                                candidates.append(("default", default_candidate, {}))
                            aggressive_len = settings.get(
                                "aggressive_summary_max_length", 120
                            )
                            aggressive_candidate = self._compress_examples(
                                before_text,
                                preserved,
                                summary_max_length=aggressive_len,
                            )
                            if aggressive_candidate != before_text:
                                candidates.append(
                                    (
                                        "aggressive",
                                        aggressive_candidate,
                                        {"summary_max_length": aggressive_len},
                                    )
                                )
                            guard_threshold = max(
                                self._resolve_semantic_guard_threshold(),
                                settings.get("min_guard_threshold", 0.0) or 0.0,
                            )
                            result, _meta = self._select_semantic_candidate(
                                before_text,
                                candidates,
                                pass_name="compress_examples",
                                guard_threshold=guard_threshold,
                                telemetry_collector=telemetry_collector,
                            )
                            _record_semantic_selection(
                                tokens_before=tokens_before,
                                tokens_after=self.count_tokens(result),
                                semantic_start=semantic_start,
                            )
                        else:
                            _record_skipped_heavy_pass(
                                "semantic_candidate_selection",
                                semantic_skip_reason or "skipped",
                            )
                            result = self._compress_examples(result, preserved)
                    else:
                        result = self._compress_examples(result, preserved)
                if len(result) < before_len:
                    self.techniques_applied.append("Example Compression")

            elif pass_name == "summarize_history":
                with profiler.step("summarize_history") if profiler else nullcontext():
                    settings = self._resolve_multi_candidate_settings(
                        "summarize_history", optimization_mode
                    )
                    max_candidates = settings.get("max_candidates", 1)
                    if max_candidates > 1 and self.semantic_guard_enabled:
                        run_semantic, semantic_skip_reason = _should_execute_heavy_pass(
                            "semantic_candidate_selection"
                        )
                        if run_semantic:
                            semantic_start = (
                                time.perf_counter() if telemetry_collector else None
                            )
                            candidates: List[Tuple[str, str, Dict[str, Any]]] = [
                                ("original", before_text, {})
                            ]
                            default_candidate = _history.summarize_history(
                                self, before_text
                            )
                            if default_candidate != before_text:
                                candidates.append(("default", default_candidate, {}))
                            aggressive_modifier = settings.get(
                                "aggressive_keep_ratio_modifier", 0.75
                            )
                            aggressive_candidate = (
                                self._summarize_history_with_modifier(
                                    before_text, aggressive_modifier
                                )
                            )
                            if aggressive_candidate != before_text:
                                candidates.append(
                                    (
                                        "aggressive",
                                        aggressive_candidate,
                                        {"keep_ratio_modifier": aggressive_modifier},
                                    )
                                )
                            guard_threshold = max(
                                self._resolve_semantic_guard_threshold(),
                                settings.get("min_guard_threshold", 0.0) or 0.0,
                            )
                            result, _meta = self._select_semantic_candidate(
                                before_text,
                                candidates,
                                pass_name="summarize_history",
                                guard_threshold=guard_threshold,
                                telemetry_collector=telemetry_collector,
                            )
                            _record_semantic_selection(
                                tokens_before=tokens_before,
                                tokens_after=self.count_tokens(result),
                                semantic_start=semantic_start,
                            )
                        else:
                            _record_skipped_heavy_pass(
                                "semantic_candidate_selection",
                                semantic_skip_reason or "skipped",
                            )
                            result = _history.summarize_history(self, result)
                    else:
                        result = _history.summarize_history(self, result)
                if self._per_pass_guard_enabled(optimization_mode):
                    result, _rollback, _similarity = (
                        self._apply_per_pass_semantic_guard(
                            "summarize_history",
                            before_text,
                            result,
                            guard_threshold=self._resolve_semantic_guard_threshold(),
                            telemetry_collector=telemetry_collector,
                        )
                    )
                if len(result) < before_len:
                    self.techniques_applied.append("History Summarization")

            elif pass_name == "prune_low_entropy":
                with profiler.step("prune_low_entropy") if profiler else nullcontext():
                    settings = self._resolve_multi_candidate_settings(
                        "prune_low_entropy", optimization_mode
                    )
                    max_candidates = settings.get("max_candidates", 1)
                    if max_candidates > 1 and self.semantic_guard_enabled:
                        run_semantic, semantic_skip_reason = _should_execute_heavy_pass(
                            "semantic_candidate_selection"
                        )
                        if run_semantic:
                            semantic_start = (
                                time.perf_counter() if telemetry_collector else None
                            )
                            candidates: List[Tuple[str, str, Dict[str, Any]]] = [
                                ("original", before_text, {})
                            ]
                            default_candidate = self._maybe_prune_low_entropy(
                                before_text
                            )
                            if default_candidate != before_text:
                                candidates.append(("default", default_candidate, {}))
                            aggressive_ratio = settings.get(
                                "aggressive_ratio_multiplier", 1.25
                            )
                            aggressive_max_ratio = settings.get(
                                "aggressive_max_ratio_multiplier", 1.15
                            )
                            aggressive_candidate = (
                                self._prune_low_entropy_with_ratio_multiplier(
                                    before_text,
                                    aggressive_ratio,
                                    aggressive_max_ratio,
                                )
                            )
                            if aggressive_candidate != before_text:
                                candidates.append(
                                    (
                                        "aggressive",
                                        aggressive_candidate,
                                        {
                                            "ratio_multiplier": aggressive_ratio,
                                            "max_ratio_multiplier": aggressive_max_ratio,
                                        },
                                    )
                                )
                            guard_threshold = max(
                                self._resolve_semantic_guard_threshold(),
                                settings.get("min_guard_threshold", 0.0) or 0.0,
                            )
                            pruned, _meta = self._select_semantic_candidate(
                                before_text,
                                candidates,
                                pass_name="prune_low_entropy",
                                guard_threshold=guard_threshold,
                                telemetry_collector=telemetry_collector,
                            )
                            _record_semantic_selection(
                                tokens_before=tokens_before,
                                tokens_after=self.count_tokens(pruned),
                                semantic_start=semantic_start,
                            )
                        else:
                            _record_skipped_heavy_pass(
                                "semantic_candidate_selection",
                                semantic_skip_reason or "skipped",
                            )
                            pruned = self._maybe_prune_low_entropy(result)
                    else:
                        pruned = self._maybe_prune_low_entropy(result)
                if self._per_pass_guard_enabled(optimization_mode):
                    pruned, _rollback, _similarity = (
                        self._apply_per_pass_semantic_guard(
                            "prune_low_entropy",
                            before_text,
                            pruned,
                            guard_threshold=self._resolve_semantic_guard_threshold(),
                            telemetry_collector=telemetry_collector,
                        )
                    )
                if pruned != before_text:
                    result = pruned
                    self._record_technique("Entropy Pruning")

            if telemetry_collector and pass_start is not None:
                pass_duration = (time.perf_counter() - pass_start) * 1000.0
                tokens_after, estimated_tokens_after, exact_tokens_after = (
                    pass_tokens_after(before_text, result)
                )
                actual_utility = (
                    ((tokens_before - tokens_after) / pass_duration)
                    if pass_duration > 0
                    else 0.0
                )
                record_pass_telemetry(
                    pass_name,
                    pass_duration,
                    tokens_before,
                    tokens_after,
                    estimated_tokens_after=estimated_tokens_after,
                    exact_tokens_after=exact_tokens_after,
                    expected_utility=heavy_expected_utilities.get(pass_name),
                    actual_utility=actual_utility,
                    content_profile=resolved_profile.name,
                    optimization_mode=optimization_mode,
                    token_bin=token_bin,
                )
                remaining_heavy_budget_ms = max(
                    0.0, remaining_heavy_budget_ms - pass_duration
                )

            update_noop(before_text, result)

        heavy_execution_order = sorted(
            heavy_prechecked_passes,
            key=lambda name: (-heavy_expected_utilities.get(name, 0.0), name),
        )
        _record_telemetry_flag(
            telemetry_collector,
            "heavy_execution_order",
            heavy_execution_order,
        )

        for heavy_pass_name in heavy_execution_order:
            run_heavy, skip_reason = _should_execute_heavy_pass(heavy_pass_name)
            if run_heavy:
                _run_heavy_pass(heavy_pass_name)
            elif skip_reason:
                _record_skipped_heavy_pass(heavy_pass_name, skip_reason)

        if (
            config.TOKEN_CLASSIFIER_POST_PASS_ENABLED
            and self.semantic_guard_enabled
            and self.token_classifier_model
        ):
            before_text = result
            post_text, post_applied, post_meta = self._optimize_with_token_classifier(
                before_text,
                force_preserve_digits=force_preserve_digits,
                json_policy=json_policy,
                content_type=content_type,
                content_profile=content_profile,
                min_confidence_override=config.TOKEN_CLASSIFIER_POST_MIN_CONFIDENCE,
                min_keep_ratio_override=config.TOKEN_CLASSIFIER_POST_MIN_KEEP_RATIO,
            )
            guard_passed = False
            similarity = None
            if post_applied and post_text != before_text:
                similarity = self._require_semantic_guard_similarity(
                    before_text,
                    post_text,
                    embedding_cache=self._get_state().embedding_cache,
                )
                guard_threshold = self._resolve_semantic_guard_threshold()
                if similarity >= guard_threshold:
                    guard_passed = True
                    result = post_text
                    self._record_technique("Token Classification Compression (Post)")
            _record_telemetry_flag(
                telemetry_collector,
                "token_classifier_post_applied",
                post_applied,
            )
            _record_telemetry_flag(
                telemetry_collector,
                "token_classifier_post_min_keep_ratio",
                config.TOKEN_CLASSIFIER_POST_MIN_KEEP_RATIO,
            )
            _record_telemetry_flag(
                telemetry_collector,
                "token_classifier_post_min_confidence",
                config.TOKEN_CLASSIFIER_POST_MIN_CONFIDENCE,
            )
            _record_telemetry_flag(
                telemetry_collector,
                "token_classifier_post_guard_passed",
                guard_passed,
            )
            if similarity is not None:
                _record_telemetry_flag(
                    telemetry_collector,
                    "token_classifier_post_guard_similarity",
                    similarity,
                )
            if post_meta:
                _record_telemetry_flag(
                    telemetry_collector,
                    "token_classifier_post_keep_ratio",
                    post_meta.get("keep_ratio"),
                )

        if glossary_collector.has_entries():
            legend = glossary_collector.build_legend(self.count_tokens)
            if legend:
                result = _lexical._prepend_legend(result, legend)
                self._record_technique("Legend Consolidation")

        # Pass 13: Final normalization (whitespace + punctuation)
        compress_punctuation_enabled = not should_skip_pass("compress_punctuation")
        final_whitespace_enabled = not should_skip_pass("final_whitespace")
        normalize_text_enabled = (
            compress_punctuation_enabled or final_whitespace_enabled
        )
        normalize_text_deferred = False

        if normalize_text_enabled:
            if self._skip_sentence_deduplication:
                normalize_text_deferred = True
                logger.debug(
                    "Deferring pass: normalize_text until deduplication completes"
                )
            else:
                if telemetry_collector:
                    tokens_before = pass_tokens_before(result)
                    pass_start = time.perf_counter()
                before_text = result
                with profiler.step("normalize_text") if profiler else nullcontext():
                    result, punctuation_changed = self._normalize_text(
                        result,
                        normalize_whitespace=final_whitespace_enabled,
                        compress_punctuation=compress_punctuation_enabled,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "normalize_text",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if compress_punctuation_enabled and punctuation_changed:
                    self._record_technique("Punctuation Compression")
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: normalize_text")

        # Pass 14: Restore preserved elements
        if not should_skip_pass("restore_preserved"):
            before_text = result
            with profiler.step("restore_preserved") if profiler else nullcontext():
                result = _preservation.restore(self, result, preserved)
            update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: restore_preserved")

        # Pass 15: Alias repeated preserved elements (post-restore)
        if not should_skip_pass("alias_preserved_elements"):
            if resolved_profile.name in {"code", "json"}:
                logger.debug(
                    "Skipping pass: alias_preserved_elements (code/json profile)"
                )
            else:
                before_text = result
                tokens_before = (
                    self.count_tokens(result) if telemetry_collector else None
                )
                pass_start = time.perf_counter() if telemetry_collector else None
                alias_applied = False
                with (
                    profiler.step("alias_preserved_elements")
                    if profiler
                    else nullcontext()
                ):
                    result, alias_applied = self._alias_preserved_elements(
                        result,
                        preserved,
                        token_counter=self.count_tokens,
                    )
                if telemetry_collector:
                    pass_duration = (time.perf_counter() - pass_start) * 1000.0
                    tokens_after, estimated_tokens_after, exact_tokens_after = (
                        pass_tokens_after(before_text, result)
                    )
                    record_pass_telemetry(
                        "alias_preserved_elements",
                        pass_duration,
                        tokens_before,
                        tokens_after,
                        estimated_tokens_after=estimated_tokens_after,
                        exact_tokens_after=exact_tokens_after,
                    )
                if alias_applied:
                    self._record_technique("Preserved Element Aliasing")
                update_noop(before_text, result)
        else:
            logger.debug("Skipping pass: alias_preserved_elements")

        if self._skip_sentence_deduplication:
            if dedup_enabled:
                before_text = result
                before = len(result)
                with (
                    profiler.step("deduplicate_content.deferred")
                    if profiler
                    else nullcontext()
                ):
                    result = self._deduplicate_content(result)
                if len(result) < before:
                    self.techniques_applied.append("Content Deduplication")
                update_noop(before_text, result)
            else:
                logger.debug(
                    "Deduplication disabled; clearing deferred deduplication request"
                )
            self._skip_sentence_deduplication = False

        if normalize_text_enabled and normalize_text_deferred:
            if telemetry_collector:
                tokens_before = pass_tokens_before(result)
                pass_start = time.perf_counter()
            before_text = result
            with profiler.step("normalize_text") if profiler else nullcontext():
                result, punctuation_changed = self._normalize_text(
                    result,
                    normalize_whitespace=final_whitespace_enabled,
                    compress_punctuation=compress_punctuation_enabled,
                )
            if telemetry_collector:
                pass_duration = (time.perf_counter() - pass_start) * 1000.0
                tokens_after, estimated_tokens_after, exact_tokens_after = (
                    pass_tokens_after(before_text, result)
                )
                record_pass_telemetry(
                    "normalize_text",
                    pass_duration,
                    tokens_before,
                    tokens_after,
                    estimated_tokens_after=estimated_tokens_after,
                    exact_tokens_after=exact_tokens_after,
                )
            if compress_punctuation_enabled and punctuation_changed:
                self._record_technique("Punctuation Compression")
            update_noop(before_text, result)

        # Only strip if final_whitespace pass was not skipped
        if final_whitespace_enabled:
            result = result.strip()

        # Final cleanup pass to remove any remaining artifacts
        if final_whitespace_enabled or compress_punctuation_enabled:
            result = _lexical.final_text_cleanup(
                result,
                normalize_whitespace=final_whitespace_enabled,
                compress_punctuation=compress_punctuation_enabled,
            )

        collapsed_tail = _lexical._collapse_consecutive_duplicates_segment(result)
        if collapsed_tail != result:
            result = collapsed_tail
            self._record_technique("Content Deduplication")

        if enable_frequency_learning:
            before_frequency_dedup = result
            result = self._deduplicate_content(result)
            if result != before_frequency_dedup:
                self._record_technique("Content Deduplication")

        if (
            resolved_profile.name not in {"code", "json"}
            and self._estimate_sentence_redundancy_ratio(result) > 0.0
        ):
            before_tail_dedup = result
            result = self._deduplicate_content(result)
            if result != before_tail_dedup:
                self._record_technique("Content Deduplication")

        # Restore original thresholds after content-profile modifiers
        state = self._get_state()
        if state.semantic_guard_threshold_override is not None:
            state.semantic_guard_threshold_override = original_semantic_guard
        else:
            self.semantic_guard_threshold = original_semantic_guard

        self.entropy_prune_ratio = original_entropy_ratio
        self.entropy_prune_max_ratio = original_entropy_max_ratio
        self.near_dup_similarity = original_near_dup_similarity
        self.summarize_keep_ratio_modifier = original_summarize_modifier

        return result

    @staticmethod
    def _needs_whitespace_normalization(text: str) -> bool:
        if not text:
            return False
        if _WHITESPACE_MULTILINE_PATTERN.search(text):
            return True
        if _WHITESPACE_INTERIOR_SPACE_PATTERN.search(text):
            return True
        if _WHITESPACE_TRAILING_PATTERN.search(text):
            return True
        if "\n" not in text:
            return False

        lines = text.split("\n")
        for line in lines[1:]:
            if not line:
                continue
            stripped = line.lstrip()
            if stripped and LIST_MARKER_PATTERN.match(stripped):
                if line != line.rstrip():
                    return True
                continue
            if line != stripped:
                return True
        return False

    def _normalize_text(
        self,
        text: str,
        *,
        normalize_whitespace: bool,
        compress_punctuation: bool,
    ) -> Tuple[str, bool]:
        """Normalize whitespace and punctuation in a single pass."""
        updated = text
        if normalize_whitespace:
            updated = self._normalize_whitespace(updated)

        punctuation_changed = False
        if compress_punctuation:
            punctuated = self._compress_punctuation(updated)
            punctuation_changed = punctuated != updated
            updated = punctuated

        return updated, punctuation_changed

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace - collapse multiple spaces, newlines"""
        # Replace multiple newlines with single newline
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Replace multiple interior spaces with single space while preserving indentation
        text = re.sub(r"(?<=\S) {2,}", " ", text)
        # Remove trailing whitespace on lines
        text = re.sub(r" +\n", "\n", text)
        # Remove leading whitespace on lines (except first line)
        lines = text.split("\n")
        if not lines:
            return text

        normalized_lines = [lines[0]]
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped and LIST_MARKER_PATTERN.match(stripped):
                normalized_lines.append(line.rstrip())
            else:
                normalized_lines.append(stripped)

        return "\n".join(normalized_lines)

    def _compress_boilerplate(self, text: str, preserved: Dict) -> str:
        """Compress boilerplate sections while preserving protected spans."""
        placeholder_ranges = self._get_placeholder_ranges(text, preserved)
        return _lexical.compress_boilerplate(
            text,
            placeholder_ranges=placeholder_ranges,
        )

    def _get_frequency_learning_params(self, token_count: int) -> Dict[str, int]:
        if token_count > 5000:
            return {"min_occurrences": 2, "max_new": 8, "min_phrase_chars": 10}
        if token_count > 2000:
            return {"min_occurrences": 2, "max_new": 6, "min_phrase_chars": 12}
        return {"min_occurrences": 3, "max_new": 5, "min_phrase_chars": 12}

    def _load_learned_phrase_dictionary(
        self, customer_id: Optional[str]
    ) -> Dict[str, str]:
        if not customer_id:
            return {}
        try:
            from database import (
                get_learned_phrase_dictionary,
                is_learned_abbreviations_enabled,
            )
        except ImportError:
            return {}
        if not is_learned_abbreviations_enabled():
            return {}
        return get_learned_phrase_dictionary(customer_id)

    def _persist_learned_phrase_dictionary(
        self, customer_id: Optional[str], learned_map: Dict[str, str]
    ) -> None:
        if not customer_id or not learned_map:
            return
        try:
            from database import (
                is_learned_abbreviations_enabled,
                upsert_learned_phrase_mappings,
            )
        except ImportError:
            return
        if not is_learned_abbreviations_enabled():
            return
        upsert_learned_phrase_mappings(customer_id, learned_map)

    def _record_learned_phrase_usage(
        self, customer_id: Optional[str], phrases: List[str]
    ) -> None:
        if not customer_id or not phrases:
            return
        try:
            from database import (
                is_learned_abbreviations_enabled,
                update_learned_phrase_usage,
            )
        except ImportError:
            return
        if not is_learned_abbreviations_enabled():
            return
        update_learned_phrase_usage(customer_id, phrases)

    def _apply_frequency_abbreviations(
        self,
        text: str,
        canonical_map: Dict[str, str],
        preserved: Dict,
    ) -> Tuple[str, Dict[str, str], Optional[List[Tuple[str, str]]], int]:
        """Learn abbreviations for frequently repeated multi-word phrases."""
        token_count = self.count_tokens(text)
        params = self._get_frequency_learning_params(token_count)
        return _lexical.apply_frequency_abbreviations(
            text,
            canonical_map,
            preserved,
            self._get_placeholder_tokens(preserved),
            config.PLACEHOLDER_PATTERN,
            token_counter=self.count_tokens,
            **params,
        )

    def _compress_coreferences(self, text: str, preserved: Optional[Dict]) -> str:
        """Replace repeated entity mentions with concise aliases or pronouns."""
        if not text or not text.strip():
            return text

        if _ARABIC_CHAR_PATTERN.search(text):
            logger.warning(
                "Skipping coreference compression: Arabic detected in input; coref is English-only"
            )
            return text

        if self.tokenizer is None:
            logger.warning(
                "Skipping coreference compression: tokenizer unavailable for accurate token counts"
            )
            return text

        total_tokens = self.count_tokens(text)
        if _COREF_MAX_TOKENS and total_tokens > _COREF_MAX_TOKENS:
            logger.warning(
                "Skipping coreference compression: %s tokens exceeds %s-token model limit",
                total_tokens,
                _COREF_MAX_TOKENS,
            )
            return text

        nlp_token_threshold = self.fastpath_token_threshold or self.chunk_threshold
        if nlp_token_threshold and total_tokens <= nlp_token_threshold:
            logger.debug(
                "Skipping coreference compression: %s tokens below threshold %s",
                total_tokens,
                nlp_token_threshold,
            )
            return text

        coref_model = self._get_coref_model()
        if coref_model is None:
            return text

        try:
            doc_preview = coref_model.make_doc(text)
        except (
            Exception
        ) as exc:  # pragma: no cover - tokenizer failure is environment-specific
            logger.debug("Coreference tokenizer failed: %s", exc)
            return text

        if _COREF_MAX_TOKENS and len(doc_preview) > _COREF_MAX_TOKENS:
            logger.warning(
                "Skipping coreference compression: %s spaCy tokens exceeds %s-token model limit",
                len(doc_preview),
                _COREF_MAX_TOKENS,
            )
            return text

        try:
            # spaCy models are thread-safe for processing (nlp() and nlp.pipe())
            # The lock is only needed for model loading, not for processing
            # Using nlp() directly is safe for concurrent reads
            doc = coref_model(text)
        except (
            Exception
        ) as exc:  # pragma: no cover - runtime failure depends on model availability
            logger.debug("Coreference resolution failed: %s", exc)
            return text

        spans_container = getattr(doc, "spans", None)
        clusters_source: Any = None
        if isinstance(spans_container, dict):
            clusters_source = spans_container.get("coref_clusters")
        elif hasattr(spans_container, "get"):
            clusters_source = spans_container.get("coref_clusters")

        if not clusters_source:
            return text

        if hasattr(clusters_source, "clusters"):
            clusters_iterable = getattr(clusters_source, "clusters")
        else:
            clusters_iterable = clusters_source

        try:
            clusters = list(clusters_iterable)
        except TypeError:
            clusters = [clusters_iterable]

        if not clusters:
            return text

        placeholder_ranges = self._get_placeholder_ranges(text, preserved)
        reserved_aliases = {
            token.lower() for token in self._get_placeholder_tokens(preserved)
        }
        replacements: List[Tuple[int, int, str]] = []

        for cluster in clusters:
            mentions = list(getattr(cluster, "mentions", []))
            if not mentions and hasattr(cluster, "__iter__"):
                mentions = list(cluster)

            if len(mentions) < 2:
                continue

            mentions.sort(key=lambda span: getattr(span, "start_char", 0))
            first_mention = mentions[0]

            if span_overlaps_placeholder(first_mention, placeholder_ranges):
                continue

            first_text = first_mention.text.strip()
            if not first_text:
                continue

            alias = build_coref_alias(first_text, reserved_aliases)
            pronoun = select_coref_pronoun(first_text)
            pronoun_allowed = len(mentions) >= 3
            last_full_end = getattr(first_mention, "end_char", None)

            for mention in mentions[1:]:
                if span_overlaps_placeholder(mention, placeholder_ranges):
                    continue

                start_char = getattr(mention, "start_char", None)
                end_char = getattr(mention, "end_char", None)
                if start_char is None or end_char is None:
                    continue

                mention_text = mention.text
                normalized = mention_text.strip()
                if not normalized:
                    continue

                replacement: Optional[str] = None
                distance = 0
                if last_full_end is not None:
                    distance = start_char - last_full_end

                if (
                    alias
                    and alias.lower() != normalized.lower()
                    and len(alias) <= len(normalized)
                ):
                    replacement = alias
                elif pronoun and pronoun_allowed and distance <= _PRONOUN_MAX_DISTANCE:
                    pronoun_form = pronoun
                    if mention_text.isupper():
                        pronoun_form = pronoun.upper()
                    elif mention_text[:1].isupper():
                        pronoun_form = pronoun.capitalize()

                    if pronoun_form.lower() != normalized.lower() and len(
                        pronoun_form
                    ) <= len(normalized):
                        replacement = pronoun_form

                if not replacement:
                    last_full_end = end_char
                    continue

                replacements.append((start_char, end_char, replacement))
                last_full_end = end_char

        if not replacements:
            return text

        replacements.sort(key=lambda item: item[0], reverse=True)
        updated = text
        for start, end, replacement in replacements:
            updated = updated[:start] + replacement + updated[end:]

        return updated

    def _get_placeholder_tokens(self, preserved: Dict) -> Set[str]:
        """Return placeholder tokens generated during preservation."""
        return _preservation.get_placeholder_tokens(self, preserved)

    def _build_placeholder_normalization_map(self, preserved: Dict) -> Dict[str, str]:
        """Return a deterministic mapping for placeholders based on their values."""
        return _preservation.build_placeholder_normalization_map(self, preserved)

    def _get_placeholder_ranges(
        self, text: str, preserved: Optional[Dict]
    ) -> List[Tuple[int, int]]:
        """Locate preserved placeholder spans to avoid modifying them."""
        return _preservation.get_placeholder_ranges(self, text, preserved)

    def _get_coref_model(self):
        """Load and cache the spaCy coreference model lazily."""
        global _COREF_NLP_SINGLETON

        if self._coref_nlp is not None:
            return self._coref_nlp

        if _COREF_NLP_SINGLETON is not None:
            self._coref_nlp = _COREF_NLP_SINGLETON
            return self._coref_nlp

        if self._coref_load_failed:
            return None

        with self._coref_lock:
            if self._coref_nlp is not None:
                return self._coref_nlp

            if self._coref_load_failed:
                return None

            spacy_module = _import_spacy()
            if spacy_module is None:
                logger.warning(
                    "spaCy is not installed; coreference compression disabled"
                )
                self._coref_load_failed = True
                return None

            try:
                with _COREF_NLP_SINGLETON_LOCK:
                    if _COREF_NLP_SINGLETON is not None:
                        self._coref_nlp = _COREF_NLP_SINGLETON
                        return self._coref_nlp

                    nlp = spacy_module.blank("en")

                    if self._coref_pipe_name not in nlp.pipe_names:
                        if _import_spacy_coref() is None:
                            raise ImportError("spacy-coref is not installed")
                        nlp.add_pipe(self._coref_pipe_name)

                    _COREF_NLP_SINGLETON = nlp
                    self._coref_nlp = nlp
            except (
                Exception
            ) as exc:  # pragma: no cover - load failures depend on environment
                logger.warning(
                    "Failed to load coreference model %s (pipe %s): %s",
                    self._coref_model_name,
                    self._coref_pipe_name,
                    exc,
                )
                self._coref_load_failed = True
                self._coref_nlp = None

            return self._coref_nlp

    def _get_repetition_params(self, token_count: int) -> Tuple[int, int]:
        if token_count > 5000:
            return 15, self.repeat_min_occurrences
        if token_count > 2000:
            return 18, self.repeat_min_occurrences
        return self.repeat_min_tokens, self.repeat_min_occurrences

    def _compress_repeated_fragments(self, text: str, preserved: Dict) -> str:
        """Compress repeated long fragments by referencing later occurrences."""

        token_matches = list(re.finditer(r"\S+", text))
        token_count = len(token_matches)
        min_tokens, min_occurrences = self._get_repetition_params(token_count)
        if token_count < min_tokens * min_occurrences:
            return text

        tokens = [match.group(0) for match in token_matches]
        spans = [match.span() for match in token_matches]

        placeholder_normalization = self._build_placeholder_normalization_map(preserved)
        analysis_tokens = [
            placeholder_normalization.get(token, token) for token in tokens
        ]

        fragments = self._repetition_detector.find_repetitions(
            analysis_tokens,
            min_length=min_tokens,
            min_occurrences=min_occurrences,
        )

        if not fragments:
            return text

        fragments.sort(
            key=lambda fragment: (
                -len(fragment.positions),
                -fragment.length,
                fragment.positions[0],
            )
        )

        replacements: List[Tuple[int, int, str]] = []
        occupied_tokens: Set[int] = set()
        protected_tokens: Set[int] = set()

        for fragment in fragments:
            if fragment.length < min_tokens or len(fragment.positions) < 2:
                continue

            first_start = fragment.positions[0]
            first_range = range(first_start, first_start + fragment.length)
            if any(index in occupied_tokens for index in first_range):
                continue

            protected_tokens.update(first_range)

            for start in fragment.positions[1:]:
                token_range = range(start, start + fragment.length)
                if any(
                    index in protected_tokens or index in occupied_tokens
                    for index in token_range
                ):
                    continue

                if start >= len(spans):
                    continue

                end_token_index = start + fragment.length - 1
                if end_token_index >= len(spans):
                    continue

                start_char = spans[start][0]
                end_char = spans[end_token_index][1]

                trailing_extension = end_char
                trailing_whitespace = ""
                while (
                    trailing_extension < len(text)
                    and text[trailing_extension].isspace()
                ):
                    trailing_whitespace += text[trailing_extension]
                    trailing_extension += 1

                prefix = ""
                if start_char > 0 and not text[start_char - 1].isspace():
                    prefix = " "

                suffix = trailing_whitespace
                if (
                    not suffix
                    and trailing_extension < len(text)
                    and not text[trailing_extension].isspace()
                ):
                    suffix = " "

                replacement_text = f"{prefix}{suffix}"

                replacements.append((start_char, trailing_extension, replacement_text))
                occupied_tokens.update(token_range)

        if not replacements:
            return text

        replacements.sort(key=lambda item: item[0], reverse=True)
        updated_text = text
        for start_char, end_char, replacement_text in replacements:
            updated_text = (
                updated_text[:start_char] + replacement_text + updated_text[end_char:]
            )

        return updated_text

    def warm_up(self) -> None:
        """Pre-load optional NLP models to avoid cold-start latency."""

        status: Dict[str, Any] = {
            "spacy": {"name": self._spacy_model_name, "loaded": False},
            "coreference": {"name": self._coref_model_name, "loaded": False},
            "semantic_guard": {"name": self.semantic_guard_model, "loaded": False},
            "semantic_rank": {"name": self.semantic_rank_model, "loaded": False},
            "entropy": {
                "name": getattr(_entropy, "_ENTROPY_MODEL_NAME", None),
                "loaded": False,
            },
            "entropy_fast": {
                "name": getattr(_entropy, "_ENTROPY_FAST_MODEL_NAME", None),
                "loaded": False,
            },
            "token_classifier": {
                "name": self.token_classifier_model,
                "loaded": False,
            },
        }

        if self._nlp_load_failed:
            self._nlp_load_failed = False

        def _warm_spacy() -> bool:
            try:
                model = self._get_nlp_model()
                if model is not None:
                    logger.info("Prompt optimizer spaCy model loaded during warm-up")
                    return True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load spaCy model: %s", exc
                )
            return False

        def _warm_coreference() -> bool:
            try:
                coref_model = self._get_coref_model()
                if coref_model is not None:
                    logger.info(
                        "Prompt optimizer coreference model loaded during warm-up"
                    )
                    return True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load coreference model: %s", exc
                )
            return False

        def _warm_semantic_guard() -> bool:
            if not self.semantic_guard_model:
                return False
            try:
                _metrics.warm_up(self.semantic_guard_model)
                semantic_probe = _metrics.score_similarity(
                    "warmup", "warmup", self.semantic_guard_model
                )
                return semantic_probe is not None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load semantic guard model: %s",
                    exc,
                )
                return False

        def _warm_semantic_rank() -> bool:
            if not self.semantic_rank_model:
                return False
            try:
                _metrics.warm_up(
                    self.semantic_rank_model,
                    model_type="semantic_rank",
                )
                semantic_rank_probe = _metrics.score_similarity(
                    "warmup",
                    "warmup",
                    self.semantic_rank_model,
                    model_type="semantic_rank",
                )
                return semantic_rank_probe is not None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load semantic rank model: %s",
                    exc,
                )
                return False

        def _warm_entropy_models() -> Tuple[bool, bool]:
            entropy_fast_loaded = False
            entropy_loaded = False
            try:
                fast_scorer = _entropy._get_fast_scorer()
                entropy_fast_loaded = bool(getattr(fast_scorer, "available", False))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load fast entropy model: %s",
                    exc,
                )

            try:
                scorer = _entropy._get_scorer()
                entropy_loaded = bool(getattr(scorer, "available", False))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load entropy model: %s", exc
                )
            return entropy_fast_loaded, entropy_loaded

        def _warm_token_classifier() -> bool:
            try:
                classifier = _token_classifier._get_classifier(
                    self.token_classifier_model
                )
                return bool(getattr(classifier, "available", False))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Prompt optimizer warm-up failed to load token classifier: %s", exc
                )
                return False

        semantic_rank_reuses_guard = (
            bool(self.semantic_guard_model)
            and self.semantic_rank_model == self.semantic_guard_model
        )

        workers = min(3, max(1, os.cpu_count() or 1))
        futures: Dict[str, Any] = {}
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="optimizer-warmup"
        ) as pool:
            futures["spacy"] = pool.submit(_warm_spacy)
            futures["coreference"] = pool.submit(_warm_coreference)
            futures["semantic_guard"] = pool.submit(_warm_semantic_guard)
            if not semantic_rank_reuses_guard and self.semantic_rank_model:
                futures["semantic_rank"] = pool.submit(_warm_semantic_rank)
            futures["entropy_pair"] = pool.submit(_warm_entropy_models)
            futures["token_classifier"] = pool.submit(_warm_token_classifier)

            semantic_guard_loaded = bool(futures["semantic_guard"].result())
            status["semantic_guard"]["loaded"] = semantic_guard_loaded
            if semantic_rank_reuses_guard:
                status["semantic_rank"]["loaded"] = semantic_guard_loaded

            status["spacy"]["loaded"] = bool(futures["spacy"].result())
            status["coreference"]["loaded"] = bool(futures["coreference"].result())

            if "semantic_rank" in futures:
                status["semantic_rank"]["loaded"] = bool(
                    futures["semantic_rank"].result()
                )

            entropy_fast_loaded, entropy_loaded = futures["entropy_pair"].result()
            status["entropy_fast"]["loaded"] = bool(entropy_fast_loaded)
            status["entropy"]["loaded"] = bool(entropy_loaded)

            status["token_classifier"]["loaded"] = bool(
                futures["token_classifier"].result()
            )

        status["__warmup_epoch"] = int(time.time())
        self._model_load_status = status

    def probe_model_readiness(self, model_type: str) -> Dict[str, Any]:
        """Lightweight readiness probe for a single optimizer-backed model type."""

        status: Dict[str, Any] = {}

        if model_type == "spacy":
            status = {"name": self._spacy_model_name, "loaded": False}
            try:
                status["loaded"] = self._get_nlp_model() is not None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("spaCy readiness probe failed: %s", exc)

        elif model_type == "coreference":
            status = {"name": self._coref_model_name, "loaded": False}
            try:
                status["loaded"] = self._get_coref_model() is not None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Coreference readiness probe failed: %s", exc)

        elif model_type in {"semantic_guard", "semantic_rank"}:
            semantic_model_name = (
                self.semantic_guard_model
                if model_type == "semantic_guard"
                else self.semantic_rank_model
            )
            if not semantic_model_name:
                try:
                    from services.model_cache_manager import get_model_configs

                    semantic_model_name = (
                        get_model_configs().get(model_type, {}).get("model_name")
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to resolve %s model name for readiness probe: %s",
                        model_type,
                        exc,
                    )
            status = {"name": semantic_model_name, "loaded": False}
            if not semantic_model_name:
                logger.warning(
                    "%s readiness probe skipped: no model configured",
                    model_type,
                )
            else:
                try:
                    _metrics.warm_up(semantic_model_name, model_type=model_type)
                    semantic_probe = _metrics.score_similarity(
                        "warmup", "warmup", semantic_model_name, model_type=model_type
                    )
                    status["loaded"] = semantic_probe is not None
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("%s readiness probe failed: %s", model_type, exc)
        elif model_type in {"entropy", "entropy_fast"}:
            entropy_key = "entropy_fast" if model_type == "entropy_fast" else "entropy"
            status = {
                "name": getattr(
                    _entropy,
                    (
                        "_ENTROPY_FAST_MODEL_NAME"
                        if entropy_key == "entropy_fast"
                        else "_ENTROPY_MODEL_NAME"
                    ),
                    None,
                ),
                "loaded": False,
            }
            try:
                if entropy_key == "entropy_fast":
                    fast_scorer = _entropy._get_fast_scorer()
                    status["loaded"] = bool(getattr(fast_scorer, "available", False))
                else:
                    scorer = _entropy._get_scorer()
                    status["loaded"] = bool(getattr(scorer, "available", False))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("%s readiness probe failed: %s", model_type, exc)

        elif model_type == "token_classifier":
            status = {"name": self.token_classifier_model, "loaded": False}
            try:
                classifier = _token_classifier._get_classifier(
                    self.token_classifier_model
                )
                status["loaded"] = bool(getattr(classifier, "available", False))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Token classifier readiness probe failed: %s", exc)

        if status:
            merged = dict(self._model_load_status)
            merged[model_type] = status
            merged["__warmup_epoch"] = int(time.time())
            self._model_load_status = merged

        return status

    def model_status(self) -> Dict[str, Any]:
        """Return cached model load status from the last warm-up."""
        return dict(self._model_load_status)

    def _get_nlp_model(self):
        """Load and cache the spaCy model for semantic deduplication."""
        if not self.enable_semantic_deduplication:
            return None

        if self._nlp is not None:
            return self._nlp

        if self._nlp_load_failed:
            return None

        with self._nlp_lock:
            if self._nlp is not None:
                return self._nlp

            if self._nlp_load_failed:
                return None

            spacy_module = _import_spacy()
            if spacy_module is None:
                logger.warning(
                    "spaCy is not installed; semantic deduplication disabled"
                )
                self._nlp_load_failed = True
                return None

            try:
                model_target = self._resolve_spacy_model_target()
                nlp = None
                load_exc = None
                try:
                    nlp = spacy_module.load(model_target)
                except Exception as exc:
                    load_exc = exc
                if nlp is None:
                    raise load_exc or RuntimeError("spaCy model load failed")
                disable_candidates = (
                    "tagger",
                    "parser",
                    "attribute_ruler",
                    "lemmatizer",
                    "ner",
                )
                available_disable = [
                    pipe for pipe in disable_candidates if pipe in nlp.pipe_names
                ]
                if available_disable:
                    nlp.disable_pipes(*available_disable)
                self._nlp_disabled_pipes = available_disable
                self._nlp_pipe_names = list(nlp.pipe_names)
                logger.debug(
                    "Loaded spaCy model %s with active pipes: %s | disabled pipes: %s",
                    self._spacy_model_name,
                    ", ".join(self._nlp_pipe_names) or "<none>",
                    ", ".join(self._nlp_disabled_pipes) or "<none>",
                )
                self._nlp = nlp
            except (
                Exception
            ) as exc:  # pragma: no cover - load failures are environment issues
                logger.warning(
                    "Failed to load spaCy model %s: %s", self._spacy_model_name, exc
                )
                self._nlp_load_failed = True
                self._nlp = None
                self._nlp_disabled_pipes = []
                self._nlp_pipe_names = []

            return self._nlp

    def _get_linguistic_nlp_model(self):
        """Load spaCy model with parser/NER enabled for linguistic trimming."""
        if self._linguistic_nlp is not None:
            return self._linguistic_nlp

        if self._linguistic_nlp_load_failed:
            return None

        with self._nlp_lock:
            if self._linguistic_nlp is not None:
                return self._linguistic_nlp

            if self._linguistic_nlp_load_failed:
                return None

            spacy_module = _import_spacy()
            if spacy_module is None:
                logger.warning("spaCy is not installed; linguistic trimming disabled")
                self._linguistic_nlp_load_failed = True
                return None

            try:
                model_target = self._resolve_spacy_model_target()
                nlp = spacy_module.load(model_target)
                if "parser" not in nlp.pipe_names:
                    logger.warning(
                        "spaCy model %s lacks parser; linguistic trimming disabled",
                        self._spacy_model_name,
                    )
                    self._linguistic_nlp_load_failed = True
                    return None
                keep_pipes = {"parser", "ner", "senter"}
                disable = [pipe for pipe in nlp.pipe_names if pipe not in keep_pipes]
                if disable:
                    nlp.disable_pipes(*disable)
                self._linguistic_nlp_pipe_names = list(nlp.pipe_names)
                self._linguistic_nlp = nlp
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning(
                    "Failed to load linguistic spaCy model %s: %s",
                    self._spacy_model_name,
                    exc,
                )
                self._linguistic_nlp_load_failed = True
                self._linguistic_nlp = None
                self._linguistic_nlp_pipe_names = []

            return self._linguistic_nlp

    def _compute_minhash_signature(self, tokens: List[str]):
        """Return a MinHash signature for the provided tokens if supported."""
        if not tokens or MinHash is None:
            return None

        signature = MinHash(num_perm=self._minhash_num_perm)
        for token in tokens:
            signature.update(token.encode("utf-8"))
        return signature

    @staticmethod
    def _cosine_similarity(
        vector_a: Optional[np.ndarray], vector_b: Optional[np.ndarray]
    ) -> float:
        if np is None or vector_a is None or vector_b is None:
            return -1.0

        # Compute proper cosine similarity: dot(a, b) / (||a|| * ||b||)
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        # Epsilon guard for zero norms
        if norm_a == 0.0 or norm_b == 0.0:
            return -1.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def _lexical_similarity(original: str, optimized: str) -> float:
        if not original or not optimized:
            return 0.0
        original_tokens = set(re.findall(r"\b\w+\b", original.lower()))
        optimized_tokens = set(re.findall(r"\b\w+\b", optimized.lower()))
        if not original_tokens or not optimized_tokens:
            return 0.0

        overlap = 0
        for token in original_tokens:
            if token in optimized_tokens:
                overlap += 1
        union = len(original_tokens) + len(optimized_tokens) - overlap
        word_similarity = overlap / union if union > 0 else 0.0

        length_ratio = (
            min(len(optimized) / len(original), 1.0) if len(original) else 0.0
        )
        original_sentence_count = len(re.findall(r"[.!?]", original)) or 1
        optimized_sentence_count = len(re.findall(r"[.!?]", optimized)) or 1
        sentence_ratio = min(optimized_sentence_count / original_sentence_count, 1.0)
        similarity = (
            (word_similarity * 0.7) + (length_ratio * 0.15) + (sentence_ratio * 0.15)
        )
        return round(similarity, 3)

    def _compute_semantic_similarity(
        self, original: str, optimized: str
    ) -> Optional[float]:
        """Semantic similarity for history/analytics storage."""
        if not original or not optimized:
            return None

        try:
            embedding_cache = self._get_state().embedding_cache
            score = _metrics.score_similarity(
                original,
                optimized,
                self.semantic_guard_model,
                embedding_cache=embedding_cache,
            )
            if score is not None:
                return float(score)
        except Exception:
            return None
        return None

    def _vectorize_sentences(
        self, sentences: Sequence[str], nlp_model
    ) -> List[Optional[np.ndarray]]:
        if np is None:
            return [None for _ in sentences]
        if not sentences:
            return []
        with self._nlp_lock:
            docs = list(nlp_model.pipe(sentences))
        vectors: List[Optional[np.ndarray]] = []
        for doc in docs:
            if not doc.has_vector:
                vectors.append(None)
                continue
            vector = doc.vector
            norm = np.linalg.norm(vector)
            if norm == 0:
                vectors.append(None)
                continue
            vectors.append(vector / norm)
        return vectors

    def _remove_verbatim_duplicate_blocks(
        self,
        text: str,
        preserved: Optional[Dict[str, Any]] = None,
        min_block_length: int = _VERBATIM_BLOCK_MIN_LENGTH,
    ) -> Tuple[str, bool]:
        """
        Remove large verbatim duplicate text blocks before any transformations.

        This catches complete paragraph duplications that would be missed by
        sentence-level deduplication after abbreviations are applied.

        Args:
            text: Input text to deduplicate
            preserved: Dictionary of preserved content with placeholders
            min_block_length: Minimum character length for a block to be considered
                             for duplicate detection (default 100 chars)

        Returns:
            Tuple of (deduplicated text, whether any duplicates were removed)
        """
        if len(text) < min_block_length * 2:
            return text, False

        # Get placeholder ranges to avoid breaking preserved content
        placeholder_ranges: List[Tuple[int, int]] = []
        if preserved:
            try:
                placeholder_ranges = self._get_placeholder_ranges(text, preserved)
            except Exception:
                placeholder_ranges = []

        def overlaps_placeholder(start: int, end: int) -> bool:
            for p_start, p_end in placeholder_ranges:
                if start < p_end and end > p_start:
                    return True
            return False

        # Strategy 1: Split by paragraph boundaries (double newlines)
        paragraph_pattern = re.compile(r"\n\s*\n")
        paragraphs = paragraph_pattern.split(text)

        if len(paragraphs) > 1:
            seen_hashes: Dict[str, int] = {}
            result_paragraphs: List[str] = []
            removed_any = False

            for i, paragraph in enumerate(paragraphs):
                stripped = paragraph.strip()
                if len(stripped) < min_block_length:
                    result_paragraphs.append(paragraph)
                    continue

                # Normalize for comparison (lowercase, collapse whitespace)
                normalized = " ".join(stripped.lower().split())
                block_hash = hashlib.md5(normalized.encode()).hexdigest()

                if block_hash not in seen_hashes:
                    seen_hashes[block_hash] = i
                    result_paragraphs.append(paragraph)
                else:
                    logger.debug(
                        "Removed verbatim duplicate paragraph (%d chars) at position %d, "
                        "first seen at position %d",
                        len(stripped),
                        i,
                        seen_hashes[block_hash],
                    )
                    removed_any = True

            if removed_any:
                return "\n\n".join(result_paragraphs), True

        # Strategy 2: Detect duplicates within continuous text using rolling comparison
        # This handles cases where duplicates are on the same line without paragraph breaks
        words = text.split()
        n = len(words)
        window_size = _VERBATIM_SLIDING_WINDOW_TOKENS

        if n < window_size * 2:
            return text, False

        # Build hash index of all windows
        window_hashes: Dict[str, List[int]] = {}
        for start in range(n - window_size + 1):
            window = words[start : start + window_size]
            window_key = " ".join(w.lower() for w in window)
            window_hash = hashlib.md5(window_key.encode()).hexdigest()
            if window_hash not in window_hashes:
                window_hashes[window_hash] = []
            window_hashes[window_hash].append(start)

        # Find duplicate sequences that can be extended
        removal_ranges: List[Tuple[int, int]] = []
        processed_starts: Set[int] = set()

        for positions in window_hashes.values():
            if len(positions) < 2:
                continue

            # Sort positions and find duplicate occurrences
            positions = sorted(positions)
            first_pos = positions[0]

            for dup_pos in positions[1:]:
                if dup_pos in processed_starts:
                    continue
                # Ensure no overlap with first occurrence
                if dup_pos < first_pos + window_size:
                    continue

                # Extend the match as far as possible
                match_len = window_size
                # Extend while the next word matches and we don't overlap
                while (
                    first_pos + match_len < n
                    and dup_pos + match_len < n
                    and words[first_pos + match_len].lower()
                    == words[dup_pos + match_len].lower()
                ):
                    # Stop if extending would cause overlap with first occurrence
                    if dup_pos + match_len <= first_pos + match_len:
                        break
                    match_len += 1

                # Only remove if it's a significant duplicate
                if match_len >= window_size:
                    removal_ranges.append((dup_pos, dup_pos + match_len))
                    for idx in range(dup_pos, dup_pos + match_len):
                        processed_starts.add(idx)

        if not removal_ranges:
            return text, False

        # Merge overlapping ranges
        removal_ranges.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in removal_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Build result excluding removed ranges
        keep_mask = [True] * n
        for start, end in merged:
            for i in range(start, min(end, n)):
                keep_mask[i] = False

        result_words = [w for w, keep in zip(words, keep_mask) if keep]

        if len(result_words) < n:
            logger.debug(
                "Removed %d words via sliding window duplicate detection",
                n - len(result_words),
            )
            return " ".join(result_words), True

        return text, False

    def _remove_multi_sentence_sequence_duplicates(
        self,
        sentences: List[str],
        directives: List[bool],
        sequence_length: int = 2,
    ) -> Tuple[List[str], List[bool]]:
        """
        Detect and remove sequences of consecutive sentences that repeat.

        Args:
            sentences: List of sentences
            directives: Boolean flags indicating directive sentences
            sequence_length: Number of consecutive sentences to check for duplication

        Returns:
            Tuple of (filtered sentences, filtered directives)
        """
        if len(sentences) < sequence_length * 2:
            return sentences, directives

        seen_sequences: Dict[str, int] = {}
        sentences_to_remove: Set[int] = set()

        for i in range(len(sentences) - sequence_length + 1):
            # Skip if any sentence in this sequence is a directive
            if any(
                i + j < len(directives) and directives[i + j]
                for j in range(sequence_length)
            ):
                continue

            seq = sentences[i : i + sequence_length]
            # Normalize each sentence and join with separator
            seq_key = "|||".join(" ".join(s.lower().split()) for s in seq)
            seq_hash = hashlib.md5(seq_key.encode()).hexdigest()

            if seq_hash in seen_sequences:
                first_occurrence = seen_sequences[seq_hash]
                # Ensure no overlap with first occurrence
                if i >= first_occurrence + sequence_length:
                    # Mark all sentences in this duplicate sequence for removal
                    for j in range(sequence_length):
                        if i + j < len(sentences):
                            sentences_to_remove.add(i + j)
                    logger.debug(
                        "Removed duplicate sentence sequence at index %d "
                        "(first seen at %d)",
                        i,
                        first_occurrence,
                    )
            else:
                seen_sequences[seq_hash] = i

        if not sentences_to_remove:
            return sentences, directives

        filtered_sentences = [
            s for idx, s in enumerate(sentences) if idx not in sentences_to_remove
        ]
        filtered_directives = [
            d for idx, d in enumerate(directives) if idx not in sentences_to_remove
        ]

        return filtered_sentences, filtered_directives

    @staticmethod
    def _deduplicate_exact_lines(text: str) -> Tuple[str, int]:
        if not text or "\n" not in text:
            return text, 0

        lines = text.splitlines(keepends=True)
        if len(lines) < 8:
            return text, 0

        seen: Set[str] = set()
        kept: List[str] = []
        removed = 0

        for line in lines:
            key = line.strip("\r\n").strip()
            if not key:
                kept.append(line)
                continue

            normalized = " ".join(key.lower().split())
            if normalized in seen:
                removed += 1
                continue

            seen.add(normalized)
            kept.append(line)

        if removed <= 0:
            return text, 0

        return "".join(kept), removed

    def _deduplicate_near_sentences(
        self,
        sentences: List[str],
        directives: List[bool],
    ) -> Tuple[List[str], List[bool]]:
        if len(sentences) < 2:
            return sentences, directives

        token_sets = [
            set(re.findall(r"\w+", sentence.lower())) for sentence in sentences
        ]
        sentence_counts = [
            len(re.findall(r"[.!?]", sentence)) or 1 for sentence in sentences
        ]
        keep = [True] * len(sentences)

        def levenshtein_with_cap(a: str, b: str, max_dist: int) -> Optional[int]:
            if a == b:
                return 0
            if not a:
                return len(b) if len(b) <= max_dist else None
            if not b:
                return len(a) if len(a) <= max_dist else None

            if abs(len(a) - len(b)) > max_dist:
                return None

            previous = list(range(len(b) + 1))
            for i, char_a in enumerate(a, start=1):
                current = [i]
                row_min = current[0]
                for j, char_b in enumerate(b, start=1):
                    insertions = current[j - 1] + 1
                    deletions = previous[j] + 1
                    substitutions = previous[j - 1] + (char_a != char_b)
                    value = min(insertions, deletions, substitutions)
                    current.append(value)
                    if value < row_min:
                        row_min = value
                if row_min > max_dist:
                    return None
                previous = current
            return previous[-1] if previous[-1] <= max_dist else None

        for idx, tokens in enumerate(token_sets):
            if not keep[idx] or not tokens:
                continue

            start = max(0, idx - _NEAR_DUP_LOOKBACK)
            for prev_idx in range(start, idx):
                if not keep[prev_idx]:
                    continue

                prev_tokens = token_sets[prev_idx]
                if not prev_tokens:
                    continue

                overlap = len(tokens & prev_tokens)
                union = len(tokens | prev_tokens)
                if not union:
                    continue

                overlap_ratio = overlap / union
                length_ratio = min(len(sentences[idx]) / len(sentences[prev_idx]), 1.0)
                sent_ratio = min(
                    sentence_counts[idx] / sentence_counts[prev_idx],
                    sentence_counts[prev_idx] / sentence_counts[idx],
                )
                if sent_ratio < 0.7:
                    continue
                score = (overlap_ratio * 0.7) + (length_ratio * 0.3)
                if score >= self.near_dup_similarity:
                    keep[idx] = False
                    break

                if (
                    overlap_ratio >= 0.75
                    and length_ratio >= 0.75
                    and max(len(sentences[idx]), len(sentences[prev_idx]))
                    <= _CHAR_NEAR_DUP_MAX_LEN
                ):
                    max_len = max(len(sentences[idx]), len(sentences[prev_idx]))
                    max_dist = max(1, int((1.0 - self.near_dup_similarity) * max_len))
                    normalized_a = " ".join(sentences[idx].lower().split())
                    normalized_b = " ".join(sentences[prev_idx].lower().split())
                    dist = levenshtein_with_cap(normalized_a, normalized_b, max_dist)
                    if dist is not None and max_len:
                        char_similarity = 1.0 - (dist / max_len)
                        if char_similarity >= self.near_dup_similarity:
                            keep[idx] = False
                            break

        if all(keep):
            return sentences, directives

        deduped_sentences: List[str] = []
        deduped_directives: List[bool] = []
        for idx, sentence in enumerate(sentences):
            if keep[idx]:
                deduped_sentences.append(sentence)
                if idx < len(directives):
                    deduped_directives.append(directives[idx])

        return deduped_sentences, deduped_directives

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        return _SENTENCE_SPLIT_PATTERN.split(text)

    def _is_deduplication_only_transform(
        self, baseline_text: str, optimized_text: str
    ) -> bool:
        """Return True when optimized text is a sentence-level dedup of baseline."""
        if not baseline_text or not optimized_text:
            return False

        baseline_sentences = [
            sentence.strip()
            for sentence in self._split_sentences(baseline_text)
            if sentence.strip()
        ]
        optimized_sentences = [
            sentence.strip()
            for sentence in self._split_sentences(optimized_text)
            if sentence.strip()
        ]

        if not baseline_sentences or not optimized_sentences:
            return False
        if len(optimized_sentences) >= len(baseline_sentences):
            return False

        baseline_signatures = [
            self._normalized_sentence_signature(sentence)
            for sentence in baseline_sentences
        ]
        optimized_signatures = [
            self._normalized_sentence_signature(sentence)
            for sentence in optimized_sentences
        ]

        cursor = 0
        for signature in optimized_signatures:
            while cursor < len(baseline_signatures):
                if baseline_signatures[cursor] == signature:
                    cursor += 1
                    break
                cursor += 1
            else:
                return False

        return len(set(baseline_signatures)) < len(baseline_signatures)

    @staticmethod
    def _normalized_sentence_signature(sentence: str) -> str:
        normalized = _NORMALIZED_SENTENCE_SIGNATURE_PATTERN.sub(" ", sentence.lower())
        return " ".join(normalized.split())

    def _sentence_has_directive_or_constraint_cues(self, sentence: str) -> bool:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in _DIRECTIVE_KEYWORDS):
            return True
        if any(pattern.search(sentence) for pattern in _DIRECTIVE_PATTERNS):
            return True
        if _CONSTRAINT_MARKER_PATTERN.search(sentence):
            return True
        head_match = re.match(r"^\s*([a-zA-Z][a-zA-Z'-]*)", sentence)
        if head_match and head_match.group(1).lower() in _INSTRUCTION_HEAD_TERMS:
            return True
        return False

    def _is_prechunk_dedup_eligible(self, sentence: str) -> bool:
        words = re.findall(r"\b\w+\b", sentence)
        if len(words) < 6:
            return False
        stripped = sentence.strip()
        if not stripped:
            return False
        if stripped[-1] not in _TERMINAL_PUNCTUATION:
            return False
        return True

    def _score_prechunk_duplicate_removal(
        self,
        sentence: str,
        *,
        seen_in_section: int,
    ) -> float:
        score = 0.0
        words = re.findall(r"\b\w+\b", sentence)
        stripped = sentence.strip()

        if self._is_prechunk_dedup_eligible(stripped):
            score += 0.65
        if len(words) >= 10:
            score += 0.15
        if stripped and stripped[-1] == ".":
            score += 0.10
        elif stripped and stripped[-1] in {"!", "?"}:
            score -= 0.15

        score += min(0.10, 0.05 * max(0, seen_in_section - 1))
        return max(0.0, min(score, 1.0))

    def _deduplicate_normalized_sentences(self, text: str) -> Tuple[str, bool]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text, False

        paragraph_pattern = re.compile(r"\n\s*\n+")
        sections = self._split_text_with_separators(text, paragraph_pattern)

        seen_sections: Dict[str, int] = {}
        boundary_kept_once: Set[str] = set()
        section_signature_counts: Dict[int, Dict[str, int]] = defaultdict(dict)
        changed = False
        rebuilt_parts: List[str] = []
        optimization_mode = self._get_state().optimization_mode
        removal_threshold = _PRECHUNK_SENTENCE_DEDUP_REMOVAL_THRESHOLD.get(
            optimization_mode, _PRECHUNK_SENTENCE_DEDUP_REMOVAL_THRESHOLD["balanced"]
        )
        keep_boundary_duplicate_once = optimization_mode != "maximum"

        for section_idx, (section, separator, _, _) in enumerate(sections):
            section_sentences = self._split_sentences(section)
            if not section_sentences:
                rebuilt_parts.append(section)
                rebuilt_parts.append(separator)
                continue

            kept: List[str] = []
            section_seen = section_signature_counts[section_idx]

            for sentence in section_sentences:
                stripped = sentence.strip()
                if not stripped:
                    continue

                signature = self._normalized_sentence_signature(stripped)
                if not signature:
                    continue

                section_seen[signature] = section_seen.get(signature, 0) + 1

                should_protect = self._sentence_has_directive_or_constraint_cues(
                    stripped
                ) or not self._is_prechunk_dedup_eligible(stripped)

                if should_protect or signature not in seen_sections:
                    kept.append(stripped)
                    seen_sections[signature] = section_idx
                    continue

                previous_section = seen_sections.get(signature, section_idx)
                if (
                    previous_section != section_idx
                    and signature not in boundary_kept_once
                    and keep_boundary_duplicate_once
                ):
                    kept.append(stripped)
                    boundary_kept_once.add(signature)
                    seen_sections[signature] = section_idx
                    continue

                removal_score = self._score_prechunk_duplicate_removal(
                    stripped,
                    seen_in_section=section_seen[signature],
                )
                if removal_score >= removal_threshold:
                    changed = True
                    continue

                kept.append(stripped)

            rebuilt_parts.append(" ".join(kept) if kept else "")
            rebuilt_parts.append(separator)

        deduped = "".join(rebuilt_parts)
        if not changed:
            return text, False

        deduped = re.sub(r"\n\s*\n+\Z", "", deduped)
        if not deduped.strip():
            return text, False

        return deduped, deduped != text

    def _estimate_sentence_redundancy_ratio(self, text: str) -> float:
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        seen: Dict[str, int] = {}
        total = 0
        repeated = 0
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            total += 1
            signature = self._normalized_sentence_signature(stripped)
            if signature in seen:
                repeated += 1
            seen[signature] = seen.get(signature, 0) + 1

        if total == 0:
            return 0.0
        return repeated / total

    def _deduplicate_content(self, text: str) -> str:
        """
        Remove duplicate sentences and repeated phrases.

        Uses hashing for efficient duplicate detection.
        """
        self._last_dedup_short_circuit = False

        # Split into sentences
        # Note: We use a simple sentence boundary pattern. More sophisticated
        # abbreviation handling (Dr., Mr., etc.) would require NLP parsing
        # which is too expensive for this pass.
        sentences = self._split_sentences(text)

        skip_exact_deduplication = self._get_state().skip_exact_deduplication

        # Track seen sentences (normalized for comparison)
        seen_hashes = set()
        unique_sentences: List[str] = []
        unique_sentence_directives: List[bool] = []
        normalized_signatures: List[str] = []

        sentence_counts: Dict[str, int] = {}
        sentence_is_directive: Dict[str, bool] = {}
        sentence_positions: Dict[str, List[int]] = {}
        directive_overflow: Dict[str, int] = {}
        found_repeat = False
        total_sentences = 0

        for sentence in sentences:
            if not sentence.strip():
                continue
            total_sentences += 1

            # Normalize for comparison (lowercase, remove extra spaces)
            normalized = " ".join(sentence.lower().split())

            # Use hash for efficient duplicate detection
            sentence_hash = hashlib.md5(normalized.encode()).hexdigest()

            is_directive = self._sentence_has_directive_or_constraint_cues(sentence)
            current_count = sentence_counts.get(sentence_hash, 0)

            if sentence_hash not in seen_hashes:
                seen_hashes.add(sentence_hash)
                unique_sentences.append(sentence)
                unique_sentence_directives.append(is_directive)
                normalized_signatures.append(normalized)
                sentence_counts[sentence_hash] = 1
                sentence_is_directive[sentence_hash] = is_directive
                sentence_positions[sentence_hash] = [len(unique_sentences) - 1]
            else:
                found_repeat = True
                sentence_counts[sentence_hash] = current_count + 1
                sentence_is_directive[sentence_hash] = (
                    sentence_is_directive.get(sentence_hash, False) or is_directive
                )

                if sentence_is_directive[sentence_hash]:
                    for position in sentence_positions.get(sentence_hash, []):
                        unique_sentence_directives[position] = True

                if skip_exact_deduplication:
                    positions = sentence_positions.setdefault(sentence_hash, [])
                    unique_sentences.append(sentence)
                    unique_sentence_directives.append(
                        sentence_is_directive.get(sentence_hash, False)
                    )
                    normalized_signatures.append(normalized)
                    positions.append(len(unique_sentences) - 1)
                elif sentence_is_directive[sentence_hash]:
                    directive_overflow[sentence_hash] = (
                        directive_overflow.get(sentence_hash, 0) + 1
                    )

        # Note: directive_overflow tracking is complete at this point
        # Extra repetitions beyond 2 are simply discarded (not labeled)
        if not skip_exact_deduplication and total_sentences:
            exact_removed = max(total_sentences - len(unique_sentences), 0)
            self._track_dedup_counts("exact", exact_removed)

        if len(unique_sentences) <= 1:
            self._last_dedup_short_circuit = True
            return " ".join(unique_sentences) if unique_sentences else text

        if skip_exact_deduplication:
            near_dedup_applied = False
            seq_dedup_applied = False
            cheap_similarity_signal = False
        else:
            before_near = len(unique_sentences)
            unique_sentences, unique_sentence_directives = (
                self._deduplicate_near_sentences(
                    unique_sentences, unique_sentence_directives
                )
            )
            near_removed = max(before_near - len(unique_sentences), 0)
            near_dedup_applied = near_removed > 0

            # Check for multi-sentence sequence duplicates (e.g., 2+ consecutive sentences repeated)
            before_seq = len(unique_sentences)
            unique_sentences, unique_sentence_directives = (
                self._remove_multi_sentence_sequence_duplicates(
                    unique_sentences, unique_sentence_directives, sequence_length=2
                )
            )
            seq_removed = max(before_seq - len(unique_sentences), 0)
            seq_dedup_applied = seq_removed > 0
            if near_removed or seq_removed:
                self._track_dedup_counts("near", near_removed + seq_removed)
            cheap_similarity_signal = near_dedup_applied or seq_dedup_applied

        if (
            not found_repeat
            and not near_dedup_applied
            and not seq_dedup_applied
            and not skip_exact_deduplication
        ):
            self._last_dedup_short_circuit = True
            return " ".join(unique_sentences)

        optimization_mode = self._get_state().optimization_mode
        semantic_deduplication_enabled = (
            self._get_state().semantic_deduplication_enabled
        )
        semantic_enabled = (
            found_repeat
            and semantic_deduplication_enabled
            and optimization_mode == "maximum"
        )

        nlp_token_threshold = self.fastpath_token_threshold or self.chunk_threshold
        if nlp_token_threshold:
            token_count = self.count_tokens(text)
            if token_count <= nlp_token_threshold:
                logger.debug(
                    "Skipping spaCy deduplication: %s tokens below threshold %s",
                    token_count,
                    nlp_token_threshold,
                )
                return " ".join(unique_sentences)

        long_sentence_count = sum(
            1
            for sentence in unique_sentences
            if len(sentence.split()) >= _MINHASH_PARAPHRASE_MIN_WORDS
        )
        has_long_sentences = long_sentence_count >= 2
        minhash_paraphrase_enabled = (
            semantic_deduplication_enabled
            and self.enable_lsh_deduplication
            and not semantic_enabled
            and (found_repeat or cheap_similarity_signal or has_long_sentences)
        )
        if minhash_paraphrase_enabled:
            keep = [True] * len(unique_sentences)
            light_scan = not (found_repeat or cheap_similarity_signal)
            paraphrase_threshold = self._resolve_minhash_paraphrase_threshold()
            if light_scan:
                paraphrase_threshold = max(
                    paraphrase_threshold, _MINHASH_LIGHT_SCAN_THRESHOLD
                )

            # Use our zero-dependency SentenceLSHIndex
            lsh_index = SentenceLSHIndex(
                threshold=paraphrase_threshold, num_perm=self._minhash_num_perm
            )

            signatures: List[Optional[MinHash]] = []
            token_sets: List[Optional[Set[str]]] = []
            for sentence in unique_sentences:
                # Only compute signatures for reasonably long sentences
                tokens = re.findall(r"\w+", sentence.lower())
                if len(tokens) < _MINHASH_PARAPHRASE_MIN_WORDS:
                    signatures.append(None)
                    token_sets.append(None)
                    continue
                token_sets.append(set(tokens))
                sig = lsh_index.create_signature(sentence, shingle_size=2)
                signatures.append(sig)

            # Build global index
            for idx, sig in enumerate(signatures):
                if sig is not None:
                    lsh_index.add_sentence(str(idx), sig)

            removed_lsh = 0
            for idx, signature in enumerate(signatures):
                if signature is None or not keep[idx]:
                    continue

                # Directives are never removed as duplicates here
                if (
                    idx < len(unique_sentence_directives)
                    and unique_sentence_directives[idx]
                ):
                    continue

                # Check for previously seen similar sentences
                matches = lsh_index.query_similar(signature)
                is_duplicate = False
                match_idx = -1
                for match_id in matches:
                    match_idx = int(match_id)
                    if match_idx < idx and keep[match_idx]:
                        if (
                            match_idx < len(unique_sentence_directives)
                            and unique_sentence_directives[match_idx]
                        ):
                            continue
                        if (
                            skip_exact_deduplication
                            and normalized_signatures[match_idx]
                            == normalized_signatures[idx]
                        ):
                            continue
                        # Double check with jaccard for precision
                        prev_sig = signatures[match_idx]
                        if (
                            prev_sig
                            and signature.jaccard(prev_sig) >= paraphrase_threshold
                        ):
                            current_tokens = token_sets[idx]
                            previous_tokens = token_sets[match_idx]
                            if not current_tokens or not previous_tokens:
                                continue
                            overlap = len(current_tokens & previous_tokens)
                            union = len(current_tokens | previous_tokens)
                            if not union:
                                continue
                            token_overlap = overlap / union
                            if token_overlap < _MINHASH_TOKEN_OVERLAP_THRESHOLD:
                                continue
                            is_duplicate = True
                            break

                if is_duplicate:
                    if (
                        self.prefer_shorter_duplicates
                        and len(unique_sentences[idx])
                        < len(unique_sentences[match_idx])
                        and not (
                            match_idx < len(unique_sentence_directives)
                            and unique_sentence_directives[match_idx]
                        )
                    ):
                        unique_sentences[match_idx] = unique_sentences[idx]
                        signatures[match_idx] = signature
                        token_sets[match_idx] = token_sets[idx]
                    keep[idx] = False
                    removed_lsh += 1

            if not all(keep):
                unique_sentences = [
                    s for idx, s in enumerate(unique_sentences) if keep[idx]
                ]
                unique_sentence_directives = [
                    d for idx, d in enumerate(unique_sentence_directives) if keep[idx]
                ]
                if removed_lsh:
                    self._track_dedup_counts("semantic", removed_lsh)
                self._record_technique("Semantic LSH Deduplication")

        semantic_sentences: List[str] = []
        semantic_sentence_directives: List[bool] = []
        semantic_vectors: List[Optional[np.ndarray]] = []
        semantic_signatures = []
        semantic_normalized_signatures: List[str] = []
        lsh_index: Optional[SentenceLSHIndex] = None

        if semantic_enabled:
            resolved_similarity_threshold = (
                self._resolve_semantic_similarity_threshold()
            )
            nlp_model = self._get_nlp_model()
            if nlp_model is None:
                threshold = max(0.75, min(0.95, resolved_similarity_threshold))
                removed_any = False
                token_sets: List[Set[str]] = [
                    set(re.findall(r"\w+", sentence.lower()))
                    for sentence in unique_sentences
                ]
                idf_weights = self._compute_idf_weights(token_sets)
                keep = [True] * len(unique_sentences)

                for idx, tokens in enumerate(token_sets):
                    if (
                        idx < len(unique_sentence_directives)
                        and unique_sentence_directives[idx]
                    ):
                        continue
                    if not tokens:
                        continue

                    start = max(0, idx - _NEAR_DUP_LOOKBACK)
                    for prev_idx in range(start, idx):
                        if not keep[prev_idx]:
                            continue
                        if (
                            prev_idx < len(unique_sentence_directives)
                            and unique_sentence_directives[prev_idx]
                        ):
                            continue
                        if (
                            skip_exact_deduplication
                            and normalized_signatures[prev_idx]
                            == normalized_signatures[idx]
                        ):
                            continue
                        prev_tokens = token_sets[prev_idx]
                        if not prev_tokens:
                            continue
                        word_similarity = self._weighted_jaccard(
                            tokens, prev_tokens, idf_weights
                        )

                        len_a = len(unique_sentences[idx])
                        len_b = len(unique_sentences[prev_idx])
                        length_ratio = (
                            min(len_a / len_b, len_b / len_a)
                            if len_a > 0 and len_b > 0
                            else 0.0
                        )

                        count_a = len(re.findall(r"[.!?]", unique_sentences[idx])) or 1
                        count_b = (
                            len(re.findall(r"[.!?]", unique_sentences[prev_idx])) or 1
                        )
                        sent_ratio = min(count_a / count_b, count_b / count_a)
                        if sent_ratio < 0.7:
                            continue

                        similarity = (
                            (word_similarity * 0.5)
                            + (length_ratio * 0.25)
                            + (sent_ratio * 0.25)
                        )
                        if similarity >= threshold:
                            keep[idx] = False
                            removed_any = True
                            break

                if removed_any:
                    removed_semantic = sum(1 for item in keep if not item)
                    self._track_dedup_counts("semantic", removed_semantic)
                    unique_sentences = [
                        s for idx, s in enumerate(unique_sentences) if keep[idx]
                    ]
                    unique_sentence_directives = [
                        d
                        for idx, d in enumerate(unique_sentence_directives)
                        if keep[idx]
                    ]
                    self._record_technique("Token Overlap Sentence Deduplication")
                semantic_enabled = False

            lsh_indexed_count = 0
            if semantic_enabled:
                candidate_index = SentenceLSHIndex(
                    self.lsh_similarity_threshold, self._minhash_num_perm
                )
                if candidate_index.is_available:
                    lsh_index = candidate_index

            vector_map: Dict[int, Optional[np.ndarray]] = {}
            if nlp_model is not None and unique_sentences:
                batch_indices = [
                    index
                    for index, sentence in enumerate(unique_sentences)
                    if sentence.strip()
                    and not (
                        index < len(unique_sentence_directives)
                        and unique_sentence_directives[index]
                    )
                ]
                if batch_indices:
                    batch_sentences = [
                        unique_sentences[index] for index in batch_indices
                    ]
                    batch_vectors = self._vectorize_sentences(
                        batch_sentences, nlp_model
                    )
                    for index, vector in zip(batch_indices, batch_vectors):
                        vector_map[index] = vector

            semantic_removed = 0
            for idx, sentence in enumerate(unique_sentences):
                if not sentence.strip():
                    continue

                if (
                    idx < len(unique_sentence_directives)
                    and unique_sentence_directives[idx]
                ):
                    semantic_sentences.append(sentence)
                    semantic_sentence_directives.append(True)
                    semantic_signatures.append(None)
                    semantic_normalized_signatures.append(normalized_signatures[idx])
                    semantic_vectors.append(None)
                    continue

                sentence_vector = vector_map.get(idx) if nlp_model is not None else None
                is_duplicate = False
                tokens = re.findall(r"\w+", sentence.lower())
                minhash_signature = self._compute_minhash_signature(tokens)

                if nlp_model is not None and semantic_sentences:
                    candidate_indices: List[int] = []
                    use_lsh = (
                        lsh_index is not None
                        and minhash_signature is not None
                        and lsh_indexed_count > 0
                    )

                    if use_lsh:
                        queried_indices = {
                            int(candidate)
                            for candidate in lsh_index.query_similar(minhash_signature)
                            if str(candidate).isdigit()
                        }
                        candidate_indices = [
                            candidate
                            for candidate in sorted(queried_indices)
                            if 0 <= candidate < len(semantic_vectors)
                        ]

                    if not use_lsh:
                        for candidate_idx, previous_vector in enumerate(
                            semantic_vectors
                        ):
                            if previous_vector is None:
                                continue

                            previous_signature = semantic_signatures[candidate_idx]
                            if (
                                minhash_signature is not None
                                and previous_signature is not None
                                and minhash_signature.jaccard(previous_signature)
                                < self.minhash_candidate_threshold
                            ):
                                continue

                            candidate_indices.append(candidate_idx)

                    doc_vector: Optional[np.ndarray] = None
                    for candidate_idx in candidate_indices:
                        previous_vector = semantic_vectors[candidate_idx]
                        if previous_vector is None:
                            continue

                        if doc_vector is None:
                            doc_vector = sentence_vector
                            if doc_vector is None:
                                break

                        similarity = self._cosine_similarity(
                            doc_vector, previous_vector
                        )
                        if similarity >= resolved_similarity_threshold:
                            if (
                                skip_exact_deduplication
                                and semantic_normalized_signatures[candidate_idx]
                                == normalized_signatures[idx]
                            ):
                                continue
                            # Check if we should prefer the shorter sentence
                            if self.prefer_shorter_duplicates:
                                prev_sentence = semantic_sentences[candidate_idx]
                                if len(sentence) < len(prev_sentence):
                                    # Current sentence is shorter - replace the previous one
                                    semantic_sentences[candidate_idx] = sentence
                                    semantic_vectors[candidate_idx] = doc_vector
                                    semantic_signatures[candidate_idx] = (
                                        minhash_signature
                                    )
                            is_duplicate = True
                            semantic_removed += 1
                            break

                if not is_duplicate:
                    semantic_sentences.append(sentence)
                    semantic_sentence_directives.append(
                        idx < len(unique_sentence_directives)
                        and unique_sentence_directives[idx]
                    )
                    semantic_signatures.append(minhash_signature)
                    semantic_normalized_signatures.append(normalized_signatures[idx])

                    if nlp_model is not None:
                        vector = sentence_vector
                        semantic_vectors.append(vector)
                    else:
                        semantic_vectors.append(None)

                    if (
                        lsh_index is not None
                        and minhash_signature is not None
                        and not semantic_sentence_directives[-1]
                    ):
                        lsh_index.add_sentence(
                            str(len(semantic_sentences) - 1), minhash_signature
                        )
                        lsh_indexed_count += 1

            if semantic_removed:
                self._track_dedup_counts("semantic", semantic_removed)

        if semantic_enabled and semantic_sentences:
            result_sentences = semantic_sentences
            result_directives = semantic_sentence_directives
        else:
            result_sentences = unique_sentences
            result_directives = unique_sentence_directives

        result = " ".join(result_sentences)

        # Also check for repeated phrases while respecting directive emphasis
        phrase_length = self.dedup_phrase_length
        if (
            not skip_exact_deduplication
            and phrase_length > 0
            and result_sentences
            and any(
                len(sentence.split()) >= phrase_length for sentence in result_sentences
            )
        ):
            seen_phrases: Dict[str, Dict[str, Any]] = {}
            processed_sentences: List[str] = []

            for sentence_idx, sentence in enumerate(result_sentences):
                is_directive_sentence = (
                    sentence_idx < len(result_directives)
                    and result_directives[sentence_idx]
                )

                if is_directive_sentence:
                    processed_sentences.append(sentence)
                    continue

                words_in_sentence = sentence.split()
                if len(words_in_sentence) < phrase_length:
                    processed_sentences.append(sentence)
                    continue

                removal_ranges: List[Tuple[int, int]] = []
                for start in range(len(words_in_sentence) - phrase_length + 1):
                    phrase_words = words_in_sentence[start : start + phrase_length]
                    phrase_key = " ".join(phrase_words).lower()
                    info = seen_phrases.get(phrase_key)

                    if info is None:
                        seen_phrases[phrase_key] = {
                            "sentence_index": sentence_idx,
                            "is_directive": is_directive_sentence,
                        }
                        continue

                    if info.get("is_directive") or is_directive_sentence:
                        continue

                    removal_ranges.append((start, start + phrase_length))

                if removal_ranges:
                    mask = [True] * len(words_in_sentence)
                    for start, end in removal_ranges:
                        for idx in range(start, end):
                            mask[idx] = False

                    words_in_sentence = [
                        word for word, keep in zip(words_in_sentence, mask) if keep
                    ]

                processed_sentences.append(" ".join(words_in_sentence))

            result = " ".join(processed_sentences)

        return result

    def _compress_punctuation(self, text: str) -> str:
        """Compress repeated punctuation (Technique 2)."""
        return _lexical.compress_punctuation(text, config.PLACEHOLDER_PATTERN)

    def _compress_examples(
        self,
        text: str,
        preserved: Optional[Dict] = None,
        *,
        summary_max_length: int = 200,
    ) -> str:
        """
        Compress multiple examples while preserving I/O shape.
        Keeps first and last examples, summarizes middle ones.
        """
        # Look for patterns like "Example 1:", "Example 2:", etc.
        example_regex = re.compile(
            r"(example\s+(?:\d+|__[^\s:]+__):)(.*?)(?=example\s+(?:\d+|__[^\s:]+__):|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        matches = list(example_regex.finditer(text))

        if len(matches) <= 2:
            return text  # Not enough examples to compress

        placeholder_tokens: Set[str] = set()
        if preserved:
            placeholder_tokens = self._get_placeholder_tokens(preserved)

        rebuilt: List[str] = []
        cursor = 0

        for index, match in enumerate(matches):
            start, end = match.span()
            rebuilt.append(text[cursor:start])

            label = match.group(1)
            content = match.group(2)

            if index == 0 or index == len(matches) - 1:
                rebuilt.append(match.group(0))
            else:
                # Use the original label without adding "(summary)"
                summarized_label = label

                summary_content = self._summarize_example_content(
                    content,
                    placeholder_tokens,
                    max_length=summary_max_length,
                )

                prefix_newline = ""
                for prev_segment in reversed(rebuilt):
                    if prev_segment:
                        if not prev_segment.endswith("\n"):
                            prefix_newline = "\n"
                        break

                rebuilt.append(f"{prefix_newline}{summarized_label}{summary_content}")

            cursor = end

        rebuilt.append(text[cursor:])
        return "".join(rebuilt)

    def _summarize_history_with_modifier(self, text: str, modifier: float) -> str:
        original_modifier = self.summarize_keep_ratio_modifier
        try:
            self.summarize_keep_ratio_modifier = modifier
            return _history.summarize_history(self, text)
        finally:
            self.summarize_keep_ratio_modifier = original_modifier

    def _summarize_example_content(
        self,
        content: str,
        placeholder_tokens: Set[str],
        *,
        max_length: int = 200,
    ) -> str:
        """Return a concise, markdown-friendly summary for a middle example."""

        if not content:
            return content

        stripped = content.strip("\n")
        if not stripped:
            return content

        leading_newline = "\n" if content.startswith("\n") else ""
        trailing_newline = "\n" if content.endswith("\n") else ""

        label_split_pattern = re.compile(
            r"(?<!^)(?<!\n)(\b(?:Input|Output|Expected|Result|Response|Answer|Analysis|Explanation|User|Assistant|"
            r"Instruction|Context)\s*[:\-])",
            re.IGNORECASE,
        )
        normalized_for_sections = label_split_pattern.sub(r"\n\1", stripped)

        sections = self._extract_example_sections(normalized_for_sections)
        summary_lines: List[str] = []

        def append_bullet(label: Optional[str], snippet: str) -> None:
            snippet = snippet.strip()
            if not snippet:
                return

            if label:
                bullet_prefix = f"- {label.rstrip(':')}:"
            else:
                bullet_prefix = "-"

            matched_tokens = [token for token in placeholder_tokens if token in snippet]
            contains_code_placeholder = any(
                token.startswith("__CODE_") for token in matched_tokens
            )

            if contains_code_placeholder or "\n" in snippet:
                summary_lines.append(bullet_prefix)
                summary_lines.append(snippet)
            else:
                summary_lines.append(f"{bullet_prefix} {snippet}".rstrip())

        if sections:
            for label, body in sections:
                snippet = self._truncate_preserving_placeholders(
                    body.strip(),
                    placeholder_tokens,
                    max_length=max_length,
                )
                append_bullet(label, snippet)
        else:
            bullet_pattern = re.compile(r"^\s*[-*•+]\s*(.+)$", re.MULTILINE)
            bullet_matches = bullet_pattern.findall(stripped)
            if bullet_matches:
                for bullet in bullet_matches[:3]:
                    snippet = self._truncate_preserving_placeholders(
                        bullet,
                        placeholder_tokens,
                        max_length=max_length,
                    )
                    append_bullet(None, snippet)
            else:
                first_lines = stripped.splitlines()
                snippet_source = " ".join(
                    line.strip() for line in first_lines[:2] if line.strip()
                )
                snippet = self._truncate_preserving_placeholders(
                    snippet_source or stripped,
                    placeholder_tokens,
                    max_length=max_length,
                )
                append_bullet("Detail", snippet)

        code_blocks = re.findall(
            r"(?P<fence>```|~~~).*?(?P=fence)", stripped, flags=re.DOTALL
        )
        code_summary = None
        if code_blocks:
            code_summary = self._summarize_code_block(code_blocks[0])

        summary_body = "\n".join(summary_lines).strip("\n")
        if code_summary:
            if summary_body:
                summary_body = f"{summary_body}\n{code_summary}"
            else:
                summary_body = code_summary

        if not summary_body:
            return content

        prefix = leading_newline if leading_newline else "\n"
        return f"{prefix}{summary_body}{trailing_newline}"

    def _extract_example_sections(self, content: str) -> List[Tuple[str, str]]:
        """Extract labeled sections (Input, Output, etc.) from example content."""

        allowed_labels = {
            "input",
            "output",
            "expected",
            "result",
            "response",
            "answer",
            "analysis",
            "explanation",
            "user",
            "assistant",
            "instruction",
            "context",
        }

        section_pattern = re.compile(
            r"^\s*([A-Za-z][A-Za-z0-9 /_-]{0,40})\s*[:\-]\s*(.*)$"
        )

        sections: List[Tuple[str, str]] = []
        current_label: Optional[str] = None
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer, current_label
            if current_label and buffer:
                body = "\n".join(buffer).strip()
                if body:
                    sections.append((current_label, body))
            current_label = None
            buffer = []

        for line in content.splitlines():
            match = section_pattern.match(line)
            if match and match.group(1).strip().lower() in allowed_labels:
                flush()
                current_label = match.group(1).strip()
                initial = match.group(2)
                buffer = [initial] if initial else []
            else:
                if current_label is not None:
                    buffer.append(line)

        flush()
        return sections

    def _truncate_preserving_placeholders(
        self,
        text: str,
        placeholder_tokens: Set[str],
        max_length: int,
    ) -> str:
        """Truncate text without cutting through preserved placeholders."""

        if len(text) <= max_length:
            return text

        tokens = re.split(r"(\s+)", text)
        rebuilt = ""

        for token in tokens:
            if len(rebuilt) + len(token) > max_length:
                break
            rebuilt += token

        if not rebuilt.strip():
            rebuilt = text[:max_length]

        for placeholder in placeholder_tokens:
            index = rebuilt.rfind(placeholder)
            if index != -1 and index + len(placeholder) > len(rebuilt):
                rebuilt = text[: index + len(placeholder)]

        rebuilt = rebuilt.rstrip()
        if len(rebuilt) >= len(text):
            return rebuilt

        return f"{rebuilt} …"

    def _summarize_code_block(self, block: str, max_lines: int = 6) -> str:
        """Return a shortened code block while keeping fences balanced."""

        lines = block.splitlines()
        if len(lines) <= max_lines:
            return block

        opening = lines[0]
        opening_stripped = lines[0].lstrip()
        fence = (
            "```"
            if opening_stripped.startswith("```")
            else "~~~" if opening_stripped.startswith("~~~") else "```"
        )
        closing = lines[-1] if lines[-1].strip().startswith(fence) else fence
        inner = lines[1:-1] if lines[-1].strip().startswith(fence) else lines[1:]
        preserved_inner = inner[: max(1, max_lines - 2)]
        truncated_block = [opening, *preserved_inner]
        truncated_block.append("...")
        truncated_block.append(closing)

        return "\n".join(truncated_block)

    def _sample_for_entropy_budget(self, text: str) -> str:
        if len(text) <= _ENTROPY_SAMPLE_MAX_CHARS:
            return text

        chunk = _ENTROPY_SAMPLE_MAX_CHARS // 3
        mid = len(text) // 2
        return text[:chunk] + text[mid - chunk // 2 : mid + chunk // 2] + text[-chunk:]

    def _entropy_backend_preference_for_mode(self, optimization_mode: str) -> str:
        if optimization_mode == "maximum":
            try:
                if _entropy._get_scorer().available:
                    return "teacher"
            except Exception:
                pass
        return "fast"

    def _estimate_avg_nll(self, text: str, backend_preference: str) -> Optional[float]:
        if not text:
            return None

        sample = self._sample_for_entropy_budget(text)
        protected = _entropy._placeholder_ranges(sample)
        scorer = (
            _entropy._get_scorer()
            if backend_preference == "teacher"
            else _entropy._get_fast_scorer()
        )
        backend_name = "entropy" if backend_preference == "teacher" else "entropy_fast"
        if not scorer.available:
            raise RuntimeError(
                f"Required entropy backend '{backend_name}' is unavailable."
            )
        try:
            token_scores = scorer.score_tokens(sample, skip_ranges=protected)
        except Exception as exc:
            raise RuntimeError(
                f"Entropy budget estimation failed for backend '{backend_name}': {exc}"
            ) from exc

        if not token_scores:
            return None

        total = sum(score.entropy for score in token_scores)
        return total / len(token_scores)

    def _boundary_protection_ranges(self, text: str) -> List[Tuple[int, int]]:
        tokens = list(re.finditer(r"\S+", text))
        total = len(tokens)
        if total < _BOUNDARY_PROTECT_MIN_TOKENS:
            return []

        head_count = max(1, int(total * _BOUNDARY_PROTECT_HEAD_RATIO))
        tail_count = max(1, int(total * _BOUNDARY_PROTECT_TAIL_RATIO))
        if head_count + tail_count >= total:
            return [(tokens[0].start(), tokens[-1].end())]

        head_span = (tokens[0].start(), tokens[head_count - 1].end())
        tail_start = total - tail_count
        tail_span = (tokens[tail_start].start(), tokens[-1].end())
        if head_span[1] >= tail_span[0]:
            return [(head_span[0], tail_span[1])]
        return [head_span, tail_span]

    def _entropy_prune_budget(self, text: str, avg_nll: Optional[float] = None) -> int:
        if not text:
            return 0

        text_length = len(text)
        if text_length < self.entropy_prune_min_length:
            return 0
        token_count = self.count_tokens(text)
        has_code_blocks = any(
            f"__{prefix}_" in text
            for key, prefix in config.PLACEHOLDER_PREFIXES.items()
            if key in {"code_blocks", "toon_blocks", "json_tokens", "json_literals"}
        )
        has_instructions = any(pattern.search(text) for pattern in _DIRECTIVE_PATTERNS)

        adaptive_ratio = self.entropy_prune_ratio
        if avg_nll is not None:
            if avg_nll >= _ENTROPY_DENSE_THRESHOLD:
                adaptive_ratio *= _ENTROPY_DENSE_MULTIPLIER
            elif avg_nll <= _ENTROPY_FLUFFY_THRESHOLD:
                adaptive_ratio *= _ENTROPY_FLUFFY_MULTIPLIER
        if has_code_blocks:
            adaptive_ratio *= 0.5
        if has_instructions:
            adaptive_ratio *= 0.7
        if token_count > 5000:
            adaptive_ratio *= 1.5
        elif token_count > 3000:
            adaptive_ratio *= 1.3
        adaptive_ratio = min(0.5, max(0.0, adaptive_ratio))

        ratio_budget = int(text_length * adaptive_ratio)
        baseline_budget = max(self.entropy_prune_min_budget, ratio_budget)

        cap_ratio_budget = int(text_length * self.entropy_prune_max_ratio)
        cap = max(self.entropy_prune_cap_floor, cap_ratio_budget)
        cap = min(cap, text_length // 2)
        budget = min(baseline_budget, cap)

        if budget < self.entropy_prune_min_budget:
            return 0

        return max(0, budget)

    def _prune_low_entropy_with_ratio_multiplier(
        self,
        text: str,
        ratio_multiplier: float,
        max_ratio_multiplier: float,
    ) -> str:
        original_ratio = self.entropy_prune_ratio
        original_max_ratio = self.entropy_prune_max_ratio
        try:
            updated_ratio = min(0.5, max(0.0, original_ratio * ratio_multiplier))
            updated_max_ratio = min(
                0.5, max(updated_ratio, original_max_ratio * max_ratio_multiplier)
            )
            self.entropy_prune_ratio = updated_ratio
            self.entropy_prune_max_ratio = updated_max_ratio
            return self._maybe_prune_low_entropy(text)
        finally:
            self.entropy_prune_ratio = original_ratio
            self.entropy_prune_max_ratio = original_max_ratio

    def _maybe_prune_low_entropy(self, text: str) -> str:
        if self._entropy_prune_budget(text) <= 0:
            return text

        optimization_mode = self._get_state().optimization_mode
        backend_preference = self._entropy_backend_preference_for_mode(optimization_mode)
        try:
            avg_nll = self._estimate_avg_nll(text, backend_preference)
        except RuntimeError as exc:
            logger.warning(
                "Skipping entropy pruning: avg NLL estimation failed (%s)",
                exc,
            )
            return text
        budget = self._entropy_prune_budget(text, avg_nll)
        if budget <= 0:
            return text

        original = text
        protected_ranges = self._boundary_protection_ranges(text)
        min_confidence = (
            self.entropy_prune_min_confidence
            if self.entropy_prune_min_confidence > 0.0
            else None
        )
        pruned, removed_chars, _ = _entropy.prune_low_entropy(
            text,
            budget,
            protected_ranges=protected_ranges,
            min_confidence=min_confidence,
            backend_preference=backend_preference,
        )
        if removed_chars <= 0 or pruned == original:
            return text

        guard_threshold = min(
            0.97,
            max(self._resolve_semantic_guard_threshold(), 0.85),
        )
        similarity = _metrics.score_similarity(
            original,
            pruned,
            self.semantic_guard_model,
            embedding_cache=self._get_state().embedding_cache,
        )
        if similarity is None:
            logger.warning(
                "Semantic guard similarity unavailable for entropy pruning; reverting entropy pass."
            )
            return original

        if similarity >= guard_threshold:
            self._record_technique("Entropy Pruning")
            return pruned

        logger.debug(
            "Entropy pruning reverted due to low similarity (%.3f < %.3f)",
            similarity,
            guard_threshold,
        )
        return original

    def _apply_reference_aliases(
        self,
        text: str,
        preserved: Dict[str, Any],
        *,
        token_counter: Callable[[str], int],
    ) -> Tuple[str, bool, Optional[List[Tuple[str, str]]], int]:
        prefix_map = _preservation._resolve_prefix_map(self)
        alias_counters: Dict[str, int] = {"U": 0, "C": 0}
        alias_entries: List[Tuple[str, str]] = []
        placeholder_to_alias: Dict[str, str] = {}
        total_original = 0
        total_alias = 0

        for storage_key, prefix_key, alias_letter in (
            ("urls", "urls", "U"),
            ("citations", "citations", "C"),
        ):
            values = preserved.get(storage_key, [])
            placeholder_prefix = prefix_map.get(prefix_key, alias_letter)
            ordered_values: "OrderedDict[str, List[str]]" = OrderedDict()
            for index, value in enumerate(values):
                if not value:
                    continue
                placeholder = f"__{placeholder_prefix}_{index}__"
                ordered_values.setdefault(value, []).append(placeholder)
            for value, placeholders in ordered_values.items():
                if len(placeholders) < 2 or "__" in value:
                    continue
                alias_counters[alias_letter] += 1
                alias_tag = f"[{alias_letter}{alias_counters[alias_letter]}]"
                original_tokens = len(placeholders) * token_counter(value)
                alias_tokens = len(placeholders) * token_counter(alias_tag)
                if original_tokens <= alias_tokens:
                    alias_counters[alias_letter] -= 1
                    continue
                alias_entries.append((alias_tag, value))
                total_original += original_tokens
                total_alias += alias_tokens
                for placeholder in placeholders:
                    placeholder_to_alias[placeholder] = alias_tag

        if not alias_entries:
            return text, False, None, 0

        legend = (
            f"Refs: {', '.join(f'{alias}={value}' for alias, value in alias_entries)}"
        )
        legend_cost = token_counter(legend)
        net_savings = total_original - total_alias - legend_cost
        if net_savings <= 0:
            return text, False, None, 0

        updated = text
        for placeholder, alias_tag in placeholder_to_alias.items():
            updated = updated.replace(placeholder, alias_tag)

        return updated, True, alias_entries, net_savings

    def _replace_subsequent_occurrences(
        self, text: str, value: str, alias: str
    ) -> Tuple[str, int]:
        if not value:
            return text, 0
        first_index = text.find(value)
        if first_index == -1:
            return text, 0
        start = first_index + len(value)
        replacements: List[Tuple[int, int]] = []
        while True:
            next_index = text.find(value, start)
            if next_index == -1:
                break
            replacements.append((next_index, next_index + len(value)))
            start = next_index + len(value)
        if not replacements:
            return text, 0
        updated = text
        for start, end in reversed(replacements):
            updated = updated[:start] + alias + updated[end:]
        return updated, len(replacements)

    def _alias_preserved_elements(
        self,
        text: str,
        preserved: Dict[str, Any],
        *,
        token_counter: Callable[[str], int],
    ) -> Tuple[str, bool]:
        if not text or not preserved:
            return text, False

        original = text
        updated = text
        prompt_tokens = token_counter(text)
        short_prompt = prompt_tokens < _PRESERVED_ALIAS_MIN_TOKENS
        alias_plan: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
        total_savings = 0

        category_specs = (
            ("code_blocks", "CODE", "Code"),
            ("urls", "U", "Refs"),
            ("citations", "C", "Refs"),
        )

        for storage_key, alias_prefix, legend_label in category_specs:
            values = preserved.get(storage_key, []) if preserved else []
            if not values:
                continue
            ordered_values: "OrderedDict[str, int]" = OrderedDict()
            for value in values:
                if not value or "__" in value:
                    continue
                ordered_values[value] = ordered_values.get(value, 0) + 1
            if not ordered_values:
                continue

            alias_counter = 0
            for value, preserved_count in ordered_values.items():
                if preserved_count < 2:
                    continue
                occurrence_count = updated.count(value)
                if occurrence_count < 2:
                    continue
                alias_counter += 1
                alias = f"{alias_prefix}{alias_counter}"
                if alias == value:
                    continue
                value_tokens = token_counter(value)
                alias_tokens = token_counter(alias)
                if value_tokens <= alias_tokens:
                    continue
                updated, replacements = self._replace_subsequent_occurrences(
                    updated, value, alias
                )
                if replacements <= 0:
                    continue
                savings = replacements * (value_tokens - alias_tokens)
                if savings <= 0:
                    continue
                alias_plan.setdefault(legend_label, []).append((alias, value))
                total_savings += savings

        if not alias_plan:
            return original, False

        legend_lines = []
        for label, entries in alias_plan.items():
            if not entries:
                continue
            legend_lines.append(
                f"{label}: " + ", ".join(f"{alias}={value}" for alias, value in entries)
            )

        if not legend_lines:
            return original, False

        legend_body = "\n".join(legend_lines)
        existing_leading = updated[: len(updated) - len(updated.lstrip())]
        existing_body = updated[len(existing_leading) :]
        if existing_body.startswith("Legend:\n"):
            insertion = f"{legend_body}\n"
            legend_cost = token_counter(insertion)
        else:
            legend = "Legend:\n" + legend_body
            legend_cost = token_counter(legend)
        net_savings = total_savings - legend_cost
        if net_savings <= 0 or (short_prompt and total_savings <= legend_cost):
            return original, False

        if existing_body.startswith("Legend:\n"):
            legend_prefix_len = len("Legend:\n")
            updated = f"{existing_leading}Legend:\n{legend_body}\n{existing_body[legend_prefix_len:]}"
        else:
            updated = _lexical._prepend_legend(updated, legend)
        return updated, True

    def _normalize_constraint_line(self, line: str) -> str:
        normalized = line.strip()
        normalized = re.sub(r"^[-*+]\s*", "", normalized)
        normalized = re.sub(r"^\d+[\.\)]\s*", "", normalized)
        return normalized.strip()

    def _hoist_constraints(self, text: str) -> Tuple[str, bool]:
        if "global constraints (applies to all sections below)" in text.lower():
            return text, False

        lines = text.splitlines()
        counts: "OrderedDict[str, int]" = OrderedDict()
        keywords = (
            "do not",
            "don't",
            "must",
            "never",
            "avoid",
            "no ",
            "ensure",
        )

        for line in lines:
            if "__" in line:
                continue
            normalized = self._normalize_constraint_line(line)
            if not normalized:
                continue
            lower = normalized.lower()
            if not any(lower.startswith(keyword) for keyword in keywords):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1

        hoisted = [line for line, count in counts.items() if count >= 2]
        if not hoisted:
            return text, False

        hoisted_set = set(hoisted)
        remaining_lines = []
        for line in lines:
            normalized = self._normalize_constraint_line(line)
            if normalized in hoisted_set:
                continue
            remaining_lines.append(line)

        insert_index = 0
        while (
            insert_index < len(remaining_lines)
            and not remaining_lines[insert_index].strip()
        ):
            insert_index += 1

        block = ["Global constraints (applies to all sections below):"]
        block.extend(f"- {constraint}" for constraint in hoisted)
        block.append("")

        updated_lines = (
            remaining_lines[:insert_index] + block + remaining_lines[insert_index:]
        )
        updated_text = "\n".join(updated_lines)
        if text.endswith("\n") and not updated_text.endswith("\n"):
            updated_text += "\n"
        return updated_text, True

    def _split_by_headings(self, text: str) -> List[str]:
        """Split text into sections based on heading-like patterns."""
        heading_pattern = re.compile(r"(?<!\w)([A-Z][A-Za-z0-9 ]{0,40}):")
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [text]

        sections: List[str] = []
        last_index = 0

        for i, match in enumerate(matches):
            start = match.start()
            if start > last_index:
                prefix = text[last_index:start].strip()
                if prefix:
                    sections.append(prefix)

            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)

            last_index = end

        return sections or [text]


# Singleton instance
optimizer = PromptOptimizer()
