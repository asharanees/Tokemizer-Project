"""Utilities for ranking and selecting prompt sections before optimization."""

from __future__ import annotations

import gzip
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import chunking as _chunking
from . import metrics as _metrics

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_SEMANTIC_RANK_MIN_TOKENS = 1200
_MMR_MIN_SECTIONS = 4
_MMR_LAMBDA = 0.7
_LUHN_KEYWORDS_LIMIT = 12
_QUERY_AWARE_MIN_TOKENS = 800
_QUERY_AWARE_MIN_SENTENCES = 4
_AUTO_TOKEN_BUDGET_RATIO = 0.7
_CONTEXT_HEADER_PATTERN = re.compile(
    r"^(#{1,6}\s+\S+|[A-Z][A-Za-z0-9 _-]{0,60}:|"
    r"(system|user|assistant|context|instructions|background|objective|task)\b[:\-]?)",
    re.IGNORECASE,
)

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore


@dataclass(frozen=True)
class SectionRankingConfig:
    mode: str = "off"
    token_budget: Optional[int] = None

    def enabled(self) -> bool:
        return self.mode not in {"", "off"} or self.token_budget is not None


def normalize_ranking_mode(mode: Optional[str]) -> str:
    if not mode:
        return "off"
    normalized = mode.strip().lower()
    if normalized in {
        "off",
        "bm25",
        "gzip",
        "tfidf",
        "semantic",
        "lexrank",
        "textrank",
        "luhn",
    }:
        return normalized
    raise ValueError(
        f"Unknown section ranking mode '{mode}'. "
        "Strict mode requires explicit valid ranking mode selection."
    )


def resolve_section_ranking(
    *,
    default_mode: str,
    default_token_budget: Optional[int],
    override: Optional[Dict[str, Any]],
) -> SectionRankingConfig:
    base = SectionRankingConfig(
        mode=normalize_ranking_mode(default_mode),
        token_budget=default_token_budget,
    )

    if not override:
        return base

    mode_override = override.get("mode")
    token_budget = override.get("token_budget")

    # Determine the resolved mode
    resolved_mode = (
        normalize_ranking_mode(mode_override)
        if mode_override is not None
        else base.mode
    )

    # Resolve budgets
    resolved_token_budget = base.token_budget
    if "token_budget" in override:
        if isinstance(token_budget, int) and token_budget > 0:
            resolved_token_budget = token_budget
        else:
            resolved_token_budget = None

    # Special handling when mode is explicitly set to "off":
    # If the caller provided ONLY mode="off" with no budgets,
    # clear inherited values to allow opting out of server defaults.
    if (
        mode_override is not None
        and normalize_ranking_mode(mode_override) == "off"
        and "token_budget" not in override
    ):
        resolved_token_budget = None

    return SectionRankingConfig(
        mode=resolved_mode,
        token_budget=resolved_token_budget,
    )


def _tokenize_for_scoring(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _sample_text_for_encoder(
    text: str,
    *,
    tokenizer: Optional[Any],
    max_tokens: int,
) -> str:
    if not text:
        return text

    max_tokens = max(max_tokens, 1)
    if (
        tokenizer is not None
        and hasattr(tokenizer, "encode")
        and hasattr(tokenizer, "decode")
    ):
        try:
            token_ids = tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
        except Exception:  # pragma: no cover - tokenizer edge cases
            token_ids = None

        if token_ids is not None:
            if len(token_ids) <= max_tokens:
                return text
            chunk = max(1, max_tokens // 3)
            mid_start = max(len(token_ids) // 2 - chunk // 2, 0)
            sampled_ids = (
                token_ids[:chunk]
                + token_ids[mid_start : mid_start + chunk]
                + token_ids[-chunk:]
            )
            try:
                return tokenizer.decode(sampled_ids)
            except Exception:  # pragma: no cover - tokenizer decode failures
                pass

    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    chunk = max(1, max_chars // 3)
    mid = len(text) // 2
    return text[:chunk] + text[mid - chunk // 2 : mid + chunk // 2] + text[-chunk:]


def _sample_texts_for_encoder(
    texts: Sequence[str],
    *,
    tokenizer: Optional[Any],
    max_tokens: int,
) -> List[str]:
    return [
        _sample_text_for_encoder(text, tokenizer=tokenizer, max_tokens=max_tokens)
        for text in texts
    ]


def _extract_header_line(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _identify_header_sections(specs: Sequence[_chunking.ChunkSpec]) -> List[int]:
    header_indices: List[int] = []
    for index, spec in enumerate(specs):
        line = _extract_header_line(spec.text)
        if not line:
            continue
        if _CONTEXT_HEADER_PATTERN.match(line):
            header_indices.append(index)
    return header_indices


def _resolve_auto_token_budget(
    *,
    prompt_token_count: int,
    candidate_size: int,
) -> Optional[int]:
    """Resolve dynamic section-ranking budget when no explicit budget is provided."""
    if prompt_token_count <= 0 or prompt_token_count <= candidate_size:
        return None

    auto_budget = max(
        candidate_size,
        int(prompt_token_count * _AUTO_TOKEN_BUDGET_RATIO),
    )
    if auto_budget >= prompt_token_count:
        return None
    return max(1, auto_budget)


def _score_sections_bm25(
    specs: Sequence[_chunking.ChunkSpec],
    prompt: str,
) -> Dict[int, float]:
    tokens_docs = [_tokenize_for_scoring(spec.text) for spec in specs]
    if not tokens_docs:
        return {}

    doc_lengths = [len(tokens) or 1 for tokens in tokens_docs]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    query_terms = Counter(_tokenize_for_scoring(prompt))
    if not query_terms:
        return {index: 0.0 for index in range(len(specs))}

    df: Counter[str] = Counter()
    for tokens in tokens_docs:
        df.update(set(tokens))

    scores: Dict[int, float] = {}
    k1 = 1.5
    b = 0.75
    for index, tokens in enumerate(tokens_docs):
        if not tokens:
            scores[index] = 0.0
            continue

        term_freq = Counter(tokens)
        doc_length = len(tokens)
        score = 0.0
        for term, q_freq in query_terms.items():
            freq = term_freq.get(term)
            if not freq:
                continue
            doc_occurrences = df.get(term, 0)
            idf = math.log(
                1.0
                + (len(tokens_docs) - doc_occurrences + 0.5) / (doc_occurrences + 0.5)
            )
            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * doc_length / avgdl)
            score += (
                idf * (numerator / max(denominator, 1e-9)) * (q_freq / (q_freq + 1.0))
            )
        scores[index] = score

    return scores


def _score_sections_gzip(
    specs: Sequence[_chunking.ChunkSpec],
    prompt: str,
) -> Dict[int, float]:
    prompt_bytes = prompt.encode("utf-8")
    compressed_prompt = len(gzip.compress(prompt_bytes)) or 1
    scores: Dict[int, float] = {}

    for index, spec in enumerate(specs):
        section_bytes = spec.text.encode("utf-8")
        if not section_bytes:
            scores[index] = 0.0
            continue
        compressed_section = len(gzip.compress(section_bytes)) or 1
        combined = len(gzip.compress(section_bytes + prompt_bytes))
        min_comp = min(compressed_prompt, compressed_section)
        max_comp = max(compressed_prompt, compressed_section) or 1
        normalized = 1.0 - ((combined - min_comp) / max_comp)
        scores[index] = max(normalized, 0.0)

    return scores


def _score_sections_tfidf(
    specs: Sequence[_chunking.ChunkSpec],
    prompt: str,
) -> Dict[int, float]:
    tokens_docs = [_tokenize_for_scoring(spec.text) for spec in specs]
    if not tokens_docs:
        return {}

    prompt_tokens = _tokenize_for_scoring(prompt)
    if not prompt_tokens:
        return {index: 0.0 for index in range(len(specs))}

    corpus = tokens_docs + [prompt_tokens]
    doc_count = len(corpus)
    df: Counter[str] = Counter()
    for tokens in corpus:
        df.update(set(tokens))

    def build_vector(tokens: List[str]) -> Tuple[Dict[str, float], float]:
        if not tokens:
            return {}, 0.0
        counts = Counter(tokens)
        length = len(tokens)
        vector: Dict[str, float] = {}
        for term, count in counts.items():
            tf = count / length
            idf = math.log((doc_count + 1) / (df.get(term, 0) + 1)) + 1.0
            vector[term] = tf * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm

    prompt_vector, prompt_norm = build_vector(prompt_tokens)
    if prompt_norm == 0.0:
        return {index: 0.0 for index in range(len(specs))}

    scores: Dict[int, float] = {}
    for index, tokens in enumerate(tokens_docs):
        doc_vector, doc_norm = build_vector(tokens)
        if doc_norm == 0.0:
            scores[index] = 0.0
            continue
        dot = sum(
            prompt_vector.get(term, 0.0) * weight for term, weight in doc_vector.items()
        )
        scores[index] = dot / (prompt_norm * doc_norm)

    return scores


def _score_sections_lexrank(
    specs: Sequence[_chunking.ChunkSpec],
) -> Dict[int, float]:
    tokens_docs = [_tokenize_for_scoring(spec.text) for spec in specs]
    if not tokens_docs:
        return {}

    corpus = tokens_docs
    doc_count = len(corpus)
    df: Counter[str] = Counter()
    for tokens in corpus:
        df.update(set(tokens))

    def build_vector(tokens: List[str]) -> Tuple[Dict[str, float], float]:
        if not tokens:
            return {}, 0.0
        counts = Counter(tokens)
        length = len(tokens)
        vector: Dict[str, float] = {}
        for term, count in counts.items():
            tf = count / length
            idf = math.log((doc_count + 1) / (df.get(term, 0) + 1)) + 1.0
            vector[term] = tf * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm

    vectors = [build_vector(tokens) for tokens in tokens_docs]
    scores: Dict[int, float] = {}
    for idx, (vec, norm) in enumerate(vectors):
        if norm == 0.0:
            scores[idx] = 0.0
            continue
        centrality = 0.0
        for jdx, (other_vec, other_norm) in enumerate(vectors):
            if jdx == idx or other_norm == 0.0:
                continue
            dot = sum(vec.get(term, 0.0) * weight for term, weight in other_vec.items())
            centrality += dot / (norm * other_norm)
        scores[idx] = centrality
    return scores


def _score_sections_luhn(
    specs: Sequence[_chunking.ChunkSpec],
) -> Dict[int, float]:
    tokens_docs = [_tokenize_for_scoring(spec.text) for spec in specs]
    if not tokens_docs:
        return {}

    global_counts: Counter[str] = Counter()
    for tokens in tokens_docs:
        global_counts.update(token for token in tokens if len(token) > 2)

    keywords = {term for term, _ in global_counts.most_common(_LUHN_KEYWORDS_LIMIT)}
    if not keywords:
        return {index: 0.0 for index in range(len(specs))}

    scores: Dict[int, float] = {}
    for index, tokens in enumerate(tokens_docs):
        if not tokens:
            scores[index] = 0.0
            continue
        keyword_hits = sum(1 for token in tokens if token in keywords)
        scores[index] = keyword_hits / max(len(tokens), 1)
    return scores


def _score_sections_semantic(
    specs: Sequence[_chunking.ChunkSpec],
    prompt: str,
    model_name: Optional[str],
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[int, float]], Optional[Any]]:
    if not model_name or np is None:
        return None, None

    encoder = _metrics._load_encoder(model_name, "semantic_rank")
    if encoder is None:
        return None, None

    max_length = _metrics._resolve_max_sequence_length(encoder)
    tokenizer = getattr(encoder, "tokenizer", None)
    prompt_text = prompt
    section_texts = [spec.text for spec in specs]
    texts = [prompt_text] + section_texts
    if not _metrics._texts_fit_encoder(encoder, texts):
        prompt_text = _sample_text_for_encoder(
            prompt_text, tokenizer=tokenizer, max_tokens=max_length
        )
        section_texts = _sample_texts_for_encoder(
            section_texts, tokenizer=tokenizer, max_tokens=max_length
        )
        texts = [prompt_text] + section_texts
        if not _metrics._texts_fit_encoder(encoder, texts):
            return None, None

    embeddings = _metrics.encode_texts_with_plan(
        texts,
        model_name,
        model_type="semantic_rank",
        embedding_cache=embedding_cache,
        semantic_plan=semantic_plan,
    )
    if embeddings is None:
        return None, None

    query_vec = embeddings[0]
    section_vecs = embeddings[1:]
    scores = {
        index: float(np.dot(query_vec, section_vecs[index]))
        for index in range(len(specs))
    }
    return scores, section_vecs


def _score_section_candidates(
    specs: Sequence[_chunking.ChunkSpec],
    prompt: str,
    mode: str,
) -> Dict[int, float]:
    if not specs:
        return {}

    if mode == "bm25":
        return _score_sections_bm25(specs, prompt)
    if mode == "gzip":
        return _score_sections_gzip(specs, prompt)
    if mode == "tfidf":
        return _score_sections_tfidf(specs, prompt)
    if mode in {"lexrank", "textrank"}:
        return _score_sections_lexrank(specs)
    if mode == "luhn":
        return _score_sections_luhn(specs)
    # Default: keep original ordering preference
    return {index: float(len(specs) - index) for index in range(len(specs))}


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(_tokenize_for_scoring(a))
    tokens_b = set(_tokenize_for_scoring(b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(overlap) / max(len(union), 1)


def _mmr_order(
    indices: Sequence[int],
    scores: Dict[int, float],
    similarity_fn: Callable[[int, int], float],
) -> List[int]:
    remaining = list(indices)
    ordered: List[int] = []

    while remaining:
        if not ordered:
            best = max(remaining, key=lambda idx: scores.get(idx, 0.0))
            ordered.append(best)
            remaining.remove(best)
            continue

        def mmr_value(idx: int) -> float:
            redundancy = max(similarity_fn(idx, chosen) for chosen in ordered)
            return _MMR_LAMBDA * scores.get(idx, 0.0) - (1.0 - _MMR_LAMBDA) * redundancy

        best = max(remaining, key=mmr_value)
        ordered.append(best)
        remaining.remove(best)

    return ordered


def _reorder_for_attention(
    indices: Sequence[int], scores: Dict[int, float]
) -> List[int]:
    ranked = sorted(indices, key=lambda idx: scores.get(idx, 0.0), reverse=True)
    if not ranked:
        return []

    split_point = max(1, len(ranked) // 3)
    high_priority = ranked[:split_point]
    medium_priority = ranked[split_point:]

    reordered: List[int] = []
    for index, idx in enumerate(high_priority):
        if index % 2 == 0:
            reordered.insert(0, idx)
        else:
            reordered.append(idx)

    mid_point = len(reordered) // 2
    for idx in medium_priority:
        reordered.insert(mid_point, idx)
        mid_point += 1

    return reordered


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    return [part.strip() for part in parts if part and part.strip()]




def _resolve_planned_units(
    semantic_plan: Optional[Dict[str, Any]],
    *,
    prompt: str,
    unit: str,
) -> Optional[List[str]]:
    if not isinstance(semantic_plan, dict):
        return None
    if semantic_plan.get("prompt") != prompt:
        return None
    units = semantic_plan.get("units")
    if not isinstance(units, dict):
        return None
    resolved = units.get(unit)
    if not isinstance(resolved, list):
        return None
    return [str(item).strip() for item in resolved if str(item).strip()]

def query_aware_compress(
    *,
    prompt: str,
    query: str,
    model_name: Optional[str],
    budget_ratio: float,
    count_tokens: Callable[[str], int],
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    if not model_name:
        return prompt, False, {}
    if not prompt.strip() or not query.strip():
        return prompt, False, {}

    if model_name is None or np is None:
        return prompt, False, {}

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens < _QUERY_AWARE_MIN_TOKENS:
        return prompt, False, {}

    sentences = _resolve_planned_units(
        semantic_plan,
        prompt=prompt,
        unit="sentences",
    ) or _split_sentences(prompt)
    if len(sentences) < _QUERY_AWARE_MIN_SENTENCES:
        return prompt, False, {}

    texts = [query] + sentences
    embeddings = _metrics.encode_texts_with_plan(
        texts,
        model_name,
        model_type="semantic_rank",
        embedding_cache=embedding_cache,
        semantic_plan=semantic_plan,
    )
    if embeddings is None:
        return prompt, False, {}

    query_vec = embeddings[0]
    sentence_vecs = embeddings[1:]
    scores = [float(np.dot(query_vec, vector)) for vector in sentence_vecs]

    budget_tokens = max(1, int(prompt_tokens * max(0.1, min(budget_ratio, 0.95))))
    tokens_per_sentence = [count_tokens(sentence) for sentence in sentences]

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    selected: List[int] = []
    total_tokens = 0
    for index, _score in ranked:
        sentence_tokens = tokens_per_sentence[index]
        if selected and total_tokens + sentence_tokens > budget_tokens:
            continue
        selected.append(index)
        total_tokens += sentence_tokens
        if total_tokens >= budget_tokens:
            break

    if not selected:
        selected = [ranked[0][0]]
        total_tokens = tokens_per_sentence[selected[0]]

    selected.sort()
    compressed = " ".join(sentences[index] for index in selected).strip()
    if not compressed:
        return prompt, False, {}

    return (
        compressed,
        True,
        {
            "selected_indices": selected,
            "budget_tokens": budget_tokens,
            "total_tokens": total_tokens,
        },
    )


def apply_section_ranking(
    *,
    optimizer,
    prompt: str,
    ranking: SectionRankingConfig,
    chunking_mode: Optional[str],
    chat_metadata: Optional[Dict[str, Any]],
    default_chunking_mode: str,
    chunk_size: int,
    semantic_model: Optional[str],
    semantic_rank_model: Optional[str],
    semantic_similarity: float,
    count_tokens: Callable[[str], int],
    content_profile: Optional[Any] = None,
    prompt_tokens: Optional[Sequence[int]] = None,
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    if not prompt:
        return prompt, False, {"selected_indices": []}

    mode = normalize_ranking_mode(ranking.mode)
    is_enabled = ranking.enabled() or mode != "off"
    if not is_enabled:
        return prompt, False, {"selected_indices": []}

    strategy = _chunking.resolve_strategy(chunking_mode or default_chunking_mode)
    if strategy == "off":
        strategy = "structured"

    candidate_size = ranking.token_budget or min(chunk_size, 4096)
    candidate_size = max(candidate_size, 1)

    chunk_specs, joiner = _chunking.chunk_prompt(
        optimizer,
        prompt,
        chunk_size=candidate_size,
        strategy=strategy,
        chat_metadata=chat_metadata,
        semantic_model=semantic_model,
        similarity_threshold=semantic_similarity,
        prompt_tokens=prompt_tokens,
        embedding_cache=embedding_cache,
        semantic_plan=semantic_plan,
    )

    if not chunk_specs:
        return prompt, False, {"selected_indices": []}

    profile_name = getattr(content_profile, "name", None)
    allow_reorder = profile_name in {
        "general_prose",
        "markdown",
        "technical_doc",
        "heavy_document",
    }

    prompt_token_count = count_tokens(prompt)
    effective_token_budget = ranking.token_budget
    if effective_token_budget is None:
        effective_token_budget = _resolve_auto_token_budget(
            prompt_token_count=prompt_token_count,
            candidate_size=candidate_size,
        )
    auto_semantic = (
        mode in {"bm25", "tfidf"}
        and allow_reorder
        and prompt_token_count >= _SEMANTIC_RANK_MIN_TOKENS
    )
    semantic_scores = None
    semantic_vectors = None
    resolved_mode = mode
    if mode == "semantic" or auto_semantic:
        semantic_scores, semantic_vectors = _score_sections_semantic(
            chunk_specs,
            prompt,
            semantic_rank_model,
            embedding_cache=embedding_cache,
            semantic_plan=semantic_plan,
        )
        if semantic_scores is not None:
            resolved_mode = "semantic"
        else:
            resolved_mode = "lexrank" if allow_reorder else "luhn"

    if semantic_scores is not None:
        scores = semantic_scores
    else:
        scores = _score_section_candidates(chunk_specs, prompt, resolved_mode)
    tokens_per_chunk = [count_tokens(spec.text) for spec in chunk_specs]

    selected: List[int] = []
    total_tokens = 0

    header_indices = _identify_header_sections(chunk_specs)
    retained_header_indices: List[int] = []
    if header_indices:
        for index in header_indices:
            prospective_tokens = total_tokens + tokens_per_chunk[index]
            if (
                effective_token_budget is not None
                and prospective_tokens > effective_token_budget
            ):
                continue
            selected.append(index)
            total_tokens = prospective_tokens
            retained_header_indices.append(index)

    sorted_candidates = sorted(
        ((index, scores.get(index, 0.0)) for index in range(len(chunk_specs))),
        key=lambda item: item[1],
        reverse=True,
    )

    candidate_indices = [index for index, _score in sorted_candidates]
    use_mmr = allow_reorder and len(candidate_indices) >= _MMR_MIN_SECTIONS
    if use_mmr:
        if semantic_vectors is not None and np is not None:

            def similarity_fn(a: int, b: int) -> float:
                return float(np.dot(semantic_vectors[a], semantic_vectors[b]))

        else:

            def similarity_fn(a: int, b: int) -> float:
                return _jaccard_similarity(chunk_specs[a].text, chunk_specs[b].text)

        candidate_indices = _mmr_order(candidate_indices, scores, similarity_fn)

    for index in candidate_indices:
        if index in selected:
            continue
        prospective_tokens = total_tokens + tokens_per_chunk[index]
        if (
            effective_token_budget is not None
            and prospective_tokens > effective_token_budget
        ):
            continue

        selected.append(index)
        total_tokens = prospective_tokens

    if not selected:
        best_index = max(sorted_candidates, key=lambda item: item[1], default=(0, 0.0))[
            0
        ]
        selected = [best_index]

    selected_sorted = sorted(selected)
    reorder_enabled = allow_reorder and len(selected_sorted) > 2
    if selected_sorted == list(range(len(chunk_specs))) and not reorder_enabled:
        return prompt, False, {"selected_indices": selected_sorted}

    if reorder_enabled:
        selected_sorted = _reorder_for_attention(selected_sorted, scores)

    merged = joiner.join(chunk_specs[index].text for index in selected_sorted)
    metadata = {
        "selected_indices": selected_sorted,
        "total_tokens": total_tokens,
        "token_budget": effective_token_budget,
        "ranking_mode": resolved_mode,
        "reordered": reorder_enabled,
        "header_indices": retained_header_indices,
    }
    return merged, True, metadata
