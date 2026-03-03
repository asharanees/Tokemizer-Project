"""Chunking helpers for extremely large prompts.

This module centralizes the logic that previously lived in ``truncation`` so the
optimizer can support multiple chunking strategies without spreading the
implementation across unrelated helpers. The strategies currently supported are:

* ``fixed``: token-aligned chunks with optional overlap
* ``structured``: recursively split on headings/paragraphs to keep related
  context together
* ``semantic``: group paragraphs using embedding similarity when available

All strategies capture lightweight metadata so the merged prompt can be restored
without losing placeholder tokens or structured chat turns.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import history, metrics


PLACEHOLDER_RE = re.compile(r"__(?P<prefix>[A-Za-z0-9]+)_(?P<index>\d+)__")
_JSON_HEURISTIC_SENTINELS = ("{", "[", "/")

DEFAULT_STRATEGY = "fixed"
CHUNKING_STRATEGIES = {"fixed", "structured", "semantic"}


@dataclass(slots=True)
class ChunkSpec:
    """Container describing a chunk extracted from the prompt."""

    text: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_strategy(strategy: Optional[str]) -> str:
    if not strategy:
        return DEFAULT_STRATEGY
    normalized = strategy.strip().lower()
    if normalized in CHUNKING_STRATEGIES:
        return normalized
    if normalized in {"none", "off", "disabled"}:
        return "off"
    return DEFAULT_STRATEGY


def _placeholder_metadata(text: str) -> Dict[str, Any]:
    placeholders = PLACEHOLDER_RE.findall(text)
    counts: Dict[str, int] = {}
    for prefix, _ in placeholders:
        counts[prefix] = counts.get(prefix, 0) + 1
    return {"placeholders": counts}


def _placeholder_tokens(text: str) -> Set[str]:
    return {match.group(0) for match in PLACEHOLDER_RE.finditer(text)}


def _find_segment(prompt: str, fragment: str, start_hint: int) -> Tuple[int, int]:
    if not fragment:
        return start_hint, start_hint
    index = prompt.find(fragment, start_hint)
    if index < 0:
        index = start_hint
    return index, index + len(fragment)


def _likely_contains_json_structures(
    text: str, *, json_policy: Optional[Dict[str, Any]] = None
) -> bool:
    if json_policy and json_policy.get("default"):
        return True
    if not text:
        return False
    return any(marker in text for marker in _JSON_HEURISTIC_SENTINELS)


def _chunk_fixed(
    prompt: str,
    tokenizer,
    chunk_size: int,
    overlap_ratio: float,
    prompt_tokens: Optional[Sequence[int]] = None,
) -> Tuple[List[ChunkSpec], str]:
    if chunk_size <= 0:
        chunk_size = max(len(prompt), 1)

    overlap_tokens = max(int(chunk_size * overlap_ratio), 0)
    step = max(chunk_size - overlap_tokens, 1)

    if tokenizer is None:
        approximate_chars = max(chunk_size * 4, 1)
        specs: List[ChunkSpec] = []
        for start in range(
            0, len(prompt), approximate_chars - max(approximate_chars // 10, 1)
        ):
            end = min(start + approximate_chars, len(prompt))
            fragment = prompt[start:end]
            metadata = {"strategy": "fixed"}
            metadata.update(_placeholder_metadata(fragment))
            specs.append(ChunkSpec(fragment, start, end, metadata))
        return (
            specs or [ChunkSpec(prompt, 0, len(prompt), {"strategy": "fixed"})],
            "\n\n",
        )

    tokens: Sequence[int]
    if prompt_tokens is not None:
        tokens = list(prompt_tokens)
    else:
        tokens = tokenizer.encode(prompt)
    specs: List[ChunkSpec] = []
    cursor = 0
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        fragment = tokenizer.decode(chunk_tokens)
        segment_start, segment_end = _find_segment(prompt, fragment, cursor)
        metadata = {
            "strategy": "fixed",
            "token_start": start,
            "token_end": start + len(chunk_tokens),
        }
        metadata.update(_placeholder_metadata(fragment))
        specs.append(ChunkSpec(fragment, segment_start, segment_end, metadata))
        cursor = segment_end

    return specs or [ChunkSpec(prompt, 0, len(prompt), {"strategy": "fixed"})], "\n\n"


def _split_by_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def _group_sections(
    prompt: str,
    sections: Iterable[str],
    *,
    chunk_size: int,
    count_tokens,
    strategy: str,
    joiner: str,
) -> List[ChunkSpec]:
    specs: List[ChunkSpec] = []
    section_list = [section for section in sections if section]
    if not section_list:
        return [ChunkSpec(prompt, 0, len(prompt), {"strategy": strategy})]

    buffer: List[str] = []
    buffer_tokens = 0
    cursor = 0
    approximate_tokens = chunk_size >= 8000

    if approximate_tokens:
        tokens_list = [max(len(section) // 4, 1) for section in section_list]
    else:
        owner = getattr(count_tokens, "__self__", None)
        use_parallel = len(section_list) > 1
        if use_parallel:
            executor = None
            shutdown_executor = False
            get_executor = (
                getattr(owner, "_get_chunk_executor", None) if owner else None
            )
            if callable(get_executor):
                try:
                    executor = get_executor()
                except TypeError:
                    executor = get_executor(None)
            if executor is None:
                max_workers = min(len(section_list), os.cpu_count() or 4)
                executor = ThreadPoolExecutor(max_workers=max_workers)
                shutdown_executor = True
            tokens_list = list(executor.map(count_tokens, section_list))
            if shutdown_executor:
                executor.shutdown(wait=True)
        else:
            tokens_list = [count_tokens(section) for section in section_list]

    for section, tokens in zip(section_list, tokens_list):
        if buffer and buffer_tokens + tokens > chunk_size:
            fragment = joiner.join(buffer).strip()
            start, end = _find_segment(prompt, fragment, cursor)
            metadata = {"strategy": strategy, "section_count": len(buffer)}
            metadata.update(_placeholder_metadata(fragment))
            specs.append(ChunkSpec(fragment, start, end, metadata))
            cursor = end
            buffer = []
            buffer_tokens = 0

        buffer.append(section)
        buffer_tokens += tokens

    if buffer:
        fragment = joiner.join(buffer).strip()
        start, end = _find_segment(prompt, fragment, cursor)
        metadata = {"strategy": strategy, "section_count": len(buffer)}
        metadata.update(_placeholder_metadata(fragment))
        specs.append(ChunkSpec(fragment, start, end, metadata))

    return specs or [ChunkSpec(prompt, 0, len(prompt), {"strategy": strategy})]


def _chunk_structured(opt, prompt: str, chunk_size: int) -> Tuple[List[ChunkSpec], str]:
    sections = opt._split_by_headings(prompt)
    if len(sections) <= 1:
        sections = _split_by_paragraphs(prompt)

    specs = _group_sections(
        prompt,
        sections,
        chunk_size=max(chunk_size, 1),
        count_tokens=opt.count_tokens,
        strategy="structured",
        joiner="\n\n",
    )
    return specs, "\n\n"


def _chunk_chat(
    prompt: str, chunk_size: int, count_tokens
) -> Tuple[List[ChunkSpec], str]:
    segments = history.parse_chat_segments(prompt)
    if not segments:
        return [ChunkSpec(prompt, 0, len(prompt), {"strategy": "structured"})], "\n"

    lines: List[str] = []
    for segment in segments:
        role = segment.get("role")
        content = (segment.get("content") or "").strip()
        if role:
            line = f"{role}: {content}".rstrip()
        else:
            line = content
        if line:
            lines.append(line)

    specs = _group_sections(
        prompt,
        lines,
        chunk_size=max(chunk_size, 1),
        count_tokens=count_tokens,
        strategy="structured",
        joiner="\n",
    )
    for spec in specs:
        spec.metadata.setdefault("separator", "\n")
    return specs, "\n"



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


def _chunk_semantic(
    opt,
    prompt: str,
    chunk_size: int,
    *,
    model_name: str,
    similarity_threshold: float,
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChunkSpec], str]:
    paragraphs = _resolve_planned_units(
        semantic_plan,
        prompt=prompt,
        unit="paragraphs",
    ) or _split_by_paragraphs(prompt)
    if not paragraphs:
        return [ChunkSpec(prompt, 0, len(prompt), {"strategy": "semantic"})], "\n\n"

    embeddings = metrics.encode_texts_with_plan(
        paragraphs,
        model_name,
        model_type="semantic_chunk",
        embedding_cache=embedding_cache,
        semantic_plan=semantic_plan,
    )
    if embeddings is None:
        raise RuntimeError(
            "Semantic chunking embeddings are unavailable; strict mode forbids "
            "fallback to structured chunking."
        )

    vectors = list(embeddings)
    if not vectors:
        raise RuntimeError(
            "Semantic chunking produced no embeddings; strict mode requires "
            "semantic chunk vectors."
        )

    def similarity(a, b) -> float:
        if metrics.np is None:
            return 1.0 if a == b else 0.0
        return float(metrics.np.dot(a, b))

    groups: List[List[str]] = []
    current: List[str] = [paragraphs[0]]
    for index in range(1, len(paragraphs)):
        prev_vec = vectors[index - 1]
        current_vec = vectors[index]
        score = similarity(prev_vec, current_vec)
        if score >= similarity_threshold:
            current.append(paragraphs[index])
        else:
            groups.append(current)
            current = [paragraphs[index]]
    groups.append(current)

    flat_sections: List[str] = []
    for group in groups:
        flat_sections.append("\n\n".join(group).strip())

    specs = _group_sections(
        prompt,
        flat_sections,
        chunk_size=max(chunk_size, 1),
        count_tokens=opt.count_tokens,
        strategy="semantic",
        joiner="\n\n",
    )
    return specs, "\n\n"


def chunk_prompt(
    opt,
    prompt: str,
    *,
    chunk_size: int,
    strategy: Optional[str],
    chat_metadata: Optional[Dict[str, Any]] = None,
    overlap_ratio: float = 0.1,
    semantic_model: Optional[str] = None,
    similarity_threshold: float = 0.6,
    prompt_tokens: Optional[Sequence[int]] = None,
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChunkSpec], str]:
    normalized = _normalize_strategy(strategy)
    if normalized == "off":
        return [ChunkSpec(prompt, 0, len(prompt), {"strategy": "off"})], "\n\n"

    if chat_metadata and not chat_metadata.get("skip_roles"):
        specs, joiner = _chunk_chat(prompt, chunk_size, opt.count_tokens)
        return specs, joiner

    if normalized == "structured":
        return _chunk_structured(opt, prompt, chunk_size)

    if normalized == "semantic" and semantic_model:
        return _chunk_semantic(
            opt,
            prompt,
            chunk_size,
            model_name=semantic_model,
            similarity_threshold=similarity_threshold,
            embedding_cache=embedding_cache,
            semantic_plan=semantic_plan,
        )

    if normalized == "semantic":
        raise RuntimeError(
            "Semantic chunking requested without a semantic model; strict mode "
            "forbids fallback to structured chunking."
        )

    return _chunk_fixed(
        prompt,
        getattr(opt, "tokenizer", None),
        chunk_size,
        overlap_ratio,
        prompt_tokens,
    )


class _PlaceholderNormalizer:
    def __init__(self, preserved: Optional[Iterable[str]] = None):
        self.preserved: Set[str] = set(preserved or [])
        self.prefix_counts: Dict[str, int] = {}
        for token in self.preserved:
            match = PLACEHOLDER_RE.fullmatch(token)
            if not match:
                continue
            prefix = match.group("prefix")
            index = int(match.group("index"))
            current = self.prefix_counts.get(prefix, 0)
            if index + 1 > current:
                self.prefix_counts[prefix] = index + 1

    def normalize(self, text: str) -> str:
        replacements: Dict[str, str] = {}

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in self.preserved:
                return token

            prefix = match.group("prefix")
            mapped = replacements.get(token)
            if mapped is None:
                next_index = self.prefix_counts.get(prefix, 0)
                mapped = f"__{prefix}_{next_index}__"
                while mapped in self.preserved:
                    next_index += 1
                    mapped = f"__{prefix}_{next_index}__"
                self.prefix_counts[prefix] = next_index + 1
                replacements[token] = mapped
            return mapped

        return PLACEHOLDER_RE.sub(repl, text)


def merge_chunks(first: str, second: str) -> str:
    """Overlap-aware merge used by the fixed chunking strategy."""

    sentences1 = first.split(". ")
    sentences2 = second.split(". ")

    overlap_size = 0
    max_overlap = min(len(sentences1), len(sentences2), 3)

    for i in range(1, max_overlap + 1):
        if sentences1[-i:] == sentences2[:i]:
            overlap_size = i

    if overlap_size > 0:
        merged_sentences = sentences1 + sentences2[overlap_size:]
        return ". ".join(merged_sentences)

    if first.endswith("\n") or first.endswith("\n\n"):
        separator = ""
    else:
        separator = "\n\n"
    return f"{first}{separator}{second}"


def _join_chunks(chunks: Sequence[str], *, strategy: str, joiner: str) -> str:
    if not chunks:
        return ""

    if strategy == "fixed":
        merged = chunks[0]
        for chunk in chunks[1:]:
            merged = merge_chunks(merged, chunk)
        return merged

    cleaned = [chunk for chunk in chunks if chunk.strip()]
    if not cleaned:
        return ""
    return joiner.join(cleaned)


def _profile_step(profiler: Optional[Any], name: str):
    if profiler is None:
        return nullcontext()
    return profiler.step(name)


def optimize_with_chunking(
    opt,
    prompt: str,
    mode: str,
    optimization_mode: str,
    chunk_size: int,
    *,
    strategy: Optional[str] = None,
    overlap_ratio: Optional[float] = None,
    enable_frequency_learning: bool = False,
    use_discourse_weighting: bool = True,
    force_preserve_digits: Optional[bool] = None,
    segment_spans: Optional[Sequence[Dict[str, Any]]] = None,
    chat_metadata: Optional[Dict[str, Any]] = None,
    semantic_model: Optional[str] = None,
    similarity_threshold: float = 0.6,
    profiler: Optional[Any] = None,
    telemetry_collector: Optional[Any] = None,
    json_policy: Optional[Dict[str, Any]] = None,
    token_cache: Optional[Dict[str, List[int]]] = None,
    enable_toon_conversion: bool = False,
    content_type: Optional[str] = None,
    content_profile: Optional[Any] = None,
    embedding_cache: Optional[Dict[Tuple[str, str, str], Any]] = None,
    semantic_plan: Optional[Dict[str, Any]] = None,
    custom_canonicals: Optional[Dict[str, str]] = None,
    force_disabled_passes: Optional[Collection[str]] = None,
    customer_id: Optional[str] = None,
) -> Tuple[str, List[ChunkSpec]]:
    # CRITICAL: Pre-preserve JSON blocks BEFORE chunking to avoid splitting JSON mid-structure
    # This ensures large JSON documents remain protected even when prompt exceeds chunk size
    from . import preservation as _preservation

    pre_preserved: Dict[str, Any] = {
        "json_tokens": [],
        "json_literals": [],
        "json_strings": [],
        "toon_blocks": [],
        "toon_stats": {"conversions": 0, "bytes_saved": 0},
    }
    prompt_with_protected_json = prompt
    should_preserve_json = _likely_contains_json_structures(
        prompt, json_policy=json_policy
    )

    with _profile_step(profiler, "pre_preserve_json"):
        if should_preserve_json:
            prompt_with_protected_json = _preservation._preserve_json_blocks(
                prompt,
                json_policy,
                pre_preserved,
                enable_toon_conversion=enable_toon_conversion,
            )
            if enable_toon_conversion:
                toon_stats = pre_preserved.get(
                    "toon_stats", {"conversions": 0, "bytes_saved": 0}
                )
                opt._get_state().toon_stats = dict(
                    pre_preserved.get(
                        "toon_stats", {"conversions": 0, "bytes_saved": 0}
                    )
                )
                if toon_stats.get("conversions", 0) > 0:
                    if "TOON Conversion" not in opt.techniques_applied:
                        opt.techniques_applied.append("TOON Conversion")

    token_getter = getattr(opt, "tokenize", None)
    encoded_prompt = (
        token_getter(prompt_with_protected_json) if callable(token_getter) else None
    )

    with _profile_step(profiler, "chunk_prompt"):
        chunk_specs, joiner = chunk_prompt(
            opt,
            prompt_with_protected_json,  # Use JSON-protected version for chunking
            chunk_size=chunk_size,
            strategy=strategy,
            chat_metadata=chat_metadata,
            overlap_ratio=overlap_ratio if overlap_ratio is not None else 0.1,
            semantic_model=semantic_model,
            similarity_threshold=similarity_threshold,
            prompt_tokens=encoded_prompt,
            embedding_cache=embedding_cache,
            semantic_plan=semantic_plan,
        )

    # Extract placeholders from JSON-protected prompt to preserve them across chunks
    original_placeholders = _placeholder_tokens(prompt_with_protected_json)
    normalizer = _PlaceholderNormalizer(original_placeholders)

    def _span_overlaps_chunk(start: int, end: int, spec: ChunkSpec) -> bool:
        return start < spec.end and end > spec.start

    def _normalize_chunk_spans(spec: ChunkSpec) -> Optional[List[Dict[str, Any]]]:
        if not segment_spans:
            return None
        normalized: List[Dict[str, Any]] = []
        for span in segment_spans:
            if not isinstance(span, dict):
                continue
            try:
                start = int(span.get("start", 0))
                end = int(span.get("end", 0))
            except (TypeError, ValueError):
                continue
            if end <= start or not _span_overlaps_chunk(start, end, spec):
                continue
            normalized.append(
                {
                    "start": max(start, spec.start) - spec.start,
                    "end": min(end, spec.end) - spec.start,
                    "label": span.get("label"),
                    "weight": span.get("weight"),
                }
            )
        return normalized or None

    def _optimize_chunk(spec: ChunkSpec) -> Tuple[str, Dict[str, Any]]:
        with _profile_step(profiler, "chunk_optimize"):
            # Pass json_policy=None to skip JSON preservation in chunks (already done pre-chunking)
            return opt._run_pipeline_threadsafe(
                spec.text,
                mode,
                optimization_mode,
                enable_frequency_learning,
                force_preserve_digits,
                profiler=profiler,
                telemetry_collector=telemetry_collector,
                segment_spans=_normalize_chunk_spans(spec),
                use_discourse_weighting=use_discourse_weighting,
                json_policy=None,  # Skip JSON preservation in chunks - already pre-preserved
                token_cache=token_cache,
                embedding_cache=embedding_cache,
                content_type=content_type,
                content_profile=content_profile,
                customer_id=customer_id,
                custom_canonicals=custom_canonicals,
                force_disabled_passes=force_disabled_passes,
            )

    optimized_chunks: List[str] = []
    chunk_snapshots: List[Dict[str, Any]] = []

    if len(chunk_specs) > 1:
        cpu_workers = os.cpu_count() or 4
        base_limit = min(len(chunk_specs), cpu_workers)
        configured_limit = getattr(opt, "max_chunk_workers", base_limit) or base_limit
        worker_limit = max(1, min(base_limit, configured_limit))
        shutdown_executor = False
        executor = getattr(opt, "_get_chunk_executor", None)
        pool: ThreadPoolExecutor
        if callable(executor):
            try:
                pool = executor(worker_limit)
            except TypeError:
                pool = executor()
        else:
            pool = ThreadPoolExecutor(max_workers=worker_limit)
            shutdown_executor = True

        futures = [pool.submit(_optimize_chunk, spec) for spec in chunk_specs]
        for future in futures:
            optimized_chunk, snapshot = future.result()
            chunk_snapshots.append(snapshot)
            optimized_chunks.append(optimized_chunk)

        if shutdown_executor:
            pool.shutdown(wait=True)
    else:
        for spec in chunk_specs:
            optimized_chunk, snapshot = _optimize_chunk(spec)
            chunk_snapshots.append(snapshot)
            optimized_chunks.append(optimized_chunk)

    normalized_chunks: List[str] = []
    for optimized_chunk, snapshot in zip(optimized_chunks, chunk_snapshots):
        with _profile_step(profiler, "chunk_normalize"):
            normalized_chunk = normalizer.normalize(optimized_chunk)
        normalized_chunks.append(normalized_chunk)
        opt._merge_pipeline_snapshot(snapshot)

    with _profile_step(profiler, "chunk_join"):
        merged = _join_chunks(
            normalized_chunks,
            strategy=chunk_specs[0].metadata.get("strategy", DEFAULT_STRATEGY),
            joiner=chunk_specs[0].metadata.get("separator", joiner),
        )

    with _profile_step(profiler, "post_chunk_dedup"):
        merged = opt._apply_post_chunk_dedup(
            merged,
            content_profile=content_profile,
            preserved=pre_preserved,
            telemetry_collector=telemetry_collector,
        )

    # Restore pre-preserved JSON blocks after all chunks are merged
    with _profile_step(profiler, "restore_pre_preserved_json"):
        final_result = _preservation.restore(opt, merged, pre_preserved)

    return final_result, chunk_specs


__all__ = [
    "ChunkSpec",
    "chunk_prompt",
    "merge_chunks",
    "optimize_with_chunking",
    "resolve_strategy",
]


def resolve_strategy(strategy: Optional[str]) -> str:
    """Public helper mirroring the internal normalization logic."""

    return _normalize_strategy(strategy)
