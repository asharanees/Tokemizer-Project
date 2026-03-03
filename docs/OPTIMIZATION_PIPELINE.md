# Tokemizer Optimization Pipeline (Code-Accurate)

This document describes **exactly what the optimizer does in this codebase**, in the order it runs, and the conditions that enable/skip each pass.

Primary orchestration lives in:
- `backend/services/optimizer/core.py` (`PromptOptimizer.optimize()` and `PromptOptimizer._optimize_pipeline()`)
- `backend/services/optimizer/chunking.py` (chunked execution)
- `backend/services/optimizer/pipeline_config.py` (mode presets: disabled passes + toggles)
- `backend/services/optimizer/router.py` (content classification + profiles + SmartContext)

## Table of Contents

1. [How Passes Are Enabled/Skipped](#how-passes-are-enabledskipped)
2. [Optimization Mode Matrix](#optimization-mode-matrix)
3. [High-Level Stages in `PromptOptimizer.optimize()`](#high-level-stages-in-promptoptimizeroptimize)
4. [Core Pipeline: Pass-by-Pass](#core-pipeline-pass-by-pass-order--logic)
5. [Chunking Pipeline](#chunking-pipeline-when-prompts-exceed-chunk-thresholds)
6. [Fast Path](#fast-path-short-prompts)
7. [Configuration Touchpoints](#configuration-touchpoints-exact-names-used-in-code)

## How Passes Are Enabled/Skipped

The pipeline computes a set `resolved_disabled` and then checks `should_skip_pass(name)` (`name in resolved_disabled`) throughout `_optimize_pipeline()`.

Disabled sets come from:
- **Optimization mode** (`conservative` / `balanced` / `maximum`): `resolve_optimization_config()` in `pipeline_config.py` provides:
  - `disabled_passes` (always disabled for that mode)
  - `pass_toggles` (adds additional names into `resolved_disabled` when `False`)
  - `enable_toon_conversion` (affects preservation behavior)
- **Content profile** (e.g., `code`, `json`): `router.merge_disabled_passes()` unions mode-disabled with profile-disabled.
- **Request overrides**: optional `force_disabled_passes` merged in `optimize()` / `_optimize_pipeline()`.
- **Heuristics**: some passes add more names to `resolved_disabled` mid-run (e.g., low lexical gain can disable heavy passes).

## Optimization Mode Matrix

This table reflects **only** the default behavior driven by `pipeline_config.py` (mode `disabled_passes` + `pass_toggles`) plus a few mode-gated steps in `optimize()` (e.g., maximum-only token classifier pre-pass). Content profiles and runtime heuristics can further disable/skip passes even when a mode shows “Enabled”.

Legend:
- **Enabled**: not disabled by the mode preset.
- **Disabled**: disabled by the mode preset (or explicitly gated off by mode checks).
- **Conditional**: may run depending on inputs, content type/profile, configuration, or model availability.

| Pass / Stage | Conservative | Balanced | Maximum | Notes |
|---|---:|---:|---:|---|
| `dedup_normalized_sentences` (pre-chunk) | Disabled | Enabled | Enabled | Pre-pipeline; skipped for `code`/`json` profiles and when frequency learning is enabled. |
| `alias_json_keys` (inside preservation) | Disabled | Disabled | Enabled | Enables JSON key aliasing during `preserve_elements`. |
| `alias_references` | Disabled | Enabled | Enabled | Conservative sets toggle off. Also skipped for `code`/`json` and TOON blocks. |
| `compress_field_labels` | Disabled | Enabled | Enabled | Conservative sets toggle off. Skipped for `code`/`json`. |
| `compress_parentheticals` | Disabled | Enabled | Enabled | Conservative sets toggle off. Skipped for `code`/`json`. |
| `hoist_constraints` | Disabled | Enabled | Enabled | Conservative sets toggle off. Skipped for `code`/`json`. |
| `apply_symbolic_replacements` | Disabled | Disabled | Enabled | Lexical sub-pass (low-weight segments only). |
| `remove_articles` | Disabled | Disabled | Enabled | Lexical sub-pass (low-weight segments only). |
| `apply_macro_dictionary` | Disabled | Disabled | Enabled | Disabled by conservative and balanced mode presets. |
| `compress_coreferences` | Disabled | Enabled | Enabled | Disabled by conservative mode preset; can also be disabled by heuristics. |
| `compress_examples` | Disabled | Disabled | Enabled | Disabled by conservative/balanced presets; can also be skipped by heuristics. |
| `summarize_history` | Disabled | Disabled | Enabled | Disabled by conservative/balanced presets; can also be skipped by heuristics. |
| `prune_low_entropy` | Disabled | Enabled | Enabled | Disabled by conservative preset; can also be skipped by heuristics. |
| `maximum_prepass` (pre-pass in `optimize()`) | Disabled | Disabled | Conditional | Runtime policy resolves enablement/thresholds from prompt signals; `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED` explicitly forces on/off. Other maximum pre-pass env vars provide defaults and hard bounds/caps; hard-preserves protected spans/placeholders/constraints. |
| `token_classifier` (pre-pass in `optimize()`) | Disabled | Disabled | Conditional | Only considered in `maximum` mode, and only under additional gating (e.g., no `segment_spans`). |
| `token_classifier_post` | Conditional | Conditional | Conditional | Controlled by `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_PASS` and model availability (not mode-specific). |
| `enable_toon_conversion` (preservation option) | Disabled | Disabled | Enabled | TOON conversion is enabled only for maximum mode preset. |

## High-Level Stages in `PromptOptimizer.optimize()`

`optimize()` performs several pre-pipeline decisions/transforms before calling `_optimize_pipeline()` (directly or via chunking):

1. **Smart Router / Profile Resolution**
   - Determines `content_type` and `content_profile` (either passed in, derived from profile name, or classified from text).
   - Resolves `SmartContext` (routing decisions such as `chunking_mode`, `preserve_digits`, and `section_ranking_enabled`).

2. **Request-Scoped Semantic Plan Initialization**
   - `optimize()` builds a request-scoped semantic plan from the source prompt once per request.
   - The plan precomputes `sections`, `paragraphs`, and `sentences`, then shares these units across semantic ranking, query-aware compression, and semantic chunking.
   - Semantic embedding lookups are routed through `metrics.encode_texts_with_plan(...)`, which reuses planned vectors first and only encodes cache misses.
   - The plan tracks instrumentation counters: `embedding_reuse_count`, `embedding_calls_saved`, and `embedding_wall_clock_savings_ms`.

3. **Query-Aware Compression (optional)**
   - When a `query` hint exists and no `segment_spans` are provided, prose-like profiles can be compressed by `section_ranking.query_aware_compress()`.
   - Budget ratio is driven by `_QUERY_AWARE_BUDGET_BY_MODE` in `core.py` and profile threshold modifiers.

4. **Redundancy Estimation → Section Ranking (optional)**
   - `redundancy_ratio = _estimate_sentence_redundancy_ratio(working_prompt)`.
   - If `redundancy_ratio >= _REDUNDANCY_SECTION_RANKING_RATIO_THRESHOLD` and tokens exceed `_REDUNDANCY_SECTION_RANKING_TOKEN_MIN`, section ranking may be enabled even when size thresholds are not exceeded.
   - Section ranking is applied via `section_ranking.apply_section_ranking()` if the resolved ranking config is enabled.
   - If no explicit section budget is configured (`token_budget` unset or `0`), ranking resolves an automatic budget from prompt size and candidate chunk size.
   - Important detail: redundancy signatures are produced by `_normalized_sentence_signature()` and **preserve digits** (punctuation/underscores are stripped; numbers are kept). This avoids false “structured data redundancy” (e.g., `id: 123` vs `id: 456`).

5. **Pre-Chunk Normalized Sentence Dedup (optional)**
   - Runs for non-`code`/`json` profiles if `dedup_normalized_sentences` is enabled.
   - Also skipped when frequency-learning is enabled to avoid interfering with abbreviation discovery.
   - Removes repeated sentences based on `_normalized_sentence_signature()`; this is intentionally cheaper than semantic dedup.

6. **Maximum Budgeted Pre-Pass (maximum-only; optional)**
   - In `maximum` mode, the optimizer may run `max_prepass.budgeted_sentence_span_prepass(...)` before section ranking/chunking.
   - Runtime policy resolver adapts enablement and effective values (`minimum_tokens`, `budget_ratio`, `max_sentences`) using prompt tokens, content profile, query hint, and redundancy estimate.
   - `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED` acts as an explicit force/override when set; related env vars are treated as defaults/hard rails rather than fixed final behavior.
   - Scores candidate sentences/spans using query relevance, protection mask signal, redundancy score, and entropy-lite score.
   - Hard-preserves any content overlapping explicit protected ranges (`segment_spans`), placeholders, and detected constraints.

7. **Token Classifier (pre-pass, maximum-only; optional)**
   - In `maximum` mode, the optimizer may run a token-classifier-based compression pass before chunking/pipeline, selecting among candidates using semantic guard thresholds.

8. **Chunking (optional)**
   - If token count exceeds chunk thresholds and the resolved chunking strategy is enabled, execution goes through `chunking.optimize_with_chunking()`.
   - Each chunk runs `_optimize_pipeline()` independently, then the output is normalized/joined and post-processed.

Everything below describes the detailed “core pass list” implemented in `_optimize_pipeline()` (which is used in both non-chunked and chunked flows).

## Core Pipeline: Pass-by-Pass (Order + Logic)

All passes below run inside `PromptOptimizer._optimize_pipeline()` unless skipped. Pass names are the strings used in `should_skip_pass("...")` and telemetry.

Note: `core.py` labels some steps as “Pass N” in inline comments, but several steps are unnumbered. This document lists every independently skippable unit in exact execution order.

### Pass 0 — `remove_verbatim_duplicates`
**Goal**: remove repeated *large blocks* before placeholders make identical text appear different.

**Implementation**:
- Calls `_remove_verbatim_duplicate_blocks(text, preserved=None)`.
- Strategy A: split by paragraph boundaries and drop duplicate paragraphs (MD5 of normalized lowercase whitespace-collapsed paragraph).
- Strategy B: sliding window duplicate detection on token windows (`_VERBATIM_SLIDING_WINDOW_TOKENS`) and removal of repeated spans.

**Why it runs first**: preservation introduces unique placeholders (`__CIT_0__`, `__CIT_1__`), which would prevent verbatim block detection.

**Additional skip behavior**:
- When frequency learning is enabled and sentence redundancy is very high, this pass is skipped to preserve repeated patterns used for abbreviation learning.

### Pass 1 — `preserve_elements`
**Goal**: protect content that should not be rewritten (code, URLs, quotes, numbers, citations, JSON tokens/strings, etc.).

**Implementation**:
- Calls `preservation.extract_and_preserve(...)`.
- Produces a `preserved` dict (e.g., `code_blocks`, `urls`, `quotes`, `numbers`, `citations`, `json_tokens`, `json_strings`, `toon_blocks`, plus bookkeeping).
- Placeholders match `config.PLACEHOLDER_PATTERN` (`__\w+_\d+__`).

**Key knobs**:
- `force_preserve_digits` (from SmartContext or request).
- `json_policy` (resolved in `optimize()`; JSON profile forces minify).
- `enable_toon_conversion` (from mode preset; enabled only for `maximum` mode).
- `enable_alias_json_keys` is enabled when `alias_json_keys` is not disabled.

### Pass 1b — `alias_references`
**Goal**: alias repeated references (URLs, citations) and prepend a compact legend (“Refs”).

**Implementation**:
- Calls `_apply_reference_aliases(...)` and records legend entries in a `GlossaryCollector`.

**Skip conditions**:
- Skipped for `code`/`json` profiles and whenever TOON blocks are present.
- Disabled by mode toggles (e.g., conservative sets `alias_references=False`).

### Pass 2 — `compress_boilerplate`
**Goal**: remove/shorten common boilerplate templates early.

**Implementation**:
- Calls `_compress_boilerplate(result, preserved)`, which routes to lexical boilerplate compression.

### Pass 3 — `normalize_whitespace`
**Goal**: reduce whitespace token waste early (without punctuation compression).

**Implementation**:
- Checks `_needs_whitespace_normalization(text)`.
- If needed, calls `_normalize_text(normalize_whitespace=True, compress_punctuation=False)`.

### Pass 3a — `compress_field_labels`
**Goal**: compress repeated “field labels” like `User:` / `Assistant:` / `Title:` / `Description:` while respecting placeholders.

**Implementation**:
- Computes placeholder ranges (`preservation.get_placeholder_ranges(...)`).
- Calls `lexical.compress_field_labels(...)`, recording a “Labels” legend when applied.

**Skip conditions**:
- Disabled by mode toggles (conservative sets `compress_field_labels=False`).
- Skipped for `code` and `json` profiles.

### Pass 3a2 — `deduplicate_exact_lines`
**Goal**: remove exact duplicated *lines* (normalized lowercase whitespace-collapsed) for line-heavy text.

**Implementation**:
- Calls `_deduplicate_exact_lines(text)`; activates only when there are at least 8 lines.

**Skip conditions**:
- Skipped for `code`/`json`.
- Disabled via `deduplicate_exact_lines`.

### Pass 3b — `compress_enumerated_prefix_suffix`
**Goal**: factor common prefixes/suffixes in enumerated lines (structured lists).

**Implementation**:
- Uses placeholder ranges.
- Calls `structural.compress_enumerated_prefix_suffix(...)`.

**Skip conditions**:
- Skipped for `code`/`json`.

### Pass 3c — `compress_repeated_prefix_suffix`
**Goal**: factor repeated prefixes/suffixes across lines (not only enumerated lists).

**Implementation**:
- Uses placeholder ranges.
- Calls `structural.compress_repeated_prefix_suffix(...)`.

**Skip conditions**:
- Skipped for `code`/`json`.

### Pass 4 — `lexical_transforms` (grouped sub-passes)
**Goal**: fast token savings using rule-based, mostly local transformations, while being content- and segment-weight-aware.

**Shared mechanics**:
- Splits around placeholders so transforms only run on non-placeholder segments.
- Uses optional segment weights (discourse weighting and/or `segment_spans`) to avoid rewriting high-importance regions.
- For a subset of operations, the pass pre-checks regex “detectors” and disables work if nothing matches (consecutive duplicates, paradoxical phrases, repeated phrase consolidation).

Sub-pass inventory (each controlled by its own pass name in `resolved_disabled`):
- `collapse_consecutive_duplicates`: collapses repeated adjacent tokens/phrases.
- `collapse_paradoxical_phrases`: collapses contradictory/paradoxical phrase patterns.
- `consolidate_repeated_phrases`: consolidates repeated short phrases.
- `compress_lists`: compacts list syntax and optionally factors prefixes/suffixes when profile is not `code`.
- `clean_instruction_noise`: applies category-ordered instruction simplification rules; records techniques per triggered category.
- `canonicalize_entities`:
  - Loads canonical mappings from DB when available (`get_combined_canonical_mappings(customer_id)`), otherwise uses a cached fallback.
  - For prose-like content types (`general_prose`, `dialogue`, `markdown`, `technical_doc`, `heavy_document`, `short`), merges in contextual dictionaries:
    - `FLUFF_CANONICALIZATIONS`
    - `CONTEXTUAL_CANONICALIZATIONS`
    - `PROMPT_SPECIFIC_CANONICALIZATIONS`
    - `SMART_DEFAULT_CANONICALIZATIONS`
  - Contextual additions are filtered by disabled tokens and gated by `_segment_allows_contextual_canon()` (skips questions and negations).
  - `custom_canonicals` (request) are sanitized and merged, allowing override.
  - Applies replacements via `trie_canonicalize(...)`.
- `shorten_synonyms`: replaces longer synonyms with shorter equivalents (token-cache assisted).
- `apply_contractions`: applies contraction mappings when shorter.
- `normalize_numbers_units`: standardizes number/unit representations.
- `reduce_numeric_precision`: reduces float precision when shorter.
- `compress_clauses`: clause-level structural shortening, weighted.
- `apply_symbolic_replacements`: only for low-weight segments (below `SEGMENT_WEIGHT_HIGH`).
- `remove_articles`: only for low-weight segments.

### Pass 4a — `trim_adjunct_clauses`
**Goal**: remove non-essential discourse/adjunct clauses (“as a reminder…”) when spaCy is available.

**Implementation**:
- Requires `self._get_linguistic_nlp_model()` to be available.
- Calls `adjunct.trim_adjunct_clauses(...)` with:
  - placeholder ranges
  - allowlist markers (`config.ADJUNCT_DISCOURSE_MARKERS`)
  - dependency allowlist (`config.ADJUNCT_ALLOWED_DEPS`)
  - negation/condition/modal tokens
- If semantic guard is enabled, performs an inline similarity check and rolls back if below guard threshold.

**Skip conditions**:
- Skipped for `code`/`json`.
- Skipped when spaCy is unavailable.

### Pass 4 (optional) — `compress_parentheticals`
**Goal**: extract parenthetical expansions into a glossary legend (reduces repeated “(…long expansion…)” noise).

**Implementation**:
- Calls `lexical.extract_parenthetical_glossary(...)` and adds “Glossary” legend entries.

**Skip conditions**:
- Disabled by mode toggles (conservative sets `compress_parentheticals=False`).
- Skipped for `code`/`json`.

### Pass 4b — `phrase_dictionary`
**Goal**: apply offline + learned phrase dictionary replacements (user/customer-specific compression).

**Implementation**:
- Loads `self.phrase_dictionary` from `PROMPT_OPTIMIZER_PHRASE_DICTIONARY_PATH` at init.
- Optionally merges learned phrase mappings from DB (`_load_learned_phrase_dictionary(customer_id)`), without overriding static entries.
- Applies replacements via `trie_replacer.apply_phrase_dictionary(...)`.
- If learned mappings were used, records usage back to DB.

### Heuristic Gate — low lexical gain disables heavy passes
After lexical work, if savings ratio is below `_LOW_GAIN_HEAVY_PASS_SAVINGS_RATIO` and token count is below `_LOW_GAIN_HEAVY_PASS_MAX_TOKENS`, the pipeline disables:
- `compress_coreferences`
- `deduplicate_content`

### Pass 5 — `learn_frequency_abbreviations`
**Goal**: learn and apply abbreviations for frequently repeated multi-word phrases.

**Implementation**:
- Calls `_apply_frequency_abbreviations(...)` (which delegates to lexical logic).
- If this pass changes text, it sets `_skip_sentence_deduplication=True` so sentence-level dedup runs **after** abbreviations settle (deferred dedup).
- Optionally persists learned mappings to DB if semantic guard passes.

**Skip conditions**:
- Requires `SmartContext.enable_frequency_learning=True`.
- Disabled by `learn_frequency_abbreviations`.

### Pass 5b — `apply_macro_dictionary`
**Goal**: generate a macro legend for repeated spans and replace them with short macros.

**Implementation**:
- Calls `lexical.apply_macro_dictionary(...)`, passing placeholder tokens to avoid colliding with placeholders.

### Heuristic Gate — “savings already high” skips heavy model passes
If current savings ratio exceeds the per-mode thresholds (`0.25/0.35/0.45` for conservative/balanced/maximum), the pipeline disables:
- `compress_coreferences`
- `compress_examples`
- `summarize_history`
- `prune_low_entropy`

### Pass 6 — `compress_coreferences`
**Goal**: replace repeated entity mentions with concise aliases/pronouns using coreference resolution.

**Implementation**:
- Calls `_compress_coreferences(text, preserved)`.
- In maximum mode, can be guarded by per-pass semantic guard rollback (`_apply_per_pass_semantic_guard()`).

**Skip conditions**:
- Disabled by mode/profile/heuristics.
- Internal gating includes language/availability/threshold checks (see `_compress_coreferences`).

### Pass 6b — `alias_named_entities`
**Goal**: alias repeated named entities (ORG/PRODUCT/etc) to short tokens and prepend legend entries.

**Implementation**:
- Calls `entity_aliasing.alias_named_entities(...)` with config thresholds (`ENTITY_ALIAS_*`) and placeholder ranges.

**Skip conditions**:
- Skipped for `code`/`json`, when TOON blocks are present, and while frequency-learning mode is active.

### Pass 7 — `compress_repeated_fragments`
**Goal**: compress repeated long fragments (beyond sentence-level duplicates).

**Implementation**:
- Calls `_compress_repeated_fragments(text, preserved)`.

### Pass 8 — `deduplicate_content`
**Goal**: remove duplicate sentences and repeated phrases with layered strategies.

**Implementation outline**:
1. Split text into sentences via `_split_sentences()` (newline boundaries and punctuation boundaries).
2. Exact sentence dedup:
   - Computes a normalized lowercase whitespace-collapsed string and MD5 hash.
   - Keeps one copy; for “directive” sentences (detected via keywords/patterns), keeps up to 2.
   - Can be disabled via thread-state `skip_exact_deduplication`.
3. Near-duplicate sentence dedup (when exact dedup is enabled):
   - Uses token set overlap + length ratio against a sliding lookback window.
   - Optional capped Levenshtein refinement for short strings.
4. Multi-sentence sequence dedup:
   - Removes repeated sequences of consecutive sentences (sequence length = 2), skipping any directive sequences.
5. MinHash + LSH paraphrase dedup (cheap semantic):
   - Uses `SentenceLSHIndex` (internal implementation) to find paraphrase candidates for longer sentences.
   - Applies token overlap thresholds and respects `prefer_shorter_duplicates`.
6. Full semantic dedup (maximum-only):
   - Uses spaCy vectors when available; otherwise falls back to weighted token overlap similarity.
   - Uses minhash signatures to prune candidate comparisons.

**Important skip behavior**:
- If the entire prompt token count is below `fastpath_token_threshold` (default `_FASTPATH_TOKEN_THRESHOLD`) the spaCy-based dedup path is skipped.
- If frequency abbreviations were learned earlier, this pass may be deferred and rerun post-restore (see “Deferred Passes” below).

### Pass 9 — `hoist_constraints`
**Goal**: move constraint-like lines to the top of the prompt to increase their survival under later trimming.

**Implementation**:
- Calls `_hoist_constraints(text)` which normalizes and hoists constraint lines.

**Skip conditions**:
- Disabled by mode toggles (conservative sets `hoist_constraints=False`).
- Skipped for `code`/`json`.

### Heavy-Pass Short-Circuit — consecutive no-ops
If two consecutive passes made no changes (`no_change_streak >= 2`) and there are no viable candidates for examples/history/entropy pruning, the pipeline skips the remaining heavy passes.

### Heavy-Pass Scheduler — utility + latency budget
Heavy passes are scheduled using telemetry-derived expected utility and a per-mode latency budget:
- Candidate order is sorted by expected utility (`compress_examples`, `summarize_history`, `prune_low_entropy`).
- Cold-start priors enforce deterministic ordering when telemetry is empty (`prune_low_entropy` > `compress_examples` > `summarize_history`).
- Execution is constrained by `_HEAVY_LATENCY_BUDGET_MS_BY_MODE` (`120/220/350ms` for conservative/balanced/maximum).
- Passes can be skipped with explicit reasons (`disabled`, `short_circuit`, `latency_budget`, `negative_expected_utility`) for telemetry analysis.

### Pass 10 — `compress_examples` (heavy)
**Goal**: compress “Example:” sections, optionally with multi-candidate selection (maximum-only).

**Implementation**:
- Calls `_compress_examples(text, preserved, ...)`.
- In maximum mode with semantic guard enabled, generates candidates (“default”, “aggressive”) and selects via `_select_semantic_candidate(...)` with a guard threshold derived from config.

### Pass 11 — `summarize_history` (heavy)
**Goal**: summarize long chat histories (role-prefixed dialogue).

**Implementation**:
- Calls `history.summarize_history(...)`.
- In maximum mode with semantic guard enabled, selects between “default” and “aggressive” candidates (keep ratio modifier) via `_select_semantic_candidate(...)`.
- Optionally guarded by per-pass semantic guard rollback.

### Pass 12 — `prune_low_entropy` (heavy)
**Goal**: remove low-information regions with a hybrid entropy stack.

**Implementation**:
- Performs a fast standalone entropy-budget pre-check and skips early when the budget is zero.
- Calls `_maybe_prune_low_entropy(text)` (and aggressive variants in multi-candidate mode).
- Runtime order inside entropy pruning:
  1. Fast ONNX token-level drop-probability scorer (`entropy_fast`) runs first.
  2. Low-confidence tokens near protected spans are re-scored with the causal-LM teacher (`entropy`) on those localized ranges only.
  3. If model artifacts are unavailable, entropy pruning falls back to heuristic token entropy.
- Optionally guarded by per-pass semantic guard rollback.

### Optional Post-Pass — token classifier post pass
**Goal**: apply a conservative, guard-protected token-classifier compression after main heuristics.

**Runs when**:
- `config.TOKEN_CLASSIFIER_POST_PASS_ENABLED` (`PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_PASS`) is true,
- semantic guard is enabled,
- `self.token_classifier_model` is available.

### Legend Consolidation (not a skip-pass)
If any glossary entries were collected (references/labels/parentheticals/macros/entities/abbreviations), a legend is prepended via `lexical._prepend_legend(...)`.

Legend output is normalized to always include the standard sections in order: `Labels`, `Aliases`, `Refs`, `Glossary` (missing sections are emitted as `-`).

### Pass 13 — `normalize_text` (final normalization; can be deferred)
**Goal**: final whitespace and punctuation normalization.

**Implementation**:
- Controlled by two independent pass names:
  - `final_whitespace` (enables whitespace normalization)
  - `compress_punctuation` (enables punctuation compression)
- If `_skip_sentence_deduplication` is set, normalization is deferred and executed after deferred dedup runs.

### Pass 14 — `restore_preserved`
**Goal**: restore all placeholders back to original content (code blocks, URLs, quotes, numbers, JSON blocks, citations, etc.).

**Implementation**:
- Calls `preservation.restore(self, text, preserved)`.

### Pass 15 — `alias_preserved_elements`
**Goal**: alias repeated preserved elements (post-restore), reducing repeated large code/URL blocks, etc.

**Skip conditions**:
- Skipped for `code`/`json`.

### Deferred Passes (when abbreviations changed text)
If `_skip_sentence_deduplication` was set in Pass 5:
- `deduplicate_content` is re-run as `deduplicate_content.deferred`.
- `normalize_text` is re-run if it was deferred.
- `_skip_sentence_deduplication` is cleared afterward.

### Final Cleanup (post passes)
If `final_whitespace` is enabled, the result is `.strip()`’d and then `lexical.final_text_cleanup(...)` runs to remove any remaining artifacts.

## Chunking Pipeline (when prompts exceed chunk thresholds)

Chunking is implemented in `backend/services/optimizer/chunking.py`:

1. `pre_preserve_json`: detect JSON-like content and preserve JSON blocks **before** chunking to avoid splitting structures.
2. `chunk_prompt`: split prompt into chunk specs (fixed/structured/semantic strategies).
3. `chunk_optimize`: run `_optimize_pipeline()` per chunk (JSON preservation inside chunks is disabled because it already happened).
   - `alias_named_entities` is force-disabled during chunking to avoid cross-chunk alias legend collisions when chunks are merged.
4. `chunk_normalize`: normalize placeholder tokens across chunks to avoid collisions.
5. `chunk_join`: join optimized chunks back into a single prompt.
6. `post_chunk_dedup`: run `opt._apply_post_chunk_dedup(...)` (post-join cleanup).
7. `restore_pre_preserved_json`: restore the JSON blocks preserved in step 1.

## Fast Path (short prompts)

Fast path uses the same `_optimize_pipeline()` but force-merges `_FASTPATH_DISABLED_PASSES` into `resolved_disabled` (see `core.py`).
- Default token threshold: `_FASTPATH_TOKEN_THRESHOLD` (1000).
- Fast path is eligible for short prompts with up to `_FASTPATH_MAX_NEWLINES` newlines (default `4`), so compact multi-line prompts can stay on the low-latency path.
- The disabled set includes: `deduplicate_content`, `learn_frequency_abbreviations`, `trim_adjunct_clauses`, `alias_named_entities`, `compress_coreferences`, `compress_examples`, `compress_repeated_fragments`, `summarize_history`, `prune_low_entropy`.

## Guardrails: Query-Aware + Section Ranking

In `maximum` mode, query-aware compression can run before section ranking. To prevent cascading over-compression:
- If query-aware already reduced the prompt below a budget floor, section ranking is skipped.
- If section ranking would reduce the prompt below a hard floor, ranking is rolled back for that request.

## Configuration Touchpoints (exact names used in code)

This doc focuses on pass logic; for a complete list, search for `get_env_*(` and `os.environ.get(` in `backend/services/optimizer/`.

Frequently used controls:
- `PROMPT_OPTIMIZER_SECTION_RANKING_MODE` (default `off`)
- `PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET` (default `0` → treated as “unset”)
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED` (default from `config.MAXIMUM_PREPASS_ENABLED`)
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS` (default from `config.MAXIMUM_PREPASS_MIN_TOKENS`)
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO` (default from `config.MAXIMUM_PREPASS_BUDGET_RATIO`)
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES` (default from `config.MAXIMUM_PREPASS_MAX_SENTENCES`)
- `PROMPT_OPTIMIZER_SEMANTIC_SIMILARITY` (default `0.92`)
- `PROMPT_OPTIMIZER_LSH_ENABLED` (default `True`)
- `PROMPT_OPTIMIZER_DEDUP_PHRASE_LENGTH` (default `5`)
- `PROMPT_OPTIMIZER_PREFER_SHORTER_DUPLICATES` (default `True`)
- `PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED` (default from `config.SEMANTIC_GUARD_ENABLED`)
- `PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD` (default from `config.SEMANTIC_GUARD_THRESHOLD`)
- `PROMPT_OPTIMIZER_SEMANTIC_GUARD_PER_PASS` (default from `config.SEMANTIC_GUARD_PER_PASS_ENABLED`)
- `PROMPT_OPTIMIZER_PHRASE_DICTIONARY_PATH` (phrase dictionary path)
- `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_MODEL` (token classifier model name)
- `ALLOW_MODEL_ENV_OVERRIDE` (allow env var model overrides for model inventory)
- `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_PASS` (enable token classifier post pass)
