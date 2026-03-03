# Tokemizer Model Catalog

This document is the source of truth for optimizer model inventory, use-case mapping, capability requirements, and strict readiness behavior.

Date: 2026-02-13

## Goal

Prompt optimization must preserve prompt meaning, context, and intent while improving token efficiency and response latency.

## Catalog (Current Target)

| Use case            | Recommended model                                                          | Why this model                                                                           | License    | Model size          | Est. latency            | Est. RAM    | CPU to run efficiently |
| ------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------- | ------------------- | ----------------------- | ----------- | ---------------------- |
| `semantic_guard`    | `BAAI/bge-small-en-v1.5`                                                   | Strong small encoder quality, ONNX artifact available, good balance for similarity guard | MIT        | ONNX ~127 MB        | ~30-70 ms / 1k tokens   | ~0.6-1.2 GB | 2-4 vCPU               |
| `semantic_rank`     | `BAAI/bge-small-en-v1.5`                                                   | Same embedding objective; sharing model reduces inventory and warmup overhead            | MIT        | ONNX ~127 MB        | ~30-70 ms / 1k tokens   | ~0.6-1.2 GB | 2-4 vCPU               |
| `entropy` (teacher) | `HuggingFaceTB/SmolLM2-360M`                                               | Better modern small LM tradeoff than older 160M class for entropy scoring                | Apache-2.0 | safetensors ~690 MB | ~450-850 ms / 1k tokens | ~2.5-4.5 GB | 4-8 vCPU               |
| `entropy_fast`      | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`           | Public token-level compression model; directly aligned with keep/drop style scoring      | Apache-2.0 | safetensors ~677 MB | ~90-180 ms / 1k tokens  | ~2.5-4.0 GB | 4-8 vCPU               |
| `token_classifier`  | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`           | One model can serve both drop-probability and keep/drop paths with one inference stack   | Apache-2.0 | safetensors ~677 MB | ~90-180 ms / 1k tokens  | ~2.5-4.0 GB | 4-8 vCPU               |
| `coreference`       | `talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large` | ONNX coref model already aligned with pipeline behavior                                  | MIT        | ONNX ~498 MB        | ~250-600 ms / 1k tokens | ~2.0-3.5 GB | 4-8 vCPU               |
| `spacy`             | `en_core_web_sm` 3.8                                                       | Best CPU-friendly linguistic baseline for parser/NER/lemmatizer passes                   | MIT        | model card: 12 MB   | ~5-20 ms / 1k tokens    | ~0.3-0.8 GB | 1-2 vCPU               |

## Operational Sizing (Recommended)

For production with all strict paths enabled and moderate concurrency:

| Deployment profile                                       | Recommended RAM | Recommended CPU |
| -------------------------------------------------------- | --------------- | --------------- |
| Minimum viable (single-worker, low concurrency)          | 24 GB           | 8 vCPU          |
| Recommended baseline (stable balanced/maximum workloads) | 32 GB           | 12 vCPU         |
| High-throughput (higher concurrent optimizations)        | 48-64 GB        | 16-24 vCPU      |

Sizing notes:

- `entropy` teacher and `token_classifier`/`entropy_fast` are the primary memory/CPU consumers.
- If `maximum` traffic is low and teacher entropy is often unavailable by design, 24-32 GB RAM is generally sufficient.
- Keep model cache on persistent storage to avoid repeated cold-download and warm-up costs.

## Use Cases and Capability Contract

### 1) `semantic_guard`

- Use case: semantic safeguard similarity checks for meaning-preserving compression.
- Required capabilities:
- Sentence embeddings for pairwise similarity.
- ONNX-capable encoder artifacts.
- Stable behavior for up to 512-token segments.

### 2) `semantic_rank`

- Use case: query-aware section ranking and ranking-driven compression.
- Required capabilities:
- Sentence embeddings for query-to-section relevance.
- ONNX-capable encoder artifacts.
- Robust pooling for variable-length sections.

### 3) `entropy`

- Use case: optional teacher quality guard for entropy-based pruning quality.
- Required capabilities:
- Causal LM token-level surprisal scoring.
- Local cached transformers checkpoint support.

### 4) `entropy_fast`

- Use case: primary required entropy backend for strict modes.
- Required capabilities:
- Token-level drop/keep probability scoring.
- Supports ONNX artifacts when available.
- Supports transformers token-classification checkpoints (safetensors).

### 5) `token_classifier`

- Use case: maximum-mode keep/drop classifier fast path.
- Required capabilities:
- Token-level keep/drop classification.
- Supports ONNX artifacts when available.
- Supports transformers token-classification checkpoints (safetensors).

### 6) `coreference`

- Use case: coreference compression for repeated entity mention reduction.
- Required capabilities:
- Mention clustering and span-level coreference predictions.
- ONNX runtime-compatible model artifact.

### 7) `spacy`

- Use case: syntactic and linguistic passes (dedup support, dependency-aware transforms).
- Required capabilities:
- English tokenizer/tagger/parser/ner/lemmatizer stack.
- CPU-friendly runtime behavior.

## Selection Criteria

Model choices were selected using these criteria:

- Commercially usable public license only.
- Strong quality-per-latency ratio for each role.
- Compatibility with existing Tokemizer runtime paths.
- Low operational overhead and cache footprint where possible.
- Shared-model opportunities to reduce model inventory.

## Consolidation Decisions

Two explicit consolidations are adopted:

- `semantic_guard` and `semantic_rank` use the same embedding model (`BAAI/bge-small-en-v1.5`).
- `entropy_fast` and `token_classifier` use the same token-classification model (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`).

Benefits:

- Fewer artifacts to download, validate, and prewarm.
- Lower model inventory complexity in admin operations.
- Better cache hit and runtime reuse.

## Seeded Default Inventory (New DB Bootstrap)

When `model_inventory` is empty, defaults seed to:

| model_type         | model_name                                                                 | min_size_bytes      | expected_files                                       | revision | notes                             |
| ------------------ | -------------------------------------------------------------------------- | -------------------:| ---------------------------------------------------- | -------- | --------------------------------- |
| `semantic_guard`   | `BAAI/bge-small-en-v1.5`                                                   | `127 * 1024 * 1024` | `model.onnx`, `tokenizer.json`, `config.json`        | `main`   | semantic similarity guard         |
| `semantic_rank`    | `BAAI/bge-small-en-v1.5`                                                   | `127 * 1024 * 1024` | `model.onnx`, `tokenizer.json`, `config.json`        | `main`   | query-aware compression + ranking |
| `entropy`          | `HuggingFaceTB/SmolLM2-360M`                                               | `690 * 1024 * 1024` | `model.safetensors`, `config.json`                   | `main`   | optional teacher entropy scorer   |
| `entropy_fast`     | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`           | `677 * 1024 * 1024` | `model.safetensors`, `tokenizer.json`, `config.json` | `main`   | required fast entropy backend     |
| `token_classifier` | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`           | `677 * 1024 * 1024` | `model.safetensors`, `tokenizer.json`, `config.json` | `main`   | maximum-mode classifier path      |
| `coreference`      | `talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large` | `498 * 1024 * 1024` | `model.onnx`, `tokenizer.json`, `config.json`        | `main`   | coreference compression           |

spaCy runtime target:

- `en_core_web_sm` (managed as spaCy cache, not as Hugging Face `model_inventory` row).

## Strict Readiness and Mode Gating

### Required backends by mode

- `conservative`: requires `semantic_guard` and `entropy_fast`.
- `balanced`: requires `semantic_guard`, `spacy`, `coreference`, `semantic_rank`, and `entropy_fast`.
- `maximum`: requires `semantic_guard`, `spacy`, `coreference`, `semantic_rank`, `entropy_fast`, and `token_classifier`.

### Entropy policy

- `entropy_fast` is the required strict entropy backend.
- `entropy` teacher is optional and treated as a quality guard.
- In `maximum`, runtime prefers teacher when available and automatically falls back to `entropy_fast` when teacher is unavailable.

## Runtime Loading Notes

- `entropy_fast` and `token_classifier` now support two runtime paths:
- ONNX Runtime path when ONNX artifacts are present.
- Transformers token-classification path when checkpoint artifacts are safetensors-only.

This keeps strict-mode behavior compatible with modern public checkpoints that do not ship ONNX artifacts by default.

## Admin and Air-Gap Operations

- Refresh and validation still run through existing model cache manager flows:
- `download_missing`
- `force_redownload`
- `recovery`
- Air-gap check validates all configured model entries plus spaCy cache readiness.

## Hugging Face Authentication Note

- Default seeded models in this catalog are public and typically download without credentials.
- If you configure any private or gated Hugging Face repository in `model_inventory`, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in backend environment before running refresh.
- Without a valid token for private/gated repos, model refresh/download will fail by design.



----

# Model Swap (Runtime)

Tokemizer supports runtime model swaps without backend/server restart.

What is supported:

- Add a new model config at runtime (`POST /api/admin/models`).
- Update the model used by an existing use case (`PUT /api/admin/models/{model_type}`).
- Trigger targeted or global refresh to download/validate cache artifacts.
- Pipeline picks up updated inventory after runtime config refresh and model warm-up.

Behavior details:

- No restart is required for model replacement.
- Refresh/warm-up runs asynchronously; during this window, strict requests may fail if required models are not yet ready.
- Deleting protected core model types requires explicit override confirmation.
- Replacing active model inventory does not require immediate cache deletion of old artifacts; old files can remain on disk until maintenance cleanup.

## How Serving Works Today (Control Plane vs Runtime)

Tokemizer uses an in-process runtime with a model-management control plane:

- **Control plane (`model_inventory` + model cache manager):**
  - The active model mapping is stored in `model_inventory`.
  - Admin routes update this mapping and trigger refresh/validation.
  - `ensure_models_cached` verifies downloads, required files, and optional ONNX export.
- **Runtime inference path (inside optimizer process):**
  - Optimization passes resolve the active `model_name` via `get_model_configs()`.
  - Passes locate model artifacts via `resolve_cached_model_path` / `resolve_cached_model_artifact`.
  - Passes load ONNX Runtime sessions (or transformer fallback paths where implemented) and run inference directly in the backend process.

This means model management is decoupled from pass logic by `model_type` indirection, while inference remains local to the optimizer service.

### Why Client Modules Don’t Break on Model Swap

- Passes request models by stable `model_type` keys (`semantic_guard`, `entropy_fast`, `token_classifier`, etc.), not by hard-coded repository names.
- Swapping the underlying `model_name` in inventory keeps the same `model_type` contract for client modules and passes.
- After refresh/warm-up, subsequent requests read the updated mapping and load the new artifacts.
- Strict readiness/mode gating prevents running a pass when required replacement artifacts are not yet ready.

## Add a New Model to Catalog (Step-by-Step)

1. Choose whether you are adding a brand new `model_type` or replacing an existing one.
2. Prepare the model metadata payload for `POST /api/admin/models`.
3. Submit the payload from Admin UI or API.
4. Monitor refresh state via `GET /api/admin/models/refresh` until state is `completed` or `failed`.
5. Validate cache/readiness in `GET /api/admin/models`.
6. If needed, trigger explicit refresh:
- Global: `POST /api/admin/models/refresh?mode=download_missing`
- Targeted: `POST /api/admin/models/{model_type}/refresh?mode=download_missing`

## ModelCreate Payload Fields

`POST /api/admin/models` accepts this shape:

```json
{
  "model_type": "semantic_guard",
  "model_name": "BAAI/bge-small-en-v1.5",
  "component": "Semantic Guard",
  "library_type": "sentence-transformers",
  "usage": "semantic safeguard similarity checks",
  "min_size_bytes": 133169152,
  "expected_files": ["model.onnx", "tokenizer.json", "config.json"],
  "revision": "main",
  "allow_patterns": []
}
```

Field reference:

| Field | Required | Type | Purpose / Intent | What values to give |
| --- | --- | --- | --- | --- |
| `model_type` | Yes | string | Logical slot in Tokemizer inventory and readiness reporting. This is the primary key for runtime replacement. | Use a stable lowercase identifier. Use existing runtime slots (`semantic_guard`, `semantic_rank`, `entropy`, `entropy_fast`, `token_classifier`, `coreference`) when replacing core behavior. Use a new custom name only when adding non-core inventory. |
| `model_name` | Yes | string | Hugging Face repo id (or configured repo alias target). Used for download and cache path resolution. | Use exact repo id format like `org/repo`. Pick a model that matches the runtime path for that slot (embedding model for `semantic_guard`, token-classification model for `entropy_fast`, etc.). |
| `component` | No | string | Human-readable component label shown in admin surfaces. | Use clear operational names, e.g. `Entropy Scoring (Fast)` or `Semantic Ranking`. If omitted, defaults to `model_type`. |
| `library_type` | No | string | Documentation/admin metadata for expected loading stack. | Use values reflecting actual runtime path such as `transformers`, `sentence-transformers`, or `onnxruntime`. Keep this aligned with your real artifacts to avoid operator confusion. |
| `usage` | No | string | Operator-facing description of why model exists. | Use concise purpose text, e.g. `Token-level drop probability scoring`. |
| `min_size_bytes` | No | integer | Lower-bound cache sanity check. Helps detect incomplete downloads. | Set to realistic minimum artifact size in bytes. Use `0` only when size is intentionally not enforced. |
| `expected_files` | No (defaults) | array[string] | Required file presence contract for cache validation. | List files that must exist, e.g. `model.onnx`, `model.safetensors`, `tokenizer.json`, `config.json`. If omitted, defaults to `["config.json"]`. |
| `revision` | No | string or null | Git ref pin for deterministic model resolution. | Use `main` or a specific tag/commit hash. Use explicit pins for reproducibility; leave empty only if you intentionally accept repo default behavior. |
| `allow_patterns` | No | array[string] or null | Download allowlist patterns for snapshot filtering. | Use empty list `[]` to rely on auto-derived defaults from `expected_files`. Set explicit patterns only when repo layout requires custom inclusion rules. |

## Field Design Rules and Common Mistakes

- Keep `model_type` stable; changing it creates a new inventory slot instead of replacing existing runtime slot.
- Do not create a separate `spacy` inventory row; spaCy is managed as runtime cache outside Hugging Face `model_inventory`.
- Ensure `expected_files` reflects the runtime backend:
- ONNX path expects files such as `model.onnx` (or `model.int8.onnx`) plus tokenizer/config when needed.
- Transformers token-classification path usually needs `model.safetensors`, `tokenizer.json`, and `config.json`.
- Set `min_size_bytes` conservatively. Too high causes false invalidation; too low weakens integrity checks.
- Use `allow_patterns` only when needed. Overly strict patterns are a common cause of partial downloads.

## Core vs Custom `model_type` Intent

- Existing core `model_type` values are wired into strict readiness gates and optimizer behavior.
- A brand new custom `model_type` can be stored and refreshed, but it will not automatically participate in strict mode gating unless code paths/capability mappings are updated.

## Refresh Modes

- `download_missing`: download only missing/invalid artifacts.
- `force_redownload`: force fresh download even if cache appears valid.
- `recovery`: cleanup incomplete artifacts first, then download missing.

## Practical Examples

Example A: Replace `semantic_guard` model in place.

```json
{
  "model_name": "intfloat/e5-small-v2",
  "component": "Semantic Guard",
  "library_type": "sentence-transformers",
  "usage": "semantic safeguard similarity checks",
  "min_size_bytes": 120000000,
  "expected_files": ["config.json", "tokenizer.json", "pytorch_model.bin"],
  "revision": "main",
  "allow_patterns": []
}
```

Apply with:

- `PUT /api/admin/models/semantic_guard`

Example B: Add a non-core experimental model entry.

```json
{
  "model_type": "semantic_guard_experimental",
  "model_name": "BAAI/bge-small-en-v1.5",
  "component": "Semantic Guard Experimental",
  "library_type": "sentence-transformers",
  "usage": "A/B validation",
  "min_size_bytes": 127000000,
  "expected_files": ["config.json", "tokenizer.json"],
  "revision": "main",
  "allow_patterns": []
}
```

Apply with:

- `POST /api/admin/models`

## Private/Gated Repository Reminder

- For private or gated repositories, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) before refresh/download.
- Public repositories should refresh without token, assuming repo availability and network reachability.
