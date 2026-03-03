# Environment Variables

This document lists the runtime environment variables used by the Tokemizer API, organized by functional category. Each variable includes its purpose, default value, and configuration guidance.

## How to Set Environment Variables

### Option 1: `.env` File (Recommended)
Create a `.env` file in the repo root:
```bash
# Example .env file
DB_PATH=./data/app.db
PORT=8000
OPTIMIZER_CACHE_SIZE=512
```

### Option 2: Docker Compose
```yaml
services:
  backend:
    environment:
      - DB_PATH=/app/data/app.db
      - OPTIMIZER_CACHE_SIZE=512
```

---

## Server Configuration

### `PORT`
- **Type**: Integer
- **Default**: `8000`
- **Description**: The port number for the API server (Uvicorn) to listen on.

### `UVICORN_HOST`
- **Type**: String
- **Default**: `0.0.0.0`
- **Description**: The host address to bind the server to. Use `0.0.0.0` to listen on all interfaces.

### `UVICORN_WORKERS`
- **Type**: Integer
- **Default**: `1`
- **Description**: Number of worker processes to spawn. Use `1` for development.

### `CORS_ORIGINS`
- **Type**: String (comma-separated list)
- **Default**: `*`
- **Description**: Allowed origins for cross-origin requests. Use `*` for all or specify (e.g., `https://app.tokemizer.com`).

---

## Database Configuration

### `DB_PATH`
- **Type**: String (path)
- **Default**: `./app.db` (relative to `backend/`)
- **Description**: Path to the SQLite database file. Parent directories are created automatically. In Docker, defaults to `/app/data/app.db`.

---

## Optimization Pipeline Control

### `OPTIMIZER_CACHE_SIZE`
- **Type**: Integer
- **Default**: `256`
- **Description**: Capacity of the LRU cache for optimized prompts. Set to `0` to disable.
- **Note**: Cached results are stored in memory for sub-millisecond lookups.

### `OPTIMIZER_PREWARM_MODELS`
- **Type**: Boolean
- **Default**: `true`
- **Description**: If `true`, loads NLP models (spaCy, Transformers) at startup to eliminate first-request latency (~20-30s).

---

## Advanced Optimization Settings

Use profile-level autotuning for most deployments; only reach for expert overrides when you need deterministic behavior for experiments.

### `PROMPT_OPTIMIZER_AUTOTUNE_PROFILE`
- **Type**: String (`safe`, `balanced`, `aggressive`)
- **Default**: `balanced`
- **Description**: Primary optimizer tuning control. Resolves mode-smart defaults for maximum prepass policy, multi-candidate heavy pass settings, and token-classifier post-pass thresholds.
- **Behavior Update (2026-02)**: In `maximum` mode, short prompts automatically cap multi-candidate selection to a single candidate to reduce fixed latency. This safeguard is automatic and applies regardless of profile.

### `PROMPT_OPTIMIZER_SECTION_RANKING_MODE`
- **Type**: String (`off`, `bm25`, `gzip`, `tfidf`)
- **Default**: `off`
- **Description**: Default section ranking algorithm for extracting high-value content before optimization. Smart Router can auto-enable ranking for massive prompts even when set to `off` (not a hard kill switch).

### `PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET`
- **Type**: Integer
- **Default**: `0`
- **Description**: Maximum tokens to extract via section ranking. `0` means auto-calculate based on prompt size.

### `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY`
- **Type**: String (`off`, `auto`, `conservative`, `aggressive`) — **expert override**
- **Default**: Derived from `PROMPT_OPTIMIZER_AUTOTUNE_PROFILE`
- **Description**: Optional expert override for maximum-mode budgeted pre-pass policy. In `auto`, resolved budget defaults are prompt-adaptive and bounded by expert floors/caps.

### Expert overrides (advanced)
The following variables are still supported and take precedence when explicitly set:
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED`
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MIN_TOKENS`
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_BUDGET_RATIO`
- `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_MAX_SENTENCES`
- `PROMPT_OPTIMIZER_MC_*` multi-candidate pass controls
- `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_*` post-pass controls

### `PROMPT_OPTIMIZER_MC_*` multi-candidate controls
- **Type**: Integer / Float (family of expert overrides)
- **Default**: Derived from `PROMPT_OPTIMIZER_AUTOTUNE_PROFILE`
- **Description**: Controls multi-candidate semantic selection for heavy passes in `maximum` mode.
- **Behavior Update (2026-02)**: For short prompts, runtime now forces single-candidate execution even if `PROMPT_OPTIMIZER_MC_*` is configured >1, preventing large latency spikes on small inputs.

### `PROMPT_OPTIMIZER_ENTROPY_TRANSFORMER_MIN_TOKENS`
- **Type**: Integer
- **Default**: `0`
- **Description**: Minimum token count for entropy-based pruning transformer models.


### `PROMPT_OPTIMIZER_ENTROPY_PROTECTED_CONFIDENCE`
- **Type**: Float
- **Default**: `0.55`
- **Description**: Confidence threshold for fast entropy predictions near protected ranges. Lower-confidence tokens are re-scored with the causal-LM teacher in localized spans.

### `PROMPT_OPTIMIZER_ENTROPY_PROTECTED_WINDOW`
- **Type**: Integer
- **Default**: `16`
- **Description**: Character window around low-confidence tokens (near protected ranges) used when selecting localized teacher re-scoring spans.

### `PROMPT_OPTIMIZER_ENTROPY_TRANSFORMER_MIN_TOKENS` and entropy runtime behavior
- **Behavior Update (2026-02)**: Entropy pruning now fails safe if entropy NLL estimation/tokenization fails at runtime (the pass is skipped instead of failing the entire optimization request).

---

## Semantic & Structural Guards

### `PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enables/disables the similarity check between original and optimized output.

### `PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD`
- **Type**: Float
- **Default**: `0.82`
- **Description**: Minimum similarity score (0.0 - 1.0) required to accept an optimization. When semantic NLP models are unavailable, runtime fallback caps this threshold to mode-safe values (`0.68` for balanced/conservative, `0.82` for maximum) unless explicitly overridden.

### `PROMPT_OPTIMIZER_SEMANTIC_GUARD_MODEL`
- **Type**: String
- **Default**: `BAAI/bge-small-en-v1.5`
- **Description**: HuggingFace model name for computing semantic embeddings.

### `PROMPT_OPTIMIZER_SEMANTIC_RANK_MODEL`
- **Type**: String
- **Default**: Inherits semantic guard model when unset
- **Description**: Optional embedding model override for section ranking and semantic chunking.

### `PROMPT_OPTIMIZER_ONNX_INT8`
- **Type**: Boolean
- **Default**: `false`
- **Description**: When `true`, the optimizer prefers the quantized `model.int8.onnx` semantic guard artifact for faster ONNX inference; falls back to `model.onnx` automatically if the INT8 file is missing.

### `EMBEDDING_CACHE_SIZE`
- **Type**: Integer
- **Default**: `512`
- **Description**: Size of the LRU cache for text embedding vectors.

---

## Deduplication & Pruning

### `PROMPT_OPTIMIZER_SEMANTIC_SIMILARITY`
- **Type**: Float
- **Default**: `0.92`
- **Description**: Threshold for identifying semantically duplicate sentences.

### `PROMPT_OPTIMIZER_LSH_ENABLED`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enables Locality-Sensitive Hashing for fast duplicate candidate detection in large prompts.

### `PROMPT_OPTIMIZER_DEDUP_PHRASE_LENGTH`
- **Type**: Integer
- **Default**: `5`
- **Description**: Minimum word count for a phrase to be considered for cross-sentence deduplication.

### `PROMPT_OPTIMIZER_PREFER_SHORTER_DUPLICATES`
- **Type**: Boolean
- **Default**: `true`
- **Description**: When duplicates are found, prioritize keeping the shorter, more concise version.

---

## Infrastructure & Connectivity

### `HF_HOME`
- **Type**: String (path)
- **Default**: `/app/.cache/huggingface`
- **Description**: Directory where HuggingFace models are cached. Use a persistent volume or bind mount so refresh downloads are retained across restarts.

### `HF_TOKEN`
- **Type**: String
- **Default**: Not set
- **Description**: Hugging Face access token used for private/gated model repositories in the model inventory.

### `HUGGINGFACE_HUB_TOKEN`
- **Type**: String
- **Default**: Not set
- **Description**: Alias of `HF_TOKEN`. Either variable can be set.

### `MODEL_UPLOAD_MAX_BODY`
- **Type**: Nginx size string
- **Default**: `512m`
- **Description**: Maximum request body accepted by frontend nginx for model archive uploads (`/api/admin/models/{model_type}/upload`).

### `LOCALAPPDATA`
- **Type**: String (path, Windows only)
- **Default**: OS-managed
- **Description**: Used only on Windows hosts to resolve model cache locations when `%LOCALAPPDATA%` is available.

### `MODEL_CACHE_VALIDATION_TTL`
- **Type**: Integer (seconds)
- **Default**: `600`
- **Description**: Time-to-live for cached model validation results to reduce repeated filesystem scans.

### `MODEL_CACHE_LOCK_TTL_SECONDS`
- **Type**: Integer (seconds)
- **Default**: `900`
- **Description**: Maximum age for model-cache refresh lock files before stale locks are ignored.

### `MODEL_CACHE_FULL_HASH_BYTES`
- **Type**: Integer (bytes)
- **Default**: `262144000`
- **Description**: Maximum total size to hash every file during manifest validation; larger caches hash only critical files.

### `TIKTOKEN_CACHE_DIR`
- **Type**: String (path)
- **Default**: Automatically points to `./vendor/tiktoken` if present, otherwise `~/.tiktoken`
- **Description**: Directory where tiktoken encodings are stored.

### `REDIS_URL`
- **Type**: String (connection URL)
- **Default**: Not set
- **Description**: Optional Redis connection URL for distributed caching (not currently used in core optimization).

### `OLLAMA_BASE_URL`
- **Type**: String (URL)
- **Default**: `http://localhost:11434`
- **Description**: Base URL used when proxy-testing Ollama models through `/api/v1/llm/test`.

### `SECRET_KEY`
- **Type**: String
- **Default**: `dev_secret_key_change_in_production`
- **Description**: Secret key for authentication and JWT signing. Must be changed in production.

### `DB_POOL_SIZE`
- **Type**: Integer
- **Default**: `10`
- **Description**: Size of the SQLite connection pool for database operations.

### `OPTIMIZER_CACHE_TTL_SECONDS`
- **Type**: Integer
- **Default**: `0`
- **Description**: Time-to-live in seconds for optimizer cache entries. `0` means no expiration.

### `BATCH_OPTIMIZE_CHUNK_SIZE`
- **Type**: Integer
- **Default**: `64`
- **Description**: Number of prompts to process in each batch for batch optimization operations.

### `BATCH_PROGRESS_UPDATE_EVERY`
- **Type**: Integer
- **Default**: `10`
- **Description**: Frequency of progress updates during batch optimization (updates every N chunks).

---

## Content Awareness & Extensions

### `PROMPT_OPTIMIZER_PHRASE_DICTIONARY_PATH`
- **Type**: String (path)
- **Default**: `None`
- **Description**: Path to a JSON file containing domain-specific phrase replacements (e.g., `{"Machine Learning": "ML"}`).

### `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_MODEL`
- **Type**: String
- **Default**: `None`
- **Description**: Path or name of a specialized token classification model to guide compression targets.

### `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_CANDIDATES`
- **Type**: String (comma-separated)
- **Default**: Internal candidate list
- **Description**: Ordered candidate model IDs/paths probed for the token classifier when no explicit model is set.

### `PROMPT_OPTIMIZER_SPACY_MODEL_PATH`
- **Type**: String (path)
- **Default**: `/app/.cache/huggingface/spacy/en_core_web_sm`
- **Description**: Path to the spaCy model used for NLP operations in the optimization pipeline.
### `SPACY_HOME`
- **Type**: String (path)
- **Default**: `/app/.cache/huggingface/spacy`
- **Description**: Base directory for spaCy model cache used by the optimizer.

---

## Telemetry & History

### `OPTIMIZATION_HISTORY_ENABLED`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Record every optimization request, result, and metadata in the database for analysis and audit trails.

Telemetry collection for per-pass timing/token gains is disabled by default and is toggled at runtime using the Settings API (`PATCH /api/v1/settings`) or the Admin Settings UI (`telemetry_enabled`). No environment variable is required for this toggle.

Maximum-mode stage timing debug logs are now available when runtime `log_level` is set to `DEBUG` via Admin Settings (not an environment variable). No new env variable was introduced for this.

---

## Email & Notifications (Optional)

### `SMTP_HOST`
- **Type**: String
- **Default**: `smtp.gmail.com`
- **Description**: SMTP server host for email notifications.

### `SMTP_PORT`
- **Type**: Integer
- **Default**: `587`
- **Description**: SMTP server port.

### `SMTP_USER`
- **Type**: String
- **Default**: Empty
- **Description**: SMTP authentication username.

### `SMTP_PASS`
- **Type**: String
- **Default**: Empty
- **Description**: SMTP authentication password.

### `FROM_EMAIL`
- **Type**: String
- **Default**: `noreply@tokemizer.com`
- **Description**: Email address to use as sender for notifications.

---

## Billing & Stripe (Optional)

### `STRIPE_SECRET_KEY`
- **Type**: String
- **Default**: Not set
- **Description**: Stripe API secret key for payment processing.

### `STRIPE_WEBHOOK_SECRET`
- **Type**: String
- **Default**: Not set
- **Description**: Stripe webhook signing secret for validating webhook events.

## Quick Reference Summary

| Variable | Default | Category |
|----------|---------|----------|
| `PORT` | 8000 | Server |
| `UVICORN_HOST` | 0.0.0.0 | Server |
| `UVICORN_WORKERS` | 1 | Server |
| `CORS_ORIGINS` | * | Server |
| `DB_PATH` | ./app.db | Database |
| `DB_POOL_SIZE` | 10 | Database |
| `OPTIMIZER_CACHE_SIZE` | 256 | Optimization |
| `OPTIMIZER_CACHE_TTL_SECONDS` | 0 | Optimization |
| `OPTIMIZER_PREWARM_MODELS` | true | Optimization |
| `BATCH_OPTIMIZE_CHUNK_SIZE` | 64 | Optimization |
| `BATCH_PROGRESS_UPDATE_EVERY` | 10 | Optimization |
| `PROMPT_OPTIMIZER_SEMANTIC_SIMILARITY` | 0.92 | Deduplication |
| `PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED` | true | Guard |
| `PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD` | 0.82 | Guard |
| `PROMPT_OPTIMIZER_SPACY_MODEL_PATH` | /app/.cache/huggingface/spacy/en_core_web_sm | Content Awareness |
| `SPACY_HOME` | /app/.cache/huggingface/spacy | Infrastructure |
| `OPTIMIZATION_HISTORY_ENABLED` | true | Telemetry |
| `HF_TOKEN` | Not set | Infrastructure |
| `HF_HOME` | /app/.cache/huggingface | Infrastructure |
| `TIKTOKEN_CACHE_DIR` | ~/.tiktoken | Infrastructure |
| `SECRET_KEY` | dev_secret_key_change_in_production | Security |

> **Telemetry note**: Per-pass telemetry collection is disabled by default and is toggled via the Settings API (`"telemetry_enabled"`) or Admin UI. No environment variable is required.

> **Boolean Guidance**: Values `1`, `true`, `yes`, `on` are treated as `True`. Case-insensitive.
