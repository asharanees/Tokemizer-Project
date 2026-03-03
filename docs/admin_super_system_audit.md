# Super Admin System Audit: Environment Variable Reassessment

## Scope
This audit re-scans the variables documented in `docs/ENV_VARIABLES.md` and reassesses which controls should move from deploy-time environment configuration into **Super Admin runtime pages** (with DB-backed settings + API), while preserving env-only controls for security/infra concerns.

## Current Runtime Coverage (Already in Super Admin)
The following areas are already available in admin-managed settings and should stay there:

- SMTP host/port/user/from/password
- Stripe keys
- Log level
- Telemetry toggle
- Token expiry windows
- Optimization history toggle
- Learned abbreviations toggle
- Model prewarm toggle
- CORS origins

These should be treated as baseline and not regress back to env-only management.

## Decision Framework Used
Variables are classified by operational intent:

- **Move to Super Admin (recommended)**: Safe to tune at runtime, frequently adjusted, optimization-quality/perf knobs, or tenant experience controls.
- **Hybrid**: Env as bootstrap/default + Super Admin override at runtime.
- **Keep env-only**: Secrets, process boot/runtime infrastructure, filesystem paths, reverse-proxy limits, or host-specific concerns.

---

## Additional Recommended Candidates to Add to Super Admin

### 1) Optimizer Strategy Controls (High Priority)
These are the strongest candidates because they directly influence compression quality/latency and are frequently tuned after deployment.

1. `PROMPT_OPTIMIZER_AUTOTUNE_PROFILE` (**Hybrid**)
   - Super Admin selector: `safe | balanced | aggressive`
   - Keep env as cold-start default.
2. `PROMPT_OPTIMIZER_SECTION_RANKING_MODE` (**Hybrid**)
   - Super Admin selector: `off | bm25 | gzip | tfidf`.
3. `PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET` (**Hybrid**)
   - Integer runtime control with guardrails (`0` auto, bounded max).
4. `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY` (**Hybrid**)
   - Runtime policy selector with safe validation.

**Why now:** These controls are explicitly presented as tuning knobs and profile-driven behavior in env docs, which makes them ideal Super Admin controls rather than restart-only deploy controls.

### 2) Expert Optimization Families (High Priority, Advanced Panel)
Expose as an **Advanced / Expert** accordion with validation + “reset to profile defaults”.

1. `PROMPT_OPTIMIZER_MC_*` family
2. `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_*` family
3. `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_*` numeric sub-controls
4. `PROMPT_OPTIMIZER_ENTROPY_PROTECTED_CONFIDENCE`
5. `PROMPT_OPTIMIZER_ENTROPY_PROTECTED_WINDOW`
6. `PROMPT_OPTIMIZER_ENTROPY_TRANSFORMER_MIN_TOKENS`

**Recommendation:** Keep profile mode as primary UX; allow expert overrides only behind explicit “expert mode enabled” confirmation.

### 3) Dedup/Semantic Heuristics (Medium Priority)
These are quality-tuning controls appropriate for super admin experimentation.

1. `PROMPT_OPTIMIZER_SEMANTIC_SIMILARITY`
2. `PROMPT_OPTIMIZER_DEDUP_PHRASE_LENGTH`
3. `PROMPT_OPTIMIZER_PREFER_SHORTER_DUPLICATES`
4. `PROMPT_OPTIMIZER_SEMANTIC_RANK_MODEL` (advanced; model availability checks required)
5. `PROMPT_OPTIMIZER_ONNX_INT8` (advanced toggle, with model artifact compatibility warning)

Note: `PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED` and `PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD` are already runtime-adjustable via `/api/v1/settings`; keep aligned and avoid split-brain between user-level vs super-admin-global policy.

### 4) Batch & Throughput Runtime Controls (Medium Priority)
Operational controls worth runtime tuning during incidents or load tests:

1. `BATCH_OPTIMIZE_CHUNK_SIZE`
2. `BATCH_PROGRESS_UPDATE_EVERY`
3. `OPTIMIZER_CACHE_TTL_SECONDS`

`OPTIMIZER_CACHE_SIZE` is already runtime-managed; these should join the same capacity/performance panel.

### 5) Model Cache Governance Knobs (Medium Priority)
Add to a “Model Cache Policy” section for platform operators:

1. `MODEL_CACHE_VALIDATION_TTL`
2. `MODEL_CACHE_LOCK_TTL_SECONDS`
3. `MODEL_CACHE_FULL_HASH_BYTES`

These are operationally important for balancing cache verification cost vs integrity during heavy model refresh cycles.

---

## Keep Env-Only (Do Not Move to Super Admin)

### Security secrets / credentials
- `SECRET_KEY`
- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `STRIPE_WEBHOOK_SECRET`
- `SMTP_PASS` / raw credential envs
- `STRIPE_SECRET_KEY` (if secret-store-backed admin setting is already used at runtime, still keep env as bootstrap fallback only)

### Boot/process/network infrastructure
- `PORT`
- `UVICORN_HOST`
- `UVICORN_WORKERS`
- `DB_PATH`
- `DB_POOL_SIZE`
- `REDIS_URL`
- `OLLAMA_BASE_URL`

### Host/path/proxy specific
- `HF_HOME`
- `SPACY_HOME`
- `PROMPT_OPTIMIZER_SPACY_MODEL_PATH`
- `TIKTOKEN_CACHE_DIR`
- `LOCALAPPDATA`
- `MODEL_UPLOAD_MAX_BODY` (nginx/proxy concern)

These controls should remain under infrastructure-as-code / deployment management, not mutable from application UI.

---

## Proposed Super Admin IA (Information Architecture)

1. **Optimization Profiles**
   - Autotune profile
   - Section ranking mode + budget
   - Maximum prepass policy
2. **Optimization Expert Overrides** (gated)
   - MC family
   - Token classifier post-pass family
   - Entropy protected controls
3. **Dedup & Semantic Controls**
   - Semantic similarity
   - Dedup phrase length
   - Prefer shorter duplicates
   - Optional semantic rank model override
4. **Performance & Batch**
   - Cache size (existing)
   - Cache TTL
   - Batch chunk size
   - Batch progress cadence
5. **Model Cache Policy**
   - Validation TTL
   - Lock TTL
   - Full hash byte ceiling

---

## Rollout Recommendations

1. Implement as **hybrid precedence**: `admin_setting` override > env default.
2. Add field-level validation and conservative min/max clamps.
3. Require audit logging for every super-admin change (actor, before/after, timestamp).
4. Add “Reset to env default” and “Reset to recommended profile defaults” actions.
5. For high-risk expert settings, add a “can increase latency” warning badge.

---

## Net-New Recommendations From This Round
Compared to prior baseline super-admin settings, the most valuable additional capabilities are:

- Profile and ranking controls: `PROMPT_OPTIMIZER_AUTOTUNE_PROFILE`, `PROMPT_OPTIMIZER_SECTION_RANKING_MODE`, `PROMPT_OPTIMIZER_SECTION_RANKING_TOKEN_BUDGET`, `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY`.
- Expert optimization families: `PROMPT_OPTIMIZER_MC_*`, `PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_POST_*`, `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_*`, entropy protected controls.
- Operational model-cache policy controls: `MODEL_CACHE_VALIDATION_TTL`, `MODEL_CACHE_LOCK_TTL_SECONDS`, `MODEL_CACHE_FULL_HASH_BYTES`.
- Throughput/runtime controls: `BATCH_OPTIMIZE_CHUNK_SIZE`, `BATCH_PROGRESS_UPDATE_EVERY`, `OPTIMIZER_CACHE_TTL_SECONDS`.

These additions expand Super Admin from basic operational toggles into a full optimization-governance control plane without exposing infra-only or secret-sensitive settings.
