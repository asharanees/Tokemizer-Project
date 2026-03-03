# Model Management Module: Governance & Operations-Only Master Plan (Tokemizer)

## 1) Executive Summary

This plan defines a **comprehensive, production-ready extraction** of Tokemizer model management into a reusable module/service focused on:

- model catalog governance,
- cache/download/validation operations,
- readiness/warmup orchestration,
- artifact and air-gap lifecycle,
- observability and controls.

> **Scope lock:** This module is **governance/operations only**. It does **not** host centralized model inference/runtime serving.

Inference guidelines are included for **consumer applications** (Tokemizer and future apps) so each app can run its own runtime safely (in-process API, ONNX runtime, etc.) while still relying on this module for governance/ops.

---

## 2) Scope, Non-Scope, and Decision Guardrails

### In scope (must ship)

- Catalog management (create/update/delete/list protected model types).
- Model cache management (download, refresh, force-redownload, recovery cleanup).
- Integrity controls (manifest, expected files, size, revision).
- Readiness snapshots (cached/loaded/intended usage readiness).
- Artifact upload (zip/tar/tgz), extraction safety, air-gap readiness checks.
- Admin APIs and UI coverage for all above concerns.
- Service-to-service contract for Tokemizer backend and future apps.

### Out of scope (must not be included in this module)

- Centralized inference gateway.
- Shared embedding/classification/generation APIs.
- Runtime load-balancing for inference traffic.
- GPU scheduling for cross-app inference workloads.

### Why scope is governance/ops only

- Keeps optimization path latency predictable.
- Minimizes migration risk and blast radius.
- Reuses current mature logic in this repo with minimal behavior drift.

---

## 3) Current-State Inventory (Repo-anchored)

### Core backend ownership today

- `backend/database_extensions.py`
  - model inventory CRUD and soft-delete behavior.
- `backend/services/model_cache_manager.py`
  - `ModelCacheValidator`, model download/refresh, manifest handling, lock files, archive ingestion.
- `backend/routers/admin_routes.py`
  - model admin APIs: list/create/update/delete, refresh status/trigger, upload, air-gap.
- `backend/services/optimizer/core.py`
  - runtime config refresh and model readiness probing hooks.
- `backend/services/optimizer/model_capabilities.py`
  - capability contract and readiness/warning semantics.

### UI ownership today

- `frontend/src/pages/admin/Models.tsx`
  - model admin screen surface.
- `frontend/src/pages/admin/__tests__/Models.test.tsx`
  - base UI behavior checks.

### Platform/API entrypoints

- `backend/server.py`
  - app composition, routers, middleware registration.

---

## 4) Target Architecture (Governance/Ops only)

## 4.1 Logical components

1. **Catalog Domain**
   - Source of truth for model definitions and policy metadata.
2. **Cache Operations Domain**
   - Download/refresh/recovery, manifest generation/validation.
3. **Readiness Domain**
   - readiness computation from cache + runtime probe snapshots.
4. **Artifact Domain**
   - upload + secure extraction + cache placement + validation.
5. **Admin API Domain**
   - stable contract for UI and client apps.
6. **Client SDK Domain**
   - typed access for Tokemizer backend and future apps.

## 4.2 Deployment topology

### Phase A (in-repo modular monolith)
- Keep one backend process.
- Extract model management code into `backend/services/model_management/*`.
- Keep existing admin endpoints backward compatible.

### Phase B (standalone governance service)
- Deploy separate service process/container.
- Tokemizer backend uses remote client adapter.
- Same API contract; no inference traffic routed to this service.

## 4.3 Middleware and cross-cutting concerns

Must cover:

- Authn/Authz for admin operations.
- Request-id propagation and structured logs.
- Rate limiting for expensive refresh/upload routes.
- Idempotency keys for refresh and upload actions.
- Timeout budgets and retry policy for inter-service calls.
- Safe payload limits for archives.

---

## 5) Detailed API Contract (v1)

Base path: `/api/model-mgmt/v1`

## 5.1 Catalog APIs

- `GET /catalog/models`
- `GET /catalog/models/protected`
- `POST /catalog/models`
- `PUT /catalog/models/{model_type}`
- `DELETE /catalog/models/{model_type}`

### Contract rules

- `model_type` is immutable key.
- soft delete only (historical audit preserved).
- protected model types require explicit override flag.
- revision and allow-patterns normalized on write.

## 5.2 Cache/Refresh APIs

- `GET /cache/refresh` (state machine: `idle|running|completed|failed`)
- `POST /cache/refresh` (global or scoped)
- `POST /cache/models/{model_type}/refresh`
- `GET /cache/snapshot`
- `POST /cache/validation/bump`

### Refresh request schema (draft)

```json
{
  "refresh_mode": "download_missing",
  "model_types": ["semantic_guard", "semantic_rank", "spacy"],
  "allow_spacy_downloads": true,
  "idempotency_key": "optional-string"
}
```

### Refresh modes

- `download_missing`
- `force_redownload`
- `recovery`

## 5.3 Readiness APIs

- `GET /readiness/models`
- `POST /readiness/probe/{model_type}`

### Readiness guarantees

- returns `intended_usage_ready` and reason for every configured model.
- exposes `hard_required`, mode/profile gates, feature intent.
- generated timestamp and snapshot version included.

## 5.4 Artifact / Air-gap APIs

- `POST /artifacts/models/{model_type}/upload`
- `GET /artifacts/airgap`

### Security/validation guarantees

- block path traversal on extraction.
- accepted archive formats: zip/tar/tgz.
- validate expected files + size + manifest consistency.
- report clear machine-readable failure reasons.

## 5.5 Health & diagnostics APIs (additions)

- `GET /health/live`
- `GET /health/ready`
- `GET /diagnostics/download-issues`
- `POST /diagnostics/download-issues/reset`

These endpoints make ops automation and dashboards reliable.

---

## 6) Backend Implementation Plan (Detailed)

## 6.1 New package layout

Create:

- `backend/services/model_management/contracts.py`
- `backend/services/model_management/catalog.py`
- `backend/services/model_management/cache_ops.py`
- `backend/services/model_management/readiness.py`
- `backend/services/model_management/artifacts.py`
- `backend/services/model_management/service.py` (facade)
- `backend/services/model_management/client.py` (remote client)

## 6.2 Refactor ownership mapping

- Move inventory CRUD wrappers out of route layer into `catalog.py`.
- Move refresh/snapshot state orchestration out of route layer into `cache_ops.py`.
- Move readiness assembly into `readiness.py` and keep compatibility with `optimizer/model_capabilities.py`.
- Keep `model_cache_manager.py` as low-level operations engine; call it through `cache_ops.py`.

## 6.3 Router changes

- Keep `backend/routers/admin_routes.py` thin (HTTP-only orchestration).
- Route handlers call `ModelManagementService` facade.
- No direct DB/cache-manipulation logic in router bodies.

## 6.4 Server wiring

- `backend/server.py`: add explicit model-management config envs and dependency wiring.
- Ensure middleware order supports auth + logging + compression safely.

## 6.5 Data persistence and migration

- Reuse current `model_inventory` schema.
- Add optional table for refresh job audit trail:
  - `model_refresh_jobs(id, requested_by, mode, scope, status, warnings, error, started_at, ended_at)`
- Add optional table for artifact uploads audit trail.

---

## 7) UI Plan (Admin + UX hardening)

## 7.1 Admin Models UI scope

Update `frontend/src/pages/admin/Models.tsx` to include:

- explicit readiness panel by model type,
- refresh job status + timestamps,
- download issue diagnostics section,
- validation version indicator,
- idempotent action UX (disable duplicate submits).

## 7.2 UI states (must-have)

- loading, empty, partial-failure, stale snapshot, success.
- retry CTA for recoverable states.
- operator-safe confirmations for destructive actions.

## 7.3 UI/API contract requirements

- show machine reason + human message for failures.
- preserve and display warnings array from API.
- avoid optimistic updates for refresh/upload side effects.

## 7.4 UI test plan

Expand:

- `frontend/src/pages/admin/__tests__/Models.test.tsx`

Add tests for:
- refresh in-progress polling,
- error/warning rendering,
- protected model delete guard UX,
- upload validation failure surfaces.

---

## 8) Middleware, Security, and Governance Controls

## 8.1 Auth/Authz

- Admin-only routes require existing admin auth path.
- Future cross-service mode: service token with audience claim.

## 8.2 Auditability

- Log actor, action, model_type, mode, outcome, request_id.
- Persist refresh/upload job summaries.

## 8.3 Safety controls

- archive size cap and MIME/file-extension verification.
- lock TTL cleanup and stale lock recovery.
- per-route timeout budgets.

## 8.4 Rate and concurrency control

- allow only one global refresh at a time.
- scoped refresh allowed when not conflicting with global refresh.
- upload concurrency bounded per model type.

---

## 9) Observability and SLOs

## 9.1 Metrics

- refresh duration histogram by mode.
- model cache hit/invalid/missing counts.
- manifest mismatch counts by reason.
- upload success/failure counts by reason.
- readiness-not-ready counts by model type.

## 9.2 Logs

- structured JSON logs with request_id, job_id, model_type.
- warn on non-fatal degradations; error on hard failures.

## 9.3 SLO draft

- P95 `GET /catalog/models` < 300ms (cached snapshot path).
- P95 `GET /readiness/models` < 250ms.
- refresh reliability: > 99% successful completion when dependencies healthy.

---

## 10) Compatibility and Migration Strategy

## 10.1 Mode flags

- `PROMPT_OPTIMIZER_MODEL_MGMT_MODE=local|remote`
- `PROMPT_OPTIMIZER_MODEL_MGMT_BASE_URL`
- `PROMPT_OPTIMIZER_MODEL_MGMT_TIMEOUT_MS`
- `PROMPT_OPTIMIZER_MODEL_MGMT_RETRY_COUNT`

## 10.2 Rollout sequence

1. Internal modularization (no API behavior change).
2. Route layer slim-down + contract tests.
3. Remote-read mode (readiness/catalog only).
4. Remote-write mode (refresh/upload).
5. Standalone deployment cutover.

## 10.3 Fallback strategy

- If remote service unavailable, read-only cached snapshot fallback allowed.
- mutating operations fail fast with explicit operator message.

---

## 11) Comprehensive Phased Backlog

## Phase 0 — Contract hardening (3–5 days)

- Define Pydantic contracts for catalog/cache/readiness/artifacts.
- Add OpenAPI examples and error codes.
- Add contract tests.

**Files touched (expected):**
- `backend/services/model_management/contracts.py` (new)
- `backend/routers/admin_routes.py`
- `backend/tests/test_admin_model_schema_validation.py`

## Phase 1 — Backend domain extraction (1–2 weeks)

- Create model-management domain package and facade.
- Move non-HTTP orchestration out of `admin_routes.py`.
- Keep endpoint compatibility.

**Files touched (expected):**
- `backend/routers/admin_routes.py`
- `backend/services/model_cache_manager.py`
- `backend/services/model_management/*.py` (new)

## Phase 2 — UI hardening (4–7 days)

- Improve admin models UX for readiness/refresh diagnostics.
- Add robust error/warning rendering.

**Files touched (expected):**
- `frontend/src/pages/admin/Models.tsx`
- `frontend/src/pages/admin/__tests__/Models.test.tsx`

## Phase 3 — Remote client integration (1–2 weeks)

- Add remote client adapter in backend.
- Add mode flags and timeout/retry policies.
- Read-only remote paths first.

**Files touched (expected):**
- `backend/services/model_management/client.py` (new)
- `backend/services/optimizer/core.py`
- `backend/server.py`

## Phase 4 — Standalone governance service (2–3 weeks)

- Extract deployable service with same contract.
- Add health/diagnostic endpoints and service auth.
- Execute controlled cutover.

## Phase 5 — Hardening and multi-app onboarding (1–3 weeks)

- SLO dashboards + alerting.
- runbooks and incident procedures.
- SDK usage guide for second application onboarding.

---

## 12) Risks, Failure Modes, and Mitigations

1. **Route-to-domain regression risk**
   - Mitigation: endpoint-level regression tests against current behavior.

2. **Stale snapshot confusion in UI**
   - Mitigation: always display `generated_at` and state freshness badges.

3. **Refresh lock deadlocks/stale lock files**
   - Mitigation: lock TTL + stale lock cleanup + explicit operator reset endpoint.

4. **HF auth churn / token expiry failures**
   - Mitigation: download-issues API + reset API + actionable error messages.

5. **Remote service outage effects**
   - Mitigation: strict timeout, bounded retries, local stale-read fallback.

---

## 13) Detailed Benefits of This Module

### Technical benefits

- **Separation of concerns**: optimizer runtime stays focused on optimization, not model operations.
- **Lower coupling**: model lifecycle changes stop forcing optimizer code churn.
- **Contract-first governance**: explicit APIs reduce hidden assumptions.
- **Improved reliability**: dedicated refresh/readiness state machine and diagnostics.

### Operational benefits

- **Central policy control** for model revisions and expected artifacts.
- **Air-gap readiness** made auditable and repeatable.
- **Faster incident triage** through diagnostics endpoints and structured logs.
- **Safer admin operations** with idempotency and conflict controls.

### Product and scaling benefits

- **Reuse across applications** with same governance contract.
- **Faster onboarding** of new apps (no repeated model-ops implementation).
- **Lower long-term maintenance cost** through shared module and SDK patterns.

### Security/compliance benefits

- **Traceable model change history** and operator actions.
- **Controlled artifact ingestion** with extraction/path safety.
- **Policy enforcement point** for protected model types and revision checks.

---

## 14) Inference Runtime Guidelines for Consumer Apps (Not part of this module)

This section defines how Tokemizer and other apps should implement runtime/inference locally while relying on model-management governance APIs.

## 14.1 Guiding principle

- **Governance is centralized, inference is app-local.**
- App runtime selects local execution strategy (Python in-process, ONNX runtime, etc.) based on latency and hardware profile.

## 14.2 Recommended runtime patterns

### Pattern A — In-process Python runtime (default)

Use for low-latency CPU workloads and simplest deployment:
- load model from local cache path resolved by governance metadata,
- warm model during app startup or background prewarm,
- maintain local model object cache with lock-protected lazy loading.

### Pattern B — ONNX Runtime in-process (performance-focused)

Use when model supports ONNX and stable CPU inference is required:
- prefer ONNX artifacts generated/validated by governance layer,
- configure execution providers explicitly,
- benchmark and pin thread counts to avoid noisy-neighbor behavior.

### Pattern C — Sidecar runtime (advanced)

Use when app process isolation is required:
- sidecar container loads model locally from shared volume,
- app calls localhost sidecar API,
- still consume governance readiness/catalog APIs centrally.

## 14.3 Runtime compliance checklist for apps

Every app using this governance module should:

- verify model revision against governance API before loading,
- verify expected artifact presence before runtime initialization,
- expose runtime readiness endpoint separate from governance readiness,
- fail fast on hard-required model mismatch,
- emit startup telemetry for loaded model versions.

## 14.4 Runtime anti-patterns to avoid

- do not silently auto-download models in inference request path.
- do not bypass expected-files/revision checks.
- do not share one global mutable model object without synchronization.
- do not mix multiple untracked model revisions in one deployment.

## 14.5 Suggested app-level interface

```python
class AppModelRuntimeProvider(Protocol):
    def load(self, model_type: str) -> None: ...
    def is_ready(self, model_type: str) -> bool: ...
    def infer(self, model_type: str, payload: dict) -> dict: ...
```

This interface remains app-owned and separate from governance APIs.

---

## 15) Final Recommendation and Acceptance Criteria

## Recommendation

Proceed with **governance/operations-only extraction** and keep inference runtime outside this module.

## Acceptance criteria (bullet-proof definition)

- Backend:
  - all model governance flows work through domain facade/services,
  - admin routes remain stable or intentionally versioned,
  - full contract tests passing.

- API:
  - catalog/cache/readiness/artifact contracts documented and validated,
  - deterministic error codes + operator messages.

- UI:
  - complete admin flows for list/create/update/delete/refresh/upload/diagnostics,
  - robust error/warning/partial-state handling.

- Middleware/Security:
  - authn/authz, idempotency, timeout, and logging controls in place.

- Operations:
  - metrics/logs/alerts/runbooks ready before standalone cutover.

- Scope discipline:
  - no centralized inference endpoints added in this module rollout.
