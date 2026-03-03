# Tokemizer API Documentation

## Overview

Tokemizer is an enterprise-grade prompt compression API that reduces LLM token usage by 30-70% while preserving semantic meaning and context. The API is **model-agnostic** and **deterministic**. Most business endpoints require authentication, while public diagnostics such as `GET /api/v1/health` remain unauthenticated.

### Key Capabilities
- **Token Reduction**: 30-70% compression depending on optimization mode
- **Three Optimization Modes**: Conservative, Balanced, Maximum
- **Content Protection**: Automatic preservation of code, URLs, numbers, structured data
- **Batch Processing**: Synchronous batch optimization support
- **Weighted Segments**: Importance-aware optimization using segment spans
- **Custom Mappings**: Domain-specific abbreviations and canonicalization
- **Semantic Guard**: Validates output maintains similarity to input
- **TOON Compression**: Maximum-mode JSON compression for structured data

---

## Base URL & Access

**Base URL**: `http://<your-host>:8000/api/v1`

**Interactive Documentation**: 
- Swagger UI: `http://<your-host>:8000/api/v1/docs`
- ReDoc: `http://<your-host>:8000/api/v1/redoc`
- OpenAPI Spec: `http://<your-host>:8000/api/v1/openapi.json`

**Authentication**: JWT Bearer Token (required for optimization, billing, usage, and admin endpoints)

**CORS**: Configurable via `CORS_ORIGINS` environment variable (default: `*`)

**Response Format**: JSON (uses ORJSON for performance)

**Compression**: Automatic gzip compression for responses >500 bytes

### Subscription Pricing Sentinel Values

For plan pricing fields (`monthly_price_cents`, `annual_price_cents`):
- `0`: free/self-serve plan pricing
- `> 0`: paid self-serve plan pricing
- `-1`: contact-sales pricing (enterprise/custom), not free

Plans marked with `-1` are intentionally excluded from self-serve registration/upgrade/payment flows and should direct users to sales.

For plan visibility field (`is_public`):
- `true` (default): plan is eligible for self-serve listing and selection (when active)
- `false`: plan remains assignable by admins but is hidden from self-serve registration/upgrade/checkout

---

## Authentication

Most Tokemizer API endpoints require **JWT Bearer Token** authentication. Public endpoints such as `GET /api/v1/health` do not require a token. Include the token in the `Authorization` header for protected endpoints.

### Authentication Header Format

```
Authorization: Bearer YOUR_API_TOKEN
```

### Obtaining Tokens

Tokens can be obtained through:
- **API Keys**: Generate long-lived API keys from your dashboard
- **User Tokens**: JWT tokens issued during user login
- **Service Accounts**: For server-to-server integrations

### Token Usage Examples

**cURL:**
```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Python Requests:**
```python
import requests

response = requests.get("http://localhost:8000/api/v1/health")
```

**JavaScript Fetch:**
```javascript
fetch('http://localhost:8000/api/v1/health')
    .then(res => res.json())
    .then(data => console.log(data));
```

### Error Responses for Invalid Authentication

**401 Unauthorized** - Token is missing, expired, or invalid:
```json
{
  "detail": "Not authenticated"
}
```

**403 Forbidden** - Token is valid but lacks required permissions:
```json
{
  "detail": "Insufficient permissions for this resource"
}
```

---

## Core Endpoints

### 1. Health Check

Check system status and available optimization techniques.

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-22T12:00:00.000Z",
  "dependencies": {
    "tiktoken": true,
    "numpy": true,
    "spacy": true,
    "datasketch": true
  },
  "models": {
    "spacy_nlp": true,
    "coreference": true,
    "semantic_guard": true,
    "entropy_model": true,
    "token_classifier": true
  },
  "tiktoken_cache": {
    "enabled": true,
    "cache_dir": "default (~/.tiktoken)"
  },
  "techniques": {
    "total": {
      "enabled": 18,
      "total": 22,
      "percentage": 81.8,
      "note": "Total techniques enabled with default settings (optimization_mode='balanced')"
    },
    "base_techniques": {
      "enabled": 16,
      "total": 16,
      "percentage": 100.0,
      "note": "Core techniques available at all optimization levels"
    },
    "maximum_level_techniques": {
      "available": 4,
      "total": 4,
      "percentage": 100.0,
      "note": "Advanced techniques available only in optimization_mode='maximum'"
    }
  },
  "warnings": []
}
```

> The health payload above is an illustrative example. Technique counts and model readiness fields are runtime-dependent (installed models, feature flags, and profile defaults).

**Status Codes**:
- `200`: System healthy

---

### 2. Optimize Prompt(s)

The primary endpoint for prompt optimization. Supports both single prompt and batch processing.

**Endpoint**: `POST /api/v1/optimize`

**Content-Type**: `application/json`

**Authentication**: Required (Bearer token)

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | `string` | Conditional* | - | Single prompt to optimize |
| `prompts` | `array[string]` | Conditional* | - | Batch of prompts (synchronous processing) |
| `optimization_mode` | `string` | No | `"balanced"` | Mode: `"conservative"`, `"balanced"`, `"maximum"` |
| `custom_canonical_map` | `object` | No | `null` | Domain-specific abbreviations (e.g., `{"Machine Learning": "ML"}`) |
| `force_preserve_patterns` | `array[string]` | No | `[]` | Literal tokens or `regex:` prefixed patterns to preserve |
| `force_preserve_digits` | `boolean` | No | `false` | Preserve all tokens containing digits |
| `enable_frequency_learning` | `boolean` | No | `true` | Learn abbreviations from repeated phrases |
| `segment_spans` | `array[object]` | No | `null` | Weighted segments for importance-aware optimization |
| `query` | `string` | No | `null` | Optional query for query-aware optimization |

**\* Conditional**: Must provide **either** `prompt` **or** `prompts`, not both.

#### Optimization Modes

| Mode | Techniques Enabled | Speed vs Maximum | Token Savings | Best For |
|------|-------------------|------------------|---------------|----------|
| `conservative` | Reduced pass set (mode + profile aware) | ~70% faster | 30-35% | Real-time applications, latency-sensitive |
| `balanced` | Default adaptive pass set | ~40% faster | 40-45% | General purpose, production workloads (default) |
| `maximum` | Full adaptive pass set + maximum-only stages | Baseline | 50-70% | Batch processing, offline optimization |

**Note**: Exact pass counts vary by content profile, dependency readiness, and runtime toggles. Maximum mode additionally enables maximum-only stages such as pre-pass budgeting, deeper heavy passes, and TOON JSON compression when applicable.

#### Segment Spans Configuration

Assign importance weights to sections of the prompt:

```json
[
  {
    "start": 0,
    "end": 100,
    "label": "instructions",
    "weight": 2.0
  },
  {
    "start": 100,
    "end": 500,
    "label": "context",
    "weight": 0.8
  }
]
```

#### Request Examples

**Single Prompt (Balanced)**:
```json
{
  "prompt": "Please could you kindly provide a detailed summary of our comprehensive onboarding process and highlight the key stakeholders involved.",
  "optimization_mode": "balanced"
}
```

**Maximum with Custom Mappings**:
```json
{
  "prompt": "Investigate the Service Level Agreement breaches for customer accounts...",
  "optimization_mode": "maximum",
  "custom_canonical_map": {
    "Service Level Agreement": "SLA",
    "Customer Relationship Management": "CRM"
  }
}
```

**Weighted Segments for Importance-Aware Optimization**:
```json
{
  "prompt": "## Instructions\nAnalyze the logs...\n## Background\nContext information...",
  "segment_spans": [
    {"start": 0, "end": 20, "label": "instructions", "weight": 2.0},
    {"start": 20, "end": 100, "label": "context", "weight": 0.8}
  ]
}
```

**Preserve Patterns**:
```json
{
  "prompt": "Investigate account ABC123, compare with ABC124 and review XYZ789.",
  "force_preserve_patterns": ["regex:ABC\\d+", "regex:XYZ\\d+"],
  "force_preserve_digits": true
}
```

**Batch Processing**:
```json
{
  "prompts": [
    "First prompt to optimize...",
    "Second prompt to optimize...",
    "Third prompt to optimize..."
  ],
  "optimization_mode": "balanced"
}
```

#### Response (Single Prompt)

```json
{
  "optimized_output": "Summarize onboarding process, key stakeholders.",
  "stats": {
    "original_chars": 214,
    "optimized_chars": 118,
    "compression_percentage": 44.86,
    "original_tokens": 52,
    "optimized_tokens": 30,
    "token_savings": 22,
    "processing_time_ms": 41.09,
    "fast_path": false,
    "semantic_similarity": 0.92,
    "semantic_similarity_source": "sentence-transformers",
    "content_profile": "prose"
  },
  "router": {
    "content_type": "prose"
  },
  "techniques_applied": [
    "Whitespace Compression",
    "Instruction Simplification",
    "Entity Canonicalization"
  ],
  "warnings": null
}
```

#### Response (Batch)

```json
{
  "batch_job_id": "job-uuid-1234",
  "results": [
    {
      "optimized_output": "First optimized result...",
      "stats": { ... },
      "techniques_applied": [...],
      "warnings": null
    },
    {
      "optimized_output": "Second optimized result...",
      "stats": { ... },
      "techniques_applied": [...],
      "warnings": null
    }
  ],
  "summary": {
    "total_items": 2,
    "avg_compression": 42.3,
    "total_processing_time_ms": 85.6,
    "throughput_prompts_per_second": 23.4
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `optimized_output` | `string` | Optimized prompt text |
| `stats.original_chars` | `integer` | Original character count |
| `stats.optimized_chars` | `integer` | Optimized character count |
| `stats.compression_percentage` | `float` | Percentage of compression achieved |
| `stats.original_tokens` | `integer` | Original token count (tiktoken-based) |
| `stats.optimized_tokens` | `integer` | Optimized token count |
| `stats.token_savings` | `integer` | Tokens saved |
| `stats.processing_time_ms` | `float` | Processing time in milliseconds |
| `stats.fast_path` | `boolean` | Whether fast-path optimization was used |
| `stats.semantic_similarity` | `float` \| `null` | Similarity score (0.0-1.0) |
| `stats.content_profile` | `string` \| `null` | Detected content type (code, prose, json, mixed) |
| `stats.maximum_prepass_selected_sentences` | `array[integer]` \| `null` | Sentence indices kept by the maximum-mode budgeted pre-pass (when applied) |
| `stats.maximum_prepass_target_tokens` | `integer` \| `null` | Target token budget used by the maximum-mode budgeted pre-pass (when applied) |
| `stats.maximum_prepass_policy_source` | `string` \| `null` | Effective pre-pass decision source (`auto`, `off`, `conservative`, `aggressive`, `forced`, or `disabled_mode`) |
| `stats.maximum_prepass_policy_enabled` | `boolean` \| `null` | Effective runtime decision for whether maximum pre-pass was enabled for this request |
| `stats.maximum_prepass_policy_mode` | `string` \| `null` | Resolved user policy mode (`off`, `auto`, `conservative`, `aggressive`) before explicit overrides |
| `stats.maximum_prepass_policy_enabled_override` | `boolean` \| `null` | Whether `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_ENABLED` explicitly overrode policy enablement |
| `stats.maximum_prepass_policy_minimum_tokens` | `integer` \| `null` | Effective runtime minimum-token threshold resolved for maximum pre-pass eligibility |
| `stats.maximum_prepass_policy_budget_ratio` | `float` \| `null` | Effective runtime budget ratio resolved for maximum pre-pass sentence/span selection |
| `stats.maximum_prepass_policy_max_sentences` | `integer` \| `null` | Effective runtime maximum sentence cap resolved for maximum pre-pass |
| `techniques_applied` | `array[string]` \| `null` | List of techniques executed |
| `warnings` | `array[string]` \| `null` | Warnings about disabled features or semantic guard fallback |
| `router` | `object` \| `null` | Content routing information |

**Note**: Warnings are also returned in the `X-Tokemizer-Warnings` response header (sanitized for HTTP compatibility).

#### Status Codes

- `200`: Success
- `400`: Invalid request (validation error, prompt/prompts missing)
- `500`: Internal server error

---

## Bulk/Batch Optimization API

### Synchronous Batch Processing

Process multiple prompts in a single API request. All prompts are optimized synchronously and results are returned immediately.

**Same Endpoint**: `POST /api/v1/optimize`

**Authentication**: Required (Bearer token)

#### Bulk Request Example

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "prompts": [
      "First prompt to optimize...",
      "Second prompt to optimize...",
      "Third prompt to optimize...",
      "Fourth prompt to optimize..."
    ],
    "optimization_mode": "maximum"
  }'
```

#### Bulk Response Structure

```json
{
  "batch_job_id": "job-uuid-5678",
  "results": [
    {
      "optimized_output": "First optimized result...",
      "stats": {
        "original_tokens": 150,
        "optimized_tokens": 85,
        "token_savings": 65,
        "compression_percentage": 43.3,
        "processing_time_ms": 125.5,
        "semantic_similarity": 0.94,
        "content_profile": "prose"
      },
      "techniques_applied": [
        "Whitespace Compression",
        "Politeness Removal",
        "Entity Canonicalization"
      ],
      "warnings": null
    },
    {
      "optimized_output": "Second optimized result...",
      "stats": { ... },
      "techniques_applied": [...],
      "warnings": null
    }
  ],
  "summary": {
    "total_items": 4,
    "avg_compression": 42.8,
    "total_processing_time_ms": 512.3,
    "throughput_prompts_per_second": 7.8
  }
}
```

#### Bulk Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompts` | `array[string]` | Yes | - | Array of prompts to optimize (1-1000 items recommended) |
| `optimization_mode` | `string` | No | `"balanced"` | Mode: `"conservative"`, `"balanced"`, `"maximum"` |
| `custom_canonical_map` | `object` | No | `null` | Domain-specific abbreviations applied to all prompts |
| `force_preserve_patterns` | `array[string]` | No | `[]` | Patterns to preserve across all prompts |
| `force_preserve_digits` | `boolean` | No | `false` | Preserve digits in all prompts |
| `enable_frequency_learning` | `boolean` | No | `true` | Learn abbreviations from repeated phrases |

#### Bulk Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `batch_job_id` | `string` | Unique identifier for this batch job |
| `results` | `array[object]` | Array of optimization results (same structure as single prompt response) |
| `summary.total_items` | `integer` | Total number of prompts processed |
| `summary.avg_compression` | `float` | Average compression percentage across all prompts |
| `summary.total_processing_time_ms` | `float` | Total time to process all prompts |
| `summary.throughput_prompts_per_second` | `float` | Average throughput |

#### Python Example (Bulk Optimization)

```python
import requests
import json

# Bulk optimization with multiple prompts
api_key = "YOUR_API_TOKEN"
url = "http://localhost:8000/api/v1/optimize"

prompts_batch = [
    "Please provide a detailed summary of the following information...",
    "Can you help me understand the key concepts in this document?",
    "Summarize the main points from this lengthy article...",
    "Extract and compress the essential facts from this text..."
]

response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "prompts": prompts_batch,
        "optimization_mode": "maximum"
    }
)

result = response.json()
print(f"Batch Job ID: {result['batch_job_id']}")
print(f"Total Prompts: {result['summary']['total_items']}")
print(f"Average Compression: {result['summary']['avg_compression']:.2f}%")
print(f"Total Processing Time: {result['summary']['total_processing_time_ms']:.2f}ms")

# Access individual results
for i, optimization in enumerate(result['results']):
    print(f"\nPrompt {i+1}:")
    print(f"  Optimized: {optimization['optimized_output']}")
    print(f"  Tokens Saved: {optimization['stats']['token_savings']}")
    print(f"  Similarity: {optimization['stats']['semantic_similarity']:.2f}")
```

#### Node.js Example (Bulk Optimization)

```javascript
const axios = require('axios');

async function batchOptimize(prompts, token) {
  const url = 'http://localhost:8000/api/v1/optimize';
  
  const response = await axios.post(url, {
    prompts: prompts,
    optimization_mode: 'maximum'
  }, {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    }
  });
  
  const result = response.data;
  console.log(`Batch ID: ${result.batch_job_id}`);
  console.log(`Total Items: ${result.summary.total_items}`);
  console.log(`Avg Compression: ${result.summary.avg_compression}%`);
  
  // Process results
  result.results.forEach((res, idx) => {
    console.log(`\nPrompt ${idx + 1}:`);
    console.log(`  Optimized: ${res.optimized_output}`);
    console.log(`  Compression: ${res.stats.compression_percentage}%`);
  });
  
  return result;
}

// Usage
const prompts = [
  "First prompt to optimize...",
  "Second prompt to optimize...",
  "Third prompt to optimize..."
];

batchOptimize(prompts, 'YOUR_API_TOKEN')
  .catch(err => console.error('Error:', err.message));
```

#### Batch Jobs Tracking

Track the status of batch jobs:

**Endpoint**: `GET /api/v1/batch-jobs`

**Authentication**: Required

**Response**:
```json
{
  "batch_jobs": [
    {
      "id": "job-uuid-5678",
      "name": "Marketing Campaign Batch",
      "status": "completed",
      "total_items": 50,
      "processed_items": 50,
      "progress_percentage": 100,
      "savings_percentage": 42.3,
      "processing_time_ms": 5234.6,
      "created_at": "2025-01-01T11:00:00Z",
      "completed_at": "2025-01-01T11:05:23Z"
    }
  ]
}
```

#### Performance Recommendations for Bulk Operations

- **Batch Size**: 1-1000 prompts per request recommended
- **Timeout**: Set client timeout to at least 30 seconds for large batches
- **Optimization Mode**: Use `balanced` mode for optimal latency/quality tradeoff
- **Memory**: Ensure sufficient memory for prompt storage (8GB+ recommended for >500 prompts)
- **Rate Limiting**: Implement exponential backoff for failed requests

---

## Canonical Mappings Management

Manage persistent abbreviations and canonicalization rules applied across all optimization requests.

### List Canonical Mappings

**Endpoint**: `GET /api/v1/canonical-mappings`

**Authentication**: Required

**Query Parameters**:
- `offset` (integer, default: 0): Number of records to skip
- `limit` (integer, default: 100): Number of records to return

**Response**:
```json
{
  "mappings": [
    {
      "id": 1,
      "source_token": "Artificial Intelligence",
      "target_token": "AI",
      "created_at": "2025-01-01T10:00:00Z",
      "updated_at": "2025-01-01T10:00:00Z"
    }
  ],
  "total": 45,
  "offset": 0,
  "limit": 100
}
```

---

### Create/Update Canonical Mapping

Creates a new mapping or updates if source_token already exists (upsert).

**Endpoint**: `POST /api/v1/canonical-mappings`

**Authentication**: Required

**Request Body**:
```json
{
  "source_token": "Machine Learning",
  "target_token": "ML"
}
```

**Response** (201):
```json
{
  "id": 2,
  "source_token": "Machine Learning",
  "target_token": "ML",
  "created_at": "2025-01-01T11:00:00Z",
  "updated_at": "2025-01-01T11:00:00Z"
}
```

---

### Bulk Create Canonical Mappings

Create or update multiple mappings in one request (upsert behavior).

**Endpoint**: `POST /api/v1/canonical-mappings/bulk`

**Authentication**: Required

**Request Body**:
```json
{
  "mappings": [
    {
      "source_token": "Natural Language Processing",
      "target_token": "NLP"
    },
    {
      "source_token": "Deep Learning",
      "target_token": "DL"
    }
  ]
}
```

**Response** (201):
```json
[
  {
    "id": 3,
    "source_token": "Natural Language Processing",
    "target_token": "NLP",
    "created_at": "2025-01-01T11:05:00Z",
    "updated_at": "2025-01-01T11:05:00Z"
  }
]
```

---

### Delete Canonical Mappings

Delete multiple mappings by their IDs.

**Endpoint**: `DELETE /api/v1/canonical-mappings`

**Authentication**: Required

**Request Body**:
```json
{
  "ids": [1, 2, 3]
}
```

**Response**:
```json
{
  "deleted_count": 3
}
```

---

## Analytics & Telemetry

### Optimization History

Query recent optimization requests with compression stats.

**Endpoint**: `GET /api/v1/history`

**Authentication**: Required

**Query Parameters**:
- `limit` (integer, default: 50, max: 500): Number of records to return

**Response**:
```json
{
  "optimizations": [
    {
      "id": "uuid-1234",
      "original_tokens": 500,
      "optimized_tokens": 250,
      "tokens_saved": 250,
      "compression_percentage": 50.0,
      "semantic_similarity": 0.92,
      "mode": "balanced",
      "created_at": "2025-01-01T12:00:00Z",
      "techniques_applied": [
        "Whitespace Compression",
        "Entity Canonicalization"
      ]
    }
  ]
}
```

---

### Aggregate Statistics

Get aggregated optimization metrics for the current customer.

**Endpoint**: `GET /api/v1/stats`

**Authentication**: Required

**Response**:
```json
{
  "tokens_saved": 120000,
  "cost_savings": 1.8,
  "avg_compression_percentage": 42.3,
  "avg_latency_ms": 180.5,
  "avg_quality_score": 0.91,
  "total_optimizations": 420,
  "estimated_monthly_savings": 54.2
}
```

**Note**: Cost savings calculated based on $0.015 per 1M input tokens. Data is customer-specific.

---

### Batch Jobs

List recent batch optimization jobs.

**Endpoint**: `GET /api/v1/batch-jobs`

**Authentication**: Required

**Query Parameters**:
- `limit` (integer, default: 20, max: 100): Number of jobs to return

**Response**:
```json
{
  "batch_jobs": [
    {
      "id": "job-uuid-5678",
      "name": "Marketing Campaign Batch",
      "status": "completed",
      "total_items": 50,
      "processed_items": 50,
      "progress_percentage": 100,
      "savings_percentage": 42.3,
      "processing_time_ms": 5234.6,
      "created_at": "2025-01-01T11:00:00Z",
      "completed_at": "2025-01-01T11:05:23Z"
    }
  ]
}
```

---

### Per-Pass Telemetry

Get detailed per-pass optimization metrics (when telemetry is enabled).

**Endpoint**: `GET /api/v1/telemetry/recent`

**Authentication**: Required

**Query Parameters**:
- `limit` (integer, default: 100, max: 500): Number of records to return

**Response**:
```json
[
  {
    "optimization_id": "uuid-1234",
    "pass_name": "normalize_whitespace",
    "pass_order": 1,
    "duration_ms": 2.3,
    "tokens_before": 500,
    "tokens_after": 490,
    "tokens_saved": 10,
    "reduction_percent": 2.0,
    "created_at": "2025-01-01T12:00:00Z"
  }
]
```

**Note**: Telemetry is disabled by default. Enable it via the Settings API (`PATCH /api/v1/settings` with `"telemetry_enabled": true`) or the Admin Settings UI. Only recorded passes appear while telemetry is enabled.

---

## Runtime Settings

### Get Settings

Retrieve current runtime configuration for the customer.

**Endpoint**: `GET /api/v1/settings`

**Authentication**: Required

**Response**:
```json
{
  "semantic_guard_threshold": 0.75,
  "semantic_guard_enabled": true,
  "semantic_guard_model": "BAAI/bge-small-en-v1.5",
  "guard_latency_ms": 600,
  "guard_tokens_saved": 20,
  "telemetry_baseline_window_days": 30,
  "optimizer_cache_size": 256,
  "telemetry_enabled": false,
  "lsh_enabled": false,
  "lsh_similarity_threshold": 0.8
}
```

---

### Update Settings

Update runtime configuration (partial updates supported).

**Endpoint**: `PATCH /api/v1/settings`

**Authentication**: Required

**Request Body** (all fields optional):
```json
{
  "semantic_guard_threshold": 0.8,
  "semantic_guard_enabled": true,
  "optimizer_cache_size": 512,
  "telemetry_enabled": true,
  "guard_latency_ms": 520,
  "guard_tokens_saved": 25,
  "telemetry_baseline_window_days": 45,
  "lsh_enabled": true,
  "lsh_similarity_threshold": 0.9
}
```

**Response**: Returns updated settings (same format as GET /settings)

**Note**: Changes to these settings may clear the optimization cache.
Guard and telemetry thresholds take effect immediately for subsequent optimization requests.

---

## LLM Provider Utility Endpoints

### List Providers

**Endpoint**: `GET /api/v1/llm/providers`

Returns provider metadata and curated model options used by the admin console/test flows.

### Test Provider Connectivity

**Endpoint**: `POST /api/v1/llm/test`

**Authentication**: Required

Proxies a single prompt to the selected upstream provider (`openai`, `anthropic`, `google`, or `ollama`) for connectivity/credential verification.

---

## Error Responses

All endpoints return errors in FastAPI's standard format:

```json
{
  "detail": "Error message describing the issue"
}
```

### Common Error Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `400` | Bad Request | Missing required fields, invalid values, validation errors |
| `401` | Unauthorized | Missing or invalid authentication token |
| `403` | Forbidden | Insufficient permissions or quota exceeded |
| `404` | Not Found | Resource doesn't exist |
| `500` | Internal Server Error | Unexpected server error |

---

## Integration Examples

### Python

```python
import requests

# Single prompt optimization
response = requests.post(
    "http://localhost:8000/api/v1/optimize",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "prompt": "Please provide detailed information about...",
        "optimization_mode": "balanced"
    }
)
result = response.json()
print(f"Optimized: {result['optimized_output']}")
print(f"Tokens saved: {result['stats']['token_savings']}")

# Batch optimization
response = requests.post(
    "http://localhost:8000/api/v1/optimize",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "optimization_mode": "maximum"
    }
)
batch_result = response.json()
print(f"Batch completed: {batch_result['summary']['total_items']} items")
```

### Node.js

```javascript
const axios = require('axios');

async function optimizePrompt(prompt, token, mode = 'balanced') {
  const response = await axios.post('http://localhost:8000/api/v1/optimize', {
    prompt: prompt,
    optimization_mode: mode
  }, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return {
    optimized: response.data.optimized_output,
    savings: response.data.stats.token_savings,
    compression: response.data.stats.compression_percentage
  };
}
```

### cURL

```bash
# Optimize single prompt
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "Please provide a detailed summary...",
    "optimization_mode": "balanced"
  }'

# Check health
curl http://localhost:8000/api/v1/health

# List canonical mappings
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/canonical-mappings?limit=50"
```

---

## Environment Configuration

Key environment variables for API customization:

```bash
# Database
DB_PATH=/app/data/app.db

# CORS
CORS_ORIGINS=*

# Optimization
OPTIMIZER_CACHE_SIZE=256
PROMPT_OPTIMIZER_MAX_WORKERS=4

# Semantic Guard
PROMPT_OPTIMIZER_SEMANTIC_GUARD_ENABLED=true
PROMPT_OPTIMIZER_SEMANTIC_GUARD_THRESHOLD=0.75
PROMPT_OPTIMIZER_SEMANTIC_GUARD_MODEL=BAAI/bge-small-en-v1.5

# Model Prewarming
OPTIMIZER_PREWARM_MODELS=true

# Tiktoken Cache
TIKTOKEN_CACHE_DIR=~/.tiktoken
HF_HOME=/app/.cache/huggingface
```

Telemetry collection is disabled by default. Toggle it via the Settings API (`PATCH /api/v1/settings` with `"telemetry_enabled"`) or the Admin Settings UI.

---

## Deployment Considerations

### Security

- **Authentication**: Protected endpoints require JWT Bearer token authentication (health remains public)
- **API Keys**: Use customer-specific API keys stored securely
- **CORS**: Configure `CORS_ORIGINS` environment variable for production (avoid `*`)
- **Network Isolation**: Restrict network access to trusted sources
- **Rate Limiting**: Implement at infrastructure level or via quota management

### Performance

- **Memory**: Recommended 8GB+ for handling large prompts (200K+ tokens)
- **CPU**: Multi-core recommended for batch processing
- **Caching**: Configure `OPTIMIZER_CACHE_SIZE` based on workload
- **Workers**: Tune `PROMPT_OPTIMIZER_MAX_WORKERS` for concurrent batch processing

### Monitoring

- **Health Endpoint**: Monitor `/api/v1/health` for dependency availability
- **Metrics**: Toggle telemetry (via the Settings API or Admin UI) to collect per-pass metrics.
- **Logging**: Structured logs available via stdout/stderr
- **Stats**: Query `/api/v1/stats` for aggregate performance metrics

---

## Version Information

- **API Version**: 1.0.0
- **FastAPI Version**: 0.110+
- **Python Version**: 3.11+
- **OpenAPI Specification**: Available at `/api/v1/openapi.json`

---

## Support & Resources

- **Interactive API Documentation**: `http://<your-host>:8000/api/v1/docs`
- **OpenAPI Specification**: `http://<your-host>:8000/api/v1/openapi.json`
- **GitHub Repository**: [tokemizer](https://github.com/nowusman/tokemizer)

---

**Last Updated**: 2026-01-22
