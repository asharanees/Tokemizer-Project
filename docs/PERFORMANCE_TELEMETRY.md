# Performance Telemetry Guide

## Overview

The Performance Telemetry system provides detailed, per-pass metrics for optimization operations. It captures timing, token counts, and reduction percentages for individual optimization passes, enabling analysis of pipeline performance and effectiveness.

Telemetry data is collected asynchronously and batched for efficient database writes, ensuring minimal performance impact on the optimization pipeline.

---

## Configuration

Telemetry collection is disabled by default to minimize overhead. Enable it on demand by toggling the Admin Settings page or patching the Settings API.

### Runtime Control

Toggle telemetry at runtime via:

```bash
# Enable telemetry
curl -X PATCH http://localhost:8000/api/v1/settings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"telemetry_enabled": true}'

# Disable telemetry
curl -X PATCH http://localhost:8000/api/v1/settings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"telemetry_enabled": false}'
```

**Note:** These toggles take effect immediately and are persisted in runtime settings, so the current value is retained across backend restarts.

---

## Database Schema

Telemetry data is stored in the `performance_telemetry` table:

```sql
CREATE TABLE performance_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_id TEXT NOT NULL,           -- Links to optimization_history.id
    pass_name TEXT NOT NULL,                 -- Name of the optimization pass
    pass_order INTEGER NOT NULL,             -- Order in which pass executed (1-based)
    duration_ms REAL NOT NULL,               -- How long the pass took (milliseconds)
    tokens_before INTEGER NOT NULL,          -- Token count before pass
    tokens_after INTEGER NOT NULL,           -- Token count after pass
    tokens_saved INTEGER NOT NULL,           -- tokens_before - tokens_after
    reduction_percent REAL NOT NULL,         -- Percentage reduction for this pass
    created_at TEXT NOT NULL,                -- ISO 8601 timestamp
    FOREIGN KEY (optimization_id) REFERENCES optimization_history(id)
);
```

**Indexes:**

- `idx_telemetry_optimization_id` - Fast lookup by optimization ID
- `idx_telemetry_pass_order` - Efficient retrieval of pass sequences
- `idx_telemetry_pass_name` - Aggregate metrics by pass type
- `idx_telemetry_created_at` - Time-based queries

---

## Recorded Telemetry Passes

The following passes may be recorded in telemetry (depending on optimization mode and configuration):

| Pass Name | Description | Typical Duration |
|-----------|-------------|------------------|
| `normalize_whitespace` | Early whitespace normalization | <10ms |
| `lexical_transforms` | Synonym replacement, list compression, entity canonicalization | 10-50ms |
| `deduplicate_content` | LSH/MinHash-based content deduplication | >50ms (if not short-circuited) |
| `normalize_text` | Final text normalization and punctuation compression | <10ms |
| `llm_based` | LLM provider optimization pass (single-pass telemetry record) | Provider/network dependent |

**Note:** Not all passes are instrumented for telemetry. Passes without explicit telemetry recording contribute to overall optimization time but are not individually tracked.

---

## Batch Writing

Telemetry records are batched for efficient database writes:

- **Batch Size**: 50 records (fixed)
- **Flush Interval**: 5.0 seconds (fixed)
- **Thread**: Background daemon thread (`TelemetryWriter`)

Records are flushed when either the batch size is reached or the flush interval elapses, whichever comes first.

**Note:** Batch configuration values are fixed in the implementation and not configurable via environment variables.

---

## Querying Telemetry Data

### API Endpoint

Retrieve recent telemetry data:

```bash
# Get recent telemetry data (default: 100 records, max: 500)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/telemetry/recent?limit=100"
```

**Response Format:**
```json
[
  {
    "optimization_id": "uuid-1234",
    "pass_name": "normalize_whitespace",
    "pass_order": 1,
    "duration_ms": 2.5,
    "tokens_before": 1000,
    "tokens_after": 950,
    "tokens_saved": 50,
    "reduction_percent": 5.0,
    "created_at": "2025-01-01T08:00:00.000Z"
  }
]
```

### Direct Database Queries

Connect to the SQLite database for advanced analytics:

```bash
# Connect to database
sqlite3 backend/app.db

# Or with Docker
docker exec -it tokemizer-backend sqlite3 /app/data/app.db
```

**Example Queries:**

1. **Average duration by pass:**
```sql
SELECT 
    pass_name,
    COUNT(*) as executions,
    AVG(duration_ms) as avg_duration_ms,
    AVG(tokens_saved) as avg_tokens_saved,
    AVG(reduction_percent) as avg_reduction_percent
FROM performance_telemetry
GROUP BY pass_name
ORDER BY avg_duration_ms DESC;
```

2. **Most effective passes (token savings):**
```sql
SELECT 
    pass_name,
    COUNT(*) as executions,
    SUM(tokens_saved) as total_tokens_saved,
    AVG(tokens_saved) as avg_tokens_saved,
    AVG(duration_ms) as avg_duration_ms
FROM performance_telemetry
GROUP BY pass_name
ORDER BY total_tokens_saved DESC
LIMIT 10;
```

3. **Value ratio (tokens saved per millisecond):**
```sql
SELECT 
    pass_name,
    COUNT(*) as executions,
    AVG(tokens_saved * 1.0 / NULLIF(duration_ms, 0)) as value_ratio,
    AVG(tokens_saved) as avg_tokens_saved,
    AVG(duration_ms) as avg_duration_ms
FROM performance_telemetry
WHERE duration_ms > 0
GROUP BY pass_name
ORDER BY value_ratio DESC;
```

4. **Performance over time:**
```sql
SELECT 
    DATE(created_at) as date,
    pass_name,
    AVG(duration_ms) as avg_duration_ms,
    AVG(tokens_saved) as avg_tokens_saved
FROM performance_telemetry
WHERE created_at >= datetime('now', '-7 days')
GROUP BY date, pass_name
ORDER BY date DESC, avg_duration_ms DESC;
```

5. **Detailed pass sequence for specific optimization:**
```sql
SELECT 
    pass_order,
    pass_name,
    duration_ms,
    tokens_before,
    tokens_after,
    tokens_saved,
    reduction_percent
FROM performance_telemetry
 WHERE optimization_id = 'your-optimization-id'
ORDER BY pass_order;
```

---

## Baseline & Guardrails

Use the telemetry tables to establish baseline latency and token-savings expectations before making sweeping optimizer changes. The provided script queries both `performance_telemetry` and `optimization_history` within a configurable time window and evaluates guardrails using the latest configured thresholds.

### Running the baseline script

```bash
python backend/scripts/telemetry_baseline.py --window-days 30
```

The script prints:

- The window cutoff timestamp and number of optimization runs sampled from `performance_telemetry`
- Average and maximum cumulative latency per run
- Average tokens saved per run (from `optimization_history`)
- Semantic similarity statistics and sample count (when available)
- Guardrail evaluation for:
  - `max_latency_ms` (must stay below `SEMANTIC_GUARD_LATENCY_GUARD_MS`)
  - `semantic_similarity` (must stay above `SEMANTIC_GUARD_THRESHOLD`)
  - `tokens_saved` (must stay above `SEMANTIC_GUARD_TOKEN_SAVINGS_BASELINE`)

You can rerun the script periodically or incorporate it into CI checks to ensure no release regresses latency, savings, or semantic fidelity beyond the configured guardrails.

---

## Understanding Metrics

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **duration_ms** | Time taken by pass in milliseconds | Lower is better for performance |
| **tokens_saved** | Number of tokens reduced by this pass | Higher indicates more compression (can be zero if tokens increase) |
| **reduction_percent** | Percentage reduction for this pass | `(tokens_saved / tokens_before) * 100` |
| **value_ratio** | tokens_saved / duration_ms | Efficiency: compression per unit time |

### Pass Performance Characteristics

**normalize_whitespace** (<10ms typical)
- Fast, deterministic whitespace compression
- Applied early in the pipeline
- Low overhead, moderate savings on verbose text

**lexical_transforms** (10-50ms typical)
- Includes synonym replacement, list compression, and entity canonicalization
- Medium overhead with good token reduction
- Effectiveness depends on input characteristics

**deduplicate_content** (>50ms typical, or <1ms if short-circuited)
- Uses LSH/MinHash for similarity detection
- Skipped automatically for prompts shorter than threshold
- Most expensive pass when active, but provides significant savings on repetitive content

**normalize_text** (<10ms typical)
- Final cleanup pass that may run multiple times
- Applies punctuation compression and whitespace cleanup
- Low overhead, incremental savings

---

## Optimization Analysis Workflow

### 1. Identify Bottlenecks

Run aggregate query to find slowest passes:

```sql
SELECT 
    pass_name,
    COUNT(*) as count,
    AVG(duration_ms) as avg_ms,
    MAX(duration_ms) as max_ms,
    AVG(tokens_saved) as avg_saved
FROM performance_telemetry
GROUP BY pass_name
ORDER BY avg_ms DESC;
```

### 2. Evaluate Effectiveness

Determine which passes provide best compression:

```sql
SELECT 
    pass_name,
    AVG(tokens_saved) as avg_tokens_saved,
    AVG(reduction_percent) as avg_reduction,
    COUNT(*) as executions
FROM performance_telemetry
WHERE tokens_saved > 0
GROUP BY pass_name
ORDER BY avg_tokens_saved DESC;
```

### 3. Optimize Configuration

Based on findings, adjust optimization level:

- **High duration, low savings** → Use `conservative` optimization mode to disable expensive passes
- **Low duration, high savings** → Keep enabled in all modes (like `normalize_whitespace`)
- **High variance** → Performance may depend on input characteristics

---

## Performance Best Practices

### 1. **Enable Selectively**

Telemetry adds minimal overhead (<2% typical), but consider:
- Enable in **development** and **staging** for analysis
- Disable in **production** for maximum performance (unless monitoring specific issues)
- Use runtime toggling via Settings API for temporary investigation

### 2. **Batch Writing**

Telemetry uses fixed batch configuration (50 records, 5.0s flush interval) for consistent performance. The background writer thread automatically manages flushing.

### 3. **Data Retention**

Implement periodic cleanup to manage database size:

```sql
-- Delete telemetry older than 30 days
DELETE FROM performance_telemetry 
WHERE created_at < datetime('now', '-30 days');

-- Vacuum to reclaim space
VACUUM;
```

Consider archiving strategy:
```bash
# Export old data
sqlite3 /app/data/app.db <<EOF
.headers on
.mode csv
.output telemetry_archive_$(date +%Y%m).csv
SELECT * FROM performance_telemetry 
WHERE created_at < datetime('now', '-30 days');
EOF

# Then delete from database
```

---

## Troubleshooting

### No telemetry data appearing

1. **Check configuration:**
```bash
docker exec tokemizer-backend env | grep TELEMETRY
```

2. **Verify database:**
```bash
docker exec tokemizer-backend sqlite3 /app/data/app.db "SELECT COUNT(*) FROM performance_telemetry;"
```

3. **Check logs:**
```bash
docker logs tokemizer-backend | grep -i telemetry
```

4. **Verify telemetry was enabled before optimization:**
Telemetry must be enabled when the optimization request is processed.

### High memory usage

If memory pressure is high:
- Disable telemetry temporarily to reduce in-memory batching overhead
- Check batch writer status in logs
- Monitor database write performance

### Database locks

If experiencing lock contention:
- Telemetry writer uses WAL mode for better concurrency
- Disable telemetry temporarily to reduce write frequency

---

## Example Analysis Report

Generate a comprehensive performance report:

```sql
-- Performance Summary Report
SELECT 
    'Total Optimizations' as metric,
    COUNT(DISTINCT optimization_id) as value
FROM performance_telemetry
UNION ALL
SELECT 
    'Total Passes Recorded',
    COUNT(*) 
FROM performance_telemetry
UNION ALL
SELECT 
    'Avg Pass Duration (ms)',
    ROUND(AVG(duration_ms), 2)
FROM performance_telemetry
UNION ALL
SELECT 
    'Avg Tokens Saved Per Pass',
    ROUND(AVG(tokens_saved), 2)
FROM performance_telemetry;
```

---

**Last Updated**: 2026-01-22
**Version**: 1.1.0
