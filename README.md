<div align="center">

# ⚡ Tokemizer

### Enterprise-Grade AI Token Optimizer

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Reduce LLM prompt tokens by up to 70% without losing quality or context**

*Production-ready prompt compression middleware for cost optimization and performance enhancement*

[Features](#-key-features) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Deployment](#-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Use Cases](#-use-cases)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Performance Metrics](#-performance-metrics)
- [Security](#-security)
- [Monitoring & Observability](#-monitoring--observability)
- [Development](#-development)
- [Testing](#-testing)
- [License](#-license)

---

## 🎯 Overview

Tokemizer is an enterprise-grade AI token optimization platform built to reduce LLM prompt token consumption by **35-45% while maintaining ~~0.85 semantic similarity**, context, and intent. Built for production environments, it provides a **model-agnostic**, **rule-based** optimization pipeline with three configurable optimization modes for fine-tuned control.

### The Problem

Modern LLM applications face critical challenges:

- **💰 High Costs**: Token-based pricing makes verbose prompts expensive
- **⏱️ Slow Response**: Long prompts increase latency
- **🚫 Context Limits**: Prompts often exceed model token limits
- **📉 Poor Performance**: Large prompts degrade model effectiveness

### The Solution

Tokemizer acts as intelligent middleware that:

- **Compresses prompts** using advanced rule-based techniques
- **Preserves meaning** through semantic analysis and content protection
- **Reduces costs** by minimizing token consumption
- **Improves speed** with sub-second optimization
- **Scales effortlessly** handling prompts up to 500K+ tokens

---

## ✨ Key Features

### 🎛️ **Three Optimization Modes**

- **Conservative**: Fast processing with 30-35% token savings (~70% faster)
- **Balanced**: Optimal balance with 40-45% savings (~40% faster)
- **Maximum**: Aggressive compression with 50-70% savings

### 🚀 **Production-Ready Performance**

- Sub-second response time for most prompts
- Handles large prompts up to 500K+ tokens with intelligent chunking
- **Memory Guard**: Automatic protection against OOM for ultra-long prompts
- Concurrent processing with ThreadPoolExecutor

### 🔒 **Content Protection**

- Automatic preservation of code blocks, URLs, numbers, quotes
- JSON structure protection with optional TOON compression
- Configurable preservation patterns
- Citation and reference protection

### 🧠 **Smart Optimization**

- **Smart Router**: Auto-detects content type (code, chat, prose) and adjusts thresholds
- **Token Classifier**: Intelligent classification of tokens for targeted compression
- **Adaptive Abbreviation Learning**: Dynamic discovery of prompt-specific patterns
- **Offline Evolution**: GEPA-inspired framework for parameter tuning

### 🌐 **Model-Agnostic Design**

- Works with any LLM provider (OpenAI, Anthropic, Google, etc.)
- No model-specific optimization logic
- Standardized REST API interface
- Easy integration as middleware

### 📊 **Enterprise Features**

- Comprehensive optimization telemetry and metrics
- Historical optimization tracking with SQLite persistence
- Semantic similarity validation with configurable thresholds
- LRU caching with thread-local storage for sub-millisecond lookups

### **🧑‍💼 Admin Console**

The `/admin` UI exposes every operational control:

* **Users**: list, create, edit, assign roles/quotas, and disable/reactivate accounts.
* **Subscription Plans**: view pricing/quota details, toggle activation, and edit plan metadata.
  * Pricing sentinel: `monthly_price_cents`/`annual_price_cents` set to `-1` means **Contact Sales** (not free tier pricing).
* **Model Inventory**: manage each entry’s component, usage, revision pin, expected files, and allow-pattern overrides; protected models require an explicit override before deletion.
* **Model Cache Status**: inspect cached vs loaded readiness per model, cached reason, load reason, and last refresh timestamp while accessing refresh/delete/edit actions.
* **Refresh Workflows**: trigger global or per-model refreshes using `download_missing`, `force_redownload`, or `recovery` and monitor the running mode, targets, and missing models in real time.
* **Air-Gap Readiness**: validate offline readiness to list missing/invalid models and manifest drift before transporting the cache.
* **Canonical Mappings**: curate enterprise normalization pairs applied during optimization.
* **System Settings**: configure SMTP/Stripe credentials, telemetry/logging toggles, and the admin-only LLM system context used by LLM-based optimization.

Use the admin console to keep caches deterministic, repair downloads, and manage tenants without touching the database directly.



---

## 💼 Use Cases

### Cost Optimization

Reduce LLM API costs by 50-70% through intelligent token compression without sacrificing output quality.

### Context Window Management

Fit more context into limited token windows by compressing prompts while preserving critical information.

### Performance Enhancement

Improve response times by reducing prompt size, leading to faster LLM inference.

### RAG Systems

Optimize retrieved documents before passing to LLM, maximizing context relevance within token limits.

### Batch Processing

Process large volumes of prompts efficiently with minimal resource overhead.

### Multi-Turn Conversations

Compress conversation history to maintain longer context windows without exceeding limits.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Node.js 18+ (for frontend development)

### Local Development

```bash
# Clone the repository
git clone https://github.com/nowusman/tokemizer.git
cd tokemizer

# Start with Docker Compose
docker-compose up -d

# Access the application
# Frontend: http://localhost:80 (or your FRONTEND_PORT override)
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/api/v1/docs
```

### Basic API Usage

```bash
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "prompt": "Please provide me with a detailed summary of the following text...",
    "optimization_mode": "balanced"
  }'
```

---

## 🏗️ Architecture Overview

Tokemizer is built with a modular, production-ready architecture:

- **Backend**: FastAPI (Python 3.11+) with async request handling
- **Frontend**: React 19 with TypeScript and Vite
- **Database**: SQLite with Write-Ahead Logging for concurrent reads
- **NLP Pipeline**: Adaptive multi-pass optimization pipeline with semantic safeguards
- **Models**: spaCy, sentence-transformers, tiktoken, transformers (optional)

For detailed architecture, pipeline design, and system requirements, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Model Runtime Requirements

Current model catalog and per-use-case sizing are documented in [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md).

Recommended host sizing when running strict optimizer paths locally:

| Deployment profile | Recommended RAM | Recommended CPU |
| ------------------ | --------------: | --------------: |
| Minimum viable (single worker, low concurrency) | 24 GB | 8 vCPU |
| Recommended baseline | 32 GB | 12 vCPU |
| High-throughput | 48-64 GB | 16-24 vCPU |

Notes:
- `entropy` teacher and classifier backends are the dominant compute/memory cost.
- `entropy_fast` is the required strict entropy backend; teacher entropy is an optional quality guard in maximum mode.



---

## ⚙️ Configuration

Tokemizer is configured via environment variables. See [docs/ENV_VARIABLES.md](docs/ENV_VARIABLES.md) for the complete configuration reference.

For large prompts in `maximum` mode, start with a single policy setting:

```bash
PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY=auto
```

`PROMPT_OPTIMIZER_MAXIMUM_PREPASS_POLICY` supports:
- `off`: disable maximum pre-pass
- `auto`: enable only for large prompts and risk-aware content profiles
- `conservative`: higher min tokens + higher keep ratio
- `aggressive`: lower min tokens + lower keep ratio

Advanced `PROMPT_OPTIMIZER_MAXIMUM_PREPASS_*` variables remain available as expert overrides and are applied only when explicitly set.

The pre-pass performs budgeted sentence/span selection while hard-preserving protected spans, placeholders, and constraint-heavy instructions.

### Optimization Modes

| Mode             | Speed       | Token Savings | Passes | Use Case               |
| ---------------- | ----------- | ------------- | ------ | ---------------------- |
| **Conservative** | ~70% faster | 1-35%         | 10/15  | Real-time, interactive |
| **Balanced**     | ~40% faster | 1-50%         | 11/15  | General purpose        |
| **Maximum**      | Baseline    | 1-70%         | 15/15  | Batch, maximum savings |

---

## 🚢 Deployment

For detailed deployment instructions, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).



---

## 🔒 Security

Tokemizer implements enterprise-grade security:

### Core Security Features

- **Input Validation**: Pydantic models validate all requests
- **Content Protection**: Automatic preservation of sensitive patterns (code, URLs, etc.)
- **Authentication**: JWT tokens and API keys for protected endpoints
- **Data Privacy**: Optional history storage with configurable retention
- **CORS**: Configurable origin restrictions
- **Rate Limiting**: Protect against abuse via API gateway

### Best Practices

1. Always use HTTPS in production
2. Store JWT secrets and API keys in environment variables
3. Configure custom preservation patterns for sensitive data
4. Regularly review optimization history for data leakage
5. Deploy behind API gateway with rate limiting and authentication
6. Keep database backups synchronized with production
7. Monitor resource usage to prevent exhaustion attacks

For security considerations and compliance, see [docs/ARCHITECTURE.md#security--compliance](docs/ARCHITECTURE.md#security--compliance).

---

## 📈 Monitoring & Observability

### Health Checks

```bash
# Check service health
curl http://localhost:8000/api/v1/health

# Response includes:
# - Service status (healthy/unhealthy)
# - Available optimization techniques
# - Model readiness status
# - Dependency availability
```

### Metrics & Analytics

Access built-in metrics via API:

- **GET** `/api/v1/stats` — Aggregate optimization statistics
- **GET** `/api/v1/history` — Individual optimization records
- **GET** `/api/v1/telemetry/recent` — Per-pass performance breakdown

### Logging

Structured JSON logging via stderr. Configure verbosity:

```bash
LOG_LEVEL=debug  # debug|info|warning|error
```

---



See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture and development guidelines.

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v --cov=services/optimizer
```

### Frontend Checks

```bash
cd frontend
npm run build
```

> Frontend test scripts are not currently defined in `package.json`; use `npm run build` as the baseline verification step.

### Manual UI Check

1. Open the History page.
2. Do not click any row.
3. Confirm no runtime error occurs in the UI or console.

### End-to-End / Docker Validation

```bash
# Start services
docker-compose up -d

# Run backend test suite inside backend container
docker compose exec -T backend sh -lc "pytest -q"
```

For benchmarking scenarios, use `tests/benchmarks/` datasets and scripts.

---

## 📄 License

**Proprietary Software - All Rights Reserved**

This software is proprietary and confidential. It is intended for commercial use within authorized organizations only. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited without express written permission.

For licensing inquiries, please contact: nowusman@gmail.com

---

## 📞 Support

- 📧 Email: support@tokemizer.io
- 📚 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/nowusman/tokemizer/issues)

---

**Built with ❤️ for the LLM community**
