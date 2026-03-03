# syntax=docker/dockerfile:1.4
# Multi-stage build for optimized Python backend with ML models
# Requires Docker BuildKit: DOCKER_BUILDKIT=1 docker build .

# ============================================================================
# Stage 1: Builder - Install dependencies and download models
# ============================================================================
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements first for better layer caching
COPY requirements.txt .

# Increase uv network timeouts for large wheels (e.g., torch)
ENV UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=3

# Force CPU-only PyTorch wheels to avoid accidentally shipping CUDA runtimes.
# Installs torch from the official PyTorch CPU wheel index, then installs the rest of the deps.
ARG TORCH_CPU_VERSION=2.6.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_CPU_VERSION}+cpu"

# Install Python dependencies with uv
# uv pip install is significantly faster than standard pip
# --system installs into the system python environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements.txt

# Build-time sanity check: fail if any CUDA runtime pip packages slipped in.
RUN python -c "import importlib.metadata as md; \
names={dist.metadata['Name'].lower() for dist in md.distributions() if dist.metadata.get('Name')}; \
bad=sorted([n for n in names if n.startswith('nvidia-') or n.startswith('cuda-') or n.startswith('cupy')]); \
assert not bad, f'CUDA-related pip packages installed: {bad}'"

# Remove tests and pycache to save space
RUN find /usr/local/lib/python3.11/site-packages \( -type d -name "tests" -o -type d -name "test" -o -type d -name "__pycache__" \) -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete

# Copy backend source so the model inventory can be read during build
COPY . /app

# Pre-download tiktoken encodings for offline operation
RUN mkdir -p /vendor/tiktoken && \
    python -c "import os; \
    os.environ['TIKTOKEN_CACHE_DIR'] = '/vendor/tiktoken'; \
    import tiktoken; \
    enc = tiktoken.get_encoding('cl100k_base'); \
    print(f'Downloaded cl100k_base encoding to /vendor/tiktoken')" && \
    ls -lh /vendor/tiktoken/*

# ============================================================================
# Stage 2: Runtime - Minimal image with only necessary components
# ============================================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOME=/home/appuser \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence-transformers \
    TIKTOKEN_CACHE_DIR=/app/vendor/tiktoken \
    SPACY_HOME=/app/.cache/huggingface/spacy \
    PROMPT_OPTIMIZER_SPACY_MODEL_PATH=/app/.cache/huggingface/spacy/en_core_web_sm

# Install runtime dependencies
# libgomp1 is required for some ML libraries
# curl is used for healthcheck
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser && \
    chown -R appuser:appuser /app /home/appuser

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy tiktoken cache
# We use --chown to ensure the user can access it
COPY --from=builder --chown=appuser:appuser /vendor/tiktoken /app/vendor/tiktoken

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for SQLite and model cache with proper permissions
RUN mkdir -p /app/data /app/.cache/huggingface /home/appuser && \
    chown -R appuser:appuser /app/data /app/.cache /home/appuser


# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with increased timeout for ML model initialization
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application with explicit uvicorn tuning for long-running optimizations
CMD ["python", "-m", "run_uvicorn"]
