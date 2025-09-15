# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for PyTorch Inference Framework
# Optimized for size, performance, and security

# Build stage for dependencies
FROM python:3.10.11-slim as builder

# Build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=cu121

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for better dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv

# Copy dependency files first for better layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies using uv for faster installation
RUN uv pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (if needed)
RUN if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
    uv pip install --no-cache-dir \
    torch==${PYTORCH_VERSION}+${CUDA_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}; \
    else \
    uv pip install --no-cache-dir torch torchvision torchaudio; \
    fi

# Production stage
FROM python:3.10.11-slim as production

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential runtime libraries
    libc6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio processing libraries for TTS
    libsndfile1 \
    ffmpeg \
    # Network utilities
    curl \
    # Clean up in same layer to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user for security
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p logs models/cache calibration_cache kernel_cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use exec form for better signal handling
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development stage (optional)
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    black \
    ruff

# Switch back to appuser
USER appuser

# Override CMD for development
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
