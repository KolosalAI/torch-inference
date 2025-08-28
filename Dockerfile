# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for torch-inference framework
# Uses uv for fast and reliable dependency management

ARG PYTHON_VERSION=3.10.11

# Base stage with common setup
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Configure uv cache directory
ENV UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# Install uv for faster package management
# Install curl and other dependencies for uv installation
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Development stage
FROM base as development

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install all dependencies including dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy source code
COPY . .

# Create cache directory for the appuser and change ownership
RUN mkdir -p /tmp/uv-cache && chown appuser:appuser /tmp/uv-cache

# Switch to non-privileged user
USER appuser

# Set environment variables for the app user
ENV UV_CACHE_DIR=/tmp/uv-cache

# Expose development port
EXPOSE 8000

# Development command with hot reload
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install only production dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy the source code into the container.
COPY . .

# Create cache directory for the appuser and change ownership
RUN mkdir -p /tmp/uv-cache && chown appuser:appuser /tmp/uv-cache

# Switch to the non-privileged user to run the application.
USER appuser

# Set environment variables for the app user
ENV UV_CACHE_DIR=/tmp/uv-cache

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application using uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Default to production stage
FROM production
