# syntax=docker/dockerfile:1
#
# Multi-stage Dockerfile for the Rust torch-inference server.
# Builds the release binary in one stage, then drops it into a slim runtime
# image with the ONNX Runtime shared library and ffmpeg/libsndfile for audio.

# ── builder ──────────────────────────────────────────────────────────────────
FROM rust:1.81-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Pre-cache deps. Copy manifests first so Cargo's dep build is cached
# independently of source changes.
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "" > src/lib.rs && \
    cargo build --release --no-default-features --features production --bin torch-inference-server || true && \
    rm -rf src

COPY src ./src
COPY benches ./benches

RUN cargo build --release --no-default-features --features production --bin torch-inference-server

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install the ONNX Runtime dynamic library. ort = "2.0.0-rc.10" with
# `load-dynamic` requires libonnxruntime.so at runtime.
ARG ORT_VERSION=1.18.0
RUN ARCH="$(uname -m)" && \
    case "$ARCH" in \
        x86_64)  ORT_PKG="onnxruntime-linux-x64-${ORT_VERSION}.tgz" ;; \
        aarch64) ORT_PKG="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" ;; \
        *) echo "unsupported arch: $ARCH" >&2; exit 1 ;; \
    esac && \
    cd /tmp && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}" && \
    tar -xzf "$ORT_PKG" && \
    cp onnxruntime-*/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/onnxruntime-* /tmp/*.tgz

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

ARG UID=10001
RUN adduser --disabled-password --gecos "" --home /app --shell /sbin/nologin --uid ${UID} appuser

WORKDIR /app

COPY --from=builder /build/target/release/torch-inference-server /app/torch-inference-server
COPY --chown=appuser:appuser config.toml /app/config.toml
RUN mkdir -p /app/models /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live >/dev/null || exit 1

CMD ["/app/torch-inference-server"]
