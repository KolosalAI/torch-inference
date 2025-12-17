# Makefile for torch-inference
# Provides convenient commands for building and running

.PHONY: help build run dev test clean install doctor

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║         Torch Inference - Available Commands            ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build release binary (recommended)
	@echo "Building release binary..."
	cargo build --release --no-default-features
	@echo ""
	@echo "✅ Build complete: ./target/release/torch-inference-server"

run: ## Run server in release mode
	@echo "Starting server (release mode)..."
	cargo run --release --no-default-features

dev: ## Run server in dev mode (faster compile)
	@echo "Starting server (dev mode)..."
	cargo run --no-default-features

test: ## Run tests
	@echo "Running tests..."
	cargo test --no-default-features

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cargo clean
	@echo "✅ Clean complete"

install: ## Install binary to ~/.cargo/bin
	@echo "Installing binary..."
	cargo install --path . --no-default-features
	@echo "✅ Installed to: ~/.cargo/bin/torch-inference-server"

doctor: ## Check system requirements
	@echo "Checking system requirements..."
	@echo ""
	@echo "Rust:"
	@rustc --version || echo "  ❌ Rust not installed"
	@echo ""
	@echo "Cargo:"
	@cargo --version || echo "  ❌ Cargo not installed"
	@echo ""
	@echo "System:"
	@uname -s -m
	@echo ""
	@echo "LibTorch:"
	@if [ -d "./libtorch" ]; then \
		echo "  ✅ Found at: ./libtorch"; \
		du -sh ./libtorch 2>/dev/null || echo "  Size: Unknown"; \
	else \
		echo "  ⚠️  Not found (optional - not needed for default build)"; \
	fi
	@echo ""

# Build variants
build-torch: ## Build with PyTorch support (requires LibTorch)
	@echo "Building with PyTorch..."
	@if [ ! -d "./libtorch" ]; then \
		echo "❌ LibTorch not found. Run: ./download_libtorch.sh"; \
		exit 1; \
	fi
	LIBTORCH="$$(pwd)/libtorch" cargo build --release --features torch

build-all: ## Build with all features
	@echo "Building with all features..."
	cargo build --release --features all-backends

# Testing variants
test-all: ## Run all tests
	@echo "Running all tests..."
	cargo test --all-features

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	cargo bench --no-default-features

# Maintenance
fmt: ## Format code
	@echo "Formatting code..."
	cargo fmt
	@echo "✅ Code formatted"

clippy: ## Run clippy linter
	@echo "Running clippy..."
	cargo clippy --no-default-features -- -D warnings

check: ## Check code without building
	@echo "Checking code..."
	cargo check --no-default-features

# Server management
start: build ## Build and start server
	@echo "Starting server..."
	./target/release/torch-inference-server

stop: ## Stop running server
	@echo "Stopping server..."
	@pkill -f torch-inference-server || echo "No server running"

restart: stop start ## Restart server

# Health check
health: ## Check server health
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "Server not running"

# Quick test
quick-test: ## Quick TTS test
	@curl -s -X POST http://localhost:8000/tts/synthesize \
		-H "Content-Type: application/json" \
		-d '{"text": "Hello from make", "voice": "af_bella"}' | \
		python3 -c "import sys, json; d=json.load(sys.stdin); print(f'✅ {d[\"duration_secs\"]}s audio, {d[\"engine_used\"]} engine')" || \
		echo "❌ Server not responding"

# Release
release: clean build test ## Clean build and test for release
	@echo "✅ Release ready: ./target/release/torch-inference-server"

# Watch mode (requires cargo-watch)
watch: ## Watch and rebuild on changes
	@command -v cargo-watch >/dev/null 2>&1 || { echo "Install: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'run --no-default-features'
