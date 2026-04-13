# Makefile for torch-inference
# Provides convenient commands for building and running

.PHONY: help build run dev test clean install doctor flamegraph

# Default target
.DEFAULT_GOAL := help

# Use cargo directly (assumes cargo is in PATH)
CARGO := cargo

help: ## Show this help message
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║         Torch Inference - Available Commands            ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build release binary (recommended)
	@echo "Building release binary..."
	$(CARGO) build --release --no-default-features
	@echo ""
	@echo "✅ Build complete: ./target/release/torch-inference-server"

run: ## Run server in release mode
	@echo "Starting server (release mode)..."
	$(CARGO) run --release --no-default-features

dev: ## Run server in dev mode (faster compile)
	@echo "Starting server (dev mode)..."
	$(CARGO) run --no-default-features

test: ## Run tests
	@echo "Running tests..."
	$(CARGO) test --no-default-features

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	@echo "✅ Clean complete"

install: ## Install binary to ~/.cargo/bin
	@echo "Installing binary..."
	$(CARGO) install --path . --no-default-features
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
	LIBTORCH="$$(pwd)/libtorch" $(CARGO) build --release --features torch

build-all: ## Build with all features
	@echo "Building with all features..."
	$(CARGO) build --release --features all-backends

# Testing
test-all: ## Run all tests including integration
	@echo "Running all tests..."
	$(CARGO) test --all-features

test-bench: ## Run benchmark unit tests
	@echo "Running benchmark tests..."
	$(CARGO) test --test benchmark_test
	@echo "✅ Benchmark tests complete"

test-bench-full: ## Full benchmark test suite
	@echo "Running full benchmark test suite..."
	@chmod +x test_benchmarks.sh
	@./test_benchmarks.sh

test-bench-quick: ## Quick benchmark validation
	@echo "Quick benchmark validation..."
	@$(CARGO) test --test benchmark_test
	@$(CARGO) bench --bench cache_bench -- --test
	@$(CARGO) bench --bench model_inference_bench -- --test
	@echo "✅ All benchmark tests passed"

# Benchmarks
bench: ## Run all benchmarks
	@echo "Running benchmarks..."
	$(CARGO) bench

bench-cache: ## Run cache benchmarks only
	@echo "Running cache benchmarks..."
	$(CARGO) bench --bench cache_bench

bench-models: ## Run model inference benchmarks
	@echo "Running model inference benchmarks..."
	$(CARGO) bench --bench model_inference_bench

bench-torch: ## Run benchmarks with torch feature
	@echo "Running benchmarks with torch support..."
	$(CARGO) bench --bench model_inference_bench --features torch

bench-report: ## View benchmark HTML report
	@if [ -f "target/criterion/report/index.html" ]; then \
		open target/criterion/report/index.html || xdg-open target/criterion/report/index.html; \
	else \
		echo "❌ No benchmark results found. Run 'make bench' first."; \
	fi

# Maintenance
fmt: ## Format code
	@echo "Formatting code..."
	$(CARGO) fmt
	@echo "✅ Code formatted"

clippy: ## Run clippy linter
	@echo "Running clippy..."
	$(CARGO) clippy --no-default-features -- -D warnings

check: ## Check code without building
	@echo "Checking code..."
	$(CARGO) check --no-default-features

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

flamegraph: ## Generate CPU flamegraph (requires: cargo install flamegraph)
	@echo "Generating flamegraph (requires cargo-flamegraph and sudo/root)..."
	cargo flamegraph --features profiling --bin torch-inference-server -- --config config.toml
	@echo "Flamegraph written to flamegraph.svg"

# ── LLM Microservice ──────────────────────────────────────────────────────────
.PHONY: llm-build llm-run llm-download

llm-download: ## Download MiniCPM-V 2.6 Q2_K model and mmproj
	bash scripts/download_llm_model.sh

llm-build: ## Build LLM service
	cd services/llm && cargo build --release

llm-run: ## Run LLM service
	cd services/llm && ./target/release/llm-service
