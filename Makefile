# Makefile for torch-inference framework testing
# Provides convenient commands for development and testing

.PHONY: help test test-unit test-integration test-all coverage lint format type-check clean install dev docs security benchmark

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "🧪 Torch Inference Framework Commands"
	@echo "===================================="
	@echo ""
	@echo "QUICK TESTS (Development):"
	@echo "  make test-smoke      - Ultra-fast smoke tests (30s-2min)"
	@echo "  make test-fast       - Fast unit tests (2-5min)"
	@echo "  make test-fast-cov   - Fast tests with coverage"
	@echo ""
	@echo "COMPREHENSIVE TESTS:"
	@echo "  make test-integration - Integration tests (8-15min)"
	@echo "  make test-gpu        - GPU tests only (10-20min)"
	@echo "  make test-full       - Full test suite (15-25min)"
	@echo "  make test-coverage   - Full tests with coverage (20-30min)"
	@echo ""
	@echo "PARALLEL EXECUTION:"
	@echo "  make test-fast-parallel     - Fast tests with parallel execution"
	@echo "  make test-integration-parallel - Integration tests with parallel execution"
	@echo ""
	@echo "OTHER COMMANDS:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v help | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Performance: Optimized from 28+ minutes to 15-25 minutes!"

# Installation and setup
install: ## Install dependencies with uv
	uv sync

install-dev: ## Install development dependencies
	uv sync --group dev

install-all: ## Install all dependencies including optional ones
	uv sync --all-extras

setup: ## Setup development environment (run this first)
	@echo "Setting up torch-inference development environment..."
	@if command -v bash >/dev/null 2>&1; then \
		bash setup.sh; \
	elif command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File setup.ps1; \
	else \
		echo "Neither bash nor PowerShell found. Please run setup manually:"; \
		echo "1. Run 'uv sync' to install dependencies"; \
		echo "2. Run 'uv run pre-commit install' to setup hooks"; \
		echo "3. Copy .env.template to .env and customize"; \
	fi

check-uv: ## Check if uv is installed and working
	@uv --version || (echo "uv not found. Install it from https://github.com/astral-sh/uv" && exit 1)
	@echo "uv is properly installed"

lock-update: ## Update and regenerate lock file
	uv lock --upgrade

# ============================================================================
# OPTIMIZED TESTING COMMANDS (Performance improved: 28+ min → 15-25 min)
# ============================================================================

# Quick tests for development
test-smoke: ## Ultra-fast smoke tests (30s-2min)
	@echo "💨 Running smoke tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 smoke

test-fast: ## Fast unit tests (2-5min) - DEFAULT for development
	@echo "🚀 Running fast unit tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 fast

test-fast-cov: ## Fast tests with coverage reporting
	@echo "🚀 Running fast tests with coverage..."
	@powershell -ExecutionPolicy Bypass -File scripts/testing/run_fast_tests.ps1 -Coverage

test-fast-parallel: ## Fast tests with parallel execution
	@echo "⚡ Running fast tests with parallel execution..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 fast -Parallel

# Comprehensive tests
test-integration: ## Integration tests (8-15min)
	@echo "🔧 Running integration tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 integration

test-integration-parallel: ## Integration tests with parallel execution
	@echo "⚡ Running integration tests with parallel execution..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 integration -Parallel

test-gpu: ## GPU tests only (10-20min)
	@echo "🎮 Running GPU tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 gpu

test-full: ## Full test suite optimized (15-25min)
	@echo "🎯 Running full test suite..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 full

test-coverage: ## Full tests with coverage (20-30min)
	@echo "📊 Running full tests with coverage..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 coverage

# Legacy commands (maintained for compatibility)
test: test-fast ## Default test command (fast unit tests)

test-unit: test-fast ## Alias for fast unit tests

test-all: test-full ## Alias for full test suite

# Clean test artifacts
clean-test: ## Clean test artifacts and cache
	@echo "🧹 Cleaning test artifacts..."
	@if exist .pytest_cache rmdir /s /q .pytest_cache 2>nul || true
	@if exist htmlcov rmdir /s /q htmlcov 2>nul || true
	@if exist .coverage del .coverage 2>nul || true
	@if exist coverage.xml del coverage.xml 2>nul || true
	@if exist test-results.xml del test-results.xml 2>nul || true
	@if exist test.log del test.log 2>nul || true
	@echo "✅ Test artifacts cleaned"

# Development workflow targets
dev-test: test-fast ## Quick development test run
	@echo "✅ Development tests complete"

pre-commit: test-smoke ## Pre-commit validation
	@echo "✅ Pre-commit tests complete"

pre-push: test-fast ## Pre-push validation  
	@echo "✅ Pre-push tests complete"

# ============================================================================
# LEGACY TEST COMMANDS (Slower, maintained for compatibility)
# ============================================================================

test-slow: ## Run slow tests only
	uv run pytest -m slow

test-gpu: ## Run GPU tests (requires CUDA)
	uv run pytest -m gpu

test-tensorrt: ## Run TensorRT tests
	uv run pytest -m tensorrt

test-onnx: ## Run ONNX tests
	uv run pytest -m onnx

test-enterprise: ## Run enterprise feature tests
	uv run pytest -m enterprise

test-parallel: ## Run tests in parallel
	uv run pytest -n auto

test-verbose: ## Run tests with verbose output
	uv run pytest -v

test-debug: ## Run tests with debugging info
	uv run pytest -vvv --tb=long --showlocals

test-failed: ## Re-run only failed tests
	uv run pytest --lf

test-new: ## Run only new/modified tests
	uv run pytest --ff

# Coverage and reporting
coverage: ## Run tests with coverage reporting
	uv run pytest --cov=framework --cov-report=html --cov-report=term-missing

coverage-xml: ## Generate XML coverage report
	uv run pytest --cov=framework --cov-report=xml

coverage-html: ## Generate HTML coverage report
	uv run pytest --cov=framework --cov-report=html
	@echo "Coverage report available at htmlcov/index.html"

benchmark: ## Run performance benchmarks
	uv run pytest -m benchmark --benchmark-only --benchmark-sort=mean

# Code quality
lint: ## Run all linting checks
	uv run black --check .
	uv run ruff check .
	uv run isort --check-only .

lint-fix: ## Fix linting issues
	uv run black .
	uv run ruff check --fix .
	uv run isort .

format: ## Format code
	uv run black .
	uv run isort .

type-check: ## Run type checking
	uv run mypy framework

security: ## Run security scans
	uv run bandit -r framework
	uv run safety check

# Documentation
docs: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

docs-deploy: ## Deploy documentation
	uv run mkdocs gh-deploy

# Environment management
clean: ## Clean up generated files
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf junit.xml
	rm -rf .tox/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-models: ## Clean test models
	rm -rf tests/models/models/
	rm -f tests/models/model_registry.json

setup-models: ## Download test models
	uv run python tests/models/create_test_models.py

# Development helpers
dev: ## Setup development environment
	$(MAKE) install-dev
	$(MAKE) setup-models
	uv run pre-commit install

dev-test: ## Quick development test run
	uv run pytest tests/unit/ -x --tb=short

watch: ## Watch for changes and run tests
	uv run pytest-watch

# CI/CD helpers (optimized)
ci-test: ## Optimized CI tests (parallel execution, 15-20min)
	@echo "🚀 Running optimized CI tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 full -Parallel -Verbose

ci-test-coverage: ## CI tests with coverage reporting
	@echo "📊 Running CI tests with coverage..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 coverage -Verbose

ci-test-fast: ## Fast CI tests for quick feedback
	@echo "⚡ Running fast CI tests..."
	@powershell -ExecutionPolicy Bypass -File test.ps1 fast -Parallel

ci-lint: ## Run linting for CI
	uv run black --check .
	uv run ruff check . --output-format=github
	uv run isort --check-only .

ci-security: ## Run security checks for CI
	uv run bandit -r framework -f json -o bandit-report.json
	uv run safety check --json --output safety-report.json

# Docker helpers
docker-build: ## Build Docker image for production
	docker build --target production -t torch-inference:latest .

docker-build-dev: ## Build Docker image for development
	docker build --target development -t torch-inference:dev .

docker-test: ## Run tests in Docker
	docker build --target development -t torch-inference-test .
	docker run --rm torch-inference-test uv run pytest

docker-run: ## Run production container
	docker run --rm -p 8000:8000 torch-inference:latest

docker-run-dev: ## Run development container with hot reload
	docker run --rm -p 8001:8000 -v $(PWD):/app torch-inference:dev

# Docker Compose helpers
compose-up: ## Start services with docker-compose (production)
	docker compose up --build

compose-dev: ## Start development environment with all dev tools
	docker compose -f compose.yaml -f compose.dev.yaml up --build

compose-prod: ## Start production environment
	docker compose -f compose.prod.yaml up --build -d

compose-down: ## Stop all docker-compose services
	docker compose down
	docker compose -f compose.dev.yaml down
	docker compose -f compose.prod.yaml down

compose-logs: ## View docker-compose logs
	docker compose logs -f

compose-logs-dev: ## View development environment logs
	docker compose -f compose.dev.yaml logs -f

# Legacy aliases (deprecated)
docker-compose-up: compose-up ## [DEPRECATED] Use compose-up instead
docker-compose-dev: compose-dev ## [DEPRECATED] Use compose-dev instead  
docker-compose-down: compose-down ## [DEPRECATED] Use compose-down instead

docker-clean: ## Clean up Docker images and containers
	docker system prune -f
	docker image prune -f

# Tox integration
tox: ## Run tox for multi-environment testing
	tox

tox-recreate: ## Recreate tox environments
	tox --recreate

# Advanced testing scenarios
stress-test: ## Run stress tests
	uv run pytest tests/ -x --count=10

memory-test: ## Run memory profiling
	uv run pytest --memray tests/

profile: ## Profile test execution
	uv run pytest --profile tests/

# Release helpers
check-release: ## Check if ready for release
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test
	$(MAKE) coverage

# Example usage targets
example: ## Run basic usage example
	uv run python examples/basic_usage.py

example-config: ## Run configuration example  
	uv run python examples/config_example.py

example-enterprise: ## Run enterprise example
	uv run python examples/enterprise_example.py

# Maintenance
update-deps: ## Update dependencies
	uv lock --upgrade

check-deps: ## Check for dependency updates
	uv tree

outdated: ## Check for outdated packages
	@echo "Use 'uv lock --upgrade' to update dependencies"

# Help for specific test files
test-config: ## Test configuration module
	uv run pytest tests/unit/test_config.py -v

test-inference: ## Test inference engine
	uv run pytest tests/unit/test_inference_engine.py -v

test-optimizers: ## Test optimizers
	uv run pytest tests/unit/test_optimizers.py -v

test-adapters: ## Test model adapters
	uv run pytest tests/unit/test_adapters.py -v

test-framework: ## Test main framework
	uv run pytest tests/unit/test_framework.py -v

# Windows-specific commands (use 'make' equivalent on Windows with compatible tools)
ifeq ($(OS),Windows_NT)
    RM = del /Q /F
    RMDIR = rmdir /Q /S
else
    RM = rm -f
    RMDIR = rm -rf
endif

# Optional: Print environment info
info: ## Show environment information
	@echo "Python version: $(shell python --version)"
	@echo "UV version: $(shell uv --version)"
	@echo "Pytest version: $(shell uv run pytest --version)"
	@echo "PyTorch version: $(shell uv run python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $(shell uv run python -c 'import torch; print(torch.cuda.is_available())')"
