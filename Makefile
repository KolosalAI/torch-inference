# Makefile for torch-inference framework testing
# Provides convenient commands for development and testing

.PHONY: help test test-unit test-integration test-all coverage lint format type-check clean install dev docs security benchmark

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install dependencies
	uv sync

install-dev: ## Install development dependencies
	uv sync --extra dev

install-all: ## Install all dependencies including optional ones
	uv sync --extra all

# Testing commands
test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit/

test-integration: ## Run integration tests only  
	uv run pytest tests/integration/

test-smoke: ## Run smoke tests for quick validation
	uv run pytest -m smoke

test-fast: ## Run fast tests only (no slow/gpu tests)
	uv run pytest -m "not slow and not gpu"

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

# CI/CD helpers
ci-test: ## Run tests for CI (with XML reports)
	uv run pytest --cov=framework --cov-report=xml --junitxml=junit.xml

ci-lint: ## Run linting for CI
	uv run black --check .
	uv run ruff check . --output-format=github
	uv run isort --check-only .

ci-security: ## Run security checks for CI
	uv run bandit -r framework -f json -o bandit-report.json
	uv run safety check --json --output safety-report.json

# Docker helpers
docker-build: ## Build Docker image for testing
	docker build -t torch-inference-test .

docker-test: ## Run tests in Docker
	docker run --rm torch-inference-test make test

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
