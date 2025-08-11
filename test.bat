@echo off
REM Batch file for torch-inference framework testing (Windows)
REM Provides convenient commands for development and testing

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-dev" goto install-dev
if "%1"=="install-all" goto install-all
if "%1"=="test" goto test
if "%1"=="test-unit" goto test-unit
if "%1"=="test-integration" goto test-integration
if "%1"=="test-smoke" goto test-smoke
if "%1"=="test-fast" goto test-fast
if "%1"=="test-slow" goto test-slow
if "%1"=="test-gpu" goto test-gpu
if "%1"=="test-tensorrt" goto test-tensorrt
if "%1"=="test-onnx" goto test-onnx
if "%1"=="test-enterprise" goto test-enterprise
if "%1"=="test-parallel" goto test-parallel
if "%1"=="test-verbose" goto test-verbose
if "%1"=="test-debug" goto test-debug
if "%1"=="test-failed" goto test-failed
if "%1"=="test-new" goto test-new
if "%1"=="coverage" goto coverage
if "%1"=="coverage-xml" goto coverage-xml
if "%1"=="coverage-html" goto coverage-html
if "%1"=="benchmark" goto benchmark
if "%1"=="lint" goto lint
if "%1"=="lint-fix" goto lint-fix
if "%1"=="format" goto format
if "%1"=="type-check" goto type-check
if "%1"=="security" goto security
if "%1"=="docs" goto docs
if "%1"=="docs-serve" goto docs-serve
if "%1"=="clean" goto clean
if "%1"=="clean-models" goto clean-models
if "%1"=="setup-models" goto setup-models
if "%1"=="dev" goto dev
if "%1"=="dev-test" goto dev-test
if "%1"=="ci-test" goto ci-test
if "%1"=="ci-lint" goto ci-lint
if "%1"=="ci-security" goto ci-security
if "%1"=="tox" goto tox
if "%1"=="stress-test" goto stress-test
if "%1"=="check-release" goto check-release
if "%1"=="example" goto example
if "%1"=="example-config" goto example-config
if "%1"=="example-enterprise" goto example-enterprise
if "%1"=="update-deps" goto update-deps
if "%1"=="info" goto info
echo Unknown command: %1
goto help

:help
echo Available commands:
echo.
echo Installation and setup:
echo   install           - Install dependencies
echo   install-dev       - Install development dependencies  
echo   install-all       - Install all dependencies including optional ones
echo.
echo Testing commands:
echo   test             - Run all tests
echo   test-unit        - Run unit tests only
echo   test-integration - Run integration tests only
echo   test-smoke       - Run smoke tests for quick validation
echo   test-fast        - Run fast tests only (no slow/gpu tests)
echo   test-slow        - Run slow tests only
echo   test-gpu         - Run GPU tests (requires CUDA)
echo   test-tensorrt    - Run TensorRT tests
echo   test-onnx        - Run ONNX tests
echo   test-enterprise  - Run enterprise feature tests
echo   test-parallel    - Run tests in parallel
echo   test-verbose     - Run tests with verbose output
echo   test-debug       - Run tests with debugging info
echo   test-failed      - Re-run only failed tests
echo   test-new         - Run only new/modified tests
echo.
echo Coverage and reporting:
echo   coverage         - Run tests with coverage reporting
echo   coverage-xml     - Generate XML coverage report
echo   coverage-html    - Generate HTML coverage report
echo   benchmark        - Run performance benchmarks
echo.
echo Code quality:
echo   lint            - Run all linting checks
echo   lint-fix        - Fix linting issues
echo   format          - Format code
echo   type-check      - Run type checking
echo   security        - Run security scans
echo.
echo Documentation:
echo   docs            - Build documentation
echo   docs-serve      - Serve documentation locally
echo.
echo Environment management:
echo   clean           - Clean up generated files
echo   clean-models    - Clean test models
echo   setup-models    - Download test models
echo.
echo Development helpers:
echo   dev             - Setup development environment
echo   dev-test        - Quick development test run
echo.
echo CI/CD helpers:
echo   ci-test         - Run tests for CI (with XML reports)
echo   ci-lint         - Run linting for CI
echo   ci-security     - Run security checks for CI
echo.
echo Release helpers:
echo   check-release   - Check if ready for release
echo.
echo Examples:
echo   example         - Run basic usage example
echo   example-config  - Run configuration example
echo   example-enterprise - Run enterprise example
echo.
echo Maintenance:
echo   update-deps     - Update dependencies
echo   info            - Show environment information
echo.
echo Tox integration:
echo   tox             - Run tox for multi-environment testing
echo.
goto end

:install
uv sync
goto end

:install-dev
uv sync --extra dev
goto end

:install-all
uv sync --extra all
goto end

:test
uv run pytest
goto end

:test-unit
uv run pytest tests/unit/
goto end

:test-integration
uv run pytest tests/integration/
goto end

:test-smoke
uv run pytest -m smoke
goto end

:test-fast
uv run pytest -m "not slow and not gpu"
goto end

:test-slow
uv run pytest -m slow
goto end

:test-gpu
uv run pytest -m gpu
goto end

:test-tensorrt
uv run pytest -m tensorrt
goto end

:test-onnx
uv run pytest -m onnx
goto end

:test-enterprise
uv run pytest -m enterprise
goto end

:test-parallel
uv run pytest -n auto
goto end

:test-verbose
uv run pytest -v
goto end

:test-debug
uv run pytest -vvv --tb=long --showlocals
goto end

:test-failed
uv run pytest --lf
goto end

:test-new
uv run pytest --ff
goto end

:coverage
uv run pytest --cov=framework --cov-report=html --cov-report=term-missing
goto end

:coverage-xml
uv run pytest --cov=framework --cov-report=xml
goto end

:coverage-html
uv run pytest --cov=framework --cov-report=html
echo Coverage report available at htmlcov/index.html
goto end

:benchmark
uv run pytest -m benchmark --benchmark-only --benchmark-sort=mean
goto end

:lint
uv run black --check .
uv run ruff check .
uv run isort --check-only .
goto end

:lint-fix
uv run black .
uv run ruff check --fix .
uv run isort .
goto end

:format
uv run black .
uv run isort .
goto end

:type-check
uv run mypy framework
goto end

:security
uv run bandit -r framework
uv run safety check
goto end

:docs
uv run mkdocs build
goto end

:docs-serve
uv run mkdocs serve
goto end

:clean
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del /q .coverage
if exist coverage.xml del /q coverage.xml
if exist junit.xml del /q junit.xml
if exist .tox rmdir /s /q .tox
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*egg-info) do if exist "%%i" rmdir /s /q "%%i"
for /d /r %%i in (__pycache__) do if exist "%%i" rmdir /s /q "%%i"
for /r %%i in (*.pyc) do if exist "%%i" del /q "%%i"
goto end

:clean-models
if exist tests\models\models rmdir /s /q tests\models\models
if exist tests\models\model_registry.json del /q tests\models\model_registry.json
goto end

:setup-models
uv run python tests/models/create_test_models.py
goto end

:dev
call %0 install-dev
call %0 setup-models
uv run pre-commit install
goto end

:dev-test
uv run pytest tests/unit/ -x --tb=short
goto end

:ci-test
uv run pytest --cov=framework --cov-report=xml --junitxml=junit.xml
goto end

:ci-lint
uv run black --check .
uv run ruff check . --output-format=github
uv run isort --check-only .
goto end

:ci-security
uv run bandit -r framework -f json -o bandit-report.json
uv run safety check --json --output safety-report.json
goto end

:tox
tox
goto end

:stress-test
uv run pytest tests/ -x --count=10
goto end

:check-release
call %0 lint
if errorlevel 1 goto end
call %0 type-check
if errorlevel 1 goto end
call %0 security
if errorlevel 1 goto end
call %0 test
if errorlevel 1 goto end
call %0 coverage
goto end

:example
uv run python examples/basic_usage.py
goto end

:example-config
uv run python examples/config_example.py
goto end

:example-enterprise
uv run python examples/enterprise_example.py
goto end

:update-deps
uv lock --upgrade
goto end

:info
echo Environment Information:
echo.
python --version 2>nul
if errorlevel 1 echo Python: Not found
uv --version 2>nul
if errorlevel 1 echo UV: Not found
uv run pytest --version 2>nul
if errorlevel 1 echo Pytest: Not found
uv run python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if errorlevel 1 echo PyTorch: Not found
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 echo CUDA check: Failed
goto end

:end
