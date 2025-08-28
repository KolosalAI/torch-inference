# UV Configuration Guide

This project uses [uv](https://github.com/astral-sh/uv) as the default package manager for fast and reliable dependency management.

## Quick Start

### 1. Install uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative methods:**
```bash
# Using pip
pip install uv

# Using pipx
pipx install uv

# Using conda
conda install -c conda-forge uv
```

### 2. Setup Development Environment

**Linux/macOS:**
```bash
make setup
# or
bash setup.sh
```

**Windows:**
```powershell
make setup
# or
pwsh -ExecutionPolicy Bypass -File setup.ps1
```

### 3. Install Dependencies

```bash
# Install main dependencies
uv sync

# Install with development dependencies  
uv sync --group dev

# Install all optional dependencies
uv sync --all-extras
```

## UV Usage

### Basic Commands

```bash
# Install a new package
uv add fastapi

# Install a development dependency
uv add --group dev pytest

# Install with specific version constraints
uv add "torch>=2.0,<3.0"

# Remove a package
uv remove fastapi

# Update dependencies
uv lock --upgrade

# Run commands in the environment
uv run python main.py
uv run pytest
uv run jupyter lab
```

### Environment Management

```bash
# Create virtual environment (automatic with uv sync)
uv venv

# Activate environment (optional, uv run works without activation)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Show installed packages
uv pip list

# Export requirements
uv export --format requirements-txt --output-file requirements.txt
```

## Docker Integration

The project Dockerfile is optimized for uv:

```bash
# Build development image
make docker-build-dev

# Build production image  
make docker-build

# Run with docker-compose
make compose-dev    # Development environment
make compose-prod   # Production environment
```

## Project Structure

```
torch-inference/
├── pyproject.toml          # Project configuration with uv settings
├── uv.lock                 # Lock file for reproducible builds
├── uv-requirements.txt     # Exported requirements for compatibility
├── .env.template           # Environment variables template
├── setup.sh               # Linux/macOS setup script
├── setup.ps1               # Windows setup script
├── Dockerfile              # Multi-stage Docker build with uv
├── compose.yaml            # Base docker-compose configuration
├── compose.dev.yaml        # Development services
├── compose.prod.yaml       # Production services
└── Makefile               # Build automation with uv commands
```

## Configuration

### pyproject.toml UV Section

```toml
[tool.uv]
# Development dependencies
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    # ... more dev deps
]

# Configuration options
index-strategy = "unsafe-best-match"
compile-bytecode = true
no-cache = false

# Dependency groups
[tool.uv.groups]
test = ["pytest>=7.0.0", "pytest-cov>=4.1.0"]
lint = ["black>=23.0.0", "ruff>=0.1.0"]
docs = ["mkdocs>=1.5.0", "mkdocs-material>=9.0.0"]

# Custom package indexes
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

## Performance Benefits

UV provides significant performance improvements:

- **10-100x faster** than pip for most operations
- **Deterministic** builds with lock files
- **Parallel** dependency resolution
- **Better caching** strategies
- **Memory efficient** operations

### Benchmarks

Typical installation times:

| Package Manager | Cold Install | Warm Install |
|----------------|--------------|--------------|
| pip            | 45s          | 12s          |
| uv             | 3s           | 0.5s         |

## Troubleshooting

### Common Issues

1. **uv not found**
   ```bash
   # Ensure uv is in PATH
   echo $PATH
   which uv
   ```

2. **Lock file conflicts**
   ```bash
   # Regenerate lock file
   rm uv.lock
   uv lock
   ```

3. **Cache issues**
   ```bash
   # Clear uv cache
   uv cache clean
   ```

4. **Index conflicts**
   ```bash
   # Use specific index
   uv add --index pytorch torch
   ```

### Environment Variables

```bash
# UV configuration
export UV_CACHE_DIR=/tmp/uv-cache
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_COMPILE_BYTECODE=1
```

## Migration from pip

If migrating from pip:

1. **Generate uv configuration**:
   ```bash
   uv init --app
   ```

2. **Import requirements.txt**:
   ```bash
   uv add -r requirements.txt
   ```

3. **Update scripts**:
   - Replace `pip install` with `uv add`
   - Replace `python` with `uv run python`
   - Replace `pytest` with `uv run pytest`

## Best Practices

1. **Always use lock files** for reproducible builds
2. **Group dependencies** logically (dev, test, docs)
3. **Pin dependency versions** in CI/CD
4. **Use uv run** instead of activating environments
5. **Leverage caching** in Docker builds
6. **Regular updates** with `uv lock --upgrade`

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Set up uv
  uses: astral-sh/setup-uv@v1
  with:
    version: "latest"

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run pytest
```

### GitLab CI Example

```yaml
before_script:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - export PATH="$HOME/.cargo/bin:$PATH"
  - uv sync

test:
  script:
    - uv run pytest
```

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging with UV](https://docs.astral.sh/uv/guides/projects/)
- [UV Docker Guide](https://docs.astral.sh/uv/guides/integration/docker/)
