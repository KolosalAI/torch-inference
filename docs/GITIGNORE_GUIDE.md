# .gitignore Configuration Guide

## Overview
The `.gitignore` file has been updated to properly handle:
- ✅ Sensitive environment files
- ✅ Build artifacts and caches
- ✅ Generated benchmark results
- ✅ CUDA/TensorRT temporary files
- ✅ Model files and downloads

## What is NOW IGNORED (Not tracked in Git)

### 🔒 Sensitive Files (CRITICAL)
```
.env                    # Contains API keys, secrets
.env.local
.env.backup
.env.production
secrets.toml
credentials.json
*.pem
*.key
```

### 🗂️ Build Artifacts
```
/target/               # Rust build output
*.o, *.a, *.so        # Compiled binaries
*.dll, *.dylib        # Dynamic libraries
Cargo.lock            # Dependency lock file
__pycache__/          # Python bytecode
```

### 🧠 Model Files (Large)
```
models/*              # All downloaded models
*.pt, *.pth          # PyTorch weights
*.onnx               # ONNX models
*.torchscript        # TorchScript files
```

### 📊 Generated Results
```
benchmark_results/    # Runtime benchmark outputs
tensorrt_cache/      # TensorRT engine cache
.cuda/               # CUDA cache
.pytorch/            # PyTorch cache
*.log                # Log files
```

### 🎵 Media Output
```
*.wav, *.mp3         # Generated audio
*.flac, *.ogg       # Audio formats
test_images/         # Downloaded test images
```

### 🗄️ Temporary Files
```
tmp/                 # Temporary directory
*.tmp, *.bak        # Backup files
*.swp, *.swo        # Editor swap files
```

---

## What is STILL TRACKED (Git tracked)

### ✅ Configuration Templates
```
.env.cuda           # CUDA config template (no secrets)
config.toml         # Main configuration
config/cuda_tensorrt.toml  # TensorRT template
```

### ✅ Documentation
```
docs/**/*.md        # All markdown documentation
docs/research/*.tex # Research paper LaTeX
CUDA_QUICK_REF.md   # Quick reference guides
```

### ✅ Scripts
```
scripts/*.ps1       # PowerShell setup scripts
benchmark_cuda.py   # Benchmark scripts
benches/**/*.rs     # Rust benchmark code
```

### ✅ Source Code
```
src/**/*.rs         # Rust source code
build.rs            # Build script
Cargo.toml          # Project manifest
```

### ✅ Benchmark Analysis Data
```
benches/data/**/*.csv   # Benchmark results (for analysis)
benches/data/**/*.json  # Structured benchmark data
```

---

## Files Currently Untracked (Need Decision)

Run `git status` to see:
```
?? .env.cuda                          # Template (should track)
?? .gitignore.backup                  # Backup (can ignore)
?? CUDA_QUICK_REF.md                  # Docs (should track)
?? QUICK_START_BENCHMARKS.md          # Docs (should track)
?? benchmark_cuda.py                  # Script (should track)
?? config/cuda_tensorrt.toml          # Template (should track)
?? docs/BENCHMARK_UPDATE_SUMMARY.md   # Docs (should track)
?? docs/CUDA_TENSORRT_SETUP.md        # Docs (should track)
?? docs/GPU_BENCHMARK_RESULTS.md      # Docs (should track)
?? docs/WINDOWS_BENCHMARK_PLAN.md     # Docs (should track)
?? scripts/                           # Scripts (should track)
```

### Recommended: Add These Files
These are documentation and configuration templates (no secrets):
```bash
git add .env.cuda
git add CUDA_QUICK_REF.md
git add QUICK_START_BENCHMARKS.md
git add benchmark_cuda.py
git add config/cuda_tensorrt.toml
git add docs/BENCHMARK_UPDATE_SUMMARY.md
git add docs/CUDA_TENSORRT_SETUP.md
git add docs/GPU_BENCHMARK_RESULTS.md
git add docs/WINDOWS_BENCHMARK_PLAN.md
git add scripts/
git add .gitignore
git add config.toml
git add docs/research/torch_inference_benchmark_paper.tex
```

### Recommended: Ignore Forever
```bash
# Already handled by .gitignore
# .env (contains secrets - already ignored)
# .gitignore.backup (temporary backup - already ignored)
# benchmark_results/ (generated data - already ignored)
```

---

## Verification Commands

### Check what's ignored
```powershell
# See ignored files
git status --ignored

# Check if specific file is ignored
git check-ignore -v .env
git check-ignore -v benchmark_results/
```

### Check what will be committed
```powershell
# See staged changes
git diff --cached

# See all untracked files
git ls-files --others --exclude-standard
```

---

## Security Checklist

### ✅ Environment Files Protected
- [x] `.env` is ignored (contains secrets)
- [x] `.env.local` is ignored
- [x] `.env.production` is ignored
- [x] `.env.backup` is ignored

### ✅ No Secrets in Tracked Files
- [x] API keys not in code
- [x] Database credentials not in code
- [x] Private keys not tracked
- [x] `.env.cuda` is template only (no actual secrets)

### ✅ Large Files Excluded
- [x] Model files ignored (`models/`)
- [x] Build artifacts ignored (`target/`)
- [x] CUDA cache ignored (`tensorrt_cache/`)
- [x] PyTorch cache ignored (`.pytorch/`)

---

## Quick Reference

### Add all documentation to git
```bash
git add docs/ CUDA_QUICK_REF.md QUICK_START_BENCHMARKS.md
```

### Add all scripts to git
```bash
git add scripts/ benchmark_cuda.py
```

### Add configuration templates
```bash
git add config.toml config/cuda_tensorrt.toml .env.cuda
```

### Commit everything
```bash
git add .
git commit -m "Add CUDA configuration, documentation, and benchmarks"
```

### Push to remote
```bash
git push origin dev-rust
```

---

## Common Issues

### Issue: "I accidentally committed .env"
```bash
# Remove from git but keep local file
git rm --cached .env
git commit -m "Remove .env from tracking"

# Add to .gitignore (already done)
echo ".env" >> .gitignore
```

### Issue: "Large files in git history"
```bash
# Use git-filter-repo or BFG Repo-Cleaner
# (Advanced - backup first!)
```

### Issue: "Models folder keeps appearing"
```bash
# Ensure models/ is in .gitignore
git rm -r --cached models/
git commit -m "Remove models directory from tracking"
```

---

## Best Practices

### ✅ DO
1. **Always** use `.env.example` or `.env.cuda` as templates
2. **Always** add new config templates to git
3. **Always** document what files contain secrets
4. **Always** review `git status` before committing

### ❌ DON'T
1. **Never** commit `.env` with actual secrets
2. **Never** commit API keys in code
3. **Never** commit large model files (use Git LFS if needed)
4. **Never** commit build artifacts (`target/`)

---

## Environment File Template Usage

### For New Users
```bash
# Copy template to create local .env
cp .env.cuda .env

# Edit .env with your actual values
# .env will be ignored by git (safe)
```

### Your .env.cuda (Template - Safe to Commit)
Contains placeholder values and documentation, no actual secrets.

### Your .env (Local - Ignored by Git)
Contains your actual API keys and secrets, never committed.

---

## Summary

✅ **`.gitignore` is now properly configured**
✅ **Sensitive files (.env) are protected**
✅ **Build artifacts are excluded**
✅ **Documentation is tracked**
✅ **Templates are tracked**
✅ **Generated results are ignored**

**Next Step:** Review untracked files and add the ones you want to commit.

---

**Last Updated:** 2026-01-06  
**Version:** 2.0 (Comprehensive CUDA/TensorRT support)
