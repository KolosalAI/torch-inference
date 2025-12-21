# Fix Large File Git Push Issue

## Problem
Your git push fails with HTTP 408 error because there are large model files (700MB+) in the Git history.

## Solution Options

### Option 1: Use BFG Repo-Cleaner (Recommended - Fastest)

1. Install BFG:
```bash
brew install bfg
```

2. Clone a fresh copy (for safety):
```bash
cd ..
git clone --mirror torch-inference torch-inference-clean.git
cd torch-inference-clean.git
```

3. Remove large files:
```bash
bfg --strip-blobs-bigger-than 100M .
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

4. Push cleaned repo:
```bash
cd ../torch-inference
git remote set-url origin <your-repo-url>
git push origin --force --all
```

### Option 2: Use the Cleanup Script (Alternative)

```bash
chmod +x cleanup_large_files.sh
./cleanup_large_files.sh
git push origin main --force
```

### Option 3: Start Fresh (Nuclear Option)

If you don't need the old history:

```bash
# Remove .git directory
rm -rf .git

# Reinitialize
git init
git add .
git commit -m "chore: fresh start with Rust implementation"

# Force push to remote
git remote add origin <your-repo-url>
git push -u origin main --force
```

## Verify Models are Ignored

Check `.gitignore` has:
```
/models/*
!/models/.gitkeep
```

## Post-Cleanup Steps

1. All team members must re-clone:
```bash
git clone <repo-url>
```

2. Verify no large files:
```bash
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {if ($3 > 10485760) print substr($0,6)}' | \
  head -10
```

## Why This Happened

Large model files were committed to Git before `.gitignore` was properly configured. Even though they're ignored now, they remain in Git history.
