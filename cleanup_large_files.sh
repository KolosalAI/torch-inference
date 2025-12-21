#!/bin/bash

# Script to remove large model files from Git history
# This fixes the HTTP 408 error when pushing

echo "🧹 Cleaning large files from Git history..."

# Create backup branch
git branch backup-before-cleanup 2>/dev/null || true

# Remove large model files from entire Git history
echo "📦 Removing model files from history..."
git filter-branch --force --index-filter \
  'git rm -rf --cached --ignore-unmatch models/' \
  --prune-empty --tag-name-filter cat -- --all

# Remove large specific file types from history
for pattern in "*.ot" "*.h5" "*.msgpack" "*.safetensors" "*.tflite" "*.pt"; do
  echo "🗑️  Removing $pattern files..."
  git filter-branch --force --index-filter \
    "git rm -rf --cached --ignore-unmatch '**/$pattern'" \
    --prune-empty --tag-name-filter cat -- --all
done

# Clean up refs and force garbage collection
echo "🧼 Cleaning up Git repository..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ Cleanup complete!"
echo ""
echo "⚠️  Important next steps:"
echo "1. Verify your repository: git log --oneline | head -10"
echo "2. Force push to remote: git push origin main --force"
echo "3. If issues persist, consider using BFG Repo-Cleaner"
echo ""
echo "📝 Note: Backup branch created: backup-before-cleanup"
