# COMPLETE GUIDE: Build, Download, and Test Image Classification

## Current Situation

✅ **Completed Analysis:**
- 12 SOTA image models identified and documented
- Download API endpoints verified
- All documentation created

⚠️ **Requires Manual Execution:**
- Bash automation is currently unavailable
- You need to run commands manually in your terminal

---

## Quick Commands (Copy All at Once)

Open your terminal and paste these commands:

```bash
cd /Users/evintleovonzko/Documents/Works/Kolosal/torch-inference

# Verify or download LibTorch
if [ ! -d "./libtorch" ]; then
    echo "Downloading LibTorch..."
    curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip
    unzip -q libtorch.zip
    rm libtorch.zip
fi

# Set environment
export LIBTORCH=$(pwd)/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH

# Build
echo "Building with PyTorch support..."
cargo build --release --features torch

# Start server
echo "Starting server..."
./target/release/torch-inference-server > server_torch.log 2>&1 &
SERVER_PID=$!
sleep 8

# Test
curl http://localhost:8000/health

# Download model
echo "Downloading MobileNetV4..."
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mobilenetv4-hybrid-large","source_type":"huggingface","repo_id":"timm/mobilenetv4_hybrid_large.e600_r448_in1k"}'

# Download test image
mkdir -p test_images
curl -o test_images/cat.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

echo "Monitor download with: curl http://localhost:8000/models/download/list"
echo "Server PID: $SERVER_PID"
echo "Server logs: tail -f server_torch.log"
```

---

## Alternative: Use Automated Script

```bash
cd /Users/evintleovonzko/Documents/Works/Kolosal/torch-inference
chmod +x test_torch_complete.sh
./test_torch_complete.sh
```

This script will:
1. ✅ Check/download LibTorch automatically
2. ✅ Build with torch feature
3. ✅ Start server
4. ✅ Download MobileNetV4 model
5. ✅ Download test image
6. ✅ Attempt classification
7. ✅ Generate summary report

---

## What Each Step Does

### 1. LibTorch Setup (30 seconds - 5 minutes)
Downloads PyTorch C++ library (~500MB) if not present

### 2. Build (5-10 minutes first time)
Compiles server with PyTorch support

**Success looks like:**
```
Finished `release` profile [optimized] target(s) in 7m 23s
```

### 3. Server Start (5 seconds)
Launches server with torch backend

**Check logs:**
```bash
grep -i "pytorch\|torch" server_torch.log
```

**Should see:**
```
[OK] PyTorch initialized successfully
   ├─ Backend: CPU
   ├─ Path: /Users/.../libtorch
   └─ Version: 2.3.0
```

### 4. Model Download (1-2 minutes)
Downloads MobileNetV4 from HuggingFace

**Monitor with:**
```bash
watch -n 2 'curl -s http://localhost:8000/models/download/list | grep -o "\"status\":\"[^\"]*\"" | head -1'
```

**Status:** Pending → Downloading → Completed

### 5. Classification Test
Attempts to classify test image

**Command:**
```bash
curl -X POST http://localhost:8000/classify \
  -F "image=@test_images/cat.jpg" \
  -F "model=mobilenetv4-hybrid-large" \
  -F "top_k=5"
```

---

## Expected Results

✅ **Build completes** - Binary at `./target/release/torch-inference-server`  
✅ **Server starts** - Health endpoint responds  
✅ **Model downloads** - Files in `./models/mobilenetv4-hybrid-large/`  
⚠️ **Classification** - Depends on endpoint implementation

---

## If Something Fails

### Build Error
```bash
# Check LibTorch path
echo $LIBTORCH
ls -la $LIBTORCH/lib

# Clean and retry
cargo clean
cargo build --release --features torch
```

### Server Won't Start
```bash
# Check logs
tail -50 server_torch.log

# Check library path
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
```

### Download Fails
```bash
# Check download status
curl http://localhost:8000/models/download/list

# Check error messages
curl http://localhost:8000/models/download/list | grep error
```

---

## Files & Documentation

**Scripts:**
- `test_torch_complete.sh` - Full automated test
- `build_with_torch.sh` - Build script only

**Docs:**
- `RUN_NOW.md` - This file (quick start)
- `BUILDING_WITH_TORCH.md` - Detailed build guide
- `IMAGE_MODELS_STATUS.md` - Status & roadmap
- `SOTA_IMAGE_MODELS_SUMMARY.md` - Model catalog
- `API_SOTA_MODELS.md` - API documentation

**Logs (created after running):**
- `build_output.log` - Build output
- `server_torch.log` - Server runtime logs

---

## Summary

**Everything is ready!** Just run the commands above in your terminal.

The automated script (`test_torch_complete.sh`) handles everything:
- Downloads LibTorch if needed
- Builds with PyTorch
- Starts server
- Downloads model
- Tests classification
- Shows results

**Estimated total time:** 15-20 minutes (first run), 5-10 minutes (subsequent runs)

---

## Next Steps After Success

1. Test more SOTA models (EVA-02, EfficientNetV2, etc.)
2. Benchmark inference speed
3. Test with different images
4. Optimize for production
5. Deploy!

---

**Ready to go! 🚀**
