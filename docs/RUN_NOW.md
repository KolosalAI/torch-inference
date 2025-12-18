# Manual Testing Instructions - Build with PyTorch & Test Image Classification

Since automated bash execution is currently unavailable, follow these manual steps in your terminal.

---

## Quick Start Commands (Copy & Paste)

```bash
# 1. Navigate to project
cd /Users/evintleovonzko/Documents/Works/Kolosal/torch-inference

# 2. Set up LibTorch (if exists)
export LIBTORCH=$(pwd)/libtorch
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH

# 3. Build with torch support
cargo build --release --features torch 2>&1 | tee build_output.log

# 4. Start server
./target/release/torch-inference-server > server_torch.log 2>&1 &
SERVER_PID=$!
sleep 8

# 5. Check health
curl http://localhost:8000/health

# 6. Download MobileNetV4 (smallest model)
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mobilenetv4-hybrid-large", "source_type": "huggingface", "repo_id": "timm/mobilenetv4_hybrid_large.e600_r448_in1k"}'

# 7. Monitor download
watch -n 3 'curl -s http://localhost:8000/models/download/list | head -50'

# 8. Download test image
mkdir -p test_images
curl -o test_images/cat.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

# 9. Test classification
curl -X POST http://localhost:8000/classify \
  -F "image=@test_images/cat.jpg" \
  -F "model=mobilenetv4-hybrid-large" \
  -F "top_k=5"

# 10. Stop server
kill $SERVER_PID
```

---

## Or Use the Automated Script

```bash
chmod +x test_torch_complete.sh
./test_torch_complete.sh
```

---

## What to Expect

1. **Build:** ~5-10 minutes first time, ~1-2 minutes subsequent
2. **Download:** ~1-2 minutes for MobileNetV4 (140MB)
3. **Classification:** Response depends on endpoint implementation

---

## Files Created

- `build_output.log` - Build output
- `server_torch.log` - Server logs  
- `test_torch_complete.sh` - Automated test script
- `test_images/cat.jpg` - Test image

---

**All documentation and scripts are ready. Run the commands above in your terminal to proceed!**
