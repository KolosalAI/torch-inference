# 🐳 Docker Troubleshooting Guide for torch-inference

## Current Issue
You're encountering a Docker Desktop connectivity issue:
```
request returned 500 Internal Server Error for API route and version http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.49/...
```

## 🔧 Quick Fixes to Try

### 1. Restart Docker Desktop
```powershell
# Stop Docker Desktop (from system tray or task manager)
# Then restart Docker Desktop
```

### 2. Check Docker Desktop Status
- Look for Docker Desktop in your system tray
- Ensure it shows "Docker Desktop is running"
- If it shows "Starting..." wait for it to complete

### 3. Reset Docker Desktop (if restart doesn't work)
```powershell
# From Docker Desktop settings:
# Settings → Troubleshoot → Reset to factory defaults
```

### 4. Alternative: Use Windows Containers (temporary)
```powershell
# Right-click Docker Desktop icon → Switch to Windows containers
# Then try: docker version
```

## 🚀 Test Commands (after Docker is fixed)

### Test Docker connectivity:
```powershell
docker version
docker ps
```

### Build and run with simple configuration:
```powershell
# Use the simple Dockerfile
docker build -f Dockerfile.simple -t torch-inference-simple .

# Use the simple compose file
docker compose -f compose.simple.yaml up

# Or run directly
docker run -p 8000:8000 torch-inference-simple
```

### Build with the optimized configuration:
```powershell
# Full build
docker compose build

# Run the complete stack
docker compose up

# Run in development mode
docker compose -f compose.yaml -f compose.dev.yaml up
```

## 📂 Files Created for Easy Testing

1. **`Dockerfile.simple`** - Simplified single-stage Dockerfile
2. **`compose.simple.yaml`** - Minimal compose configuration
3. **Updated `compose.yaml`** - Fixed volume and dependency issues

## 🐛 Common Docker Issues on Windows

### Issue: Docker Desktop not starting
**Solution**: 
- Check Windows Subsystem for Linux (WSL2) is installed
- Enable Virtualization in BIOS
- Run as Administrator

### Issue: Volume mounting problems
**Solution**:
- Use Docker volumes instead of bind mounts
- Ensure directories exist before mounting

### Issue: Memory/Resource limits
**Solution**:
- Increase Docker Desktop memory allocation
- Settings → Resources → Memory (recommend 8GB+)

## ✅ Once Docker is Working

1. **Test simple build first**:
   ```powershell
   docker build -f Dockerfile.simple -t torch-inference-test .
   ```

2. **Test simple run**:
   ```powershell
   docker run -p 8000:8000 torch-inference-test
   ```

3. **Then try full compose stack**:
   ```powershell
   docker compose up
   ```

4. **Access the application**:
   - Health check: http://localhost:8000/health
   - API docs: http://localhost:8000/docs
   - Main endpoint: http://localhost:8000/

## 🔍 Debugging Commands

```powershell
# Check Docker logs
docker system events

# Check container logs
docker logs torch-inference-server

# Check system resources
docker system df
docker system prune  # Clean up unused resources
```

The Docker configuration is optimized and ready - just need to resolve the Docker Desktop connectivity issue first!