# Deployment Guide

Production deployment guide for Torch Inference Server.

## Deployment Options

1. **Binary Deployment** - Direct binary execution
2. **Docker Deployment** - Container-based
3. **Kubernetes** - Orchestrated containers
4. **Systemd Service** - Linux service
5. **Cloud Platforms** - AWS, GCP, Azure

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- OS: Linux (Ubuntu 20.04+), macOS, Windows Server

**Recommended**:
- CPU: 16+ cores
- RAM: 32+ GB
- Storage: 100+ GB SSD
- GPU: NVIDIA GPU with CUDA 11.7+ (optional)

### Software Requirements

- Rust 1.70+ (for building)
- LibTorch 2.1.0+ (if using PyTorch backend)
- ONNX Runtime (if using ONNX backend)
- CUDA Toolkit 11.7+ (for GPU support)

## Binary Deployment

### 1. Build Release Binary

```bash
# Clone repository
git clone https://github.com/your-org/torch-inference.git
cd torch-inference

# Build optimized binary
cargo build --release --features all-backends

# Binary location
./target/release/torch-inference-server
```

### 2. Prepare Directory Structure

```bash
# Create directory structure
sudo mkdir -p /opt/torch-inference/{bin,config,models,logs}

# Copy binary
sudo cp target/release/torch-inference-server /opt/torch-inference/bin/

# Copy config
sudo cp config.toml /opt/torch-inference/config/production.toml

# Set permissions
sudo chown -R torch-inference:torch-inference /opt/torch-inference
sudo chmod +x /opt/torch-inference/bin/torch-inference-server
```

### 3. Production Configuration

```toml
# /opt/torch-inference/config/production.toml

[server]
host = "0.0.0.0"
port = 8000
workers = 16
log_level = "info"

[device]
device_type = "cuda"
device_ids = [0, 1]  # Multi-GPU
use_fp16 = true
cudnn_benchmark = true

[performance]
cache_size_mb = 4096
max_batch_size = 64
enable_worker_pool = true
max_workers = 32

[auth]
enabled = true
jwt_secret = "${JWT_SECRET}"  # From environment

[guard]
enable_guards = true
enable_auto_mitigation = true
max_memory_mb = 30720

[security]
rate_limit_per_ip = 1000
enable_security_headers = true
```

### 4. Environment Variables

```bash
# /opt/torch-inference/config/env
export RUST_LOG=info
export LOG_JSON=true
export LOG_DIR=/opt/torch-inference/logs
export JWT_SECRET=your-secret-from-vault
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
```

### 5. Run Server

```bash
# Load environment
source /opt/torch-inference/config/env

# Run server
/opt/torch-inference/bin/torch-inference-server \
  --config /opt/torch-inference/config/production.toml
```

## Systemd Service

### 1. Create Service File

```ini
# /etc/systemd/system/torch-inference.service

[Unit]
Description=Torch Inference Server
After=network.target

[Service]
Type=simple
User=torch-inference
Group=torch-inference
WorkingDirectory=/opt/torch-inference

# Environment
EnvironmentFile=/opt/torch-inference/config/env

# Execution
ExecStart=/opt/torch-inference/bin/torch-inference-server \
          --config /opt/torch-inference/config/production.toml

# Restart policy
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65535
LimitNPROC=32768

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/torch-inference/logs /opt/torch-inference/models

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable torch-inference

# Start service
sudo systemctl start torch-inference

# Check status
sudo systemctl status torch-inference

# View logs
sudo journalctl -u torch-inference -f
```

### 3. Service Management

```bash
# Stop service
sudo systemctl stop torch-inference

# Restart service
sudo systemctl restart torch-inference

# Reload config (if supported)
sudo systemctl reload torch-inference

# Disable service
sudo systemctl disable torch-inference
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 torch && \
    mkdir -p /app /models /logs && \
    chown -R torch:torch /app /models /logs

# Copy binary
COPY --chown=torch:torch target/release/torch-inference-server /app/
COPY --chown=torch:torch config.toml /app/

# Copy libtorch (if needed)
COPY --chown=torch:torch libtorch /app/libtorch

# Set environment
ENV LD_LIBRARY_PATH=/app/libtorch/lib:$LD_LIBRARY_PATH
ENV RUST_LOG=info

# Switch to non-root user
USER torch
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Run server
CMD ["/app/torch-inference-server", "--config", "/app/config.toml"]
```

### 2. Build Image

```bash
# Build for CPU
docker build -t torch-inference:latest .

# Build for GPU
docker build -t torch-inference:gpu -f Dockerfile.gpu .

# Multi-arch build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t torch-inference:latest .
```

### 3. Run Container

```bash
# CPU
docker run -d \
  --name torch-inference \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  -v $(pwd)/logs:/logs \
  -e JWT_SECRET=your-secret \
  torch-inference:latest

# GPU
docker run -d \
  --name torch-inference \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  -v $(pwd)/logs:/logs \
  -e JWT_SECRET=your-secret \
  torch-inference:gpu

# Multi-GPU
docker run -d \
  --name torch-inference \
  --gpus '"device=0,1"' \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  torch-inference:gpu
```

### 4. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  torch-inference:
    image: torch-inference:latest
    container_name: torch-inference
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./logs:/logs
      - ./config.toml:/app/config.toml:ro
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - RUST_LOG=info
      - LOG_JSON=true
    deploy:
      resources:
        limits:
          cpus: '16'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

  # Optional: Redis for distributed caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

```bash
# Start stack
docker-compose up -d

# View logs
docker-compose logs -f torch-inference

# Stop stack
docker-compose down
```

## Kubernetes Deployment

### 1. Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torch-inference
  labels:
    app: torch-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: torch-inference
  template:
    metadata:
      labels:
        app: torch-inference
    spec:
      containers:
      - name: torch-inference
        image: torch-inference:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: torch-inference-secrets
              key: jwt-secret
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /models
        - name: config
          mountPath: /app/config.toml
          subPath: config.toml
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: torch-inference-models
      - name: config
        configMap:
          name: torch-inference-config
```

### 2. Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: torch-inference
  labels:
    app: torch-inference
spec:
  type: LoadBalancer
  selector:
    app: torch-inference
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
```

### 3. ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: torch-inference-config
data:
  config.toml: |
    [server]
    host = "0.0.0.0"
    port = 8000
    workers = 8
    
    [device]
    device_type = "cuda"
    use_fp16 = true
    
    [performance]
    cache_size_mb = 4096
    max_batch_size = 64
```

### 4. Deploy

```bash
# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods -l app=torch-inference
kubectl get svc torch-inference

# View logs
kubectl logs -f deployment/torch-inference

# Scale
kubectl scale deployment torch-inference --replicas=5
```

## Load Balancing

### NGINX Configuration

```nginx
# /etc/nginx/sites-available/torch-inference
upstream torch_inference {
    least_conn;
    server 10.0.1.10:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name inference.example.com;

    location / {
        proxy_pass http://torch_inference;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # Health check
    location /health {
        access_log off;
        proxy_pass http://torch_inference/api/health;
    }
}
```

## Monitoring

### Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'torch-inference'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import dashboard from `docs/grafana/dashboard.json`

Key metrics:
- Request rate
- Response time (p50/p95/p99)
- Error rate
- Cache hit rate
- GPU utilization
- Memory usage

## Security

### SSL/TLS (HTTPS)

```nginx
server {
    listen 443 ssl http2;
    server_name inference.example.com;
    
    ssl_certificate /etc/letsencrypt/live/inference.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/inference.example.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://torch_inference;
    }
}
```

### Firewall

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH (be careful!)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

## Backup & Recovery

### Backup Models

```bash
# Backup models directory
tar -czf models-$(date +%Y%m%d).tar.gz /opt/torch-inference/models

# Upload to S3
aws s3 cp models-$(date +%Y%m%d).tar.gz s3://backups/torch-inference/
```

### Backup Configuration

```bash
# Backup config
cp /opt/torch-inference/config/production.toml \
   /backups/config-$(date +%Y%m%d).toml
```

## Troubleshooting

See [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues and solutions.

## Scaling

### Horizontal Scaling

- Deploy multiple instances
- Use load balancer
- Shared model storage (NFS/S3)
- Distributed caching (Redis)

### Vertical Scaling

- Increase CPU cores
- Add more RAM
- Upgrade GPU
- Faster storage (NVMe SSD)

---

**Next**: See [Performance Tuning](TUNING.md) for optimization strategies.
