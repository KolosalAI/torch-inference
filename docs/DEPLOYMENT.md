# Deployment — Developer Guide

Production deployment reference for `torch-inference` (Rust/Actix-Web ML inference server).

---

## Table of Contents

1. [Deployment Topology](#deployment-topology)
2. [Docker Deployment Pipeline](#docker-deployment-pipeline)
3. [Docker Compose Service Graph](#docker-compose-service-graph)
4. [Bare-Metal / Binary Deployment](#bare-metal--binary-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Health Check Configuration](#health-check-configuration)
7. [Environment-Specific Configuration](#environment-specific-configuration)
8. [Monitoring Integration](#monitoring-integration)
9. [Security Hardening](#security-hardening)

---

## Deployment Topology

### Single Instance

```mermaid
graph TB
    Client([External Client])

    subgraph Host["Single Host / Container"]
        Nginx["nginx<br/>(TLS termination, port 443)"]
        Server["torch-inference<br/>Actix-Web, port 8000<br/>Tokio multi-thread runtime"]
        Models[("./models<br/>volume")]
        Logs[("./logs<br/>volume")]
    end

    Prometheus["Prometheus<br/>(optional, /metrics scrape)"]
    Grafana["Grafana<br/>(dashboards)"]

    Client -->|HTTPS 443| Nginx
    Nginx -->|HTTP 8000| Server
    Server --- Models
    Server --- Logs
    Prometheus -->|scrape :8000/metrics| Server
    Prometheus --> Grafana

    style Nginx fill:#f39c12,color:#fff
    style Server fill:#2980b9,color:#fff
    style Prometheus fill:#e74c3c,color:#fff
    style Grafana fill:#f39c12,color:#fff
```

### Multi-Instance with nginx Load Balancer

```mermaid
graph TB
    Client([External Clients])

    subgraph LB["Load Balancer Host"]
        Nginx["nginx<br/>upstream least_conn<br/>TLS termination"]
    end

    subgraph Node1["Inference Node 1"]
        S1["torch-inference :8000<br/>GPU 0"]
        M1[("models NFS")]
    end

    subgraph Node2["Inference Node 2"]
        S2["torch-inference :8000<br/>GPU 1"]
        M2[("models NFS")]
    end

    subgraph Node3["Inference Node 3"]
        S3["torch-inference :8000<br/>CPU-only"]
        M3[("models NFS")]
    end

    subgraph Monitoring["Monitoring Stack"]
        Prom["Prometheus"]
        Graf["Grafana"]
        Alert["Alertmanager"]
    end

    NFS[("NFS / S3<br/>Shared Model Store")]

    Client -->|HTTPS 443| Nginx
    Nginx -->|HTTP 8000| S1
    Nginx -->|HTTP 8000| S2
    Nginx -->|HTTP 8000| S3

    NFS --- M1
    NFS --- M2
    NFS --- M3

    Prom -->|scrape /metrics| S1
    Prom -->|scrape /metrics| S2
    Prom -->|scrape /metrics| S3
    Prom --> Graf
    Prom --> Alert

    style Nginx fill:#f39c12,color:#fff
    style S1 fill:#2980b9,color:#fff
    style S2 fill:#2980b9,color:#fff
    style S3 fill:#27ae60,color:#fff
    style Prom fill:#e74c3c,color:#fff
```

---

## Docker Deployment Pipeline

```mermaid
flowchart TD
    A([Developer pushes code]) --> B[CI: cargo test --all-features]
    B --> C{Tests pass?}
    C -->|No| D([Fix and re-push])
    C -->|Yes| E[cargo build --release\nwith target features]

    E --> F{Target environment?}

    F -->|CPU only| G[docker build\nDockerfile]
    F -->|GPU / CUDA| H[docker build\nDockerfile --build-arg GPU=1]
    F -->|Production| I[docker build\nDockerfile\nmulti-stage optimised]

    G --> J[Tag: torch-inference:latest]
    H --> K[Tag: torch-inference:gpu]
    I --> L[Tag: torch-inference:prod-vX.Y.Z]

    J --> M[docker push → registry]
    K --> M
    L --> M

    M --> N{Deploy target}

    N -->|Local dev| O[docker compose\n-f compose.dev.yaml up]
    N -->|GPU workstation| P[docker compose\n-f compose.gpu.yaml up]
    N -->|Production| Q[docker compose\n-f compose.prod.yaml up -d]
    N -->|Kubernetes| R[kubectl apply -f k8s/]

    O --> S([Server running])
    P --> S
    Q --> S
    R --> S

    S --> T[curl http://localhost:8000/health]
    T --> U{Healthy?}
    U -->|No| V([Check logs: docker compose logs -f])
    U -->|Yes| W([Deployment complete ✓])

    style A fill:#27ae60,color:#fff
    style W fill:#27ae60,color:#fff
    style D fill:#e74c3c,color:#fff
    style V fill:#e74c3c,color:#fff
```

---

## Docker Compose Service Graph

### Default (`compose.yaml`)

```mermaid
graph LR
    subgraph compose.yaml["compose.yaml — Default Stack"]
        Nginx["nginx<br/>port 80/443 → 8000<br/>config: nginx.conf"]
        Server["torch-inference<br/>port 8000<br/>image: torch-inference:latest"]
        Config[("config.toml<br/>(bind-mount :ro)")]
        Models[("./models<br/>(bind-mount)")]
        Logs[("./logs<br/>(bind-mount)")]
    end

    Nginx -->|proxy_pass :8000| Server
    Server --- Config
    Server --- Models
    Server --- Logs

    style Nginx fill:#f39c12,color:#fff
    style Server fill:#2980b9,color:#fff
```

### GPU Variant (`compose.gpu.yaml`)

```mermaid
graph LR
    subgraph compose.gpu.yaml["compose.gpu.yaml — GPU Stack"]
        Nginx2["nginx<br/>port 80/443"]
        ServerGPU["torch-inference<br/>port 8000<br/>--gpus all<br/>CUDA_VISIBLE_DEVICES=0"]
        LibTorch[("./libtorch<br/>(CUDA libraries)")]
        Models2[("./models")]
    end

    subgraph nvidia["NVIDIA Runtime"]
        GPU["GPU 0 (e.g. A100)"]
    end

    Nginx2 -->|proxy_pass| ServerGPU
    ServerGPU --- LibTorch
    ServerGPU --- Models2
    ServerGPU <-->|cuda:0| GPU

    style ServerGPU fill:#2980b9,color:#fff
    style GPU fill:#76b900,color:#fff
```

### Production (`compose.prod.yaml`)

```mermaid
graph LR
    subgraph compose.prod.yaml["compose.prod.yaml — Production Stack"]
        Nginx3["nginx<br/>TLS 443\nHTTP→HTTPS redirect"]
        Server3["torch-inference<br/>restart: always\nresource limits\nhealthcheck"]
        Secrets[("env secrets\n.env file")]
        Models3[("models volume")]
        Logs3[("logs volume")]
    end

    Nginx3 -->|proxy_pass :8000| Server3
    Server3 --- Secrets
    Server3 --- Models3
    Server3 --- Logs3

    style Nginx3 fill:#f39c12,color:#fff
    style Server3 fill:#e74c3c,color:#fff
```

### Compose commands

```bash
# Development (hot config reload)
docker compose -f compose.dev.yaml up

# GPU workstation
docker compose -f compose.gpu.yaml up -d

# Production (detached, with restart policy)
docker compose -f compose.prod.yaml up -d

# View logs
docker compose logs -f torch-inference

# Scale inference service (3 replicas behind nginx)
docker compose -f compose.prod.yaml up -d --scale torch-inference=3
```

---

## Bare-Metal / Binary Deployment

### Build

```bash
# Clone
git clone https://github.com/your-org/torch-inference.git
cd torch-inference

# CPU + ONNX (most common)
cargo build --release --features "metrics"

# Full GPU stack (requires libtorch)
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release --features "torch,metrics,cuda"

# All backends
cargo build --release --features "all-backends,metrics,telemetry"
```

### Directory layout

```
/opt/torch-inference/
├── bin/
│   └── torch-inference-server     # compiled binary
├── config/
│   └── production.toml
├── models/                        # model files (.pt, .onnx)
├── logs/
└── data/
    ├── users.json                 # user store
    └── sessions.json              # API keys / sessions
```

### Systemd service

```ini
# /etc/systemd/system/torch-inference.service
[Unit]
Description=Torch Inference Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=torch-inference
Group=torch-inference
WorkingDirectory=/opt/torch-inference

EnvironmentFile=/opt/torch-inference/config/env
ExecStart=/opt/torch-inference/bin/torch-inference-server \
          --config /opt/torch-inference/config/production.toml

Restart=always
RestartSec=10

LimitNOFILE=65535
LimitNPROC=32768

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/torch-inference/logs /opt/torch-inference/models /opt/torch-inference/data

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now torch-inference
sudo journalctl -u torch-inference -f
```

---

## Kubernetes Deployment

```mermaid
graph TB
    subgraph k8s["Kubernetes Cluster"]
        subgraph ingress["Ingress Layer"]
            Ingress["nginx Ingress Controller<br/>TLS termination"]
        end

        subgraph svc["Service"]
            Svc["Service: torch-inference<br/>ClusterIP :8000"]
        end

        subgraph deploy["Deployment (replicas: 3)"]
            Pod1["Pod 1<br/>torch-inference:prod<br/>GPU limit: 1"]
            Pod2["Pod 2<br/>torch-inference:prod<br/>GPU limit: 1"]
            Pod3["Pod 3<br/>torch-inference:prod<br/>GPU limit: 1"]
        end

        subgraph storage["Storage"]
            PVC["PVC: models<br/>ReadOnlyMany"]
            CM["ConfigMap: config.toml"]
            Secret["Secret: jwt-secret\napi-key-salt"]
        end

        subgraph autoscale["Autoscaling"]
            HPA["HPA<br/>min: 2 / max: 10<br/>CPU > 70% or\nlatency p95 > 500ms"]
        end
    end

    Client([External Traffic]) --> Ingress
    Ingress --> Svc
    Svc --> Pod1
    Svc --> Pod2
    Svc --> Pod3
    Pod1 --- PVC
    Pod2 --- PVC
    Pod3 --- PVC
    Pod1 --- CM
    Pod1 --- Secret
    HPA -.->|scales| deploy

    style Ingress fill:#f39c12,color:#fff
    style Pod1 fill:#2980b9,color:#fff
    style Pod2 fill:#2980b9,color:#fff
    style Pod3 fill:#2980b9,color:#fff
    style HPA fill:#9b59b6,color:#fff
```

### Kubernetes manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torch-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: torch-inference
  template:
    metadata:
      labels:
        app: torch-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: torch-inference
        image: torch-inference:prod-v1.0.0
        ports:
        - containerPort: 8000
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
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: config
          mountPath: /app/config.toml
          subPath: config.toml
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 10
          failureThreshold: 3
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: torch-inference-models
      - name: config
        configMap:
          name: torch-inference-config
```

```bash
kubectl apply -f k8s/
kubectl rollout status deployment/torch-inference
kubectl logs -f deployment/torch-inference
kubectl scale deployment torch-inference --replicas=5
```

---

## Health Check Configuration

The server exposes `/health` (and `/api/health`) returning JSON:

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "models_loaded": 2,
  "device": "cuda:0"
}
```

### Docker health check

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Docker Compose health check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 60s   # allow model loading time
```

### Kubernetes probes

| Probe       | Path      | Initial Delay | Period | Failure Threshold | Purpose                          |
|-------------|-----------|---------------|--------|-------------------|----------------------------------|
| `liveness`  | `/health` | 60s           | 30s    | 3                 | Restart if server deadlocked     |
| `readiness` | `/health` | 20s           | 10s    | 3                 | Remove from LB until models load |

> **Tip**: Set `start_period` / `initialDelaySeconds` to at least the time needed to load the largest model. Monitor `preload_models_on_startup` in `config.toml`.

---

## Environment-Specific Configuration

| Setting                        | Development            | Staging                  | Production                  |
|--------------------------------|------------------------|--------------------------|-----------------------------|
| `server.log_level`             | `debug`                | `info`                   | `warn`                      |
| `server.workers`               | `2`                    | `8`                      | `num_cpus` (auto)           |
| `device.device_type`           | `cpu`                  | `cuda` or `cpu`          | `auto` (CUDA preferred)     |
| `device.use_fp16`              | `false`                | `true`                   | `true`                      |
| `performance.cache_size_mb`    | `256`                  | `1024`                   | `4096`                      |
| `performance.max_workers`      | `4`                    | `8`                      | `32`                        |
| `auth.enabled`                 | `false` (optional)     | `true`                   | `true`                      |
| `auth.jwt_secret`              | dev-only-secret        | from env/vault           | from env/vault (rotate)     |
| `guard.enable_guards`          | `false`                | `true`                   | `true`                      |
| `guard.max_requests_per_second`| `10000`                | `500`                    | `1000`                      |
| `performance.enable_profiling` | `true`                 | `false`                  | `false`                     |
| `LOG_JSON`                     | `false`                | `true`                   | `true`                      |
| TLS                            | None                   | Self-signed              | Let's Encrypt / ACM         |
| Replicas                       | 1                      | 2                        | 3–10 (HPA)                  |

### Environment variables

```bash
# Required in production
JWT_SECRET=<256-bit-hex>
RUST_LOG=warn

# Optional overrides
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=16
LOG_JSON=true
LOG_DIR=/var/log/torch-inference
LIBTORCH=/opt/libtorch            # if using torch feature
LD_LIBRARY_PATH=/opt/libtorch/lib
CUDA_VISIBLE_DEVICES=0,1          # GPU selection
```

---

## Monitoring Integration

```mermaid
graph TB
    subgraph App["torch-inference"]
        Tracing["tracing crate<br/>(structured logs)"]
        Prom["prometheus crate<br/>(feature: metrics)<br/>GET /metrics"]
        OTel["opentelemetry-otlp<br/>(feature: telemetry)"]
    end

    subgraph Logging["Log Pipeline"]
        Stdout["stdout / stderr<br/>JSON format"]
        FileLog["tracing-appender<br/>./logs/server.log"]
    end

    subgraph MetricsPipeline["Metrics Pipeline"]
        PromServer["Prometheus Server<br/>scrape :8000/metrics every 15s"]
        Grafana["Grafana<br/>dashboards"]
        Alert["Alertmanager<br/>PagerDuty / Slack"]
    end

    subgraph TracePipeline["Trace Pipeline"]
        OTelCol["OpenTelemetry Collector"]
        Jaeger["Jaeger / Tempo<br/>(distributed tracing)"]
    end

    subgraph LogPipeline["Log Aggregation"]
        Loki["Loki / ELK<br/>log aggregation"]
        GrafanaLogs["Grafana Logs"]
    end

    Tracing --> Stdout
    Tracing --> FileLog
    Prom --> PromServer
    OTel --> OTelCol

    Stdout --> Loki
    FileLog --> Loki
    PromServer --> Grafana
    PromServer --> Alert
    OTelCol --> Jaeger
    Loki --> GrafanaLogs
    Grafana --> GrafanaLogs

    style App fill:#2980b9,color:#fff
    style Grafana fill:#f39c12,color:#fff
    style Alert fill:#e74c3c,color:#fff
```

### Enable Prometheus metrics

```toml
# Build with: cargo build --release --features "metrics"
# Metrics available at GET /metrics (Prometheus text format)
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: torch-inference
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

### Key metrics exposed

| Metric                              | Type      | Description                         |
|-------------------------------------|-----------|-------------------------------------|
| `inference_requests_total`          | Counter   | Total inference requests            |
| `inference_duration_seconds`        | Histogram | Request latency (p50/p95/p99)       |
| `cache_hits_total`                  | Counter   | LRU cache hits                      |
| `cache_misses_total`                | Counter   | LRU cache misses                    |
| `active_workers`                    | Gauge     | Current worker pool size            |
| `batch_size`                        | Histogram | Observed batch sizes                |
| `circuit_breaker_state`             | Gauge     | 0=Closed, 1=Open, 2=HalfOpen        |
| `model_load_duration_seconds`       | Histogram | Model load time                     |

### Enable OpenTelemetry tracing

```bash
cargo build --release --features "telemetry"

# Set OTEL endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Security Hardening

```bash
# Nginx — enforce HTTPS
server {
    listen 80;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    ssl_certificate     /etc/letsencrypt/live/your-domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header Content-Security-Policy "default-src 'none'" always;

    location / {
        proxy_pass http://torch_inference;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Firewall — expose only necessary ports
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw allow 443/tcp
sudo ufw enable

# Do NOT expose port 8000 directly — all traffic must go through nginx
```

---

**See also**: [`AUTHENTICATION.md`](AUTHENTICATION.md) · [`CONFIGURATION.md`](CONFIGURATION.md) · [`TESTING.md`](TESTING.md)
