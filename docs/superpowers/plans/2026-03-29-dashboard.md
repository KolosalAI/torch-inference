# Dashboard Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live Dashboard panel to the existing playground with SSE-pushed metrics, resource bars, a sparkline chart, and a model download manager — all without changing any existing panels.

**Architecture:** One new Rust file (`src/api/dashboard.rs`) implements a `GET /dashboard/stream` SSE endpoint that aggregates Monitor, sysinfo, GpuManager, and ModelDownloadManager data into a single JSON event every 3 seconds. The frontend adds a new "Dashboard" sidebar item and panel with two tabs: Overview (metrics + downloads) and Playground (TTS/Classify/LLM/Completion sub-tabs).

**Tech Stack:** Rust / actix-web 4.8, `tokio::sync::mpsc` + `tokio_stream::wrappers::ReceiverStream` for SSE streaming, `sysinfo 0.31`, browser `EventSource` API, HTML5 Canvas for sparkline.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/api/dashboard.rs` | Create | SSE handler + `DashboardEvent` types |
| `src/api/mod.rs` | Modify (add 1 line) | Export `dashboard` module |
| `src/main.rs` | Modify (add 1 route) | Register `/dashboard/stream` |
| `src/api/playground.html` | Modify | Dashboard nav item, panel HTML, CSS, JS |

---

### Task 1: Create `src/api/dashboard.rs` with types + SSE handler

**Files:**
- Create: `src/api/dashboard.rs`

- [ ] **Step 1: Write the full file including a serialization test**

```rust
// src/api/dashboard.rs
use actix_web::{web, HttpResponse};
use actix_web::web::Bytes;
use serde::Serialize;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::monitor::Monitor;
use crate::api::system::SystemInfoState;
use crate::api::model_download::ModelDownloadState;

#[derive(Debug, Serialize)]
pub struct DashboardMetrics {
    pub uptime_s: u64,
    pub active_req: u64,
    pub total_req: u64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub throughput_per_s: f64,
    pub cpu_pct: f32,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct DashboardGpu {
    pub name: String,
    pub util_pct: Option<u32>,
    pub temp_c: Option<u32>,
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
}

#[derive(Debug, Serialize)]
pub struct DashboardDownload {
    pub id: String,
    pub model_name: String,
    pub status: String,
    pub progress: f32,
    pub downloaded_mb: u64,
    pub total_mb: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DashboardEvent {
    pub metrics: DashboardMetrics,
    pub gpu: Vec<DashboardGpu>,
    pub downloads: Vec<DashboardDownload>,
}

pub async fn dashboard_stream(
    monitor: web::Data<Arc<Monitor>>,
    system_state: web::Data<SystemInfoState>,
    download_state: web::Data<ModelDownloadState>,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, actix_web::Error>>(4);

    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(3));
        loop {
            ticker.tick().await;

            // Metrics from Monitor
            let m = monitor.get_metrics();
            let h = monitor.get_health_status();
            let error_rate = if m.total_requests > 0 {
                m.total_errors as f64 / m.total_requests as f64
            } else {
                0.0
            };

            // CPU / RAM from sysinfo
            let mut sys = sysinfo::System::new_all();
            sys.refresh_all();
            let cpu_pct = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
                / sys.cpus().len().max(1) as f32;
            let mem_used_mb  = sys.used_memory()  / 1024 / 1024;
            let mem_total_mb = sys.total_memory()  / 1024 / 1024;

            // GPU from GpuManager
            let gpu = match system_state.gpu_manager.get_info() {
                Ok(info) => info
                    .devices
                    .into_iter()
                    .map(|d| DashboardGpu {
                        name:         d.name,
                        util_pct:     d.utilization,
                        temp_c:       d.temperature,
                        vram_free_mb: d.free_memory  / 1024 / 1024,
                        vram_total_mb: d.total_memory / 1024 / 1024,
                    })
                    .collect(),
                Err(_) => vec![],
            };

            // Active/recent downloads
            let downloads = download_state
                .manager
                .list_tasks()
                .into_iter()
                .map(|t| DashboardDownload {
                    id:           t.id,
                    model_name:   t.model_name,
                    status:       format!("{:?}", t.status),
                    progress:     t.progress,
                    downloaded_mb: t.downloaded_size / 1024 / 1024,
                    total_mb:     t.total_size.map(|s| s / 1024 / 1024),
                })
                .collect();

            let event = DashboardEvent {
                metrics: DashboardMetrics {
                    uptime_s:        m.uptime_seconds,
                    active_req:      h.active_requests,
                    total_req:       m.total_requests,
                    avg_latency_ms:  m.avg_latency_ms,
                    error_rate,
                    throughput_per_s: m.throughput_rps,
                    cpu_pct,
                    mem_used_mb,
                    mem_total_mb,
                },
                gpu,
                downloads,
            };

            let json = match serde_json::to_string(&event) {
                Ok(j)  => j,
                Err(_) => break,
            };
            let frame = Bytes::from(format!("data: {}\n\n", json));

            if tx.send(Ok(frame)).await.is_err() {
                break; // client disconnected
            }
        }
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_event_serializes_all_fields() {
        let event = DashboardEvent {
            metrics: DashboardMetrics {
                uptime_s: 3600,
                active_req: 2,
                total_req: 500,
                avg_latency_ms: 38.5,
                error_rate: 0.01,
                throughput_per_s: 12.3,
                cpu_pct: 45.0,
                mem_used_mb: 1024,
                mem_total_mb: 8192,
            },
            gpu: vec![DashboardGpu {
                name: "TestGPU".to_string(),
                util_pct: Some(30),
                temp_c: Some(65),
                vram_free_mb: 8000,
                vram_total_mb: 10000,
            }],
            downloads: vec![DashboardDownload {
                id: "abc-123".to_string(),
                model_name: "test/model".to_string(),
                status: "Downloading".to_string(),
                progress: 0.5,
                downloaded_mb: 200,
                total_mb: Some(400),
            }],
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"uptime_s\":3600"));
        assert!(json.contains("\"total_req\":500"));
        assert!(json.contains("\"cpu_pct\":45.0"));
        assert!(json.contains("\"TestGPU\""));
        assert!(json.contains("\"test/model\""));
        assert!(json.contains("\"progress\":0.5"));
    }
}
```

- [ ] **Step 2: Run the test in isolation to verify it compiles and passes**

The module isn't wired yet, so run directly:
```bash
cargo test dashboard_event_serializes_all_fields 2>&1 | tail -10
```

Expected: compile error about missing module — that's expected until Task 2 wires it. The test logic itself is valid; compilation confirms types are correct once wired.

- [ ] **Step 3: Commit the new file**

```bash
git add src/api/dashboard.rs
git commit -m "feat: add DashboardEvent types and SSE handler"
```

---

### Task 2: Wire module and route

**Files:**
- Modify: `src/api/mod.rs` (add 1 line)
- Modify: `src/main.rs` (add 1 route)

- [ ] **Step 1: Add `dashboard` to `src/api/mod.rs`**

Open `src/api/mod.rs`. After `pub mod handlers;` add:

```rust
pub mod dashboard;
```

The top of the file currently reads:
```rust
pub mod handlers;
pub mod types;
```

Change to:
```rust
pub mod handlers;
pub mod dashboard;
pub mod types;
```

- [ ] **Step 2: Register the route in `src/main.rs`**

Find the block registering routes (around line 420 — look for `.configure(handlers::configure_routes)`). Add one line **before** it:

```rust
            .route("/dashboard/stream", web::get().to(crate::api::dashboard::dashboard_stream))
            .configure(handlers::configure_routes)
```

- [ ] **Step 3: Verify it compiles and the test passes**

```bash
cargo test dashboard_event_serializes_all_fields -- --nocapture
```

Expected:
```
test api::dashboard::tests::dashboard_event_serializes_all_fields ... ok
test result: ok. 1 passed
```

If you see a borrow/lifetime error inside the `tokio::spawn` closure, `web::Data<T>` is an `Arc` wrapper — add `.clone()` on each `web::Data` before the closure:
```rust
let monitor       = monitor.clone();
let system_state  = system_state.clone();
let download_state = download_state.clone();
tokio::spawn(async move { ... });
```

- [ ] **Step 4: Commit**

```bash
git add src/api/mod.rs src/main.rs
git commit -m "feat: wire /dashboard/stream SSE route"
```

---

### Task 3: Smoke-test the SSE endpoint

**Files:**
- No changes — manual verification

- [ ] **Step 1: Build and start the server**

```bash
cargo build --release 2>&1 | tail -5
./target/release/torch_inference &
```

- [ ] **Step 2: Curl the SSE endpoint**

```bash
curl -N --max-time 7 http://localhost:8080/dashboard/stream 2>&1 | head -3
```

Expected — one event every 3 seconds:
```
data: {"metrics":{"uptime_s":1,"active_req":0,"total_req":0,...},"gpu":[...],"downloads":[]}
```

- [ ] **Step 3: Kill the server**

```bash
kill %1
```

No commit needed — verification only.

---

### Task 4: Frontend — CSS + sidebar nav item

**Files:**
- Modify: `src/api/playground.html`

- [ ] **Step 1: Add CSS**

In `src/api/playground.html`, find the line `::-webkit-scrollbar { width: 5px; height: 5px; }` near the end of the `<style>` block. Insert the following CSS **before** that line:

```css
  /* ── Dashboard tabs ─────────────────────────────────────── */
  .tab-bar {
    display: flex; gap: 2px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 9px; padding: 3px; width: fit-content; margin-bottom: 20px;
  }
  .tab {
    padding: 6px 18px; border-radius: 7px;
    font-size: 13px; font-weight: 500; color: var(--text-muted);
    cursor: pointer; border: none; background: none; transition: all .15s;
    display: flex; align-items: center; gap: 7px; font-family: var(--font);
  }
  .tab.active {
    background: var(--accent-bg); color: var(--accent-hi);
    border: 1px solid rgba(99,102,241,.25);
  }
  .tab-icon { font-size: 14px; }
  .subtab-bar {
    display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 18px;
  }
  .subtab {
    padding: 7px 16px; font-size: 13px; font-weight: 500; color: var(--text-muted);
    cursor: pointer; border: none; background: none;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
    font-family: var(--font); transition: color .15s;
  }
  .subtab:hover { color: var(--text); }
  .subtab.active { color: var(--accent-hi); border-bottom-color: var(--accent); }
  .live-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 20px;
    background: rgba(34,197,94,.1); color: var(--green);
    border: 1px solid rgba(34,197,94,.2);
  }
  .live-dot {
    width: 5px; height: 5px; border-radius: 50%; background: var(--green);
    animation: livepulse 2s ease-in-out infinite;
  }
  @keyframes livepulse {
    0%,100% { opacity: 1; box-shadow: 0 0 4px var(--green); }
    50%      { opacity: .4; box-shadow: none; }
  }
  .dash-stat-grid {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
  }
  .dash-tile {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
  }
  .dash-tile-label {
    font-size: 10px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 5px;
  }
  .dash-tile-val {
    font-size: 22px; font-weight: 700; font-family: var(--mono); color: var(--accent-hi);
  }
  .dash-tile-sub { font-size: 10px; color: var(--text-dim); margin-top: 3px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .res-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .res-row:last-child { margin-bottom: 0; }
  .res-name { font-size: 12px; color: var(--text-muted); width: 44px; flex-shrink: 0; }
  .res-bar-track { flex: 1; height: 6px; background: var(--border2); border-radius: 3px; overflow: hidden; }
  .res-bar-fill  { height: 100%; border-radius: 3px; transition: width .6s ease; }
  .res-val { font-size: 11px; font-family: var(--mono); color: var(--text); width: 56px; text-align: right; }
  .gpu-device {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 7px; padding: 12px; margin-top: 12px;
  }
  .gpu-device-name { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  .gpu-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 3px; }
  .gpu-row span:last-child { color: var(--text); font-family: var(--mono); }
  .sparkline-wrap { position: relative; }
  #dash-spark { width: 100%; height: 72px; display: block; }
  .spark-hint { font-size: 10px; color: var(--text-dim); margin-top: 5px; font-family: var(--mono); }
  .dl-form-row { display: flex; gap: 10px; align-items: flex-end; }
  .dl-form-row .field { flex: 1; margin-bottom: 0; }
  .dl-form-row .field.narrow { flex: 0 0 130px; }
  .dl-form-row .field.slim   { flex: 0 0 150px; }
  .dl-task {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; margin-bottom: 8px;
  }
  .dl-task:last-child { margin-bottom: 0; }
  .dl-task-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .dl-task-name { font-size: 13px; font-weight: 500; font-family: var(--mono); }
  .dl-status-badge { font-size: 11px; padding: 2px 8px; border-radius: 12px; }
  .dl-status-badge.downloading { background: rgba(99,102,241,.15); color: var(--accent-hi); }
  .dl-status-badge.completed   { background: rgba(34,197,94,.12);  color: var(--green); }
  .dl-status-badge.failed      { background: rgba(239,68,68,.12);  color: var(--red); }
  .dl-status-badge.pending     { background: rgba(245,158,11,.12); color: var(--yellow); }
  .dl-progress-track { height: 5px; background: var(--border2); border-radius: 3px; overflow: hidden; margin-bottom: 5px; }
  .dl-progress-fill  { height: 100%; border-radius: 3px; background: var(--accent); transition: width .4s ease; }
  .dl-progress-fill.done   { background: var(--green); }
  .dl-progress-fill.failed { background: var(--red); }
  .dl-meta { font-size: 10px; color: var(--text-dim); display: flex; justify-content: space-between; }
  #dl-error { font-size: 12px; color: var(--red); margin-top: 6px; display: none; }
  #dl-tasks-empty { font-size: 12px; color: var(--text-dim); text-align: center; padding: 16px 0; }
```

- [ ] **Step 2: Add Dashboard nav item to sidebar**

Find this in the sidebar:
```html
      <button class="nav-item" onclick="show('completion',this)">
        <span class="nav-icon">⌨</span> Completion
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
```

Replace with:
```html
      <button class="nav-item" onclick="show('completion',this)">
        <span class="nav-icon">⌨</span> Completion
      </button>
      <button class="nav-item" onclick="showDashboard(this)">
        <span class="nav-icon">⬡</span> Dashboard
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
```

- [ ] **Step 3: Commit**

```bash
git add src/api/playground.html
git commit -m "feat: add dashboard CSS and sidebar nav item"
```

---

### Task 5: Frontend — Overview tab HTML

**Files:**
- Modify: `src/api/playground.html`

- [ ] **Step 1: Insert the Dashboard panel before the Endpoints panel**

Find in `playground.html`:
```html
    <!-- Endpoints -->
    <section class="panel" id="panel-endpoints">
```

Insert the following **before** that line:

```html
    <!-- Dashboard -->
    <section class="panel" id="panel-dashboard">

      <div class="tab-bar">
        <button class="tab active" id="dash-tab-overview"    onclick="switchDashTab('overview')">
          <span class="tab-icon">◈</span> Overview
        </button>
        <button class="tab"        id="dash-tab-playground"  onclick="switchDashTab('playground')">
          <span class="tab-icon">⚡</span> Playground
        </button>
      </div>

      <!-- ── Overview ─────────────────────────────────────── -->
      <div id="dash-overview">

        <div class="panel-header" style="margin-bottom:16px">
          <div class="panel-title" style="font-size:18px;font-weight:600;display:flex;align-items:center;gap:10px">
            Dashboard
            <span class="live-badge" id="dash-live-badge">
              <span class="live-dot"></span> SSE live
            </span>
          </div>
          <div class="panel-desc">Real-time inference metrics pushed every 3 s.</div>
        </div>

        <div class="dash-stat-grid">
          <div class="dash-tile"><div class="dash-tile-label">Uptime</div><div class="dash-tile-val" id="d-uptime">—</div><div class="dash-tile-sub">since restart</div></div>
          <div class="dash-tile"><div class="dash-tile-label">Total Req</div><div class="dash-tile-val" id="d-total">—</div><div class="dash-tile-sub">all time</div></div>
          <div class="dash-tile"><div class="dash-tile-label">Active</div><div class="dash-tile-val" id="d-active">—</div><div class="dash-tile-sub">in flight</div></div>
          <div class="dash-tile"><div class="dash-tile-label">Avg Latency</div><div class="dash-tile-val" id="d-latency">—</div><div class="dash-tile-sub">rolling avg</div></div>
          <div class="dash-tile"><div class="dash-tile-label">Error Rate</div><div class="dash-tile-val" id="d-errors">—</div><div class="dash-tile-sub">last N req</div></div>
          <div class="dash-tile"><div class="dash-tile-label">Throughput</div><div class="dash-tile-val" id="d-rps">—</div><div class="dash-tile-sub">req / sec</div></div>
        </div>

        <div class="two-col" style="margin-top:16px">
          <div class="card">
            <div class="card-title">Request Throughput</div>
            <div class="sparkline-wrap">
              <canvas id="dash-spark"></canvas>
            </div>
            <div class="spark-hint">60-point rolling window · updated every 3 s via SSE</div>
          </div>
          <div class="card">
            <div class="card-title">Resource Usage</div>
            <div class="res-row">
              <span class="res-name">CPU</span>
              <div class="res-bar-track"><div class="res-bar-fill" id="bar-cpu" style="width:0%;background:#6366f1"></div></div>
              <span class="res-val" id="val-cpu">—</span>
            </div>
            <div class="res-row">
              <span class="res-name">RAM</span>
              <div class="res-bar-track"><div class="res-bar-fill" id="bar-ram" style="width:0%;background:#f59e0b"></div></div>
              <span class="res-val" id="val-ram">—</span>
            </div>
            <div class="res-row">
              <span class="res-name">GPU</span>
              <div class="res-bar-track"><div class="res-bar-fill" id="bar-gpu" style="width:0%;background:#22c55e"></div></div>
              <span class="res-val" id="val-gpu">—</span>
            </div>
            <div id="gpu-device-info"></div>
          </div>
        </div>

        <div class="card" style="margin-top:16px">
          <div class="card-title">Model Download Manager</div>
          <div class="dl-form-row">
            <div class="field narrow">
              <label>Source</label>
              <select id="dl-source">
                <option value="huggingface">HuggingFace</option>
                <option value="url">URL</option>
              </select>
            </div>
            <div class="field">
              <label>Repo ID / URL</label>
              <input type="text" id="dl-repo" placeholder="e.g. meta-llama/Llama-3.2-1B" />
            </div>
            <div class="field slim" id="dl-revision-wrap">
              <label>Revision</label>
              <input type="text" id="dl-revision" placeholder="main" />
            </div>
            <div class="field" style="flex:0 0 auto">
              <label>&nbsp;</label>
              <button class="btn btn-primary" id="dl-btn" onclick="submitDownload()">↓ Download</button>
            </div>
          </div>
          <div id="dl-error"></div>
          <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border)">
            <div class="card-title" style="margin-bottom:10px">Active &amp; Recent Downloads</div>
            <div id="dl-tasks-list">
              <div id="dl-tasks-empty">No downloads yet.</div>
            </div>
          </div>
        </div>

      </div><!-- /#dash-overview -->
```

- [ ] **Step 2: Commit**

```bash
git add src/api/playground.html
git commit -m "feat: add Dashboard Overview tab HTML"
```

---

### Task 6: Frontend — Playground tab HTML

**Files:**
- Modify: `src/api/playground.html`

- [ ] **Step 1: Append Playground tab directly after `</div><!-- /#dash-overview -->`**

```html
      <!-- ── Playground ────────────────────────────────────── -->
      <div id="dash-playground" style="display:none">

        <div class="subtab-bar">
          <button class="subtab active" id="pg-tab-tts"        onclick="switchPgTab('tts')">♪ TTS Stream</button>
          <button class="subtab"        id="pg-tab-classify"   onclick="switchPgTab('classify')">⊞ Classify</button>
          <button class="subtab"        id="pg-tab-llm"        onclick="switchPgTab('llm')">✦ LLM Chat</button>
          <button class="subtab"        id="pg-tab-completion" onclick="switchPgTab('completion')">⌨ Completion</button>
        </div>

        <!-- TTS sub-tab -->
        <div id="pg-tts">
          <div class="panel-header">
            <div class="panel-title">TTS Stream</div>
            <div class="panel-desc">Sentence-level streaming synthesis via <span class="ep-code">POST /tts/stream</span>.</div>
          </div>
          <div class="card">
            <div class="card-title">Input</div>
            <div class="field">
              <label>Text</label>
              <textarea id="pg-tts-text">Hello! This is a streaming TTS demo from Kolosal Inference.</textarea>
            </div>
            <div class="row">
              <div class="field"><label>Engine (optional)</label><input type="text" id="pg-tts-engine" placeholder="default" /></div>
              <div class="field"><label>Voice (optional)</label><input type="text" id="pg-tts-voice" placeholder="default" /></div>
              <div class="field"><label>Speed</label><input type="number" id="pg-tts-speed" value="1.0" min="0.5" max="2.0" step="0.1" /></div>
            </div>
            <div style="display:flex;gap:8px;margin-top:6px">
              <button class="btn btn-primary" id="pg-tts-btn" onclick="runPgTTS()">▶  Synthesise</button>
              <button class="btn btn-ghost" onclick="stopPgTTS()">■  Stop</button>
            </div>
          </div>
          <div class="card">
            <div class="card-title">Output</div>
            <div class="response-box" id="pg-tts-status">waiting…</div>
            <div id="pg-tts-audio-wrap" style="margin-top:12px;display:none">
              <audio id="pg-tts-audio" controls style="width:100%"></audio>
            </div>
          </div>
        </div>

        <!-- Classify sub-tab -->
        <div id="pg-classify" style="display:none">
          <div class="panel-header">
            <div class="panel-title">Image Classification</div>
            <div class="panel-desc">SIMD-fused preprocessing + batched ORT inference via <span class="ep-code">POST /classify/batch</span>.</div>
          </div>
          <div class="card">
            <div class="card-title">Upload Image</div>
            <div class="dropzone" id="pg-dropzone"
                 onclick="document.getElementById('pg-img-file').click()"
                 ondragover="event.preventDefault();this.classList.add('over')"
                 ondragleave="this.classList.remove('over')"
                 ondrop="handlePgDrop(event)">
              <div class="drop-icon">⊞</div>
              <div>Click to upload or drag &amp; drop</div>
              <div style="font-size:11px;color:var(--text-dim);margin-top:4px">JPEG / PNG</div>
              <input type="file" id="pg-img-file" accept="image/jpeg,image/png" onchange="handlePgFile(this.files[0])" />
            </div>
            <img id="pg-img-preview" alt="preview" style="max-width:200px;border-radius:8px;margin-top:10px;display:none" />
            <div class="row" style="margin-top:12px">
              <div class="field"><label>Top-K</label><input type="number" id="pg-cls-topk" value="5" min="1" max="1000" /></div>
              <div class="field"><label>Width</label><input type="number" id="pg-cls-w" value="224" /></div>
              <div class="field"><label>Height</label><input type="number" id="pg-cls-h" value="224" /></div>
            </div>
            <button class="btn btn-primary" style="margin-top:8px" id="pg-cls-btn" onclick="runPgClassify()" disabled>⊞  Classify</button>
          </div>
          <div class="card">
            <div class="card-title">Predictions</div>
            <div class="response-box" id="pg-cls-out">waiting for image…</div>
          </div>
        </div>

        <!-- LLM Chat sub-tab -->
        <div id="pg-llm" style="display:none">
          <div class="panel-header">
            <div class="panel-title">LLM Chat</div>
            <div class="panel-desc">OpenAI-compatible chat via <span class="ep-code">POST /v1/chat/completions</span>.</div>
          </div>
          <div class="card">
            <div class="card-title">Conversation</div>
            <div class="chat-history" id="pg-chat-history"></div>
            <div class="chat-input-row">
              <textarea id="pg-chat-input" placeholder="Type a message… (Enter to send)" rows="1"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendPgChat();}"></textarea>
              <button class="btn btn-primary" id="pg-chat-btn" onclick="sendPgChat()">Send</button>
            </div>
          </div>
          <div class="card">
            <div class="card-title">Parameters</div>
            <div class="row">
              <div class="field"><label>Model</label><input type="text" id="pg-chat-model" value="default" /></div>
              <div class="field"><label>Max Tokens</label><input type="number" id="pg-chat-maxtok" value="256" min="1" /></div>
              <div class="field"><label>Temperature</label><input type="number" id="pg-chat-temp" value="0.7" min="0" max="2" step="0.05" /></div>
            </div>
            <button class="btn btn-ghost" style="margin-top:4px" onclick="clearPgChat()">✕  Clear chat</button>
          </div>
        </div>

        <!-- Completion sub-tab -->
        <div id="pg-completion" style="display:none">
          <div class="panel-header">
            <div class="panel-title">Text Completion</div>
            <div class="panel-desc">Raw prompt continuation via <span class="ep-code">POST /v1/completions</span>.</div>
          </div>
          <div class="card">
            <div class="card-title">Prompt</div>
            <div class="field">
              <textarea id="pg-cmp-prompt" style="min-height:130px">The key advantages of PagedAttention for LLM inference are:</textarea>
            </div>
            <div class="row">
              <div class="field"><label>Model</label><input type="text" id="pg-cmp-model" value="default" /></div>
              <div class="field"><label>Max Tokens</label><input type="number" id="pg-cmp-maxtok" value="128" min="1" /></div>
              <div class="field"><label>Temperature</label><input type="number" id="pg-cmp-temp" value="0.7" min="0" max="2" step="0.05" /></div>
              <div class="field"><label>Top-P</label><input type="number" id="pg-cmp-topp" value="1.0" min="0" max="1" step="0.05" /></div>
            </div>
            <button class="btn btn-primary" style="margin-top:4px" id="pg-cmp-btn" onclick="runPgCompletion()">▶  Complete</button>
          </div>
          <div class="card">
            <div class="card-title">Result</div>
            <div class="response-box" id="pg-cmp-out">waiting…</div>
          </div>
        </div>

      </div><!-- /#dash-playground -->

    </section><!-- /#panel-dashboard -->
```

- [ ] **Step 2: Verify HTML structure**

```bash
grep -c "panel-dashboard\|dash-overview\|dash-playground" src/api/playground.html
```

Expected: `6` (open + close for each of the three IDs).

- [ ] **Step 3: Commit**

```bash
git add src/api/playground.html
git commit -m "feat: add Dashboard Playground tab with TTS/Classify/LLM/Completion sub-tabs"
```

---

### Task 7: Frontend — JavaScript

**Files:**
- Modify: `src/api/playground.html`

All JS goes inside the existing `<script>` block at the bottom of the file.

- [ ] **Step 1: Patch `show()` to close SSE on navigate away**

Find the existing `show()` function:
```js
function show(id, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  btn.classList.add('active');
}
```

Replace with:
```js
function show(id, btn) {
  closeDashboardStream();
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  btn.classList.add('active');
}
```

- [ ] **Step 2: Add all dashboard JavaScript**

At the **end** of the `<script>` block (just before `</script>`), append:

```js
/* ── XSS-safe helper ────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── Dashboard ─────────────────────────────────────────── */
let dashES      = null;
const sparkData = new Array(60).fill(0);
let   sparkCtx  = null;

function showDashboard(btn) {
  closeDashboardStream();
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-dashboard').classList.add('active');
  btn.classList.add('active');
  switchDashTab('overview');
  openDashboardStream();
}

function openDashboardStream() {
  if (dashES) return;
  dashES = new EventSource('/dashboard/stream');
  dashES.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      updateDashTiles(ev.metrics   || {});
      updateSparkline(ev.metrics   ? (ev.metrics.throughput_per_s || 0) : 0);
      updateResourceBars(ev.metrics || {}, ev.gpu || []);
      updateDownloadsList(ev.downloads || []);
      const badge = document.getElementById('dash-live-badge');
      if (badge) badge.style.opacity = '1';
    } catch (_) {}
  };
  dashES.onerror = () => {
    const badge = document.getElementById('dash-live-badge');
    if (badge) badge.style.opacity = '0.4';
  };
}

function closeDashboardStream() {
  if (dashES) { dashES.close(); dashES = null; }
}

function fmtUptimeDash(s) {
  if (!s) return '0s';
  if (s < 60)   return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

function updateDashTiles(m) {
  const sv = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  sv('d-uptime',  fmtUptimeDash(m.uptime_s));
  sv('d-total',   m.total_req   != null ? m.total_req.toLocaleString()         : '—');
  sv('d-active',  m.active_req  != null ? m.active_req                         : '—');
  sv('d-latency', m.avg_latency_ms != null ? m.avg_latency_ms.toFixed(1) + 'ms': '—');
  sv('d-errors',  m.error_rate  != null ? (m.error_rate * 100).toFixed(1) + '%': '—');
  sv('d-rps',     m.throughput_per_s != null ? m.throughput_per_s.toFixed(1) + '/s' : '—');
}

function updateSparkline(rps) {
  sparkData.shift();
  sparkData.push(rps);
  const canvas = document.getElementById('dash-spark');
  if (!canvas) return;
  if (!sparkCtx) sparkCtx = canvas.getContext('2d');
  const ctx = sparkCtx;
  const W   = canvas.clientWidth || canvas.offsetWidth || 300;
  const H   = 72;
  canvas.width  = W;
  canvas.height = H;
  const max  = Math.max(...sparkData, 1);
  const step = W / (sparkData.length - 1);
  ctx.clearRect(0, 0, W, H);
  ctx.beginPath(); ctx.moveTo(0, H);
  sparkData.forEach((v, i) => ctx.lineTo(i * step, H - (v / max) * (H - 6)));
  ctx.lineTo(W, H); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(99,102,241,0.25)');
  grad.addColorStop(1, 'rgba(99,102,241,0.02)');
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  sparkData.forEach((v, i) => {
    const x = i * step, y = H - (v / max) * (H - 6);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#818cf8'; ctx.lineWidth = 1.5; ctx.stroke();
  const lx = (sparkData.length - 1) * step;
  const ly = H - (sparkData[sparkData.length - 1] / max) * (H - 6);
  ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#818cf8'; ctx.fill();
}

function updateResourceBars(m, gpuList) {
  function setBar(barId, valId, pct, text) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    if (bar) bar.style.width = Math.min(100, pct || 0) + '%';
    if (val) val.textContent = text;
  }
  setBar('bar-cpu', 'val-cpu', m.cpu_pct,
    m.cpu_pct != null ? m.cpu_pct.toFixed(1) + '%' : '—');
  const ramPct = m.mem_total_mb > 0 ? (m.mem_used_mb / m.mem_total_mb) * 100 : 0;
  setBar('bar-ram', 'val-ram', ramPct,
    m.mem_used_mb != null
      ? (m.mem_used_mb >= 1024 ? (m.mem_used_mb / 1024).toFixed(1) + ' GB' : m.mem_used_mb + ' MB')
      : '—');
  const g0 = gpuList[0];
  setBar('bar-gpu', 'val-gpu',
    g0 ? (g0.util_pct || 0) : 0,
    g0 && g0.util_pct != null ? g0.util_pct + '%' : '—');

  // GPU device cards — built with safe DOM methods, no innerHTML for untrusted values
  const container = document.getElementById('gpu-device-info');
  if (!container) return;
  while (container.firstChild) container.removeChild(container.firstChild);
  gpuList.forEach(g => {
    const card  = document.createElement('div'); card.className = 'gpu-device';
    const name  = document.createElement('div'); name.className = 'gpu-device-name';
    name.textContent = g.name;
    card.appendChild(name);
    const rows = [
      ['VRAM free', (g.vram_free_mb / 1024).toFixed(1) + ' GB / ' + (g.vram_total_mb / 1024).toFixed(1) + ' GB'],
    ];
    if (g.temp_c   != null) rows.push(['Temperature', g.temp_c   + ' °C']);
    if (g.util_pct != null) rows.push(['Utilisation', g.util_pct + '%']);
    rows.forEach(([label, value]) => {
      const row = document.createElement('div'); row.className = 'gpu-row';
      const l   = document.createElement('span'); l.textContent = label;
      const v   = document.createElement('span'); v.textContent = value;
      row.appendChild(l); row.appendChild(v); card.appendChild(row);
    });
    container.appendChild(card);
  });
}

function updateDownloadsList(downloads) {
  const list  = document.getElementById('dl-tasks-list');
  const empty = document.getElementById('dl-tasks-empty');
  if (!list) return;
  const visible = downloads.slice(0, 8);
  if (!visible.length) {
    if (empty) empty.style.display = 'block';
    list.querySelectorAll('.dl-task').forEach(el => el.remove());
    return;
  }
  if (empty) empty.style.display = 'none';
  const newIds = new Set(visible.map(t => t.id));
  list.querySelectorAll('.dl-task').forEach(el => {
    if (!newIds.has(el.dataset.id)) el.remove();
  });
  visible.forEach(t => {
    const statusLower = t.status.toLowerCase();
    const badgeCls    = statusLower.includes('download') ? 'downloading'
                      : statusLower.includes('complet')  ? 'completed'
                      : statusLower.includes('fail')     ? 'failed' : 'pending';
    const fillCls     = badgeCls === 'completed' ? ' done' : badgeCls === 'failed' ? ' failed' : '';
    const pct         = Math.round((t.progress || 0) * 100);
    const dlMB        = t.downloaded_mb || 0;
    const totStr      = t.total_mb != null ? t.total_mb + ' MB' : '?';

    let el = list.querySelector('.dl-task[data-id="' + t.id + '"]');
    if (!el) {
      el = document.createElement('div');
      el.className  = 'dl-task';
      el.dataset.id = t.id;
      list.prepend(el);
    }
    // Build using safe DOM manipulation — model_name is user-supplied data
    while (el.firstChild) el.removeChild(el.firstChild);

    const header = document.createElement('div'); header.className = 'dl-task-header';
    const nameEl = document.createElement('span'); nameEl.className = 'dl-task-name';
    nameEl.textContent = t.model_name;                      // textContent — XSS safe
    const badge  = document.createElement('span'); badge.className  = 'dl-status-badge ' + badgeCls;
    badge.textContent  = t.status;                          // textContent — XSS safe
    header.appendChild(nameEl); header.appendChild(badge); el.appendChild(header);

    const track = document.createElement('div'); track.className = 'dl-progress-track';
    const fill  = document.createElement('div'); fill.className  = 'dl-progress-fill' + fillCls;
    fill.style.width = pct + '%';
    track.appendChild(fill); el.appendChild(track);

    const meta = document.createElement('div'); meta.className = 'dl-meta';
    const ml   = document.createElement('span'); ml.textContent = dlMB + ' MB / ' + totStr;
    const mr   = document.createElement('span'); mr.textContent = pct + '%';
    meta.appendChild(ml); meta.appendChild(mr); el.appendChild(meta);
  });
}

async function submitDownload() {
  const source   = document.getElementById('dl-source').value;
  const repo     = document.getElementById('dl-repo').value.trim();
  const revision = document.getElementById('dl-revision').value.trim();
  const errEl    = document.getElementById('dl-error');
  const btn      = document.getElementById('dl-btn');
  if (!repo) { showDlError('Repo ID or URL is required.'); return; }
  btn.disabled = true;
  if (errEl) errEl.style.display = 'none';
  const body = {
    model_name:  repo.split('/').pop() || repo,
    source_type: source,
  };
  if (source === 'huggingface') {
    body.repo_id = repo;
    if (revision) body.revision = revision;
  } else {
    body.url = repo;
  }
  try {
    const res = await fetch('/api/models/download', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const t = await res.text();
      showDlError(res.status + ' — ' + t);
    }
  } catch (e) {
    showDlError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

function showDlError(msg) {
  const el = document.getElementById('dl-error');
  if (!el) return;
  el.textContent  = msg;   // textContent — XSS safe
  el.style.display = 'block';
}

/* ── Dashboard tab switching ───────────────────────────── */
function switchDashTab(tab) {
  const overview   = document.getElementById('dash-overview');
  const playground = document.getElementById('dash-playground');
  const tOv        = document.getElementById('dash-tab-overview');
  const tPg        = document.getElementById('dash-tab-playground');
  if (!overview || !playground) return;
  if (tab === 'overview') {
    overview.style.display   = '';     playground.style.display = 'none';
    tOv.classList.add('active');       tPg.classList.remove('active');
  } else {
    overview.style.display   = 'none'; playground.style.display = '';
    tOv.classList.remove('active');    tPg.classList.add('active');
  }
}

/* ── Playground sub-tab switching ──────────────────────── */
const PG_SUBTABS = ['tts', 'classify', 'llm', 'completion'];
function switchPgTab(tab) {
  PG_SUBTABS.forEach(id => {
    const el  = document.getElementById('pg-' + id);
    const btn = document.getElementById('pg-tab-' + id);
    if (el)  el.style.display  = id === tab ? '' : 'none';
    if (btn) btn.classList[id === tab ? 'add' : 'remove']('active');
  });
}

/* ── Playground TTS ─────────────────────────────────────── */
let pgTtsController = null;
async function runPgTTS() {
  const text   = document.getElementById('pg-tts-text').value.trim();
  const engine = document.getElementById('pg-tts-engine').value.trim() || null;
  const voice  = document.getElementById('pg-tts-voice').value.trim()  || null;
  const speed  = parseFloat(document.getElementById('pg-tts-speed').value) || 1.0;
  if (!text) return;
  document.getElementById('pg-tts-btn').disabled = true;
  setBox('pg-tts-status', 'Connecting to /tts/stream…', 'stream');
  pgTtsController = new AbortController();
  const chunks = [];
  try {
    const body = { text, speed };
    if (engine) body.engine = engine;
    if (voice)  body.voice  = voice;
    const resp = await fetch('/tts/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: pgTtsController.signal,
    });
    if (!resp.ok) throw new Error(resp.status + ' — ' + await resp.text());
    const reader = resp.body.getReader();
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); received += value.byteLength;
      setBox('pg-tts-status', 'Received ' + received + ' bytes (' + chunks.length + ' chunks)…', 'stream');
    }
    const total = chunks.reduce((s, c) => s + c.byteLength, 0);
    const pcm   = new Uint8Array(total);
    let off = 0; for (const c of chunks) { pcm.set(c, off); off += c.byteLength; }
    const wav  = buildWav(pcm, 24000, 1);
    const url  = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
    const audio = document.getElementById('pg-tts-audio');
    audio.src  = url;
    document.getElementById('pg-tts-audio-wrap').style.display = 'block';
    audio.play();
    setBox('pg-tts-status', 'Done — ' + total + ' bytes. Playing.', 'ok');
  } catch (e) {
    if (e.name === 'AbortError') setBox('pg-tts-status', 'Stopped.');
    else setBox('pg-tts-status', 'Error: ' + e.message, 'error');
  } finally { document.getElementById('pg-tts-btn').disabled = false; }
}
function stopPgTTS() { pgTtsController && pgTtsController.abort(); }

/* ── Playground Classify ────────────────────────────────── */
let pgImgBase64 = null;
function handlePgDrop(e) {
  e.preventDefault();
  document.getElementById('pg-dropzone').classList.remove('over');
  const f = e.dataTransfer.files[0]; if (f) handlePgFile(f);
}
function handlePgFile(f) {
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const dataUrl = ev.target.result;
    pgImgBase64   = dataUrl.split(',')[1];
    const preview = document.getElementById('pg-img-preview');
    preview.src   = dataUrl; preview.style.display = 'block';
    document.getElementById('pg-cls-btn').disabled = false;
    setBox('pg-cls-out', 'Image loaded — click Classify.');
  };
  reader.readAsDataURL(f);
}
async function runPgClassify() {
  if (!pgImgBase64) return;
  const topk = parseInt(document.getElementById('pg-cls-topk').value) || 5;
  const w    = parseInt(document.getElementById('pg-cls-w').value)    || 224;
  const h    = parseInt(document.getElementById('pg-cls-h').value)    || 224;
  document.getElementById('pg-cls-btn').disabled = true;
  setBox('pg-cls-out', 'Classifying…', 'stream');
  try {
    const res = await fetch('/classify/batch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: [pgImgBase64], top_k: topk, width: w, height: h }),
    });
    const data = await res.json();
    setBox('pg-cls-out', JSON.stringify(data, null, 2), res.ok ? 'ok' : 'error');
  } catch (e) { setBox('pg-cls-out', 'Error: ' + e.message, 'error'); }
  finally { document.getElementById('pg-cls-btn').disabled = false; }
}

/* ── Playground LLM Chat ────────────────────────────────── */
let pgChatHistory = [];
function clearPgChat() {
  pgChatHistory = [];
  const h = document.getElementById('pg-chat-history');
  if (h) while (h.firstChild) h.removeChild(h.firstChild);
}
function appendPgMsg(role, text) {
  const container = document.getElementById('pg-chat-history');
  if (!container) return;
  const div  = document.createElement('div');  div.className  = 'msg ' + (role === 'user' ? 'user' : 'asst');
  const rlEl = document.createElement('div');  rlEl.className = 'msg-role';
  rlEl.textContent = role === 'user' ? 'You' : 'Assistant';
  const body = document.createElement('div');  body.className = 'msg-body';
  body.textContent = text;   // textContent — XSS safe
  div.appendChild(rlEl); div.appendChild(body); container.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}
async function sendPgChat() {
  const input  = document.getElementById('pg-chat-input');
  const text   = input.value.trim(); if (!text) return;
  input.value  = '';
  const model  = document.getElementById('pg-chat-model').value.trim()    || 'default';
  const maxtok = parseInt(document.getElementById('pg-chat-maxtok').value) || 256;
  const temp   = parseFloat(document.getElementById('pg-chat-temp').value) || 0.7;
  pgChatHistory.push({ role: 'user', content: text });
  appendPgMsg('user', text);
  document.getElementById('pg-chat-btn').disabled = true;
  try {
    const res  = await fetch('/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages: pgChatHistory, max_tokens: maxtok, temperature: temp }),
    });
    const data  = await res.json();
    const reply = data.choices?.[0]?.message?.content || JSON.stringify(data);
    pgChatHistory.push({ role: 'assistant', content: reply });
    appendPgMsg('assistant', reply);
  } catch (e) { appendPgMsg('assistant', 'Error: ' + e.message); }
  finally { document.getElementById('pg-chat-btn').disabled = false; }
}

/* ── Playground Completion ──────────────────────────────── */
async function runPgCompletion() {
  const prompt  = document.getElementById('pg-cmp-prompt').value;
  const model   = document.getElementById('pg-cmp-model').value.trim()    || 'default';
  const maxtok  = parseInt(document.getElementById('pg-cmp-maxtok').value) || 128;
  const temp    = parseFloat(document.getElementById('pg-cmp-temp').value) || 0.7;
  const topp    = parseFloat(document.getElementById('pg-cmp-topp').value) || 1.0;
  document.getElementById('pg-cmp-btn').disabled = true;
  setBox('pg-cmp-out', 'Generating…', 'stream');
  try {
    const res  = await fetch('/v1/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, max_tokens: maxtok, temperature: temp, top_p: topp }),
    });
    const data = await res.json();
    const txt  = data.choices?.[0]?.text || JSON.stringify(data, null, 2);
    setBox('pg-cmp-out', txt, res.ok ? 'ok' : 'error');
  } catch (e) { setBox('pg-cmp-out', 'Error: ' + e.message, 'error'); }
  finally { document.getElementById('pg-cmp-btn').disabled = false; }
}
```

- [ ] **Step 3: Manual smoke-test**
  1. Start the server: `cargo run`
  2. Open `http://localhost:8080` in a browser; open DevTools Console
  3. Click "Dashboard" in sidebar — Overview tab appears with `—` tiles; no console errors
  4. After ~3 s, tiles populate with live values; sparkline animates
  5. Switch to "Playground" tab → verify sub-tabs work (TTS / Classify / LLM / Completion)
  6. Click "Status" in the sidebar — SSE closes (no active EventSource in Network tab)
  7. Click "Dashboard" again — SSE reopens

- [ ] **Step 4: Commit**

```bash
git add src/api/playground.html
git commit -m "feat: add Dashboard JS (SSE lifecycle, tiles, sparkline, resource bars, downloads, playground sub-tabs)"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run all tests**

```bash
cargo test 2>&1 | tail -20
```

Expected: `test result: ok` — all prior tests pass plus `dashboard_event_serializes_all_fields`.

- [ ] **Step 2: Release build**

```bash
cargo build --release 2>&1 | grep -E "^error|Finished"
```

Expected: `Finished` — no error lines.

- [ ] **Step 3: End-to-end SSE curl**

```bash
./target/release/torch_inference &
sleep 1
curl -N --max-time 7 http://localhost:8080/dashboard/stream | head -3
kill %1
```

Expected: two `data: {...}` lines containing `metrics`, `gpu`, and `downloads` keys.

- [ ] **Step 4: Commit plan doc**

```bash
git add docs/superpowers/plans/2026-03-29-dashboard.md
git commit -m "docs: add dashboard implementation plan"
```

---

## Self-Review

**Spec coverage:**
- `GET /dashboard/stream` SSE endpoint → Task 1 + 2 ✅
- `src/api/dashboard.rs` → Task 1 ✅
- `src/api/mod.rs` + `src/main.rs` wiring → Task 2 ✅
- Dashboard sidebar nav item → Task 4 ✅
- Overview tab: tiles, sparkline, resource bars, GPU card → Tasks 5 + 7 ✅
- Download manager form + active downloads list → Tasks 5 + 7 ✅
- Playground tab with TTS/Classify/LLM/Completion sub-tabs → Tasks 6 + 7 ✅
- SSE open on navigate-to, close on navigate-away → Task 7 ✅
- Error states: disconnected badge, form inline error → Task 7 ✅
- All existing panels untouched → additive changes only ✅

**Type consistency:**
- `DashboardMetrics.throughput_per_s` → used in handler (Task 1) and `updateSparkline` / `updateDashTiles` (Task 7) ✅
- `DashboardDownload.downloaded_mb` / `total_mb` → handler (Task 1) and `updateDownloadsList` (Task 7) ✅
- `DashboardGpu.vram_free_mb` / `vram_total_mb` → handler (Task 1) and `updateResourceBars` (Task 7) ✅
- All DOM IDs referenced in JS declared in Tasks 4–6 HTML ✅

**Security:**
- User-supplied strings (`model_name`, `status`) rendered with `textContent` only — XSS safe ✅
- GPU device cards built with safe DOM methods — no untrusted `innerHTML` ✅
- `escHtml` helper available but DOM-based rendering preferred ✅
