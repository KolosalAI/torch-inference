# Dashboard Panel — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## Overview

Add a new **Dashboard** panel to the existing Kolosal Inference playground (`src/api/playground.html`). The panel provides real-time inference metrics pushed via Server-Sent Events (SSE) and a model download manager with live progress tracking. The existing Status, TTS, Classify, LLM Chat, Completion, and Endpoints panels are preserved unchanged.

---

## Scope

### In scope
- New `GET /dashboard/stream` SSE endpoint in Rust
- New `src/api/dashboard.rs` handler module
- New "Dashboard" sidebar nav item in `playground.html`
- Dashboard panel with two tabs: **Overview** and **Playground**
- Overview tab: stat tiles, sparkline chart, resource bars, GPU info, download manager
- Playground tab: sub-tabs for TTS Stream, Classify, LLM Chat, Completion (reuses existing panel HTML)
- Wiring `dashboard` route in `main.rs` and `src/api/mod.rs`

### Out of scope
- Changes to existing panels (Status, TTS, Classify, LLM, Completion, Endpoints)
- New download endpoints (reuses existing `POST /api/models/download`)
- WebSocket support
- Authentication on the SSE endpoint
- Persistent metrics history (no database)

---

## Architecture

### Backend — `src/api/dashboard.rs`

Single new handler: `GET /dashboard/stream`

**Response:** `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`

**Loop (every 3 seconds):**
1. Call `monitor.get_health_status()` → uptime, active_requests, total_requests, avg_latency_ms, error_rate
2. Call `sysinfo::System` (already used in `performance.rs`) → cpu_pct, mem_used_mb, mem_total_mb
3. Call `gpu_manager.get_info()` → GPU device list (name, util_pct, temp_c, vram_free_mb, vram_total_mb)
4. Call `download_manager.list_tasks()` → active/recent download tasks with progress
5. Serialize combined `DashboardEvent` struct and write `data: <json>\n\n` to SSE stream
6. Sleep 3 seconds; exit loop on sender drop (client disconnect)

**`DashboardEvent` shape:**
```json
{
  "metrics": {
    "uptime_s": 7284,
    "active_req": 3,
    "total_req": 1204,
    "avg_latency_ms": 42.1,
    "error_rate": 0.0,
    "throughput_per_s": 18.0,
    "cpu_pct": 34.2,
    "mem_used_mb": 2148,
    "mem_total_mb": 16384
  },
  "gpu": [
    {
      "name": "RTX 4090",
      "util_pct": 22,
      "temp_c": 62,
      "vram_free_mb": 18432,
      "vram_total_mb": 24576
    }
  ],
  "downloads": [
    {
      "id": "uuid",
      "model_name": "meta-llama/Llama-3.2-1B",
      "status": "Downloading",
      "progress": 0.62,
      "downloaded_mb": 258,
      "total_mb": 415
    }
  ]
}
```

**Dependencies used:** `actix-web` (already present), `sysinfo` (already present), `serde_json` (already present). No new crates required.

**Handler signature:**
```rust
pub async fn dashboard_stream(
    monitor: web::Data<Arc<Monitor>>,
    system_state: web::Data<crate::api::system::SystemInfoState>,  // contains gpu_manager
    download_state: web::Data<ModelDownloadState>,
) -> impl Responder
```

---

### Frontend — `src/api/playground.html`

#### Sidebar changes
Add one new nav item between "Completion" and the Reference section:
```html
<button class="nav-item" onclick="show('dashboard', this)">
  <span class="nav-icon">⬡</span> Dashboard
</button>
```
All existing nav items (Status, TTS Stream, Classify, LLM Chat, Completion, Endpoints) remain.

#### New panel: `#panel-dashboard`

Top-level tabs at panel header level: **Overview** and **Playground**.

**Overview tab contains:**

1. **Stat tiles row** (6 tiles, CSS grid): Uptime, Total Req, Active, Avg Latency, Error Rate, Throughput/s
2. **Two-column row:**
   - Left: `<canvas>` sparkline — 60-point rolling window, client-side ring buffer, drawn on each SSE event
   - Right: Resource bars (CPU, RAM, GPU) + GPU device info card (name, VRAM, temp, utilisation)
3. **Model Download Manager card:**
   - Form: Source dropdown (HuggingFace / URL), Repo ID / URL input, Revision input, Download button
   - On submit: `POST /api/models/download` with JSON body
   - Active downloads list: rendered from `event.downloads`, shows name, progress bar, bytes, percentage
   - Completed downloads show green bar; failed show red with error text

**Playground tab contains:**

Sub-tab bar: TTS Stream | Classify | LLM Chat | Completion

Each sub-tab renders the **identical HTML content** as the corresponding existing standalone panel (copy, not reference). The existing standalone panels in the sidebar continue to work unchanged.

#### SSE connection lifecycle

```js
let dashboardES = null;

function openDashboardStream() {
  if (dashboardES) return;
  dashboardES = new EventSource('/dashboard/stream');
  dashboardES.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    updateTiles(ev.metrics);
    updateSparkline(ev.metrics.throughput_per_s);
    updateResourceBars(ev.metrics, ev.gpu);
    updateDownloads(ev.downloads);
  };
  dashboardES.onerror = () => {
    // show disconnected state in UI, retry is automatic via EventSource
  };
}

function closeDashboardStream() {
  if (dashboardES) { dashboardES.close(); dashboardES = null; }
}
```

- `openDashboardStream()` called when Dashboard nav item is clicked
- `closeDashboardStream()` called when any other nav item is clicked (patched into `show()`)

---

## Routing

In `main.rs` / `src/api/mod.rs`, add:
```rust
.route("/dashboard/stream", web::get().to(dashboard::dashboard_stream))
```

`ModelDownloadState`, `Monitor`, and `SystemInfoState` (which wraps `GpuManager`) are already registered as `web::Data` in `main.rs`.

---

## Error Handling

- If `gpu_manager.get_info()` fails: emit `gpu: []` — no crash, no retry noise
- If `download_manager.list_tasks()` fails: emit `downloads: []`
- If client disconnects mid-stream: sender drop exits loop cleanly
- Download form submit errors: shown inline below the form in red text
- SSE `onerror`: stat tiles show stale values with a "disconnected" badge; `EventSource` retries automatically

---

## Testing

- Unit test `DashboardEvent` serialization (all fields present, correct types)
- Integration test: `GET /dashboard/stream` returns `200`, `text/event-stream` content-type, and at least one well-formed `data:` frame within 5 seconds
- Manual: open Dashboard panel, verify tiles update every ~3s, verify download progress bar advances during an active download

---

## Files Changed

| File | Change |
|------|--------|
| `src/api/dashboard.rs` | New — SSE handler |
| `src/api/mod.rs` | Add `pub mod dashboard;` |
| `src/main.rs` | Register `/dashboard/stream` route |
| `src/api/playground.html` | Add Dashboard nav item + panel |
