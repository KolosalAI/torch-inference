# UI–Backend Wiring: Full Audit & New Tabs

**Date:** 2026-04-05  
**Status:** Approved  
**Scope:** Fix the one confirmed bug, wire all orphaned endpoints into the UI, add three new sidebar tabs

---

## Overview

`src/api/playground.html` is the single embedded HTML/JS UI served at `GET /`. It has 8 sidebar tabs today. The backend exposes 46 API endpoints; only 12 are called by the UI. This spec covers:

1. **Bug fix** — classify request field name mismatch
2. **TTS enhancement** — dynamic voice selector per engine
3. **Models enhancements** — delete button + SOTA sub-tab
4. **New Audio tab** — transcription via multipart upload
5. **New Logs tab** — log file browser + inline viewer + clear
6. **New System tab** — system info, performance metrics, server config

All changes are confined to `src/api/playground.html` (the single `include_str!` file). No Rust changes required.

---

## 1. Bug Fix — Classify Field Names

**File:** `src/api/playground.html`  
**Location:** `runClassify()` function, inside the `POST /classify/batch` request body

**Current (broken):**
```js
{ images: [b64], top_k: topK, width: w, height: h }
```

**Fixed:**
```js
{ images: [b64], top_k: topK, model_width: w, model_height: h }
```

The backend handler (`classify::batch_classify`) expects `model_width` and `model_height`. This has been silently sending wrong field names, causing the server to use default dimensions regardless of user input.

---

## 2. TTS Tab — Dynamic Voice Selector

**Endpoints used:**
- `GET /tts/engines/{engine_id}/voices` → `{voices: [{id, name, language, gender, quality}], total}`

**Behaviour:**
- Replace the free-text voice input with a `<select>` dropdown
- On page load and on engine change, call `GET /tts/engines/{id}/voices`
- Populate select with `{name} ({language})` as label, `{id}` as value
- Show "Loading voices…" while fetching; show "No voices available" if empty or error
- The selected voice id is passed as the `voice` field in `POST /tts/stream`

---

## 3. Models Tab Enhancements

### 3a. Delete Model Button

**Endpoint:** `DELETE /models/download/{model_name}`  
**Response:** `200 OK` on success, `404` if not found

**Behaviour:**
- Each row in the "Downloaded" sub-tab gets a "🗑 Delete" button (red outline, right-aligned)
- On click: show `window.confirm("Delete model '{name}'? This cannot be undone.")` 
- On confirm: call `DELETE /models/download/{name}`, show inline "Deleting…" spinner
- On success: remove the row from the list and refresh the model selects in TTS/Classify/LLM tabs
- On error: show red error text inline next to the button

### 3b. SOTA Models Sub-Tab

**Endpoints:**
- `GET /models/sota` → `{models: [{id, name, architecture, task, size_estimate, url, ...}], total}`
- `POST /models/sota/{model_id}` → `{task_id, status, message}`

**Behaviour:**
- New fourth sub-tab "SOTA ★" in the Models tab (alongside Available, Downloaded, Download)
- Shows a grid of model cards identical in style to the "Available" tab
- Each card: name, task badge, architecture, size estimate, "Download" button
- "Download" calls `POST /models/sota/{model_id}`, then switches to the "Download" sub-tab and starts polling the task (same `pollMdlDownload` flow)
- If `GET /models/sota` returns an empty list, show a "No SOTA models found" placeholder

---

## 4. New Audio Tab

**Sidebar position:** Between "Completion" and "Dashboard"  
**Icon:** 🎙 (microphone)

**Endpoint:** `POST /audio/transcribe` (multipart/form-data)  
Fields: `audio` (file), `model` (string, default "default"), `timestamps` (bool string "true"/"false")  
Response: `{text, language, confidence, segments?: [{text, start, end, confidence}]}`

**Health indicator:** `GET /audio/health` called on tab open; shows "● Audio healthy" or "● Audio unavailable" badge in panel header

### Layout

Single panel with two columns:

**Left — input:**
- Dropzone (drag-and-drop + click-to-browse), accept `audio/*`
- Shows file name and duration hint after selection
- Model select (populated from `/api/models/downloaded`, filtered to `task == "speech-to-text"` or shows "default")
- Checkbox: "Include timestamps"
- "Transcribe" button (disabled until file selected)
- "Stop" button (AbortController, same pattern as TTS)

**Right — output:**
- Transcribed text in a scrollable box
- Language and confidence shown below (`Language: en | Confidence: 94.2%`)
- If timestamps enabled: segment list below text — each segment shows `[start–end]` badge + text
- If no result yet: placeholder text "Transcription will appear here"

**Error handling:** Show red error box for 400 (invalid audio), 404 (model not found), 500 (transcription failed)

---

## 5. New Logs Tab

**Sidebar position:** Between "Models" and "Endpoints"  
**Icon:** 📋 (clipboard)

**Endpoints:**
- `GET /logs` → `{available_log_files: [{name, path, size_bytes, size_mb, line_count, modified}], log_directory, log_level, total_log_size_mb}`
- `GET /logs/{log_file}` → plain text (log file content)
- `DELETE /logs/{log_file}` → `200 OK`

### Layout

Two-panel split:

**Left — file list (220px fixed):**
- Called on tab open; shows file name, size, line count, last modified
- "View" button: loads file content into right panel, highlights selected file
- "✕ Clear" button (red): `window.confirm("Clear '{name}'?")` → `DELETE /logs/{name}` → remove from list
- "Refresh" button at top of panel
- If log directory doesn't exist or is empty: "No log files found"

**Right — viewer:**
- Dark monospace code area (matches server log theme: dark bg, coloured log levels)
- Header: file name + "last N lines" note
- Content from `GET /logs/{file}` — displayed as-is in `<pre>`
- ANSI colour codes stripped (regex replace) so terminal output renders cleanly
- "Copy" button top-right of viewer panel
- If no file selected: placeholder "Select a file to view"

---

## 6. New System Tab

**Sidebar position:** Between "Logs" and "Endpoints"  
**Icon:** ⚙️

**Three sub-tabs:** Info | Performance | Config

### 6a. Info Sub-Tab (default)

**Endpoints:**
- `GET /system/info` → `{hostname, os, arch, cpu_count, total_memory_mb, available_memory_mb, rust_version, server_version, uptime_seconds, ...}`
- `GET /system/gpu/stats` → `[{name, util_pct, temp_c, vram_free_mb, vram_total_mb}, ...]`

Layout: 2×2 card grid (System, Server, GPU, Runtime). Each card shows key/value pairs. "Refresh" button top-right. GPU card shows "No GPU detected" gracefully if empty array.

### 6b. Performance Sub-Tab

**Endpoints:**
- `GET /performance` → `{avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, total_requests, requests_per_second, error_rate, ...}`
- `GET /performance/optimize` → `{suggestions: [{category, description, impact, action}], ...}`

Layout:
- Stat row: Avg / P50 / P95 / P99 latency, throughput, error rate (same tile style as Dashboard)
- "Get Optimization Tips" button → calls `GET /performance/optimize`, shows suggestion cards below (category badge, description, impact badge)
- "Refresh Metrics" button

### 6c. Config Sub-Tab

**Endpoint:** `GET /system/config` → JSON object (server configuration)

Layout:
- Formatted JSON in a `<pre>` code block (syntax-highlighted via simple regex: strings in green, numbers in blue, keys in white)
- "Copy Config" button
- Called on sub-tab open

---

## Data Flow

```
playground.html
  ├── Status tab        → GET /health (10s poll)
  ├── TTS tab           → GET /tts/engines, GET /tts/engines/{id}/voices, POST /tts/stream
  ├── Classify tab      → POST /classify/batch [FIXED], GET /api/models/downloaded
  ├── LLM tab           → POST /v1/chat/completions, GET /api/models/downloaded
  ├── Completion tab    → POST /v1/completions, GET /api/models/downloaded
  ├── Audio tab [NEW]   → GET /audio/health, POST /audio/transcribe
  ├── Dashboard tab     → GET /dashboard/stream (SSE), POST /api/models/download, GET /api/models/download/{id}
  ├── Models tab        → GET /api/models, GET /api/models/downloaded, DELETE /models/download/{name}
  │                       GET /models/sota, POST /models/sota/{model_id}
  ├── Logs tab [NEW]    → GET /logs, GET /logs/{file}, DELETE /logs/{file}
  ├── System tab [NEW]  → GET /system/info, GET /system/gpu/stats, GET /performance,
  │                       GET /performance/optimize, GET /system/config
  └── Endpoints tab     → (static, no API calls)
```

---

## Implementation Notes

- All changes are in `src/api/playground.html` only — no Rust modifications needed
- New tabs follow the exact same DOM/CSS pattern as existing tabs: `<div id="panel-X" class="panel">` with sidebar `<div class="nav-item" onclick="showPanel('X')">`
- New JS functions follow existing naming: `runAudio()`, `loadLogs()`, `viewLog(name)`, `clearLog(name)`, `loadSystem()`, etc.
- `refreshModelSelects()` extended to also refresh the Audio tab's STT model dropdown
- `DELETE /models/download/{name}` success path calls `refreshModelSelects()` and `loadDownloadedModels()` to keep all views in sync
- The playground.html file will grow from ~32KB to approximately ~55KB — still a single embedded file, no bundler required

---

## Testing Criteria

- Classify sends `model_width`/`model_height` (verify in browser DevTools Network tab)
- TTS voice dropdown populates after engine selection; selecting a voice passes its id to `/tts/stream`
- Delete model: confirm dialog appears, model removed from all dropdowns on success
- SOTA tab: model cards load, Download button starts a tracked task
- Audio tab: file upload + transcribe returns text; timestamps toggle shows/hides segment list
- Logs tab: file list loads, View shows content, Clear removes file with confirm
- System Info: cards populate from `/system/info`; GPU card shows gracefully if no GPU
- Performance: metrics load; "Get Optimization Tips" shows suggestion cards
- Config: JSON renders formatted; Copy button works
