# Playground UX Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `src/api/playground.html` — rename nav labels, move Live TTS WebSocket pane into the TTS panel, keep Live STT in the STT panel with no sub-tabs, and fix the STT AudioContext sample rate to guarantee 16kHz mono f32le.

**Architecture:** Single-file HTML change only. No backend changes. The WebSocket connection controls (Connect button, status dot/label) must appear in both the TTS panel and the STT panel since either page can initiate a connection. Live TTS pane moves out of `panel-audio` into `panel-tts`. After the move, `panel-audio` shows the Live STT pane directly with no sub-tab strip.

**Tech Stack:** Plain HTML/CSS/JS inside `src/api/playground.html` (embedded via `include_str!` in Rust). Build: `cargo build --release`.

---

### Task 1: Rename sidebar nav labels

**Files:**
- Modify: `src/api/playground.html` lines 633–644

- [ ] **Step 1: Change "TTS Stream" → "TTS", "Audio" → "STT", "Detection" → "Detect"**

Find these three nav buttons in the sidebar (lines ~633–644) and apply the label changes:

```html
<!-- Before -->
<button class="nav-item" onclick="show('tts',this)">
  <i class="nav-icon ri-music-2-line"></i> TTS Stream
</button>
<button class="nav-item" onclick="show('audio',this)">
  <i class="nav-icon ri-mic-line"></i> Audio
</button>
<button class="nav-item" onclick="show('detect',this)">
  <i class="nav-icon ri-focus-3-line"></i> Detection
</button>

<!-- After -->
<button class="nav-item" onclick="show('tts',this)">
  <i class="nav-icon ri-music-2-line"></i> TTS
</button>
<button class="nav-item" onclick="show('audio',this)">
  <i class="nav-icon ri-mic-line"></i> STT
</button>
<button class="nav-item" onclick="show('detect',this)">
  <i class="nav-icon ri-focus-3-line"></i> Detect
</button>
```

- [ ] **Step 2: Update `panel-tts` header title**

In `panel-tts` header (line ~695), change:
```
TTS Stream  →  TTS
```

- [ ] **Step 3: Update `panel-audio` header title and description**

In `panel-audio` header (lines ~903–907), change:
- Title: `Audio Transcription` → `STT`
- Desc: append `· live stream via GET /audio/ws`

- [ ] **Step 4: Build and verify**

```bash
cargo build --release 2>&1 | grep -E "^error"
```
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/api/playground.html
git commit -m "feat(playground): rename nav labels TTS Stream->TTS, Audio->STT, Detection->Detect"
```

---

### Task 2: Move Live TTS pane to `panel-tts`, Live STT directly in `panel-audio`

The current "WebSocket Audio Stream" card (lines ~968–1051) lives inside `panel-audio` and contains a sub-tab strip with both `ws-pane-tts` and `ws-pane-stt`. We will:
1. Add a new WebSocket card inside `panel-tts` with only the connection controls + `ws-pane-tts` content.
2. Replace the full WebSocket card in `panel-audio` with a simpler card containing only the connection controls + `ws-pane-stt` content (no sub-tab strip).

**Files:**
- Modify: `src/api/playground.html` — end of `panel-tts` (before line ~749 `</section>`), and `panel-audio` WebSocket block (lines ~968–1051)

- [ ] **Step 1: Insert Live TTS card at end of `panel-tts`** (before its closing `</section>`)

```html
      <!-- Live TTS WebSocket -->
      <div class="card" style="margin-top:16px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
          <div>
            <div class="card-title" style="margin-bottom:2px"><i class="ri-wifi-line" style="color:var(--accent)"></i> Live TTS Stream</div>
            <div style="font-size:12px;color:var(--text-muted)">Low-latency duplex audio via <span class="ep-code">GET /audio/ws</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="ws-status-dot-tts" style="width:9px;height:9px;border-radius:50%;background:#555;display:inline-block"></span>
            <span id="ws-status-label-tts" style="font-size:12px;color:var(--text-muted)">disconnected</span>
            <button class="btn btn-ghost" id="ws-connect-btn-tts" onclick="wsAudioToggle()" style="padding:4px 12px;font-size:12px">Connect</button>
          </div>
        </div>
        <div id="ws-pane-tts">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <div class="field">
                <label>Text to speak</label>
                <textarea id="ws-tts-text" rows="4" style="width:100%;resize:vertical;font-size:13px;padding:8px;background:var(--input-bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:inherit" placeholder="Enter text..."></textarea>
              </div>
              <div class="row" style="margin-top:10px">
                <div class="field">
                  <label>Voice / Engine</label>
                  <input id="ws-tts-voice" type="text" placeholder="af_heart (or blank for default)" style="width:100%;padding:6px 8px;background:var(--input-bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px" />
                </div>
                <div class="field" style="flex:0 0 100px">
                  <label>Speed</label>
                  <input id="ws-tts-speed" type="number" value="1.0" min="0.25" max="4.0" step="0.05" style="width:100%;padding:6px 8px;background:var(--input-bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px" />
                </div>
              </div>
              <div style="display:flex;gap:8px;margin-top:12px">
                <button class="btn btn-primary" id="ws-tts-btn" onclick="wsTtsSpeak()"><i class="ri-play-circle-line"></i> Speak</button>
                <button class="btn btn-ghost" id="ws-tts-stop-btn" onclick="wsTtsStop()" disabled><i class="ri-stop-circle-line"></i> Stop</button>
              </div>
            </div>
            <div>
              <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-bottom:6px">Waveform</div>
              <canvas id="ws-tts-canvas" width="320" height="80" style="width:100%;border-radius:6px;background:#0d1117"></canvas>
              <div id="ws-tts-status" style="font-size:12px;color:var(--text-muted);margin-top:6px">-</div>
              <div id="ws-tts-metrics" style="font-size:11px;color:var(--text-dim);margin-top:4px;display:none">
                <span id="ws-tts-dur">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
```

- [ ] **Step 2: Replace the entire WebSocket Audio Stream card in `panel-audio`** (lines ~968–1051, the block starting with `<div class="card" style="margin-top:16px">` through its closing `</div>`) with this simpler card (no sub-tabs, STT only):

```html
      <!-- Live STT WebSocket -->
      <div class="card" style="margin-top:16px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
          <div>
            <div class="card-title" style="margin-bottom:2px"><i class="ri-wifi-line" style="color:var(--accent)"></i> Live STT Stream</div>
            <div style="font-size:12px;color:var(--text-muted)">Low-latency duplex audio via <span class="ep-code">GET /audio/ws</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <span id="ws-status-dot" style="width:9px;height:9px;border-radius:50%;background:#555;display:inline-block"></span>
            <span id="ws-status-label" style="font-size:12px;color:var(--text-muted)">disconnected</span>
            <button class="btn btn-ghost" id="ws-connect-btn" onclick="wsAudioToggle()" style="padding:4px 12px;font-size:12px">Connect</button>
          </div>
        </div>
        <div id="ws-pane-stt">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
            <span id="stt-model-badge" style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:2px 8px;border-radius:12px;background:var(--border);color:var(--text-muted)">
              <span id="stt-model-dot" style="width:7px;height:7px;border-radius:50%;background:#555;display:inline-block"></span>
              <span id="stt-model-label">Checking STT model...</span>
            </span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <div style="font-size:13px;color:var(--text-muted);margin-bottom:12px">
                Press <strong>Record</strong> to stream microphone audio to the server for real-time transcription.
              </div>
              <div style="display:flex;gap:8px;align-items:center">
                <button class="btn btn-primary" id="ws-stt-btn" onclick="wsSttToggle()">
                  <i class="ri-mic-line"></i> Record
                </button>
                <span id="ws-stt-rec-dot" style="width:9px;height:9px;border-radius:50%;background:#555;display:inline-block"></span>
                <span id="ws-stt-rec-label" style="font-size:12px;color:var(--text-muted)">idle</span>
              </div>
              <div style="font-size:12px;color:var(--text-muted);margin-top:10px">Sends raw PCM f32le @16kHz mono.</div>
            </div>
            <div>
              <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-bottom:6px">Mic waveform</div>
              <canvas id="ws-stt-canvas" width="320" height="80" style="width:100%;border-radius:6px;background:#0d1117"></canvas>
              <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-top:10px;margin-bottom:4px">Transcript</div>
              <div class="response-box" id="ws-stt-result" style="min-height:60px">Transcript appears here.</div>
            </div>
          </div>
        </div>
      </div>
```

- [ ] **Step 3: Build and verify**

```bash
cargo build --release 2>&1 | grep -E "^error"
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add src/api/playground.html
git commit -m "feat(playground): move Live TTS pane to TTS panel, Live STT directly in STT panel"
```

---

### Task 3: Remove `wsTab()`, sync WebSocket status across both panels

The `wsTab()` function and sub-tab logic are no longer needed. Connection status must now update both panels (TTS has `ws-status-dot-tts`/`ws-status-label-tts`/`ws-connect-btn-tts`; STT has the original IDs).

**Files:**
- Modify: `src/api/playground.html` — JS block around line 3075 (`wsTab`), and `wsAudioConnect()` (~line 3122)

- [ ] **Step 1: Delete `wsTab()` function** (~lines 3075–3081)

Remove:
```js
function wsTab(t) {
  document.getElementById('ws-pane-tts').style.display = t === 'tts' ? '' : 'none';
  document.getElementById('ws-pane-stt').style.display = t === 'stt' ? '' : 'none';
  document.querySelectorAll('#panel-audio .ws-tab').forEach(b => b.classList.remove('active-ws-tab'));
  document.getElementById('ws-tab-' + t).classList.add('active-ws-tab');
  if (t === 'stt') sttCheckModel();
}
```

- [ ] **Step 2: Add `wsSetStatus()` helper just before `wsAudioToggle()`**

```js
function wsSetStatus(state) {
  const color   = state === 'connected' ? '#2EA043' : state === 'connecting' ? '#E3A228' : '#555';
  const text    = state === 'connected' ? 'connected' : state === 'connecting' ? 'connecting...' : 'disconnected';
  const btnText = state === 'connected' ? 'Disconnect' : 'Connect';
  ['ws-status-dot', 'ws-status-dot-tts'].forEach(id => {
    const el = document.getElementById(id); if (el) el.style.background = color;
  });
  ['ws-status-label', 'ws-status-label-tts'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = text;
  });
  ['ws-connect-btn', 'ws-connect-btn-tts'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = btnText;
  });
}
```

- [ ] **Step 3: Replace inline status-dot/label/button writes in `wsAudioConnect()` with `wsSetStatus()` calls**

In `wsAudioConnect()`, find the three update patterns and replace:

Before `new WebSocket(...)` call — replace lines like:
```js
document.getElementById('ws-status-dot').style.background = '#E3A228';
document.getElementById('ws-status-label').textContent = 'connecting...';
document.getElementById('ws-connect-btn').textContent = 'Disconnect';
```
With: `wsSetStatus('connecting');`

In `wsAudio.onopen` handler — replace similar lines with: `wsSetStatus('connected');`

In `wsAudio.onclose` and `wsAudio.onerror` handlers — replace similar lines with: `wsSetStatus('disconnected');`

- [ ] **Step 4: Call `sttCheckModel()` when STT panel is shown**

Find the `show(id, btn)` function. Add at the end of its body:
```js
  if (id === 'audio') sttCheckModel();
```

- [ ] **Step 5: Build and verify**

```bash
cargo build --release 2>&1 | grep -E "^error"
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/api/playground.html
git commit -m "refactor(playground): remove wsTab(), sync WS status across TTS+STT panels"
```

---

### Task 4: Fix STT AudioContext sample rate (resample to 16kHz)

`wsSttStart()` creates `new AudioContext({ sampleRate: 16000 })` but browsers may ignore this hint. If `ctx.sampleRate !== 16000` we resample using linear interpolation.

**Files:**
- Modify: `src/api/playground.html` — `wsSttStart()` function (lines ~3263–3304)

- [ ] **Step 1: Replace the body of `wsSttStart()` with the version below**

Key changes:
- `getUserMedia` no longer requests `sampleRate` (that constraint can cause failure on some browsers)
- After creating `AudioContext`, capture `ctx.sampleRate` as `deviceRate`
- In `onaudioprocess`, if `deviceRate !== 16000` resample via linear interpolation before sending

```js
async function wsSttStart() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    document.getElementById('ws-stt-result').textContent = 'Microphone access requires a secure context (https:// or localhost).';
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
  } catch (err) {
    document.getElementById('ws-stt-result').textContent = 'Microphone error: ' + err.message;
    return;
  }
  wsMicStream = stream;

  const TARGET_RATE = 16000;
  // sampleRate hint helps Chrome; actual rate confirmed via ctx.sampleRate after creation
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_RATE });
  const deviceRate = ctx.sampleRate;

  const src = ctx.createMediaStreamSource(stream);
  wsAnalyserStt = ctx.createAnalyser();
  wsAnalyserStt.fftSize = 512;
  src.connect(wsAnalyserStt);

  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (e) => {
    if (!wsRecording) return;
    const raw = e.inputBuffer.getChannelData(0); // Float32Array at deviceRate
    let pcm;
    if (deviceRate === TARGET_RATE) {
      pcm = raw.buffer.slice(0);
    } else {
      // Linear interpolation: deviceRate -> 16000
      const ratio = deviceRate / TARGET_RATE;
      const outLen = Math.round(raw.length / ratio);
      const out = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const pos  = i * ratio;
        const lo   = Math.floor(pos);
        const hi   = Math.min(lo + 1, raw.length - 1);
        const frac = pos - lo;
        out[i] = raw[lo] * (1 - frac) + raw[hi] * frac;
      }
      pcm = out.buffer;
    }
    wsAudio.send(pcm);
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  wsMicProc = { proc, ctx };

  wsAudio.send(JSON.stringify({ type: 'stt_begin', sample_rate: TARGET_RATE }));
  wsRecording = true;

  const dot   = document.getElementById('ws-stt-rec-dot');
  const label = document.getElementById('ws-stt-rec-label');
  dot.style.background = '#E74C3C';
  label.textContent    = 'recording...';
  const sttBtn = document.getElementById('ws-stt-btn');
  sttBtn.textContent = '';
  const stopIcon = document.createElement('i');
  stopIcon.className = 'ri-stop-circle-line';
  sttBtn.appendChild(stopIcon);
  sttBtn.appendChild(document.createTextNode(' Stop'));

  wsDrawLoop('ws-stt-canvas', wsAnalyserStt);
}
```

Note: The button text update uses DOM methods instead of setting markup directly to avoid the security hook false-positive on safe literal strings.

- [ ] **Step 2: Build and verify**

```bash
cargo build --release 2>&1 | grep -E "^error"
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add src/api/playground.html
git commit -m "fix(playground): resample STT mic audio to 16kHz when device rate differs"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| Rename "TTS Stream" → "TTS" | Task 1 |
| Rename "Audio" → "STT" | Task 1 |
| Rename "Detection" → "Detect" | Task 1 |
| Move Live TTS pane to TTS panel | Task 2 |
| Live STT directly in STT panel (no sub-tabs) | Task 2 |
| Connection controls present in both panels | Task 2 + Task 3 |
| Remove sub-tab strip from audio panel | Task 2 |
| Fix STT sample rate / resample | Task 4 |
| No UI redesign — component moves only | All tasks |
| No backend changes | All tasks |
