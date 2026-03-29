# Kolosal Vanilla UI Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `src/api/playground.html` from a custom dark indigo theme to the Kolosal vanilla light design system using Remix Icons.

**Architecture:** Single-file edit. The existing CSS uses custom properties throughout, so updating `:root` handles ~80% of the color swap. Remaining hardcoded `rgba()` colors and component-specific CSS blocks (buttons, badges) need explicit rewrites. HTML body requires icon class swaps and 3 inline color fixes.

**Tech Stack:** HTML/CSS/JS — no build step. Remix Icons v4.7.0 via CDN. Google Fonts: Inter + Geist Mono.

**Spec:** `docs/superpowers/specs/2026-03-29-kolosal-ui-rework-design.md`

**Visual verification:** Open `src/api/playground.html` directly in a browser (no server needed for layout/style review — JS API calls will fail but layout renders fine).

---

### Task 1: Swap font imports and add Remix Icons CDN

**Files:**
- Modify: `src/api/playground.html` lines 8–10

- [ ] **Step 1: Replace the font `<link>` and add Remix Icons in `<head>`**

Find this block (lines 8–10):
```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
```

Replace with:
```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet" />
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css" rel="stylesheet" />
```

- [ ] **Step 2: Verify**

Open `src/api/playground.html` in a browser. Open DevTools → Network tab. Confirm two font requests: `fonts.googleapis.com` (Inter + Geist Mono) and `cdn.jsdelivr.net` (remixicon).

---

### Task 2: Replace CSS `:root` custom properties with Kolosal tokens

**Files:**
- Modify: `src/api/playground.html` lines 14–33 (the `:root { }` block inside `<style>`)

- [ ] **Step 1: Replace the `:root` block**

Find:
```css
  :root {
    --bg:        #08080f;
    --surface:   #0e0e1a;
    --surface2:  #13131f;
    --border:    #1f1f32;
    --border2:   #2a2a42;
    --accent:    #6366f1;
    --accent-hi: #818cf8;
    --accent-lo: #4f46e5;
    --accent-bg: rgba(99,102,241,.08);
    --green:     #22c55e;
    --red:       #ef4444;
    --yellow:    #f59e0b;
    --text:      #e2e8f0;
    --text-muted:#8892a4;
    --text-dim:  #4b5568;
    --radius:    10px;
    --font:      'Inter', system-ui, sans-serif;
    --mono:      'JetBrains Mono', monospace;
  }
```

Replace with:
```css
  :root {
    --bg:        #F8F9F9;
    --surface:   #FFFFFF;
    --surface2:  #FFFFFF;
    --border:    #DDE1E3;
    --border2:   #E4E7E9;
    --accent:    #0066F5;
    --accent-hi: #3C8AF7;
    --accent-lo: #003D93;
    --accent-bg: #F0F6FE;
    --green:     #3ABC3F;
    --green-bg:  #F3FBF4;
    --green-text:#2E9632;
    --red:       #FF3131;
    --red-bg:    #FFF3F3;
    --red-text:  #CC2727;
    --yellow:    #FFA931;
    --yellow-bg: #FFFAF3;
    --yellow-text:#CC8727;
    --text:      #0D0E0F;
    --text-muted:#6A6F73;
    --text-dim:  #9C9FA1;
    --radius:    10px;
    --font:      'Inter', system-ui, sans-serif;
    --mono:      'Geist Mono', monospace;
  }
```

- [ ] **Step 2: Verify**

Open the file in a browser. The page should now be light: white sidebar, grey-50 page background, dark text. Most elements should already look correct from the var swap alone.

---

### Task 3: Rewrite button CSS

**Files:**
- Modify: `src/api/playground.html` — the button section inside `<style>` (lines ~148–165)

The existing `.btn-primary` uses `var(--accent)` (now blue). Per Kolosal design, primary buttons are **black** with a gradient hover. `.btn-ghost` becomes an outline-style button.

- [ ] **Step 1: Replace button CSS block**

Find:
```css
  /* Buttons */
  .btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 16px; border-radius: 7px; border: none; cursor: pointer;
    font-family: var(--font); font-size: 13px; font-weight: 500;
    transition: opacity .15s, transform .08s;
  }
  .btn:active { transform: scale(.97); }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--accent-hi); }
  .btn-ghost { background: var(--surface2); color: var(--text-muted); border: 1px solid var(--border2); }
  .btn-ghost:hover:not(:disabled) { color: var(--text); }
  .spinner {
    display: inline-block; width: 13px; height: 13px;
    border: 2px solid rgba(255,255,255,.25);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
```

Replace with:
```css
  /* Buttons — Kolosal vanilla */
  .btn {
    display: inline-flex; align-items: center; gap: 6px;
    height: 36px; padding: 0 14px;
    border-radius: 10px; border: none; cursor: pointer;
    font-family: var(--font); font-size: 14px; font-weight: 500;
    line-height: 20px; letter-spacing: 0.3px;
    text-decoration: none;
    transition: background .15s, border-color .15s;
  }
  .btn:disabled { pointer-events: none; opacity: 0.65; cursor: not-allowed; }
  .btn-primary {
    background-color: #0D0E0F;
    color: #FFFFFF;
    box-shadow: inset 0px 2px 0px 0px rgba(255,255,255,0.25);
    border: 1px solid #0D0E0F;
  }
  .btn-primary:hover:not(:disabled) {
    background: linear-gradient(90deg, #0D0E0F, #FF3131, #0066F5);
    background-size: 300% 300%;
    animation: gradientMove 3s ease infinite;
  }
  @keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  .btn-ghost {
    background-color: #FFFFFF;
    border: 1px solid #E4E7E9;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
    color: #0D0E0F;
  }
  .btn-ghost:hover:not(:disabled) {
    background-color: #F8F9F9;
    border-color: #DDE1E3;
  }
  .spinner {
    display: inline-block; width: 13px; height: 13px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
```

- [ ] **Step 2: Verify**

In the browser, check the Status panel. The "↺ Refresh" button should be white with a grey border. Run a visual check on TTS panel — "▶ Synthesise" should be black. Hover it to see the gradient animation.

---

### Task 4: Rewrite badge and status indicator CSS

**Files:**
- Modify: `src/api/playground.html` — badge, dot, live-badge, method badge, dl-status-badge sections

Multiple sections have hardcoded `rgba()` colors. All need explicit Kolosal token replacements.

- [ ] **Step 1: Replace `.badge` and `.dot` block**

Find:
```css
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 12px; padding: 3px 9px; border-radius: 20px;
    background: var(--accent-bg); color: var(--accent-hi);
    border: 1px solid rgba(99,102,241,.2);
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); flex-shrink: 0;
    box-shadow: 0 0 6px var(--green);
  }
```

Replace with:
```css
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 12px; padding: 3px 9px; border-radius: 20px;
    background: var(--accent-bg); color: var(--accent-hi);
    border: 1px solid rgba(0,102,245,.2);
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); flex-shrink: 0;
    box-shadow: 0 0 6px var(--green);
  }
```

- [ ] **Step 2: Replace `.live-badge` and `.live-dot` block**

Find:
```css
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
```

Replace with:
```css
  .live-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 20px;
    background: #F3FBF4; color: #2E9632;
    border: 1px solid rgba(58,188,63,.25);
  }
  .live-dot {
    width: 5px; height: 5px; border-radius: 50%; background: #3ABC3F;
    animation: livepulse 2s ease-in-out infinite;
  }
  @keyframes livepulse {
    0%,100% { opacity: 1; box-shadow: 0 0 4px #3ABC3F; }
    50%      { opacity: .4; box-shadow: none; }
  }
```

- [ ] **Step 3: Replace method badge CSS**

Find:
```css
  .method.get  { background: rgba(34,197,94,.12); color: var(--green); }
  .method.post { background: rgba(99,102,241,.12); color: var(--accent-hi); }
```

Replace with:
```css
  .method.get  { background: #F3FBF4; color: #2E9632; }
  .method.post { background: #F0F6FE; color: #0052C4; }
```

- [ ] **Step 4: Replace download status badge CSS**

Find:
```css
  .dl-status-badge.downloading { background: rgba(99,102,241,.15); color: var(--accent-hi); }
  .dl-status-badge.completed   { background: rgba(34,197,94,.12);  color: var(--green); }
  .dl-status-badge.failed      { background: rgba(239,68,68,.12);  color: var(--red); }
  .dl-status-badge.pending     { background: rgba(245,158,11,.12); color: var(--yellow); }
```

Replace with:
```css
  .dl-status-badge.downloading { background: #F0F6FE; color: #0052C4; }
  .dl-status-badge.completed   { background: #F3FBF4; color: #2E9632; }
  .dl-status-badge.failed      { background: #FFF3F3; color: #CC2727; }
  .dl-status-badge.pending     { background: #FFFAF3; color: #CC8727; }
```

- [ ] **Step 5: Verify**

In the browser, navigate to Dashboard → Overview. The SSE "live" badge should be green on white. Check Endpoints panel — GET badges should be green/white, POST badges blue/white.

---

### Task 5: Refine card, input, and response box styles

**Files:**
- Modify: `src/api/playground.html` — card, form inputs, response box sections in `<style>`

- [ ] **Step 1: Update card border-radius and shadow**

Find:
```css
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px;
  }
```

Replace with:
```css
  .card {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 12px; padding: 20px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
```

- [ ] **Step 2: Update input styles**

Find:
```css
  input[type=text], input[type=number], textarea, select {
    width: 100%; padding: 8px 11px;
    background: var(--surface2); color: var(--text);
    border: 1px solid var(--border2); border-radius: 7px;
    font-family: var(--font); font-size: 13px;
    outline: none; transition: border-color .15s;
    resize: vertical;
  }
  input[type=text]:focus, input[type=number]:focus, textarea:focus {
    border-color: var(--accent);
  }
  textarea { min-height: 90px; }
```

Replace with:
```css
  input[type=text], input[type=number], select {
    width: 100%; height: 36px;
    padding: 0 14px;
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    font-family: var(--font); font-size: 14px; font-weight: 400;
    outline: none; transition: border-color .15s, box-shadow .15s;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
  }
  input[type=text]:hover, input[type=number]:hover, select:hover {
    border-color: var(--text-dim);
    box-shadow: none;
  }
  input[type=text]::placeholder, input[type=number]::placeholder {
    color: var(--text-dim);
  }
  input[type=text]:focus, input[type=number]:focus, select:focus {
    border-color: var(--text-dim);
    outline: none;
    box-shadow: 0px 0px 0px 2px var(--border2);
  }
  textarea {
    width: 100%; min-height: 90px;
    padding: 8px 12px;
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: 12px;
    font-family: var(--font); font-size: 14px; font-weight: 400;
    line-height: 1.6; outline: none; resize: vertical;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
    transition: border-color .15s, box-shadow .15s;
  }
  textarea:hover { border-color: var(--text-dim); box-shadow: none; }
  textarea::placeholder { color: var(--text-dim); }
  textarea:focus {
    border-color: var(--text-dim);
    outline: none;
    box-shadow: 0px 0px 0px 2px var(--border2);
  }
```

- [ ] **Step 3: Update response box styles**

Find:
```css
  /* Response */
  .response-box {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 7px; padding: 14px;
    font-family: var(--mono); font-size: 12.5px; line-height: 1.7;
    color: var(--text-muted); min-height: 80px;
    white-space: pre-wrap; word-break: break-all;
    overflow-y: auto; max-height: 340px;
  }
  .response-box.ok     { color: var(--text); }
  .response-box.error  { color: var(--red); }
  .response-box.stream { color: var(--green); }
```

Replace with:
```css
  /* Response */
  .response-box {
    background: #F1F3F4; border: 1px solid var(--border2);
    border-radius: 10px; padding: 14px;
    font-family: var(--mono); font-size: 12.5px; line-height: 1.7;
    color: var(--text-muted); min-height: 80px;
    white-space: pre-wrap; word-break: break-all;
    overflow-y: auto; max-height: 340px;
  }
  .response-box.ok     { color: var(--text); }
  .response-box.error  { color: var(--red); }
  .response-box.stream { color: #2E9632; }
```

- [ ] **Step 4: Verify**

In the browser, check the TTS panel — the "Output" response box should be light grey (`#F1F3F4`). Check input fields — they should have a white background with light border and 36px height.

---

### Task 6: Update stat grid, dashboard tile, and chat bubble styles

**Files:**
- Modify: `src/api/playground.html` — stat-card, dash-tile, chat message sections in `<style>`

- [ ] **Step 1: Update stat-card styles**

Find:
```css
  /* Stats */
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(140px,1fr)); gap: 12px;
  }
  .stat-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
  }
  .stat-label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 600; font-family: var(--mono); }
  .stat-value.ok     { color: var(--green); }
  .stat-value.warn   { color: var(--yellow); }
  .stat-value.err    { color: var(--red); }
  .stat-value.accent { color: var(--accent-hi); }
```

Replace with:
```css
  /* Stats */
  .stat-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(140px,1fr)); gap: 12px;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
  .stat-label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-value { font-size: 20px; font-weight: 700; font-family: var(--mono); }
  .stat-value.ok     { color: #2E9632; }
  .stat-value.warn   { color: #CC8727; }
  .stat-value.err    { color: var(--red); }
  .stat-value.accent { color: var(--accent); }
```

- [ ] **Step 2: Update dashboard tile styles**

Find:
```css
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
```

Replace with:
```css
  .dash-tile {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
  .dash-tile-label {
    font-size: 10px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 5px;
  }
  .dash-tile-val {
    font-size: 22px; font-weight: 700; font-family: var(--mono); color: var(--accent);
  }
  .dash-tile-sub { font-size: 10px; color: var(--text-dim); margin-top: 3px; }
```

- [ ] **Step 3: Update chat bubble styles**

Find:
```css
  .chat-history {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 7px; padding: 14px; min-height: 180px;
    max-height: 320px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 10px;
  }
```

Replace with:
```css
  .chat-history {
    background: #F1F3F4; border: 1px solid var(--border2);
    border-radius: 10px; padding: 14px; min-height: 180px;
    max-height: 320px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 10px;
  }
```

Find:
```css
  .msg.user .msg-body { background: var(--accent-lo); color: #fff; }
  .msg.asst .msg-body { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
```

Replace with:
```css
  .msg.user .msg-body { background: var(--accent); color: #fff; }
  .msg.asst .msg-body { background: var(--surface); color: var(--text); border: 1px solid var(--border2); }
```

- [ ] **Step 4: Verify**

Check the Dashboard panel → Overview tab. The 6 metric tiles should be white cards with blue values. Navigate to LLM Chat and send a test message — user bubble should be blue, assistant bubble white.

---

### Task 7: Update GPU device, endpoint row, and download task styles

**Files:**
- Modify: `src/api/playground.html` — gpu-device, endpoint-row, dl-task sections in `<style>`

- [ ] **Step 1: Update GPU device card**

Find:
```css
  .gpu-device {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 7px; padding: 12px; margin-top: 12px;
  }
```

Replace with:
```css
  .gpu-device {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 10px; padding: 12px; margin-top: 12px;
  }
```

- [ ] **Step 2: Update endpoint row**

Find:
```css
  .endpoint-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 7px; font-size: 12px;
  }
```

Replace with:
```css
  .endpoint-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; background: var(--surface);
    border: 1px solid var(--border2); border-radius: 8px; font-size: 12px;
  }
```

- [ ] **Step 3: Update download task card**

Find:
```css
  .dl-task {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; margin-bottom: 8px;
  }
```

Replace with:
```css
  .dl-task {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 10px; padding: 12px; margin-bottom: 8px;
  }
```

- [ ] **Step 4: Update tab bar and subtab styles**

Find:
```css
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
```

Replace with:
```css
  .tab-bar {
    display: flex; gap: 2px;
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 10px; padding: 3px; width: fit-content; margin-bottom: 20px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
  .tab {
    padding: 6px 18px; border-radius: 8px;
    font-size: 13px; font-weight: 500; color: var(--text-muted);
    cursor: pointer; border: none; background: none; transition: all .15s;
    display: flex; align-items: center; gap: 7px; font-family: var(--font);
  }
  .tab.active {
    background: var(--accent-bg); color: var(--accent);
    border: 1px solid rgba(0,102,245,.2);
  }
```

Find:
```css
  .subtab.active { color: var(--accent-hi); border-bottom-color: var(--accent); }
```

Replace with:
```css
  .subtab.active { color: var(--accent); border-bottom-color: var(--accent); }
```

- [ ] **Step 5: Verify**

Navigate to Endpoints panel. Each row should be white with a light border. Dashboard tab bar should have a white background with a blue-tinted active tab.

---

### Task 8: Replace Unicode nav icons with Remix Icons in HTML

**Files:**
- Modify: `src/api/playground.html` — `<nav class="sidebar">` block (~lines 373–401)

- [ ] **Step 1: Replace nav item icons**

Find this entire nav block:
```html
  <nav class="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Playground</div>
      <button class="nav-item active" onclick="show('status',this)">
        <span class="nav-icon">◈</span> Status
      </button>
      <button class="nav-item" onclick="show('tts',this)">
        <span class="nav-icon">♪</span> TTS Stream
      </button>
      <button class="nav-item" onclick="show('classify',this)">
        <span class="nav-icon">⊞</span> Classify
      </button>
      <button class="nav-item" onclick="show('llm',this)">
        <span class="nav-icon">✦</span> LLM Chat
      </button>
      <button class="nav-item" onclick="show('completion',this)">
        <span class="nav-icon">⌨</span> Completion
      </button>
      <button class="nav-item" onclick="showDashboard(this)">
        <span class="nav-icon">⬡</span> Dashboard
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
      <button class="nav-item" onclick="show('endpoints',this)">
        <span class="nav-icon">⊕</span> Endpoints
      </button>
    </div>
  </nav>
```

Replace with:
```html
  <nav class="sidebar">
    <div class="sidebar-section">
      <div class="sidebar-label">Playground</div>
      <button class="nav-item active" onclick="show('status',this)">
        <i class="nav-icon ri-pulse-line"></i> Status
      </button>
      <button class="nav-item" onclick="show('tts',this)">
        <i class="nav-icon ri-music-2-line"></i> TTS Stream
      </button>
      <button class="nav-item" onclick="show('classify',this)">
        <i class="nav-icon ri-image-line"></i> Classify
      </button>
      <button class="nav-item" onclick="show('llm',this)">
        <i class="nav-icon ri-chat-3-line"></i> LLM Chat
      </button>
      <button class="nav-item" onclick="show('completion',this)">
        <i class="nav-icon ri-edit-line"></i> Completion
      </button>
      <button class="nav-item" onclick="showDashboard(this)">
        <i class="nav-icon ri-dashboard-3-line"></i> Dashboard
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
      <button class="nav-item" onclick="show('endpoints',this)">
        <i class="nav-icon ri-link-m"></i> Endpoints
      </button>
    </div>
  </nav>
```

- [ ] **Step 2: Update nav-icon CSS to target `<i>` elements**

Find:
```css
  .nav-icon { font-size: 16px; width: 18px; text-align: center; flex-shrink: 0; }
```

Replace with:
```css
  .nav-icon { font-size: 16px; width: 18px; text-align: center; flex-shrink: 0; font-style: normal; }
```

- [ ] **Step 3: Verify**

The sidebar should show Remix Icon glyphs for all 7 nav items. Active item (Status) should have blue text/bg.

---

### Task 9: Replace Unicode icons in dashboard tab bar, subtabs, and buttons

**Files:**
- Modify: `src/api/playground.html` — dashboard section and playground sub-tab section

- [ ] **Step 1: Replace dashboard tab bar icons**

Find:
```html
      <div class="tab-bar">
        <button class="tab active" id="dash-tab-overview"    onclick="switchDashTab('overview')">
          <span class="tab-icon">◈</span> Overview
        </button>
        <button class="tab"        id="dash-tab-playground"  onclick="switchDashTab('playground')">
          <span class="tab-icon">⚡</span> Playground
        </button>
      </div>
```

Replace with:
```html
      <div class="tab-bar">
        <button class="tab active" id="dash-tab-overview"    onclick="switchDashTab('overview')">
          <i class="tab-icon ri-bar-chart-line"></i> Overview
        </button>
        <button class="tab"        id="dash-tab-playground"  onclick="switchDashTab('playground')">
          <i class="tab-icon ri-flashlight-line"></i> Playground
        </button>
      </div>
```

- [ ] **Step 2: Replace playground subtab Unicode icons**

Find:
```html
          <button class="subtab active" id="pg-tab-tts"        onclick="switchPgTab('tts')">♪ TTS Stream</button>
          <button class="subtab"        id="pg-tab-classify"   onclick="switchPgTab('classify')">⊞ Classify</button>
          <button class="subtab"        id="pg-tab-llm"        onclick="switchPgTab('llm')">✦ LLM Chat</button>
          <button class="subtab"        id="pg-tab-completion" onclick="switchPgTab('completion')">⌨ Completion</button>
```

Replace with:
```html
          <button class="subtab active" id="pg-tab-tts"        onclick="switchPgTab('tts')"><i class="ri-music-2-line"></i> TTS Stream</button>
          <button class="subtab"        id="pg-tab-classify"   onclick="switchPgTab('classify')"><i class="ri-image-line"></i> Classify</button>
          <button class="subtab"        id="pg-tab-llm"        onclick="switchPgTab('llm')"><i class="ri-chat-3-line"></i> LLM Chat</button>
          <button class="subtab"        id="pg-tab-completion" onclick="switchPgTab('completion')"><i class="ri-edit-line"></i> Completion</button>
```

- [ ] **Step 3: Update tab-icon CSS**

Find:
```css
  .tab-icon { font-size: 14px; }
```

Replace with:
```css
  .tab-icon { font-size: 14px; font-style: normal; }
```

- [ ] **Step 4: Replace Unicode symbols in action buttons**

These are inline in various panels. Make the following replacements throughout the HTML (use find-and-replace for each):

| Find | Replace |
|---|---|
| `▶  Synthesise` | `<i class="ri-play-line"></i> Synthesise` |
| `■  Stop` | `<i class="ri-stop-line"></i> Stop` |
| `⊞  Classify` | `<i class="ri-image-line"></i> Classify` |
| `↺  Refresh` | `<i class="ri-refresh-line"></i> Refresh` |
| `✕  Clear chat` | `<i class="ri-close-line"></i> Clear chat` |
| `▶  Complete` | `<i class="ri-play-line"></i> Complete` |
| `↓ Download` | `<i class="ri-download-line"></i> Download` |
| `▶  Synthesise` (pg-tts panel) | `<i class="ri-play-line"></i> Synthesise` |
| `■  Stop` (pg-tts panel) | `<i class="ri-stop-line"></i> Stop` |
| `✕  Clear chat` (pg-llm panel) | `<i class="ri-close-line"></i> Clear chat` |
| `▶  Complete` (pg-completion panel) | `<i class="ri-play-line"></i> Complete` |

Also replace the dropzone icon:

Find (appears twice — in `#dropzone` and `#pg-dropzone`):
```html
          <div class="drop-icon">⊞</div>
```
Replace with:
```html
          <div class="drop-icon"><i class="ri-image-add-line"></i></div>
```

- [ ] **Step 5: Verify**

All buttons should now show Remix Icon glyphs. Dashboard tab icons should be a bar chart and lightning bolt. Subtabs should show music/image/chat/edit icons.

---

### Task 10: Fix inline hardcoded colors in HTML

**Files:**
- Modify: `src/api/playground.html` — resource bar fills (3 `style` attributes in the Dashboard overview)

- [ ] **Step 1: Update resource bar fill colors**

Find:
```html
              <div class="res-bar-fill" id="bar-cpu" style="width:0%;background:#6366f1"></div>
```
Replace with:
```html
              <div class="res-bar-fill" id="bar-cpu" style="width:0%;background:#0066F5"></div>
```

Find:
```html
              <div class="res-bar-fill" id="bar-ram" style="width:0%;background:#f59e0b"></div>
```
Replace with:
```html
              <div class="res-bar-fill" id="bar-ram" style="width:0%;background:#FFA931"></div>
```

Find:
```html
              <div class="res-bar-fill" id="bar-gpu" style="width:0%;background:#22c55e"></div>
```
Replace with:
```html
              <div class="res-bar-fill" id="bar-gpu" style="width:0%;background:#3ABC3F"></div>
```

- [ ] **Step 2: Fix the download section divider border**

Find:
```html
          <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border)">
```
Replace with:
```html
          <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--border2)">
```

- [ ] **Step 3: Fix classify panel info text color**

Find (appears twice — in `#dropzone` and `#pg-dropzone`):
```html
          <div style="font-size:11px;color:var(--text-dim);margin-top:4px">JPEG / PNG</div>
```
This already uses `var(--text-dim)` — no change needed. ✓

- [ ] **Step 4: Verify**

Open Dashboard → Overview. The CPU bar should be blue, RAM bar orange, GPU bar green when the server is running. At rest (all 0%) verify the bars are grey track (`#EBEDEE`).

---

### Task 11: Final visual sweep and commit

**Files:**
- Modify: `src/api/playground.html` (final review)

- [ ] **Step 1: Full visual sweep — check each panel**

Open `src/api/playground.html` in a browser and click through every panel:

| Panel | Things to verify |
|---|---|
| Status | White cards, blue stat values, grey "Refresh" outline button |
| TTS Stream | Black "Synthesise" btn, grey "Stop" btn, light grey output box |
| Classify | Dropzone with dashed border, white bg, Remix Icon upload icon |
| LLM Chat | Chat history light grey bg, blue user bubble, white assistant bubble |
| Completion | Textarea with 12px radius, black "Complete" btn |
| Dashboard → Overview | White tiles with blue values, live badge green/white, bar chart canvas |
| Dashboard → Playground | Subtab bar with Remix Icons, all sub-panels render same as standalone panels |
| Endpoints | White rows, green GET badges, blue POST badges |

- [ ] **Step 2: Verify no dark backgrounds remain**

In DevTools, use the color picker or inspect `computed styles` on: `.shell`, `body`, `.sidebar`, `.card`. None should have dark backgrounds (`#0`, `#1`, `#08`, `#0e`, etc.).

- [ ] **Step 3: Commit**

```bash
git add src/api/playground.html
git commit -m "feat: rework dashboard UI to Kolosal vanilla light design system

- Replace dark indigo theme with Kolosal vanilla light tokens
- Swap JetBrains Mono for Geist Mono, add Remix Icons v4.7.0
- Rewrite button CSS: primary=black with gradient hover, outline style for secondary
- Update all hardcoded rgba() colors to Kolosal token equivalents
- Replace all Unicode symbols with ri- Remix Icon classes
- Cards: 12px radius, subtle shadow; inputs: 36px height, 10px radius

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Color tokens — Task 2 updates all `:root` vars
- ✅ Typography — Task 1 swaps fonts
- ✅ Layout shell (sidebar/topbar/main) — vars handle these automatically
- ✅ Buttons (primary black + gradient, outline) — Task 3
- ✅ Badges + status indicators — Task 4
- ✅ Cards (12px radius, shadow) — Task 5
- ✅ Inputs (36px, 10px radius, focus ring) — Task 5
- ✅ Response boxes (grey-600 bg) — Task 5
- ✅ Stat grid + dashboard tiles — Task 6
- ✅ Chat bubbles (blue user, white asst) — Task 6
- ✅ GPU device + endpoint rows + dl tasks + tab bar — Task 7
- ✅ Nav icons (Remix) — Task 8
- ✅ Tab/subtab/button icons (Remix) — Task 9
- ✅ Resource bar fill inline colors — Task 10
- ✅ No dark mode required (out of scope) ✓

**Placeholder scan:** No TBDs, no "implement later", no vague steps. Every step shows exact strings to find and exact replacement content.

**Type consistency:** All CSS selector names are consistent across tasks. `var(--accent)` resolves to `#0066F5` (blue) throughout. `.btn-primary` is black, `.btn-ghost` is outline-white. No conflicts.
