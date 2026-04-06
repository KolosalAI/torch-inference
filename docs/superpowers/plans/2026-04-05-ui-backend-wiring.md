# UI–Backend Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all orphaned backend endpoints into the playground UI, fix the classify field-name bug, and add three new tabs (Audio, Logs, System).

**Architecture:** All changes are confined to `src/api/playground.html` (the single embedded HTML/JS file served at `GET /`) plus Playwright test files. No Rust changes required. Each task adds one feature, commits, then moves on.

**Tech Stack:** Vanilla JS, HTML5, Playwright for E2E tests. Server must be running at `http://localhost:8000` before running playwright tests (`cargo run --no-default-features -- --config config.toml`).

**Security note:** All `innerHTML` usage in this plan is limited to hardcoded icon strings (no user-supplied data). All dynamic content (server responses, error messages) uses `mkEl()` or `textContent` assignments, which prevent XSS.

---

## File Map

| File | Role |
|------|------|
| `src/api/playground.html` | All UI changes (HTML structure + embedded JS) |
| `tests/playwright/utils/selectors.js` | Add new element IDs for each new feature |
| `tests/playwright/tests/classify.spec.js` | Add request-interception test for bug fix |
| `tests/playwright/tests/tts.spec.js` | Add voice select test |
| `tests/playwright/tests/models-panel.spec.js` | Add delete + SOTA tests |
| `tests/playwright/tests/audio.spec.js` | New — Audio tab tests |
| `tests/playwright/tests/logs.spec.js` | New — Logs tab tests |
| `tests/playwright/tests/system.spec.js` | New — System tab tests |

---

## Task 1: Fix classify field names in `runPgClassify`

The `runPgClassify()` function in the Dashboard Playground sub-tab sends `width`/`height` instead of `model_width`/`model_height`. The main `runClassify()` is already correct.

**Files:**
- Modify: `src/api/playground.html:1727`
- Modify: `tests/playwright/tests/classify.spec.js`

- [ ] **Step 1: Add a failing test** that intercepts the classify request body

Open `tests/playwright/tests/classify.spec.js` and add this test at the end of the file (inside `test.describe('Image Classification', () => {`):

```js
test('classify request body uses model_width and model_height', async ({ page }) => {
  let capturedBody = null;
  await page.route('/classify/batch', async route => {
    capturedBody = JSON.parse(route.request().postData() || '{}');
    await route.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify({ results: [[{ label: 'cat', confidence: 0.9, class_id: 0 }]], batch_size: 1 }) });
  });
  await page.locator(S.imgFile).setInputFiles(pngPath);
  await expect(page.locator(S.clsBtn)).toBeEnabled({ timeout: 5000 });
  await page.locator(S.clsBtn).click();
  await expect(page.locator(S.clsOut)).not.toHaveText('waiting for image…', { timeout: 5000 });
  expect(capturedBody).toHaveProperty('model_width');
  expect(capturedBody).toHaveProperty('model_height');
  expect(capturedBody).not.toHaveProperty('width');
  expect(capturedBody).not.toHaveProperty('height');
});
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd tests/playwright && npx playwright test tests/classify.spec.js --project=chromium 2>&1 | tail -20
```

Expected: the last test (`classify request body uses model_width...`) fails with the properties check.

- [ ] **Step 3: Fix the bug in `runPgClassify`**

In `src/api/playground.html`, find line 1727 (inside `runPgClassify`):

```js
      body: JSON.stringify({ images: [pgImgBase64], top_k: topk, width: w, height: h }),
```

Replace with:

```js
      body: JSON.stringify({ images: [pgImgBase64], top_k: topk, model_width: w, model_height: h }),
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd tests/playwright && npx playwright test tests/classify.spec.js --project=chromium 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/classify.spec.js
git commit -m "fix(ui): use model_width/model_height in playground classify request"
```

---

## Task 2: Wire TTS voice selector dynamically from `/tts/engines/{id}/voices`

Replace the free-text `#tts-voice` and `#pg-tts-voice` inputs with `<select>` dropdowns that repopulate when the engine changes.

**Files:**
- Modify: `src/api/playground.html` (lines 573, 579, 823, 827, 1836)
- Modify: `tests/playwright/tests/tts.spec.js`

- [ ] **Step 1: Add failing tests for voice select behaviour**

Open `tests/playwright/tests/tts.spec.js` and add these tests inside the describe block:

```js
test('tts-voice is a <select> element', async ({ page }) => {
  await expect(page.locator('#tts-voice')).toHaveJSProperty('tagName', 'SELECT');
});

test('voice select repopulates when engine changes', async ({ page }) => {
  await page.route('/tts/engines/*', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify({ voices: [
        { id: 'af_heart', name: 'Heart', language: 'en-US', gender: 'Female', quality: 'Neural' },
        { id: 'af_nova',  name: 'Nova',  language: 'en-US', gender: 'Male',   quality: 'Neural' },
      ], total: 2 }) });
  });
  await page.evaluate(() => {
    const sel = document.getElementById('tts-engine');
    sel.value = sel.options[0]?.value || '';
    sel.dispatchEvent(new Event('change'));
  });
  await expect(page.locator('#tts-voice option[value="af_heart"]')).toBeAttached({ timeout: 3000 });
  await expect(page.locator('#tts-voice option[value="af_nova"]')).toBeAttached({ timeout: 3000 });
});
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd tests/playwright && npx playwright test tests/tts.spec.js --project=chromium 2>&1 | tail -15
```

Expected: "tts-voice is a select" fails (it is currently an `<input>`).

- [ ] **Step 3: Change the TTS voice text inputs to selects in the HTML**

In `src/api/playground.html`, find line 579:

```html
            <input type="text" id="tts-voice" placeholder="default" />
```

Replace with:

```html
            <select id="tts-voice"><option value="">default</option></select>
```

Find line 827 (pg-tts panel):

```html
              <div class="field"><label>Voice (optional)</label><input type="text" id="pg-tts-voice" placeholder="default" /></div>
```

Replace with:

```html
              <div class="field"><label>Voice (optional)</label><select id="pg-tts-voice"><option value="">default</option></select></div>
```

- [ ] **Step 4: Add `onchange` to the engine selects**

Find line 573:

```html
              <select id="tts-engine"><option value="">default</option></select>
```

Replace with:

```html
              <select id="tts-engine" onchange="loadVoicesForEngine(this.value,'tts-voice')"><option value="">default</option></select>
```

Find line 823:

```html
                  <select id="pg-tts-engine"><option value="">default</option></select>
```

Replace with:

```html
                  <select id="pg-tts-engine" onchange="loadVoicesForEngine(this.value,'pg-tts-voice')"><option value="">default</option></select>
```

- [ ] **Step 5: The `runTTS` / `runPgTTS` voice reads need no change**

Both already use `document.getElementById('tts-voice').value.trim()` — reading `.value` from a `<select>` works identically to an `<input>`. No JS change needed here.

- [ ] **Step 6: Add `loadVoicesForEngine` and update `refreshTtsSelects`**

In `src/api/playground.html`, find the `refreshTtsSelects` function (around line 1836):

```js
async function refreshTtsSelects() {
  try {
    const r = await fetch('/tts/engines');
    if (!r.ok) return;
    const data    = await r.json();
    const engines = Array.isArray(data) ? data : (data.engines || []);
    const idOf    = e => (typeof e === 'string' ? e : (e.name || e.id || ''));
    TTS_SEL.forEach(id => fillSelect(id, engines, idOf, idOf, 'default'));
  } catch (_) {}
}
```

Replace with:

```js
async function refreshTtsSelects() {
  try {
    const r = await fetch('/tts/engines');
    if (!r.ok) return;
    const data    = await r.json();
    const engines = Array.isArray(data) ? data : (data.engines || []);
    const idOf    = e => (typeof e === 'string' ? e : (e.name || e.id || ''));
    TTS_SEL.forEach(id => fillSelect(id, engines, idOf, idOf, 'default'));
    const firstId = engines.length ? idOf(engines[0]) : '';
    if (firstId) {
      loadVoicesForEngine(firstId, 'tts-voice');
      loadVoicesForEngine(firstId, 'pg-tts-voice');
    }
  } catch (_) {}
}

async function loadVoicesForEngine(engineId, voiceSelectId) {
  const sel = document.getElementById(voiceSelectId);
  if (!sel) return;
  clearEl(sel);
  const def = document.createElement('option');
  def.value = ''; def.textContent = 'default'; sel.appendChild(def);
  if (!engineId) return;
  try {
    const r = await fetch('/tts/engines/' + encodeURIComponent(engineId) + '/voices');
    if (!r.ok) return;
    const data   = await r.json();
    const voices = Array.isArray(data) ? data : (data.voices || []);
    voices.forEach(v => {
      const opt = document.createElement('option');
      opt.value       = v.id || v.name || '';
      opt.textContent = (v.name || v.id || '') + (v.language ? ' (' + v.language + ')' : '');
      sel.appendChild(opt);
    });
  } catch (_) {}
}
```

- [ ] **Step 7: Run the tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/tts.spec.js --project=chromium 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/tts.spec.js
git commit -m "feat(ui): wire TTS voice dropdown dynamically from /tts/engines/{id}/voices"
```

---

## Task 3: Add delete button to downloaded models

Each row in the "Downloaded" sub-tab gets a "Delete" button that calls `DELETE /models/download/{name}`.

**Files:**
- Modify: `src/api/playground.html` (inside `loadDownloadedModels` ~line 1985; add `deleteModel` function)
- Modify: `tests/playwright/tests/models-panel.spec.js`

- [ ] **Step 1: Add failing tests for the delete button**

Open `tests/playwright/tests/models-panel.spec.js` and add this describe block at the end:

```js
test.describe('Downloaded models — delete', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
  });

  test('delete button is visible on each model row', async ({ page }) => {
    await page.route('/api/models/downloaded', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify([{ id: 'resnet50', name: 'ResNet-50', task: 'classification', status: 'cached' }]),
    }));
    await page.reload();
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
    await expect(page.locator('#mdl-downloaded-list button:has-text("Delete")')).toBeVisible({ timeout: 3000 });
  });

  test('clicking delete calls DELETE /models/download/{name}', async ({ page }) => {
    let deleteCalled = '';
    await page.route('/api/models/downloaded', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify([{ id: 'resnet50', name: 'resnet50', task: 'classification', status: 'cached' }]),
    }));
    await page.route('/models/download/**', async route => {
      if (route.request().method() === 'DELETE') {
        deleteCalled = route.request().url();
        await route.fulfill({ status: 200 });
      } else { await route.continue(); }
    });
    await page.reload();
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
    page.once('dialog', d => d.accept());
    await page.locator('#mdl-downloaded-list button:has-text("Delete")').first().click();
    await expect.poll(() => deleteCalled, { timeout: 5000 }).toContain('/models/download/resnet50');
  });
});
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd tests/playwright && npx playwright test tests/models-panel.spec.js --project=chromium 2>&1 | tail -15
```

Expected: the delete tests fail (no delete button found).

- [ ] **Step 3: Add delete button into `loadDownloadedModels`**

In `src/api/playground.html`, inside `loadDownloadedModels`, find:

```js
      row.appendChild(mkEl('span', 'dl-model-status ' + stCls, status));
      list.appendChild(row);
```

Replace with:

```js
      row.appendChild(mkEl('span', 'dl-model-status ' + stCls, status));
      const delBtn = document.createElement('button');
      delBtn.className = 'btn btn-ghost';
      // Safe: icon is hardcoded, not user data
      const delIco = document.createElement('i'); delIco.className = 'ri-delete-bin-line';
      delBtn.appendChild(delIco);
      delBtn.appendChild(document.createTextNode(' Delete'));
      delBtn.style.cssText = 'margin-left:auto;font-size:12px;padding:3px 10px;color:var(--red);border-color:var(--red)';
      const modelName = m.name || m.id || m.model_id || '';
      delBtn.onclick = () => deleteModel(modelName);
      row.appendChild(delBtn);
      list.appendChild(row);
```

- [ ] **Step 4: Add the `deleteModel` function**

Add before `formatBytes` (near end of `<script>` block):

```js
async function deleteModel(name) {
  if (!name) return;
  if (!confirm('Delete model "' + name + '"? This cannot be undone.')) return;
  try {
    const r = await fetch('/models/download/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) {
      const msg = await r.text().catch(() => String(r.status));
      alert('Delete failed: ' + msg);
      return;
    }
    loadDownloadedModels();
    refreshModelSelects();
  } catch (e) { alert('Delete error: ' + e.message); }
}
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/models-panel.spec.js --project=chromium 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/models-panel.spec.js
git commit -m "feat(ui): add delete button to downloaded models list"
```

---

## Task 4: Add SOTA models sub-tab

A new "SOTA" sub-tab in the Models panel loads `GET /models/sota` and renders model cards.

**Files:**
- Modify: `src/api/playground.html` (models panel subtab-bar ~line 957, `switchMdlTab` ~line 1852, add functions)
- Modify: `tests/playwright/utils/selectors.js`
- Modify: `tests/playwright/tests/models-panel.spec.js`

- [ ] **Step 1: Add SOTA selectors to `selectors.js`**

In `tests/playwright/utils/selectors.js`, add at the end of the Models panel section:

```js
  // SOTA sub-tab
  mdlTabSota:   '#mdl-tab-sota',
  mdlSota:      '#mdl-sota',
  mdlSotaGrid:  '#mdl-sota-grid',
```

- [ ] **Step 2: Add failing SOTA tests**

Add to `tests/playwright/tests/models-panel.spec.js`:

```js
test.describe('SOTA models sub-tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator('button.nav-item:has-text("Models")').click();
  });

  test('SOTA sub-tab button is visible', async ({ page }) => {
    await expect(page.locator('#mdl-tab-sota')).toBeVisible();
  });

  test('SOTA tab renders model cards from /models/sota', async ({ page }) => {
    await page.route('/models/sota', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ models: [
        { id: 'efficientnet_b0', name: 'EfficientNet-B0', task: 'classification',
          architecture: 'EfficientNet', size_estimate: '20MB' },
      ], total: 1 }),
    }));
    await page.locator('#mdl-tab-sota').click();
    await expect(page.locator('#mdl-sota-grid .model-card')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('#mdl-sota-grid .model-card-name')).toHaveText('EfficientNet-B0');
  });
});
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
cd tests/playwright && npx playwright test tests/models-panel.spec.js --project=chromium 2>&1 | grep -E "FAILED|passed|failed"
```

- [ ] **Step 4: Add SOTA sub-tab button to the subtab-bar HTML**

In `src/api/playground.html`, find (line 957):

```html
      <div class="subtab-bar">
        <button class="subtab active" id="mdl-tab-available" onclick="switchMdlTab('available')"><i class="ri-store-2-line"></i> Catalog</button>
        <button class="subtab"        id="mdl-tab-downloaded" onclick="switchMdlTab('downloaded')"><i class="ri-hard-drive-2-line"></i> Downloaded</button>
        <button class="subtab"        id="mdl-tab-download" onclick="switchMdlTab('download')"><i class="ri-download-cloud-line"></i> Download</button>
      </div>
```

Replace with:

```html
      <div class="subtab-bar">
        <button class="subtab active" id="mdl-tab-available"  onclick="switchMdlTab('available')"><i class="ri-store-2-line"></i> Catalog</button>
        <button class="subtab"        id="mdl-tab-downloaded" onclick="switchMdlTab('downloaded')"><i class="ri-hard-drive-2-line"></i> Downloaded</button>
        <button class="subtab"        id="mdl-tab-sota"       onclick="switchMdlTab('sota')"><i class="ri-star-line"></i> SOTA</button>
        <button class="subtab"        id="mdl-tab-download"   onclick="switchMdlTab('download')"><i class="ri-download-cloud-line"></i> Download</button>
      </div>
```

- [ ] **Step 5: Add the SOTA container div**

Find just before `</section><!-- /#panel-models -->`:

```html
      </div>

    </section><!-- /#panel-models -->
```

Replace with:

```html
      </div>

      <!-- SOTA sub-tab -->
      <div id="mdl-sota" style="display:none">
        <div style="display:flex;justify-content:flex-end;margin-bottom:14px">
          <button class="btn btn-ghost" style="height:36px;padding:0 12px;font-size:13px" onclick="loadSotaModels()">
            <i class="ri-refresh-line"></i> Refresh
          </button>
        </div>
        <div id="mdl-sota-loading" style="display:none">Loading SOTA models…</div>
        <div class="model-grid" id="mdl-sota-grid"></div>
      </div>

    </section><!-- /#panel-models -->
```

- [ ] **Step 6: Update `MDL_SUBTABS` and `switchMdlTab`**

Find (around line 1852):

```js
const MDL_SUBTABS = ['available', 'downloaded', 'download'];
function switchMdlTab(tab) {
  MDL_SUBTABS.forEach(id => {
    const pane = document.getElementById('mdl-' + id);
    const btn  = document.getElementById('mdl-tab-' + id);
    if (pane) pane.style.display = id === tab ? '' : 'none';
    if (btn)  btn.classList[id === tab ? 'add' : 'remove']('active');
  });
  if (tab === 'available')  loadAvailableModels();
  if (tab === 'downloaded') loadDownloadedModels();
}
```

Replace with:

```js
const MDL_SUBTABS = ['available', 'downloaded', 'sota', 'download'];
function switchMdlTab(tab) {
  MDL_SUBTABS.forEach(id => {
    const pane = document.getElementById('mdl-' + id);
    const btn  = document.getElementById('mdl-tab-' + id);
    if (pane) pane.style.display = id === tab ? '' : 'none';
    if (btn)  btn.classList[id === tab ? 'add' : 'remove']('active');
  });
  if (tab === 'available')  loadAvailableModels();
  if (tab === 'downloaded') loadDownloadedModels();
  if (tab === 'sota')       loadSotaModels();
}
```

- [ ] **Step 7: Add `loadSotaModels` and `downloadSotaModel` functions**

Add before `formatBytes`:

```js
async function loadSotaModels() {
  const grid = document.getElementById('mdl-sota-grid');
  const load = document.getElementById('mdl-sota-loading');
  if (!grid) return;
  clearEl(grid); load.style.display = '';
  try {
    const r = await fetch('/models/sota');
    if (!r.ok) throw new Error(String(r.status));
    const data   = await r.json();
    const models = Array.isArray(data) ? data : (data.models || []);
    load.style.display = 'none';
    if (!models.length) { grid.appendChild(mkEl('div', 'mdl-empty', 'No SOTA models found.')); return; }
    models.forEach(m => {
      const card = buildModelCard(m);
      // Override the download button to use the SOTA endpoint
      const btn = card.querySelector('button.btn-primary');
      if (btn) {
        btn.onclick = null;
        const id   = m.id || m.model_id || m.name || '';
        const name = m.name || id;
        btn.addEventListener('click', () => downloadSotaModel(id, name));
      }
      grid.appendChild(card);
    });
  } catch (e) {
    load.style.display = 'none';
    grid.appendChild(mkEl('div', 'mdl-empty', 'Failed: ' + e.message));
  }
}

async function downloadSotaModel(modelId, name) {
  try {
    const r = await fetch('/models/sota/' + encodeURIComponent(modelId), { method: 'POST' });
    if (!r.ok) { alert('Download failed: ' + r.status); return; }
    const data = await r.json().catch(() => ({}));
    switchMdlTab('download');
    if (data.task_id || data.id) pollMdlDownload(data.task_id || data.id, name || modelId);
  } catch (e) { alert('Network error: ' + e.message); }
}
```

- [ ] **Step 8: Run the tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/models-panel.spec.js --project=chromium 2>&1 | tail -10
```

- [ ] **Step 9: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/models-panel.spec.js tests/playwright/utils/selectors.js
git commit -m "feat(ui): add SOTA models sub-tab to Models panel"
```

---

## Task 5: Add Audio tab (transcription)

New sidebar tab that uploads an audio file and calls `POST /audio/transcribe`.

**Files:**
- Modify: `src/api/playground.html` (nav item, panel HTML after line 698, JS functions, `show()`)
- Modify: `tests/playwright/utils/selectors.js`
- Create: `tests/playwright/tests/audio.spec.js`

- [ ] **Step 1: Add Audio selectors**

In `tests/playwright/utils/selectors.js`, add after the Models panel section:

```js
  // ── Audio panel ─────────────────────────────────────────
  navAudio:         'button.nav-item:has-text("Audio")',
  panelAudio:       '#panel-audio',
  audioHealthBadge: '#audio-health-badge',
  audioDropzone:    '#audio-dropzone',
  audioFile:        '#audio-file',
  audioModel:       '#audio-model',
  audioTimestamps:  '#audio-timestamps',
  audioBtn:         '#audio-btn',
  audioResultText:  '#audio-result-text',
  audioSegments:    '#audio-segments',
  audioMeta:        '#audio-meta',
```

- [ ] **Step 2: Write the failing test file**

Create `tests/playwright/tests/audio.spec.js`:

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Audio Transcription', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navAudio).click();
    await expect(page.locator(S.panelAudio)).toHaveClass(/active/);
  });

  test('Audio nav item is present in sidebar', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(S.navAudio)).toBeVisible();
  });

  test('audio dropzone is visible', async ({ page }) => {
    await expect(page.locator(S.audioDropzone)).toBeVisible();
  });

  test('Transcribe button is disabled before file upload', async ({ page }) => {
    await expect(page.locator(S.audioBtn)).toBeDisabled();
  });

  test('model select defaults to "default"', async ({ page }) => {
    await expect(page.locator(S.audioModel)).toHaveValue('default');
  });

  test('timestamps checkbox is unchecked by default', async ({ page }) => {
    await expect(page.locator(S.audioTimestamps)).not.toBeChecked();
  });

  test('transcription request sends multipart form with audio field', async ({ page }) => {
    let requestMade = false;
    await page.route('/audio/transcribe', async route => {
      requestMade = true;
      const headers = route.request().headers();
      expect(headers['content-type']).toContain('multipart/form-data');
      await route.fulfill({ status: 200, contentType: 'application/json',
        body: JSON.stringify({ text: 'Hello world', language: 'en', confidence: 0.95, segments: null }) });
    });
    const wavBytes = Buffer.alloc(44); wavBytes.write('RIFF', 0); wavBytes.write('WAVE', 8);
    await page.locator(S.audioFile).setInputFiles({
      name: 'test.wav', mimeType: 'audio/wav', buffer: wavBytes,
    });
    await expect(page.locator(S.audioBtn)).toBeEnabled({ timeout: 3000 });
    await page.locator(S.audioBtn).click();
    await expect.poll(() => requestMade, { timeout: 5000 }).toBe(true);
    await expect(page.locator(S.audioResultText)).toContainText('Hello world', { timeout: 5000 });
  });

  test('timestamps toggle shows segment list when response includes segments', async ({ page }) => {
    await page.route('/audio/transcribe', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ text: 'Hello', language: 'en', confidence: 0.9,
        segments: [{ text: 'Hello', start: 0.0, end: 1.2, confidence: 0.9 }] }),
    }));
    const wavBytes = Buffer.alloc(44); wavBytes.write('RIFF', 0); wavBytes.write('WAVE', 8);
    await page.locator(S.audioFile).setInputFiles({ name: 'test.wav', mimeType: 'audio/wav', buffer: wavBytes });
    await page.locator(S.audioTimestamps).check();
    await expect(page.locator(S.audioBtn)).toBeEnabled({ timeout: 3000 });
    await page.locator(S.audioBtn).click();
    await expect(page.locator(S.audioSegments)).toBeVisible({ timeout: 5000 });
  });
});
```

- [ ] **Step 3: Run the test to confirm it fails**

```bash
cd tests/playwright && npx playwright test tests/audio.spec.js --project=chromium 2>&1 | grep -E "FAILED|Error|passed|failed" | head -10
```

Expected: fails because Audio nav item doesn't exist.

- [ ] **Step 4: Add the Audio nav item**

In `src/api/playground.html`, find lines 516-519:

```html
      <button class="nav-item" onclick="show('completion',this)">
        <i class="nav-icon ri-edit-line"></i> Completion
      </button>
      <button class="nav-item" onclick="showDashboard(this)">
```

Replace with:

```html
      <button class="nav-item" onclick="show('completion',this)">
        <i class="nav-icon ri-edit-line"></i> Completion
      </button>
      <button class="nav-item" onclick="show('audio',this)">
        <i class="nav-icon ri-mic-line"></i> Audio
      </button>
      <button class="nav-item" onclick="showDashboard(this)">
```

- [ ] **Step 5: Add the Audio panel HTML**

Find (around line 699):

```html
    </section>

    <!-- Dashboard -->
    <section class="panel" id="panel-dashboard">
```

Replace with:

```html
    </section>

    <!-- Audio -->
    <section class="panel" id="panel-audio">
      <div class="panel-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="panel-title">Audio Transcription</div>
          <span id="audio-health-badge" style="font-size:12px;padding:2px 9px;border-radius:12px;background:var(--accent-bg);color:var(--accent)">checking…</span>
        </div>
        <div class="panel-desc">Upload audio and transcribe via <span class="ep-code">POST /audio/transcribe</span>.</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div class="card">
          <div class="card-title">Input</div>
          <div class="dropzone" id="audio-dropzone"
               onclick="document.getElementById('audio-file').click()"
               ondragover="event.preventDefault();this.classList.add('over')"
               ondragleave="this.classList.remove('over')"
               ondrop="handleAudioDrop(event)">
            <div class="drop-icon"><i class="ri-mic-line"></i></div>
            <div>Click to upload or drag &amp; drop</div>
            <div style="font-size:11px;color:var(--text-dim);margin-top:4px">WAV · MP3 · OGG · FLAC · M4A</div>
            <input type="file" id="audio-file" accept="audio/*" onchange="handleAudioFile(this.files[0])" />
          </div>
          <div id="audio-file-name" style="font-size:12px;color:var(--text-muted);margin-top:6px;display:none"></div>
          <div class="field" style="margin-top:12px">
            <label>Model</label>
            <select id="audio-model"><option value="default">default</option></select>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
            <input type="checkbox" id="audio-timestamps" />
            <label for="audio-timestamps" style="font-size:13px;cursor:pointer">Include timestamps</label>
          </div>
          <div style="display:flex;gap:8px;margin-top:12px">
            <button class="btn btn-primary" id="audio-btn" onclick="runAudio()" disabled>
              <i class="ri-file-music-line"></i> Transcribe
            </button>
            <button class="btn btn-ghost" onclick="stopAudio()"><i class="ri-stop-line"></i> Stop</button>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Result</div>
          <div class="response-box" id="audio-result-text" style="min-height:80px">Upload a file to transcribe.</div>
          <div id="audio-meta" style="font-size:12px;color:var(--text-muted);margin-top:6px;display:none"></div>
          <div id="audio-segments" style="display:none;margin-top:10px">
            <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-bottom:6px">Segments</div>
            <div id="audio-segments-list"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Dashboard -->
    <section class="panel" id="panel-dashboard">
```

- [ ] **Step 6: Add Audio JS functions**

Add before `formatBytes` (near end of `<script>`):

```js
/* ── Audio tab ──────────────────────────────────────────── */
let audioFile = null;
let audioAbort = null;

async function loadAudioHealth() {
  const badge = document.getElementById('audio-health-badge');
  if (!badge) return;
  try {
    const r = await fetch('/audio/health');
    badge.textContent       = r.ok ? '● Audio healthy'     : '● Audio unavailable';
    badge.style.background  = r.ok ? 'var(--green-bg)'     : 'var(--red-bg)';
    badge.style.color       = r.ok ? 'var(--green-text)'   : 'var(--red-text)';
  } catch (_) {
    badge.textContent = '● Audio unavailable';
    badge.style.background = 'var(--red-bg)'; badge.style.color = 'var(--red-text)';
  }
}

function handleAudioFile(file) {
  if (!file) return;
  audioFile = file;
  const nameEl = document.getElementById('audio-file-name');
  if (nameEl) {
    nameEl.textContent = file.name + ' (' + formatBytes(file.size) + ')';
    nameEl.style.display = '';
  }
  const drop = document.getElementById('audio-dropzone');
  if (drop) { drop.style.borderColor = 'var(--accent)'; drop.style.background = 'var(--accent-bg)'; }
  document.getElementById('audio-btn').disabled = false;
}

function handleAudioDrop(e) {
  e.preventDefault();
  document.getElementById('audio-dropzone')?.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (file) handleAudioFile(file);
}

async function runAudio() {
  if (!audioFile) return;
  audioAbort = new AbortController();
  const btn = document.getElementById('audio-btn');
  btn.disabled = true;
  setBox('audio-result-text', 'Transcribing…', 'stream');
  document.getElementById('audio-meta').style.display = 'none';
  document.getElementById('audio-segments').style.display = 'none';
  try {
    const form = new FormData();
    form.append('audio', audioFile);
    form.append('model', document.getElementById('audio-model').value || 'default');
    form.append('timestamps', document.getElementById('audio-timestamps').checked ? 'true' : 'false');
    const res  = await fetch('/audio/transcribe', { method: 'POST', body: form, signal: audioAbort.signal });
    const data = await res.json();
    if (!res.ok) {
      setBox('audio-result-text', String(data.error || data.message || res.status), 'error');
      return;
    }
    setBox('audio-result-text', data.text || '(empty)', 'ok');
    const meta  = document.getElementById('audio-meta');
    const parts = [];
    if (data.language) parts.push('Language: ' + data.language);
    if (data.confidence != null) parts.push('Confidence: ' + (data.confidence * 100).toFixed(1) + '%');
    if (parts.length) { meta.textContent = parts.join(' · '); meta.style.display = ''; }
    if (data.segments && data.segments.length) {
      const list = document.getElementById('audio-segments-list');
      clearEl(list);
      data.segments.forEach(seg => {
        const row = mkEl('div', null);
        row.style.cssText = 'display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;font-size:12px';
        const ts = mkEl('span', null, seg.start.toFixed(2) + 's–' + seg.end.toFixed(2) + 's');
        ts.style.cssText = 'flex-shrink:0;background:var(--accent-bg);color:var(--accent);padding:1px 7px;border-radius:10px';
        row.appendChild(ts);
        row.appendChild(mkEl('span', null, seg.text));
        list.appendChild(row);
      });
      document.getElementById('audio-segments').style.display = '';
    }
  } catch (e) {
    if (e.name !== 'AbortError') setBox('audio-result-text', 'Error: ' + e.message, 'error');
  } finally { btn.disabled = false; }
}

function stopAudio() { if (audioAbort) { audioAbort.abort(); audioAbort = null; } }
```

- [ ] **Step 7: Update `show()` to hook audio, logs, and system tab-open callbacks**

Find `show()` (around line 1081):

```js
  if (id === 'models') switchMdlTab('available');
}
```

Replace with:

```js
  if (id === 'models') switchMdlTab('available');
  if (id === 'audio')  loadAudioHealth();
  if (id === 'logs')   loadLogs();
  if (id === 'system') switchSysTab('info');
}
```

(`loadLogs` and `switchSysTab` are defined in Tasks 6 and 7; calling them here is safe because each guards with `if (!el) return`.)

- [ ] **Step 8: Run the tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/audio.spec.js --project=chromium 2>&1 | tail -15
```

- [ ] **Step 9: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/audio.spec.js tests/playwright/utils/selectors.js
git commit -m "feat(ui): add Audio tab with transcription via POST /audio/transcribe"
```

---

## Task 6: Add Logs tab

New sidebar tab that browses log files from `GET /logs`, views them with `GET /logs/{file}`, and clears them with `DELETE /logs/{file}`.

**Files:**
- Modify: `src/api/playground.html` (new "Tools" sidebar section, panel HTML after model panel, JS)
- Modify: `tests/playwright/utils/selectors.js`
- Create: `tests/playwright/tests/logs.spec.js`

- [ ] **Step 1: Add Logs selectors**

In `tests/playwright/utils/selectors.js`, add:

```js
  // ── Logs panel ───────────────────────────────────────────
  navLogs:           'button.nav-item:has-text("Logs")',
  panelLogs:         '#panel-logs',
  logsFileList:      '#logs-file-list',
  logsViewerHeader:  '#logs-viewer-header',
  logsViewerContent: '#logs-viewer-content',
```

- [ ] **Step 2: Write failing tests**

Create `tests/playwright/tests/logs.spec.js`:

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Logs', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navLogs).click();
    await expect(page.locator(S.panelLogs)).toHaveClass(/active/);
  });

  test('Logs nav item is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(S.navLogs)).toBeVisible();
  });

  test('log file list loads and shows files', async ({ page }) => {
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        available_log_files: [{ name: 'server.log', size_mb: 2.4, line_count: 18420, modified: '2026-04-05 10:00:00' }],
        log_directory: 'logs', log_level: 'info', total_log_size_mb: 2.4,
      }),
    }));
    await page.reload();
    await page.locator(S.navLogs).click();
    await expect(page.locator(S.logsFileList)).toContainText('server.log', { timeout: 5000 });
  });

  test('View button loads log content into viewer', async ({ page }) => {
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ available_log_files: [
        { name: 'server.log', size_mb: 0.1, line_count: 3, modified: '2026-04-05 10:00:00' },
      ], log_directory: 'logs', log_level: 'info', total_log_size_mb: 0.1 }),
    }));
    await page.route('/logs/server.log', route => route.fulfill({
      status: 200, contentType: 'text/plain', body: '[INFO] server started\n[INFO] listening on :8000',
    }));
    await page.reload();
    await page.locator(S.navLogs).click();
    await page.locator(S.logsFileList + ' button:has-text("View")').first().click();
    await expect(page.locator(S.logsViewerContent)).toContainText('server started', { timeout: 5000 });
  });

  test('Clear button calls DELETE /logs/{file} after confirmation', async ({ page }) => {
    let deleteCalled = '';
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ available_log_files: [
        { name: 'server.log', size_mb: 0.1, line_count: 3, modified: '2026-04-05 10:00:00' },
      ], log_directory: 'logs', log_level: 'info', total_log_size_mb: 0.1 }),
    }));
    await page.route('/logs/**', async route => {
      if (route.request().method() === 'DELETE') {
        deleteCalled = route.request().url();
        await route.fulfill({ status: 200 });
      } else { await route.continue(); }
    });
    await page.reload();
    await page.locator(S.navLogs).click();
    page.once('dialog', d => d.accept());
    await page.locator(S.logsFileList + ' button:has-text("Clear")').first().click();
    await expect.poll(() => deleteCalled, { timeout: 5000 }).toContain('/logs/server.log');
  });
});
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd tests/playwright && npx playwright test tests/logs.spec.js --project=chromium 2>&1 | grep -E "FAILED|passed|failed" | head -5
```

- [ ] **Step 4: Add a "Tools" sidebar section with Logs and System nav items**

In `src/api/playground.html`, find (around line 524):

```html
      <button class="nav-item" onclick="show('models',this)">
        <i class="nav-icon ri-box-3-line"></i> Models
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
```

Replace with:

```html
      <button class="nav-item" onclick="show('models',this)">
        <i class="nav-icon ri-box-3-line"></i> Models
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Tools</div>
      <button class="nav-item" onclick="show('logs',this)">
        <i class="nav-icon ri-file-list-3-line"></i> Logs
      </button>
      <button class="nav-item" onclick="show('system',this)">
        <i class="nav-icon ri-settings-3-line"></i> System
      </button>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-label">Reference</div>
```

- [ ] **Step 5: Add the Logs panel HTML**

Find (just before `<!-- Endpoints -->`):

```html
    </section><!-- /#panel-models -->

    <!-- Endpoints -->
```

Replace with:

```html
    </section><!-- /#panel-models -->

    <!-- Logs -->
    <section class="panel" id="panel-logs">
      <div class="panel-header">
        <div class="panel-title">Log Viewer</div>
        <div class="panel-desc">Browse and inspect server log files via <span class="ep-code">GET /logs</span>.</div>
      </div>
      <div style="display:grid;grid-template-columns:240px 1fr;gap:16px;align-items:start">
        <div class="card" style="padding:12px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div class="card-title" style="margin:0">Files</div>
            <button class="btn btn-ghost" style="padding:3px 10px;font-size:12px" onclick="loadLogs()">
              <i class="ri-refresh-line"></i>
            </button>
          </div>
          <div id="logs-file-list"><div class="mdl-empty">Loading…</div></div>
          <div id="logs-total" style="font-size:11px;color:var(--text-dim);margin-top:8px"></div>
        </div>
        <div class="card" style="padding:12px;min-height:240px">
          <div id="logs-viewer-header" style="font-size:12px;color:var(--text-muted);margin-bottom:8px;font-family:var(--mono)">Select a file to view</div>
          <pre id="logs-viewer-content" style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px;font-family:var(--mono);max-height:400px;overflow:auto;white-space:pre-wrap;word-break:break-all;color:var(--text)">—</pre>
        </div>
      </div>
    </section>

    <!-- Endpoints -->
```

- [ ] **Step 6: Add Logs JS functions**

Add before `formatBytes`:

```js
/* ── Logs tab ───────────────────────────────────────────── */
async function loadLogs() {
  const list  = document.getElementById('logs-file-list');
  const total = document.getElementById('logs-total');
  if (!list) return;
  clearEl(list);
  list.appendChild(mkEl('div', 'mdl-empty', 'Loading…'));
  try {
    const r = await fetch('/logs');
    if (!r.ok) {
      clearEl(list);
      list.appendChild(mkEl('div', 'mdl-empty', 'Failed: ' + r.status));
      return;
    }
    const data  = await r.json();
    const files = data.available_log_files || [];
    if (total) total.textContent = 'Total: ' + (data.total_log_size_mb || 0).toFixed(2) + ' MB · Level: ' + (data.log_level || '—');
    clearEl(list);
    if (!files.length) { list.appendChild(mkEl('div', 'mdl-empty', 'No log files found.')); return; }
    files.forEach(f => {
      const row = mkEl('div', null);
      row.style.cssText = 'padding:8px 0;border-bottom:1px solid var(--border2);display:flex;justify-content:space-between;align-items:center';
      const info = mkEl('div', null);
      const nameEl = mkEl('div', null, f.name); nameEl.style.cssText = 'font-weight:500;font-size:13px';
      const subEl  = mkEl('div', null, (f.size_mb || 0).toFixed(2) + ' MB · ' + (f.line_count || 0).toLocaleString() + ' lines');
      subEl.style.cssText = 'font-size:11px;color:var(--text-dim)';
      info.appendChild(nameEl); info.appendChild(subEl); row.appendChild(info);
      const btns = mkEl('div', null); btns.style.cssText = 'display:flex;gap:4px;flex-shrink:0';
      const vBtn = mkEl('button', 'btn btn-ghost', 'View');
      vBtn.style.cssText = 'font-size:12px;padding:3px 10px';
      vBtn.onclick = () => viewLog(f.name);
      const cBtn = mkEl('button', 'btn btn-ghost', 'Clear');
      cBtn.style.cssText = 'font-size:12px;padding:3px 10px;color:var(--red);border-color:var(--red)';
      cBtn.onclick = () => clearLog(f.name);
      btns.appendChild(vBtn); btns.appendChild(cBtn); row.appendChild(btns);
      list.appendChild(row);
    });
  } catch (e) {
    clearEl(list);
    list.appendChild(mkEl('div', 'mdl-empty', 'Error: ' + e.message));
  }
}

async function viewLog(name) {
  const hdr = document.getElementById('logs-viewer-header');
  const pre = document.getElementById('logs-viewer-content');
  if (!pre) return;
  pre.textContent = 'Loading…';
  if (hdr) hdr.textContent = name;
  try {
    const r = await fetch('/logs/' + encodeURIComponent(name));
    if (!r.ok) { pre.textContent = 'Failed: ' + r.status; return; }
    const text = await r.text();
    pre.textContent = text.replace(/\x1b\[[0-9;]*m/g, '');
    pre.scrollTop   = pre.scrollHeight;
  } catch (e) { pre.textContent = 'Error: ' + e.message; }
}

async function clearLog(name) {
  if (!confirm('Clear log file "' + name + '"?')) return;
  try {
    const r = await fetch('/logs/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) { alert('Clear failed: ' + r.status); return; }
    loadLogs();
    const hdr = document.getElementById('logs-viewer-header');
    const pre = document.getElementById('logs-viewer-content');
    if (hdr && hdr.textContent === name) { pre.textContent = '—'; hdr.textContent = 'Select a file to view'; }
  } catch (e) { alert('Error: ' + e.message); }
}
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/logs.spec.js --project=chromium 2>&1 | tail -10
```

- [ ] **Step 8: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/logs.spec.js tests/playwright/utils/selectors.js
git commit -m "feat(ui): add Logs tab with file viewer and clear via GET|DELETE /logs"
```

---

## Task 7: Add System tab (Info · Performance · Config)

**Files:**
- Modify: `src/api/playground.html` (panel HTML just before Endpoints, JS functions)
- Modify: `tests/playwright/utils/selectors.js`
- Create: `tests/playwright/tests/system.spec.js`

- [ ] **Step 1: Add System selectors**

In `tests/playwright/utils/selectors.js`, add:

```js
  // ── System panel ─────────────────────────────────────────
  navSystem:        'button.nav-item:has-text("System")',
  panelSystem:      '#panel-system',
  sysTabInfo:       '#sys-tab-info',
  sysTabPerf:       '#sys-tab-perf',
  sysTabConfig:     '#sys-tab-config',
  sysInfoPane:      '#sys-info-pane',
  sysPerfPane:      '#sys-perf-pane',
  sysConfigPane:    '#sys-config-pane',
  sysInfoCards:     '#sys-info-cards',
  sysPerfTiles:     '#sys-perf-tiles',
  sysOptTips:       '#sys-opt-tips',
  sysConfigContent: '#sys-config-content',
```

- [ ] **Step 2: Write failing tests**

Create `tests/playwright/tests/system.spec.js`:

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('System tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navSystem).click();
    await expect(page.locator(S.panelSystem)).toHaveClass(/active/);
  });

  test('System nav item is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(S.navSystem)).toBeVisible();
  });

  test('three sub-tabs are visible', async ({ page }) => {
    await expect(page.locator(S.sysTabInfo)).toBeVisible();
    await expect(page.locator(S.sysTabPerf)).toBeVisible();
    await expect(page.locator(S.sysTabConfig)).toBeVisible();
  });

  test('Info sub-tab shows system data', async ({ page }) => {
    await page.route('/system/info', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ hostname: 'testhost', os: 'macOS', arch: 'aarch64',
        cpu_count: 8, total_memory_mb: 16384, server_version: '0.5.0', uptime_seconds: 1200 }),
    }));
    await page.route('/system/gpu/stats', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify([]),
    }));
    await page.locator(S.sysTabInfo).click();
    await expect(page.locator(S.sysInfoCards)).toContainText('macOS', { timeout: 5000 });
  });

  test('Performance sub-tab loads metrics', async ({ page }) => {
    await page.route('/performance', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ avg_latency_ms: 14, p50_latency_ms: 12, p95_latency_ms: 38,
        p99_latency_ms: 62, total_requests: 500, requests_per_second: 142, error_rate: 0.01 }),
    }));
    await page.locator(S.sysTabPerf).click();
    await expect(page.locator(S.sysPerfTiles)).toContainText('14', { timeout: 5000 });
  });

  test('Optimization Tips button calls /performance/optimize', async ({ page }) => {
    await page.route('/performance', route => route.fulfill({
      status: 200, contentType: 'application/json', body: '{}',
    }));
    await page.route('/performance/optimize', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ suggestions: [
        { category: 'caching', description: 'Increase cache size', impact: 'high', action: 'Set cache_size=1000' },
      ]}),
    }));
    await page.locator(S.sysTabPerf).click();
    await page.locator(S.sysPerfPane + ' button:has-text("Optimization Tips")').click();
    await expect(page.locator(S.sysOptTips)).toContainText('Increase cache size', { timeout: 5000 });
  });

  test('Config sub-tab renders JSON from /system/config', async ({ page }) => {
    await page.route('/system/config', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ workers: 8, port: 8000, log_level: 'info' }),
    }));
    await page.locator(S.sysTabConfig).click();
    await expect(page.locator(S.sysConfigContent)).toContainText('"workers"', { timeout: 5000 });
  });
});
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd tests/playwright && npx playwright test tests/system.spec.js --project=chromium 2>&1 | grep -E "FAILED|passed|failed" | head -5
```

- [ ] **Step 4: Add the System panel HTML**

Find (just before `<!-- Endpoints -->`):

```html
    <!-- Endpoints -->
    <section class="panel" id="panel-endpoints">
```

Add the System panel before it:

```html
    <!-- System -->
    <section class="panel" id="panel-system">
      <div class="panel-header">
        <div class="panel-title">System</div>
        <div class="panel-desc">Server info, performance metrics, and configuration.</div>
      </div>

      <div class="subtab-bar">
        <button class="subtab active" id="sys-tab-info"   onclick="switchSysTab('info')"><i class="ri-server-line"></i> Info</button>
        <button class="subtab"        id="sys-tab-perf"   onclick="switchSysTab('perf')"><i class="ri-speed-line"></i> Performance</button>
        <button class="subtab"        id="sys-tab-config" onclick="switchSysTab('config')"><i class="ri-code-s-slash-line"></i> Config</button>
      </div>

      <!-- Info sub-tab -->
      <div id="sys-info-pane">
        <div style="display:flex;justify-content:flex-end;margin-bottom:12px">
          <button class="btn btn-ghost" style="padding:4px 12px;font-size:13px" onclick="loadSystemInfo()">
            <i class="ri-refresh-line"></i> Refresh
          </button>
        </div>
        <div id="sys-info-cards" style="display:grid;grid-template-columns:1fr 1fr;gap:12px"></div>
      </div>

      <!-- Performance sub-tab -->
      <div id="sys-perf-pane" style="display:none">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <button class="btn btn-ghost" style="padding:4px 12px;font-size:13px" onclick="loadPerformance()">
            <i class="ri-refresh-line"></i> Refresh
          </button>
          <button class="btn btn-ghost" style="padding:4px 12px;font-size:13px;color:var(--green);border-color:var(--green)"
                  onclick="getOptimizationTips()">
            <i class="ri-lightbulb-line"></i> Optimization Tips
          </button>
        </div>
        <div id="sys-perf-tiles" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;margin-bottom:16px"></div>
        <div id="sys-opt-tips" style="display:none"></div>
      </div>

      <!-- Config sub-tab -->
      <div id="sys-config-pane" style="display:none">
        <div style="display:flex;justify-content:flex-end;margin-bottom:12px">
          <button class="btn btn-ghost" style="padding:4px 12px;font-size:13px" onclick="copyConfig()">
            <i class="ri-file-copy-line"></i> Copy Config
          </button>
        </div>
        <pre id="sys-config-content" style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;font-size:12px;font-family:var(--mono);max-height:500px;overflow:auto;white-space:pre-wrap;word-break:break-all">Loading…</pre>
      </div>
    </section>

    <!-- Endpoints -->
    <section class="panel" id="panel-endpoints">
```

- [ ] **Step 5: Add System JS functions**

Add before `formatBytes`:

```js
/* ── System tab ─────────────────────────────────────────── */
const SYS_SUBTABS = ['info', 'perf', 'config'];

function switchSysTab(tab) {
  SYS_SUBTABS.forEach(id => {
    const pane = document.getElementById('sys-' + id + '-pane');
    const btn  = document.getElementById('sys-tab-' + id);
    if (pane) pane.style.display = id === tab ? '' : 'none';
    if (btn)  btn.classList[id === tab ? 'add' : 'remove']('active');
  });
  if (tab === 'info')   loadSystemInfo();
  if (tab === 'perf')   loadPerformance();
  if (tab === 'config') loadConfig();
}

function makeSysCard(title, rows) {
  const card = mkEl('div', 'card'); card.style.padding = '12px';
  card.appendChild(mkEl('div', 'card-title', title));
  rows.forEach(([label, value]) => {
    const row = mkEl('div', null);
    row.style.cssText = 'display:flex;justify-content:space-between;margin-bottom:3px;font-size:13px';
    const lEl = mkEl('span', null, label); lEl.style.color = 'var(--text-muted)';
    const vEl = mkEl('span', null, value != null ? String(value) : '—'); vEl.style.fontWeight = '500';
    row.appendChild(lEl); row.appendChild(vEl); card.appendChild(row);
  });
  return card;
}

async function loadSystemInfo() {
  const cards = document.getElementById('sys-info-cards');
  if (!cards) return;
  clearEl(cards);
  try {
    const [ri, rg] = await Promise.all([fetch('/system/info'), fetch('/system/gpu/stats')]);
    const info = ri.ok ? await ri.json() : {};
    const gpus = rg.ok ? await rg.json() : [];
    const ramGB   = info.total_memory_mb     ? (info.total_memory_mb / 1024).toFixed(1) + ' GB'       : null;
    const ramFree = info.available_memory_mb ? (info.available_memory_mb / 1024).toFixed(1) + ' GB'   : null;
    cards.appendChild(makeSysCard('System',  [['OS', info.os], ['Arch', info.arch], ['Hostname', info.hostname], ['CPUs', info.cpu_count]]));
    cards.appendChild(makeSysCard('Server',  [['Version', info.server_version], ['Workers', info.worker_count], ['Uptime', info.uptime_seconds != null ? fmtUptime(info.uptime_seconds) : null], ['Rust', info.rust_version]]));
    cards.appendChild(makeSysCard('Memory',  [['Total', ramGB], ['Free', ramFree]]));
    if (gpus && gpus.length) {
      gpus.forEach((g, i) => cards.appendChild(makeSysCard('GPU' + (gpus.length > 1 ? ' ' + (i + 1) : ''), [
        ['Device', g.name],
        ['Util',   g.util_pct  != null ? g.util_pct + '%'  : null],
        ['Temp',   g.temp_c    != null ? g.temp_c + '°C'   : null],
        ['VRAM',   g.vram_total_mb != null ? ((g.vram_total_mb - (g.vram_free_mb || 0)) / 1024).toFixed(1) + ' / ' + (g.vram_total_mb / 1024).toFixed(1) + ' GB' : null],
      ])));
    } else {
      cards.appendChild(makeSysCard('GPU', [['Status', 'No GPU detected']]));
    }
  } catch (e) {
    cards.appendChild(mkEl('div', 'mdl-empty', 'Failed to load system info: ' + e.message));
  }
}

async function loadPerformance() {
  const tiles = document.getElementById('sys-perf-tiles');
  if (!tiles) return;
  clearEl(tiles);
  document.getElementById('sys-opt-tips').style.display = 'none';
  try {
    const r = await fetch('/performance');
    if (!r.ok) { tiles.appendChild(mkEl('div', 'mdl-empty', 'Failed: ' + r.status)); return; }
    const d = await r.json();
    [
      ['Avg Latency',  d.avg_latency_ms  != null ? d.avg_latency_ms  + ' ms' : '—'],
      ['P50 Latency',  d.p50_latency_ms  != null ? d.p50_latency_ms  + ' ms' : '—'],
      ['P95 Latency',  d.p95_latency_ms  != null ? d.p95_latency_ms  + ' ms' : '—'],
      ['P99 Latency',  d.p99_latency_ms  != null ? d.p99_latency_ms  + ' ms' : '—'],
      ['Throughput',   d.requests_per_second != null ? d.requests_per_second + ' req/s' : '—'],
      ['Error Rate',   d.error_rate != null ? (d.error_rate * 100).toFixed(2) + '%' : '—'],
      ['Total Reqs',   d.total_requests != null ? d.total_requests.toLocaleString() : '—'],
    ].forEach(([label, value]) => {
      const tile = mkEl('div', 'stat-tile');
      tile.appendChild(mkEl('div', 'stat-label', label));
      tile.appendChild(mkEl('div', 'stat-value', value));
      tiles.appendChild(tile);
    });
  } catch (e) { tiles.appendChild(mkEl('div', 'mdl-empty', 'Error: ' + e.message)); }
}

async function getOptimizationTips() {
  const box = document.getElementById('sys-opt-tips');
  if (!box) return;
  box.style.display = '';
  clearEl(box);
  box.appendChild(mkEl('div', 'mdl-empty', 'Loading suggestions…'));
  try {
    const r = await fetch('/performance/optimize');
    if (!r.ok) {
      clearEl(box); box.appendChild(mkEl('div', 'mdl-empty', 'Failed: ' + r.status)); return;
    }
    const data        = await r.json();
    const suggestions = Array.isArray(data) ? data : (data.suggestions || []);
    clearEl(box);
    if (!suggestions.length) { box.appendChild(mkEl('div', 'mdl-empty', 'No suggestions available.')); return; }
    suggestions.forEach(s => {
      const card = mkEl('div', 'card'); card.style.cssText = 'padding:10px;margin-bottom:8px';
      const hdr  = mkEl('div', null);  hdr.style.cssText  = 'display:flex;gap:8px;align-items:center;margin-bottom:4px';
      const impact  = (s.impact   || '').toLowerCase();
      const impCls  = impact === 'high' ? 'task-detection' : impact === 'medium' ? 'task-tts' : 'task-other';
      hdr.appendChild(mkEl('span', 'model-task-badge ' + impCls, s.category || 'general'));
      hdr.appendChild(mkEl('span', 'model-task-badge task-classify', impact + ' impact'));
      card.appendChild(hdr);
      card.appendChild(mkEl('div', null, s.description || ''));
      if (s.action) {
        const a = mkEl('div', null, s.action);
        a.style.cssText = 'font-size:12px;color:var(--text-muted);margin-top:4px;font-family:var(--mono)';
        card.appendChild(a);
      }
      box.appendChild(card);
    });
  } catch (e) {
    clearEl(box); box.appendChild(mkEl('div', 'mdl-empty', 'Error: ' + e.message));
  }
}

async function loadConfig() {
  const pre = document.getElementById('sys-config-content');
  if (!pre) return;
  pre.textContent = 'Loading…';
  try {
    const r = await fetch('/system/config');
    if (!r.ok) { pre.textContent = 'Failed: ' + r.status; return; }
    const data = await r.json();
    pre.textContent = JSON.stringify(data, null, 2);
  } catch (e) { pre.textContent = 'Error: ' + e.message; }
}

function copyConfig() {
  const pre = document.getElementById('sys-config-content');
  if (!pre || !navigator.clipboard) return;
  navigator.clipboard.writeText(pre.textContent).then(() => alert('Config copied to clipboard.'));
}
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cd tests/playwright && npx playwright test tests/system.spec.js --project=chromium 2>&1 | tail -10
```

- [ ] **Step 7: Run the full playwright suite to catch regressions**

```bash
cd tests/playwright && npx playwright test --project=chromium 2>&1 | tail -15
```

Expected: all tests pass (or only pre-existing failures from earlier sessions).

- [ ] **Step 8: Commit**

```bash
git add src/api/playground.html tests/playwright/tests/system.spec.js tests/playwright/utils/selectors.js
git commit -m "feat(ui): add System tab with Info, Performance, and Config sub-tabs"
```

---

## Self-Review

**Spec coverage:**
- Bug fix → Task 1 ✓
- TTS voice select → Task 2 ✓
- Delete model → Task 3 ✓
- SOTA sub-tab → Task 4 ✓
- Audio tab → Task 5 ✓
- Logs tab → Task 6 ✓
- System tab (Info + Performance + Config) → Task 7 ✓
- `show()` hooks for audio/logs/system on tab open → Task 5 Step 7 ✓

**Type consistency:**
- `loadVoicesForEngine(engineId, voiceSelectId)` — called in engine `onchange` (Task 2 Step 4) and `refreshTtsSelects` (Task 2 Step 6) ✓
- `deleteModel(name)` — called from `delBtn.onclick` in Task 3 Step 3, defined in Task 3 Step 4 ✓
- `loadSotaModels()` / `downloadSotaModel(modelId, name)` — wired in Task 4 Steps 5–7 ✓
- `loadAudioHealth()` / `handleAudioFile()` / `runAudio()` / `stopAudio()` — all in Task 5 Step 6 ✓
- `loadLogs()` / `viewLog(name)` / `clearLog(name)` — all in Task 6 Step 6 ✓
- `switchSysTab(tab)` / `makeSysCard()` / `loadSystemInfo()` / `loadPerformance()` / `getOptimizationTips()` / `loadConfig()` / `copyConfig()` — all in Task 7 Step 5 ✓
- All `document.getElementById` calls reference IDs defined in the HTML steps ✓
- `formatBytes` referenced in Task 5 — already exists in the file ✓
- `fmtUptime` referenced in Task 7 — already exists in the file ✓
- `clearEl` / `mkEl` / `setBox` — already exist in the file ✓

**Placeholder scan:** None found.
