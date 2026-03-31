# Playwright UI Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tests/playwright/` alongside `tests/jest/` with full E2E browser tests for every section of `playground.html`, auto-building and starting the dashboard server before tests run.

**Architecture:** Playwright project mirrors Jest structure — one spec file per UI section, a `global-setup.js` that builds and starts the Rust binary from `.worktrees/dashboard/`, and `utils/selectors.js` for shared element selectors. Tests hit the real server and tolerate error responses from unloaded models.

**Tech Stack:** `@playwright/test`, Node.js, Chromium (headless), dotenv

---

## File Map

| File | Purpose |
|---|---|
| `tests/playwright/package.json` | npm project, scripts, `@playwright/test` dep |
| `tests/playwright/.env.test` | `BASE_URL`, `BUILD_SERVER`, `SERVER_BINARY` |
| `tests/playwright/.gitignore` | ignore results/, node_modules/, .server.pid |
| `tests/playwright/global-setup.js` | build + spawn server, poll `/health` |
| `tests/playwright/global-teardown.js` | kill server via PID file |
| `tests/playwright/playwright.config.js` | Chromium, reporters, timeouts, globalSetup/Teardown |
| `tests/playwright/utils/selectors.js` | all `#id` / `.class` selectors for playground.html |
| `tests/playwright/tests/navigation.spec.js` | sidebar nav, panel switching, active state |
| `tests/playwright/tests/health.spec.js` | topbar health badge + status panel content |
| `tests/playwright/tests/tts.spec.js` | TTS form fields, submit, audio/error output |
| `tests/playwright/tests/classify.spec.js` | dropzone, file upload, classify button, output |
| `tests/playwright/tests/chat.spec.js` | LLM chat input/send/history + completion panel |
| `tests/playwright/tests/dashboard.spec.js` | metrics tiles, spark canvas, resource bars |
| `tests/playwright/tests/models.spec.js` | download form, source selector, task list |
| `tests/playwright/tests/playground.spec.js` | playground tab: TTS/classify/chat/completion sub-tabs |

---

## Task 1: Project scaffold

**Files:**
- Create: `tests/playwright/package.json`
- Create: `tests/playwright/.env.test`
- Create: `tests/playwright/.gitignore`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "torch-inference-playwright-tests",
  "version": "1.0.0",
  "description": "Playwright E2E UI tests for the Kolosal Inference dashboard",
  "private": true,
  "scripts": {
    "test":            "playwright test",
    "test:headed":     "HEADED=1 playwright test",
    "test:ui":         "playwright test --ui",
    "test:navigation": "playwright test tests/navigation.spec.js",
    "test:health":     "playwright test tests/health.spec.js",
    "test:tts":        "playwright test tests/tts.spec.js",
    "test:classify":   "playwright test tests/classify.spec.js",
    "test:chat":       "playwright test tests/chat.spec.js",
    "test:dashboard":  "playwright test tests/dashboard.spec.js",
    "test:models":     "playwright test tests/models.spec.js",
    "test:playground": "playwright test tests/playground.spec.js"
  },
  "dependencies": {
    "dotenv": "^16.4.5"
  },
  "devDependencies": {
    "@playwright/test": "^1.44.0"
  }
}
```

- [ ] **Step 2: Create .env.test**

```
# Playwright UI test configuration
BASE_URL=http://localhost:8000

# Set to "false" to skip cargo build (use pre-built binary)
BUILD_SERVER=true

# Override path to pre-built binary
# (default: .worktrees/dashboard/target/release/torch-inference)
# SERVER_BINARY=/path/to/binary
```

- [ ] **Step 3: Create .gitignore**

```
node_modules/
results/
.server.pid
```

- [ ] **Step 4: Install dependencies**

```bash
cd tests/playwright
npm install
npx playwright install chromium
```

Expected: `node_modules/` created, chromium browser downloaded.

- [ ] **Step 5: Commit**

```bash
git add tests/playwright/package.json tests/playwright/.env.test tests/playwright/.gitignore tests/playwright/package-lock.json
git commit -m "feat(playwright): scaffold project with package.json and env config"
```

---

## Task 2: Server lifecycle — global-setup.js and global-teardown.js

**Files:**
- Create: `tests/playwright/global-setup.js`
- Create: `tests/playwright/global-teardown.js`

- [ ] **Step 1: Create global-setup.js**

Note: uses `execFileSync` (not `execSync`) and `execFile` (not `exec`) — no shell involvement, no injection surface.

```js
require('dotenv').config({ path: __dirname + '/.env.test' });
const { execFile, execFileSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.BASE_URL || 'http://localhost:8000';
const DASHBOARD_DIR = path.resolve(__dirname, '../../.worktrees/dashboard');
const PID_FILE = path.join(__dirname, '.server.pid');

function checkHealth() {
  return new Promise(resolve => {
    const req = http.get(`${BASE_URL}/health`, { timeout: 2000 }, res => {
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function waitForServer(maxWaitMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    if (await checkHealth()) return true;
    await new Promise(r => setTimeout(r, 500));
  }
  return false;
}

module.exports = async function globalSetup() {
  const resultsDir = path.join(__dirname, 'results');
  if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

  const alreadyRunning = await checkHealth();
  if (alreadyRunning) {
    console.log(`[playwright-setup] Server already running at ${BASE_URL}`);
    return;
  }

  const buildServer = process.env.BUILD_SERVER !== 'false';
  if (buildServer) {
    console.log('[playwright-setup] Building dashboard server (cargo build --release)…');
    // execFileSync avoids a shell — arguments are passed as an array, not a string
    execFileSync('cargo', ['build', '--release'], {
      cwd: DASHBOARD_DIR,
      stdio: 'inherit',
    });
    console.log('[playwright-setup] Build complete.');
  }

  const binaryPath = process.env.SERVER_BINARY
    ? path.resolve(process.env.SERVER_BINARY)
    : path.join(DASHBOARD_DIR, 'target', 'release', 'torch-inference');

  if (!fs.existsSync(binaryPath)) {
    throw new Error(
      `Server binary not found at ${binaryPath}. ` +
      `Set BUILD_SERVER=true or provide SERVER_BINARY in .env.test.`
    );
  }

  console.log(`[playwright-setup] Starting server: ${binaryPath}`);
  // execFile — no shell, binary path is a resolved absolute path
  const serverProcess = execFile(binaryPath, [], {
    cwd: DASHBOARD_DIR,
    env: { ...process.env },
    detached: false,
  });
  serverProcess.stdout?.on('data', d => process.stdout.write(`[server] ${d}`));
  serverProcess.stderr?.on('data', d => process.stderr.write(`[server] ${d}`));
  serverProcess.on('exit', code => {
    if (code !== null && code !== 0) {
      console.error(`[server] exited with code ${code}`);
    }
  });

  fs.writeFileSync(PID_FILE, String(serverProcess.pid));

  const ready = await waitForServer(30000);
  if (!ready) throw new Error('Server failed to become healthy within 30s');
  console.log('[playwright-setup] Server is ready');
};
```

- [ ] **Step 2: Create global-teardown.js**

```js
const fs = require('fs');
const path = require('path');

const PID_FILE = path.join(__dirname, '.server.pid');

module.exports = async function globalTeardown() {
  if (!fs.existsSync(PID_FILE)) return;
  const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8'), 10);
  try {
    process.kill(pid, 'SIGTERM');
    console.log(`[playwright-teardown] Sent SIGTERM to PID ${pid}`);
  } catch (e) {
    console.warn(`[playwright-teardown] Could not kill PID ${pid}: ${e.message}`);
  }
  fs.unlinkSync(PID_FILE);
};
```

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/global-setup.js tests/playwright/global-teardown.js
git commit -m "feat(playwright): add global setup/teardown with server auto-build and health poll"
```

---

## Task 3: Playwright config

**Files:**
- Create: `tests/playwright/playwright.config.js`

- [ ] **Step 1: Create playwright.config.js**

```js
require('dotenv').config({ path: __dirname + '/.env.test' });
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  workers: 1,
  reporter: [
    ['html',  { outputFolder: 'results/report', open: 'never' }],
    ['junit', { outputFile: 'results/junit.xml' }],
    ['list'],
  ],
  use: {
    baseURL:    process.env.BASE_URL || 'http://localhost:8000',
    headless:   process.env.HEADED !== '1',
    screenshot: 'only-on-failure',
  },
  globalSetup:    './global-setup.js',
  globalTeardown: './global-teardown.js',
  outputDir: 'results/test-results',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
```

- [ ] **Step 2: Verify config**

```bash
cd tests/playwright
npx playwright test --list
```

Expected: no config errors, lists zero tests (no spec files yet).

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/playwright.config.js
git commit -m "feat(playwright): add playwright.config.js with Chromium, reporters, timeouts"
```

---

## Task 4: Selectors utility

**Files:**
- Create: `tests/playwright/utils/selectors.js`

- [ ] **Step 1: Create utils/selectors.js**

```js
/**
 * Shared Playwright selectors for playground.html.
 * All IDs are taken directly from the HTML source in
 * .worktrees/dashboard/src/api/playground.html.
 */
module.exports = {
  // ── Topbar ──────────────────────────────────────────────
  healthDot:  '#health-dot',
  healthText: '#health-text',

  // ── Sidebar nav — uses Playwright :has-text() pseudo-class
  navStatus:     'button.nav-item:has-text("Status")',
  navTTS:        'button.nav-item:has-text("TTS Stream")',
  navClassify:   'button.nav-item:has-text("Classify")',
  navLLM:        'button.nav-item:has-text("LLM Chat")',
  navCompletion: 'button.nav-item:has-text("Completion")',
  navDashboard:  'button.nav-item:has-text("Dashboard")',
  navEndpoints:  'button.nav-item:has-text("Endpoints")',

  // ── Panels ───────────────────────────────────────────────
  panelStatus:     '#panel-status',
  panelTTS:        '#panel-tts',
  panelClassify:   '#panel-classify',
  panelLLM:        '#panel-llm',
  panelCompletion: '#panel-completion',
  panelDashboard:  '#panel-dashboard',
  panelEndpoints:  '#panel-endpoints',

  // ── Status panel ────────────────────────────────────────
  sStatus:   '#s-status',
  sUptime:   '#s-uptime',
  sActive:   '#s-active',
  sTotal:    '#s-total',
  sLatency:  '#s-latency',
  sErrors:   '#s-errors',
  healthRaw: '#health-raw',

  // ── TTS panel ───────────────────────────────────────────
  ttsText:      '#tts-text',
  ttsEngine:    '#tts-engine',
  ttsVoice:     '#tts-voice',
  ttsSpeed:     '#tts-speed',
  ttsBtn:       '#tts-btn',
  ttsStatus:    '#tts-status',
  ttsAudioWrap: '#tts-audio-wrap',
  ttsAudio:     '#tts-audio',

  // ── Classify panel ──────────────────────────────────────
  dropzone:   '#dropzone',
  imgFile:    '#img-file',
  imgPreview: '#img-preview',
  clsTopk:    '#cls-topk',
  clsW:       '#cls-w',
  clsH:       '#cls-h',
  clsBtn:     '#cls-btn',
  clsOut:     '#cls-out',

  // ── LLM Chat panel ──────────────────────────────────────
  chatHistory: '#chat-history',
  chatInput:   '#chat-input',
  chatBtn:     '#chat-btn',
  chatModel:   '#chat-model',
  chatMaxtok:  '#chat-maxtok',
  chatTemp:    '#chat-temp',

  // ── Completion panel ────────────────────────────────────
  cmpPrompt: '#cmp-prompt',
  cmpModel:  '#cmp-model',
  cmpMaxtok: '#cmp-maxtok',
  cmpTemp:   '#cmp-temp',
  cmpTopp:   '#cmp-topp',
  cmpBtn:    '#cmp-btn',
  cmpOut:    '#cmp-out',

  // ── Dashboard panel ─────────────────────────────────────
  dashTabOverview:   '#dash-tab-overview',
  dashTabPlayground: '#dash-tab-playground',
  dashOverview:      '#dash-overview',
  dashPlayground:    '#dash-playground',
  dUptime:       '#d-uptime',
  dTotal:        '#d-total',
  dActive:       '#d-active',
  dLatency:      '#d-latency',
  dErrors:       '#d-errors',
  dRps:          '#d-rps',
  dashSpark:     '#dash-spark',
  dashLiveBadge: '#dash-live-badge',
  barCpu:  '#bar-cpu',
  barRam:  '#bar-ram',
  barGpu:  '#bar-gpu',
  valCpu:  '#val-cpu',
  valRam:  '#val-ram',
  valGpu:  '#val-gpu',
  gpuDeviceInfo: '#gpu-device-info',

  // ── Model download manager ──────────────────────────────
  dlSource:       '#dl-source',
  dlRepo:         '#dl-repo',
  dlRevision:     '#dl-revision',
  dlRevisionWrap: '#dl-revision-wrap',
  dlBtn:          '#dl-btn',
  dlError:        '#dl-error',
  dlTasksEmpty:   '#dl-tasks-empty',
  dlTasksList:    '#dl-tasks-list',

  // ── Playground sub-tabs ─────────────────────────────────
  pgTabTTS:        '#pg-tab-tts',
  pgTabClassify:   '#pg-tab-classify',
  pgTabLLM:        '#pg-tab-llm',
  pgTabCompletion: '#pg-tab-completion',

  // ── Playground TTS ──────────────────────────────────────
  pgTTS:          '#pg-tts',
  pgTtsText:      '#pg-tts-text',
  pgTtsEngine:    '#pg-tts-engine',
  pgTtsVoice:     '#pg-tts-voice',
  pgTtsSpeed:     '#pg-tts-speed',
  pgTtsBtn:       '#pg-tts-btn',
  pgTtsStatus:    '#pg-tts-status',
  pgTtsAudioWrap: '#pg-tts-audio-wrap',

  // ── Playground Classify ─────────────────────────────────
  pgClassify:   '#pg-classify',
  pgDropzone:   '#pg-dropzone',
  pgImgFile:    '#pg-img-file',
  pgImgPreview: '#pg-img-preview',
  pgClsTopk:    '#pg-cls-topk',
  pgClsW:       '#pg-cls-w',
  pgClsH:       '#pg-cls-h',
  pgClsBtn:     '#pg-cls-btn',
  pgClsOut:     '#pg-cls-out',

  // ── Playground LLM Chat ─────────────────────────────────
  pgLLM:         '#pg-llm',
  pgChatHistory: '#pg-chat-history',
  pgChatInput:   '#pg-chat-input',
  pgChatBtn:     '#pg-chat-btn',
  pgChatModel:   '#pg-chat-model',
  pgChatMaxtok:  '#pg-chat-maxtok',
  pgChatTemp:    '#pg-chat-temp',

  // ── Playground Completion ───────────────────────────────
  pgCompletion: '#pg-completion',
  pgCmpPrompt:  '#pg-cmp-prompt',
  pgCmpModel:   '#pg-cmp-model',
  pgCmpMaxtok:  '#pg-cmp-maxtok',
  pgCmpTemp:    '#pg-cmp-temp',
  pgCmpTopp:    '#pg-cmp-topp',
  pgCmpBtn:     '#pg-cmp-btn',
  pgCmpOut:     '#pg-cmp-out',
};
```

- [ ] **Step 2: Commit**

```bash
git add tests/playwright/utils/selectors.js
git commit -m "feat(playwright): add shared selectors utility for playground.html"
```

---

## Task 5: Navigation tests

**Files:**
- Create: `tests/playwright/tests/navigation.spec.js`

- [ ] **Step 1: Create navigation.spec.js**

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('page title contains Kolosal', async ({ page }) => {
    await expect(page).toHaveTitle(/Kolosal/);
  });

  test('sidebar renders 7 nav items', async ({ page }) => {
    await expect(page.locator('.nav-item')).toHaveCount(7);
  });

  test('Status is active and its panel visible by default', async ({ page }) => {
    await expect(page.locator(S.navStatus)).toHaveClass(/active/);
    await expect(page.locator(S.panelStatus)).toHaveClass(/active/);
  });

  test('clicking TTS Stream shows TTS panel', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator(S.panelTTS)).toHaveClass(/active/);
    await expect(page.locator(S.navTTS)).toHaveClass(/active/);
  });

  test('clicking Classify shows Classify panel', async ({ page }) => {
    await page.locator(S.navClassify).click();
    await expect(page.locator(S.panelClassify)).toHaveClass(/active/);
  });

  test('clicking LLM Chat shows LLM panel', async ({ page }) => {
    await page.locator(S.navLLM).click();
    await expect(page.locator(S.panelLLM)).toHaveClass(/active/);
  });

  test('clicking Completion shows Completion panel', async ({ page }) => {
    await page.locator(S.navCompletion).click();
    await expect(page.locator(S.panelCompletion)).toHaveClass(/active/);
  });

  test('clicking Dashboard shows Dashboard panel', async ({ page }) => {
    await page.locator(S.navDashboard).click();
    await expect(page.locator(S.panelDashboard)).toHaveClass(/active/);
  });

  test('clicking Endpoints shows Endpoints panel', async ({ page }) => {
    await page.locator(S.navEndpoints).click();
    await expect(page.locator(S.panelEndpoints)).toHaveClass(/active/);
  });

  test('only one panel is active at a time', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator('.panel.active')).toHaveCount(1);
    await page.locator(S.navClassify).click();
    await expect(page.locator('.panel.active')).toHaveCount(1);
  });

  test('only one nav item is active at a time', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator('.nav-item.active')).toHaveCount(1);
    await page.locator(S.navDashboard).click();
    await expect(page.locator('.nav-item.active')).toHaveCount(1);
  });

  test('Endpoints panel lists endpoint rows', async ({ page }) => {
    await page.locator(S.navEndpoints).click();
    await expect(
      page.locator(S.panelEndpoints).locator('.endpoint-row').first()
    ).toBeVisible();
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/navigation.spec.js
```

Expected: all 11 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/navigation.spec.js
git commit -m "feat(playwright): add navigation spec — sidebar nav, panel switching, active state"
```

---

## Task 6: Health and status panel tests

**Files:**
- Create: `tests/playwright/tests/health.spec.js`

- [ ] **Step 1: Create health.spec.js**

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Health badge and Status panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Status panel is active by default — no nav click needed
  });

  test('health badge is visible in topbar', async ({ page }) => {
    await expect(page.locator(S.healthText)).toBeVisible();
  });

  test('health badge leaves "checking" state within 5s', async ({ page }) => {
    await expect(page.locator(S.healthText)).not.toHaveText('checking', { timeout: 5000 });
  });

  test('status panel shows all 6 stat cards', async ({ page }) => {
    await expect(page.locator(S.sStatus)).toBeVisible();
    await expect(page.locator(S.sUptime)).toBeVisible();
    await expect(page.locator(S.sActive)).toBeVisible();
    await expect(page.locator(S.sTotal)).toBeVisible();
    await expect(page.locator(S.sLatency)).toBeVisible();
    await expect(page.locator(S.sErrors)).toBeVisible();
  });

  test('stat cards populate from "—" within 5s', async ({ page }) => {
    await expect(page.locator(S.sStatus)).not.toHaveText('—', { timeout: 5000 });
  });

  test('health-raw box shows valid JSON from /health', async ({ page }) => {
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    const text = await page.locator(S.healthRaw).textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
  });

  test('Refresh button re-fetches and populates health-raw', async ({ page }) => {
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    await page.locator('button:has-text("Refresh")').click();
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    const text = await page.locator(S.healthRaw).textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/health.spec.js
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/health.spec.js
git commit -m "feat(playwright): add health spec — topbar badge and status panel"
```

---

## Task 7: TTS tests

**Files:**
- Create: `tests/playwright/tests/tts.spec.js`

- [ ] **Step 1: Create tts.spec.js**

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('TTS Stream', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navTTS).click();
    await expect(page.locator(S.panelTTS)).toHaveClass(/active/);
  });

  test('all input fields render', async ({ page }) => {
    await expect(page.locator(S.ttsText)).toBeVisible();
    await expect(page.locator(S.ttsEngine)).toBeVisible();
    await expect(page.locator(S.ttsVoice)).toBeVisible();
    await expect(page.locator(S.ttsSpeed)).toBeVisible();
  });

  test('Synthesise button is enabled', async ({ page }) => {
    await expect(page.locator(S.ttsBtn)).toBeEnabled();
  });

  test('text area accepts custom input', async ({ page }) => {
    await page.locator(S.ttsText).fill('Test synthesis text.');
    await expect(page.locator(S.ttsText)).toHaveValue('Test synthesis text.');
  });

  test('speed field accepts 1.5', async ({ page }) => {
    await page.locator(S.ttsSpeed).fill('1.5');
    await expect(page.locator(S.ttsSpeed)).toHaveValue('1.5');
  });

  test('status box starts as "waiting…"', async ({ page }) => {
    await expect(page.locator(S.ttsStatus)).toHaveText('waiting…');
  });

  test('Stop button is present', async ({ page }) => {
    await expect(page.locator('button:has-text("Stop")')).toBeVisible();
  });

  test('clicking Synthesise moves status away from "waiting…"', async ({ page }) => {
    await page.locator(S.ttsText).fill('Hello world.');
    await page.locator(S.ttsBtn).click();
    await expect(page.locator(S.ttsStatus)).not.toHaveText('waiting…', { timeout: 10000 });
  });

  test('Synthesise reaches a terminal state — audio or error class on status', async ({ page }) => {
    await page.locator(S.ttsText).fill('Hello.');
    await page.locator(S.ttsBtn).click();
    await expect(async () => {
      const cls = await page.locator(S.ttsStatus).getAttribute('class') ?? '';
      const audioVisible = await page.locator(S.ttsAudioWrap).isVisible();
      expect(cls.includes('ok') || cls.includes('error') || audioVisible).toBe(true);
    }).toPass({ timeout: 30000 });
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/tts.spec.js
```

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/tts.spec.js
git commit -m "feat(playwright): add TTS spec — form fields, submit, terminal state"
```

---

## Task 8: Classify tests

**Files:**
- Create: `tests/playwright/tests/classify.spec.js`

- [ ] **Step 1: Create classify.spec.js**

A 1×1 white PNG is constructed in-memory from base64 — no external fixture file needed.

```js
const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const os = require('os');
const S = require('../utils/selectors');

// Minimal 1×1 white PNG encoded in base64
const PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8' +
  'z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==';

test.describe('Image Classification', () => {
  let pngPath;

  test.beforeAll(async () => {
    pngPath = path.join(os.tmpdir(), 'pw-classify-test.png');
    fs.writeFileSync(pngPath, Buffer.from(PNG_B64, 'base64'));
  });

  test.afterAll(() => {
    if (fs.existsSync(pngPath)) fs.unlinkSync(pngPath);
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navClassify).click();
    await expect(page.locator(S.panelClassify)).toHaveClass(/active/);
  });

  test('dropzone is visible', async ({ page }) => {
    await expect(page.locator(S.dropzone)).toBeVisible();
  });

  test('Classify button is disabled before upload', async ({ page }) => {
    await expect(page.locator(S.clsBtn)).toBeDisabled();
  });

  test('output box shows "waiting for image…" initially', async ({ page }) => {
    await expect(page.locator(S.clsOut)).toHaveText('waiting for image…');
  });

  test('Top-K/Width/Height fields have correct defaults', async ({ page }) => {
    await expect(page.locator(S.clsTopk)).toHaveValue('5');
    await expect(page.locator(S.clsW)).toHaveValue('224');
    await expect(page.locator(S.clsH)).toHaveValue('224');
  });

  test('uploading an image enables the Classify button', async ({ page }) => {
    await page.locator(S.imgFile).setInputFiles(pngPath);
    await expect(page.locator(S.clsBtn)).toBeEnabled({ timeout: 5000 });
  });

  test('uploading an image shows the preview', async ({ page }) => {
    await page.locator(S.imgFile).setInputFiles(pngPath);
    await expect(page.locator(S.imgPreview)).toBeVisible({ timeout: 5000 });
  });

  test('Top-K field accepts a custom value', async ({ page }) => {
    await page.locator(S.clsTopk).fill('10');
    await expect(page.locator(S.clsTopk)).toHaveValue('10');
  });

  test('clicking Classify updates the output box', async ({ page }) => {
    await page.locator(S.imgFile).setInputFiles(pngPath);
    await expect(page.locator(S.clsBtn)).toBeEnabled({ timeout: 5000 });
    await page.locator(S.clsBtn).click();
    await expect(page.locator(S.clsOut)).not.toHaveText('waiting for image…', { timeout: 15000 });
    await expect(page.locator(S.clsOut)).not.toHaveText('Image loaded — click Classify.', { timeout: 15000 });
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/classify.spec.js
```

Expected: all 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/classify.spec.js
git commit -m "feat(playwright): add classify spec — dropzone, file upload, classify button"
```

---

## Task 9: Chat and Completion tests

**Files:**
- Create: `tests/playwright/tests/chat.spec.js`

- [ ] **Step 1: Create chat.spec.js**

The Completion panel is included here as a second `describe` block — it is a closely related LLM section not listed as a separate file in the spec.

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('LLM Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navLLM).click();
    await expect(page.locator(S.panelLLM)).toHaveClass(/active/);
  });

  test('chat input and Send button render', async ({ page }) => {
    await expect(page.locator(S.chatInput)).toBeVisible();
    await expect(page.locator(S.chatBtn)).toBeVisible();
  });

  test('chat history is empty on load', async ({ page }) => {
    await expect(page.locator(S.chatHistory)).toBeEmpty();
  });

  test('parameter fields have correct defaults', async ({ page }) => {
    await expect(page.locator(S.chatModel)).toHaveValue('default');
    await expect(page.locator(S.chatMaxtok)).toHaveValue('256');
    await expect(page.locator(S.chatTemp)).toHaveValue('0.7');
  });

  test('sending a message adds it to history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello there');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
    await expect(
      page.locator(S.chatHistory).locator('.msg.user .msg-body')
    ).toHaveText('Hello there');
  });

  test('Enter key (without Shift) sends the message', async ({ page }) => {
    await page.locator(S.chatInput).fill('Enter key test');
    await page.locator(S.chatInput).press('Enter');
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
  });

  test('after round-trip, Send button re-enables', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
  });

  test('history contains 2 messages after a round-trip', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
    await expect(page.locator(S.chatHistory).locator('.msg')).toHaveCount(2);
  });

  test('Clear chat button empties history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
    await page.locator('button:has-text("Clear chat")').click();
    await expect(page.locator(S.chatHistory)).toBeEmpty();
  });
});

test.describe('Text Completion', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navCompletion).click();
    await expect(page.locator(S.panelCompletion)).toHaveClass(/active/);
  });

  test('prompt textarea renders with default content', async ({ page }) => {
    await expect(page.locator(S.cmpPrompt)).toBeVisible();
    const value = await page.locator(S.cmpPrompt).inputValue();
    expect(value.length).toBeGreaterThan(0);
  });

  test('parameter fields render with defaults', async ({ page }) => {
    await expect(page.locator(S.cmpModel)).toHaveValue('default');
    await expect(page.locator(S.cmpMaxtok)).toHaveValue('128');
    await expect(page.locator(S.cmpTemp)).toHaveValue('0.7');
    await expect(page.locator(S.cmpTopp)).toHaveValue('1.0');
  });

  test('Complete button is enabled', async ({ page }) => {
    await expect(page.locator(S.cmpBtn)).toBeEnabled();
  });

  test('output starts as "waiting…"', async ({ page }) => {
    await expect(page.locator(S.cmpOut)).toHaveText('waiting…');
  });

  test('prompt textarea accepts custom input', async ({ page }) => {
    await page.locator(S.cmpPrompt).fill('The quick brown fox');
    await expect(page.locator(S.cmpPrompt)).toHaveValue('The quick brown fox');
  });

  test('clicking Complete updates the output box', async ({ page }) => {
    await page.locator(S.cmpBtn).click();
    await expect(page.locator(S.cmpOut)).not.toHaveText('waiting…', { timeout: 30000 });
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/chat.spec.js
```

Expected: all 14 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/chat.spec.js
git commit -m "feat(playwright): add chat and completion spec — input, send, history, round-trip"
```

---

## Task 10: Dashboard tests

**Files:**
- Create: `tests/playwright/tests/dashboard.spec.js`

- [ ] **Step 1: Create dashboard.spec.js**

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navDashboard).click();
    await expect(page.locator(S.panelDashboard)).toHaveClass(/active/);
  });

  test('Overview and Playground tabs render', async ({ page }) => {
    await expect(page.locator(S.dashTabOverview)).toBeVisible();
    await expect(page.locator(S.dashTabPlayground)).toBeVisible();
  });

  test('Overview tab is active by default', async ({ page }) => {
    await expect(page.locator(S.dashTabOverview)).toHaveClass(/active/);
    await expect(page.locator(S.dashOverview)).toBeVisible();
  });

  test('all 6 metrics tiles render', async ({ page }) => {
    await expect(page.locator(S.dUptime)).toBeVisible();
    await expect(page.locator(S.dTotal)).toBeVisible();
    await expect(page.locator(S.dActive)).toBeVisible();
    await expect(page.locator(S.dLatency)).toBeVisible();
    await expect(page.locator(S.dErrors)).toBeVisible();
    await expect(page.locator(S.dRps)).toBeVisible();
  });

  test('metrics tiles update from "—" within 5s via SSE', async ({ page }) => {
    await expect(page.locator(S.dUptime)).not.toHaveText('—', { timeout: 5000 });
  });

  test('spark canvas is rendered', async ({ page }) => {
    await expect(page.locator(S.dashSpark)).toBeVisible();
  });

  test('SSE live badge is visible', async ({ page }) => {
    await expect(page.locator(S.dashLiveBadge)).toBeVisible();
  });

  test('CPU, RAM, GPU resource bars render', async ({ page }) => {
    await expect(page.locator(S.barCpu)).toBeVisible();
    await expect(page.locator(S.barRam)).toBeVisible();
    await expect(page.locator(S.barGpu)).toBeVisible();
  });

  test('CPU, RAM, GPU value labels render', async ({ page }) => {
    await expect(page.locator(S.valCpu)).toBeVisible();
    await expect(page.locator(S.valRam)).toBeVisible();
    await expect(page.locator(S.valGpu)).toBeVisible();
  });

  test('GPU device info element is in the DOM', async ({ page }) => {
    await expect(page.locator(S.gpuDeviceInfo)).toBeAttached();
  });

  test('clicking Playground tab shows playground, hides overview', async ({ page }) => {
    await page.locator(S.dashTabPlayground).click();
    await expect(page.locator(S.dashPlayground)).toBeVisible();
    await expect(page.locator(S.dashOverview)).toBeHidden();
  });

  test('clicking Overview tab restores overview, hides playground', async ({ page }) => {
    await page.locator(S.dashTabPlayground).click();
    await page.locator(S.dashTabOverview).click();
    await expect(page.locator(S.dashOverview)).toBeVisible();
    await expect(page.locator(S.dashPlayground)).toBeHidden();
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/dashboard.spec.js
```

Expected: all 11 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/dashboard.spec.js
git commit -m "feat(playwright): add dashboard spec — metrics tiles, spark canvas, resource bars, tabs"
```

---

## Task 11: Model download tests

**Files:**
- Create: `tests/playwright/tests/models.spec.js`

- [ ] **Step 1: Create models.spec.js**

```js
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Model Download Manager', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navDashboard).click();
    // Download manager lives inside the overview tab (visible by default)
    await expect(page.locator(S.dashOverview)).toBeVisible();
  });

  test('download form elements are visible', async ({ page }) => {
    await expect(page.locator(S.dlSource)).toBeVisible();
    await expect(page.locator(S.dlRepo)).toBeVisible();
    await expect(page.locator(S.dlRevision)).toBeVisible();
    await expect(page.locator(S.dlBtn)).toBeVisible();
  });

  test('source selector has HuggingFace and URL options', async ({ page }) => {
    await expect(
      page.locator(`${S.dlSource} option[value="huggingface"]`)
    ).toHaveCount(1);
    await expect(
      page.locator(`${S.dlSource} option[value="url"]`)
    ).toHaveCount(1);
  });

  test('"No downloads yet." shown initially', async ({ page }) => {
    await expect(page.locator(S.dlTasksEmpty)).toBeVisible();
    await expect(page.locator(S.dlTasksEmpty)).toHaveText('No downloads yet.');
  });

  test('repo input accepts text', async ({ page }) => {
    await page.locator(S.dlRepo).fill('meta-llama/Llama-3.2-1B');
    await expect(page.locator(S.dlRepo)).toHaveValue('meta-llama/Llama-3.2-1B');
  });

  test('revision field is visible for HuggingFace source', async ({ page }) => {
    await page.locator(S.dlSource).selectOption('huggingface');
    await expect(page.locator(S.dlRevisionWrap)).toBeVisible();
  });

  test('selecting URL source hides revision field', async ({ page }) => {
    await page.locator(S.dlSource).selectOption('url');
    await expect(page.locator(S.dlRevisionWrap)).toBeHidden();
  });

  test('Download button is enabled', async ({ page }) => {
    await expect(page.locator(S.dlBtn)).toBeEnabled();
  });

  test('revision input accepts custom value', async ({ page }) => {
    await page.locator(S.dlRevision).fill('v1.0');
    await expect(page.locator(S.dlRevision)).toHaveValue('v1.0');
  });
});
```

- [ ] **Step 2: Run and verify**

```bash
cd tests/playwright
npx playwright test tests/models.spec.js
```

Expected: all 9 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/models.spec.js
git commit -m "feat(playwright): add models spec — download form, source selector, task list"
```

---

## Task 12: Playground tab tests

**Files:**
- Create: `tests/playwright/tests/playground.spec.js`

- [ ] **Step 1: Create playground.spec.js**

```js
const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
const os = require('os');
const S = require('../utils/selectors');

// Same 1×1 white PNG used in classify.spec.js
const PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8' +
  'z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==';

test.describe('Dashboard Playground tab', () => {
  let pngPath;

  test.beforeAll(async () => {
    pngPath = path.join(os.tmpdir(), 'pw-pg-test.png');
    fs.writeFileSync(pngPath, Buffer.from(PNG_B64, 'base64'));
  });

  test.afterAll(() => {
    if (fs.existsSync(pngPath)) fs.unlinkSync(pngPath);
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navDashboard).click();
    await page.locator(S.dashTabPlayground).click();
    await expect(page.locator(S.dashPlayground)).toBeVisible();
  });

  // ── TTS sub-tab ────────────────────────────────────────
  test.describe('TTS sub-tab', () => {
    test('TTS sub-tab is active by default in playground', async ({ page }) => {
      await expect(page.locator(S.pgTabTTS)).toHaveClass(/active/);
      await expect(page.locator(S.pgTTS)).toBeVisible();
    });

    test('TTS form fields render', async ({ page }) => {
      await expect(page.locator(S.pgTtsText)).toBeVisible();
      await expect(page.locator(S.pgTtsEngine)).toBeVisible();
      await expect(page.locator(S.pgTtsVoice)).toBeVisible();
      await expect(page.locator(S.pgTtsSpeed)).toBeVisible();
      await expect(page.locator(S.pgTtsBtn)).toBeVisible();
    });

    test('status starts as "waiting…"', async ({ page }) => {
      await expect(page.locator(S.pgTtsStatus)).toHaveText('waiting…');
    });

    test('Synthesise moves status from "waiting…"', async ({ page }) => {
      await page.locator(S.pgTtsText).fill('Hello from playground.');
      await page.locator(S.pgTtsBtn).click();
      await expect(page.locator(S.pgTtsStatus)).not.toHaveText('waiting…', { timeout: 10000 });
    });

    test('Synthesise reaches terminal state — audio or error', async ({ page }) => {
      await page.locator(S.pgTtsText).fill('Hi.');
      await page.locator(S.pgTtsBtn).click();
      await expect(async () => {
        const cls = await page.locator(S.pgTtsStatus).getAttribute('class') ?? '';
        const audioVisible = await page.locator(S.pgTtsAudioWrap).isVisible();
        expect(cls.includes('ok') || cls.includes('error') || audioVisible).toBe(true);
      }).toPass({ timeout: 30000 });
    });
  });

  // ── Classify sub-tab ──────────────────────────────────
  test.describe('Classify sub-tab', () => {
    test.beforeEach(async ({ page }) => {
      await page.locator(S.pgTabClassify).click();
      await expect(page.locator(S.pgClassify)).toBeVisible();
    });

    test('dropzone renders', async ({ page }) => {
      await expect(page.locator(S.pgDropzone)).toBeVisible();
    });

    test('Classify button disabled before upload', async ({ page }) => {
      await expect(page.locator(S.pgClsBtn)).toBeDisabled();
    });

    test('output starts as "waiting for image…"', async ({ page }) => {
      await expect(page.locator(S.pgClsOut)).toHaveText('waiting for image…');
    });

    test('uploading image enables Classify button', async ({ page }) => {
      await page.locator(S.pgImgFile).setInputFiles(pngPath);
      await expect(page.locator(S.pgClsBtn)).toBeEnabled({ timeout: 5000 });
    });

    test('clicking Classify updates output', async ({ page }) => {
      await page.locator(S.pgImgFile).setInputFiles(pngPath);
      await expect(page.locator(S.pgClsBtn)).toBeEnabled({ timeout: 5000 });
      await page.locator(S.pgClsBtn).click();
      await expect(page.locator(S.pgClsOut)).not.toHaveText('waiting for image…', { timeout: 15000 });
    });
  });

  // ── LLM Chat sub-tab ──────────────────────────────────
  test.describe('LLM Chat sub-tab', () => {
    test.beforeEach(async ({ page }) => {
      await page.locator(S.pgTabLLM).click();
      await expect(page.locator(S.pgLLM)).toBeVisible();
    });

    test('chat input and Send button render', async ({ page }) => {
      await expect(page.locator(S.pgChatInput)).toBeVisible();
      await expect(page.locator(S.pgChatBtn)).toBeVisible();
    });

    test('history is empty on load', async ({ page }) => {
      await expect(page.locator(S.pgChatHistory)).toBeEmpty();
    });

    test('sending message adds it to history', async ({ page }) => {
      await page.locator(S.pgChatInput).fill('Hello from playground');
      await page.locator(S.pgChatBtn).click();
      await expect(
        page.locator(S.pgChatHistory).locator('.msg.user')
      ).toHaveCount(1, { timeout: 5000 });
    });

    test('after round-trip Send re-enables', async ({ page }) => {
      await page.locator(S.pgChatInput).fill('Hi');
      await page.locator(S.pgChatBtn).click();
      await expect(page.locator(S.pgChatBtn)).toBeEnabled({ timeout: 30000 });
    });
  });

  // ── Completion sub-tab ────────────────────────────────
  test.describe('Completion sub-tab', () => {
    test.beforeEach(async ({ page }) => {
      await page.locator(S.pgTabCompletion).click();
      await expect(page.locator(S.pgCompletion)).toBeVisible();
    });

    test('prompt textarea and Complete button render', async ({ page }) => {
      await expect(page.locator(S.pgCmpPrompt)).toBeVisible();
      await expect(page.locator(S.pgCmpBtn)).toBeVisible();
    });

    test('output starts as "waiting…"', async ({ page }) => {
      await expect(page.locator(S.pgCmpOut)).toHaveText('waiting…');
    });

    test('clicking Complete updates output', async ({ page }) => {
      await page.locator(S.pgCmpBtn).click();
      await expect(page.locator(S.pgCmpOut)).not.toHaveText('waiting…', { timeout: 30000 });
    });
  });

  // ── Sub-tab switching ─────────────────────────────────
  test('switching sub-tabs shows correct section', async ({ page }) => {
    await page.locator(S.pgTabClassify).click();
    await expect(page.locator(S.pgClassify)).toBeVisible();
    await expect(page.locator(S.pgTTS)).toBeHidden();

    await page.locator(S.pgTabTTS).click();
    await expect(page.locator(S.pgTTS)).toBeVisible();
    await expect(page.locator(S.pgClassify)).toBeHidden();
  });
});
```

- [ ] **Step 2: Run full suite**

```bash
cd tests/playwright
npx playwright test
```

Expected: all tests pass. HTML report written to `results/report/`.

- [ ] **Step 3: Commit**

```bash
git add tests/playwright/tests/playground.spec.js
git commit -m "feat(playwright): add playground spec — TTS, classify, chat, completion sub-tabs"
```

---

## Task 13: Final verification

- [ ] **Step 1: Run full suite with verbose output**

```bash
cd tests/playwright
npx playwright test --reporter=list
```

Expected: all spec files run, all tests pass.

- [ ] **Step 2: View HTML report**

```bash
npx playwright show-report results/report
```

- [ ] **Step 3: Final commit**

```bash
git add tests/playwright/
git commit -m "feat(playwright): complete Playwright UI test suite for dashboard playground"
```

---

## Self-Review

- **Spec coverage:** all sections covered — navigation, health/status, TTS, classify, chat, completion, dashboard, model downloads, playground sub-tabs (TTS/classify/chat/completion).
- **Addition beyond spec:** Completion panel tests added to `chat.spec.js` and `playground.spec.js` — the UI has a Completion section not listed in the original spec table.
- **No placeholders:** all steps have exact code, commands, and expected output.
- **Type consistency:** every selector key used in spec files (`S.pgTtsStatus`, `S.barRam`, etc.) is defined in `utils/selectors.js` in Task 4.
- **`execSync` replaced** with `execFileSync('cargo', ['build', '--release'], ...)` — no shell string interpolation.
