# Playwright UI Testing Design

**Date:** 2026-03-31
**Status:** Approved

## Overview

Add Playwright end-to-end UI tests for the `playground.html` dashboard served by the Rust inference server (`feature/dashboard` branch). Tests live at `tests/playwright/` alongside the existing `tests/jest/` API tests. A global setup script auto-builds and starts the server before tests run.

## Directory Structure

```
tests/
├── jest/                         (existing)
└── playwright/
    ├── package.json
    ├── playwright.config.js
    ├── global-setup.js           ← builds & starts server
    ├── global-teardown.js        ← kills server
    ├── utils/
    │   └── selectors.js          ← shared element selectors
    └── tests/
        ├── navigation.spec.js
        ├── health.spec.js
        ├── tts.spec.js
        ├── classify.spec.js
        ├── chat.spec.js
        ├── dashboard.spec.js
        ├── models.spec.js
        └── playground.spec.js
```

## Global Setup / Server Lifecycle

`global-setup.js`:
1. Reads `BASE_URL` from `.env.test` (default: `http://localhost:3000`)
2. If `BUILD_SERVER=true`, runs `cargo build` targeting `.worktrees/dashboard/Cargo.toml`
3. If `SERVER_BINARY` is set, uses that path; otherwise uses the dashboard worktree's built binary
4. Spawns the server process, saves PID to a temp file
5. Polls `GET /health` with 30s timeout until 200 OK

`global-teardown.js`:
- Reads PID file and sends `SIGTERM`

`.env.test` variables:
- `BASE_URL` — server URL (default `http://localhost:3000`)
- `BUILD_SERVER` — `true`/`false` (default `true`), skip build if binary is pre-built
- `SERVER_BINARY` — optional path override to pre-built binary (default: `.worktrees/dashboard/target/release/torch-inference`)

## Test Coverage

| File | Coverage |
|---|---|
| `navigation.spec.js` | Page loads, sidebar nav items present, clicking each shows correct section, active state updates |
| `health.spec.js` | Health badge visible in topbar, reflects `/health` status (green/red), updates on poll |
| `tts.spec.js` | Textarea accepts input, engine/voice/speed fields present, submit fires request, audio element appears or error box shown |
| `classify.spec.js` | Dropzone renders, file input accepts image, preview renders, classify button triggers request, output box shows result or error |
| `chat.spec.js` | Input accepts text, Enter/button sends, user message in history, assistant response or error renders |
| `dashboard.spec.js` | Overview panel loads, metrics cards visible, GPU info section renders, spark canvas present |
| `models.spec.js` | Download panel visible, repo input accepts text, source selector works, submit triggers task, task appears in list |
| `playground.spec.js` | Playground sub-sections (TTS/classify/chat) render, forms functional, same assertions as main sections |

Tests that require a loaded model assert on the *attempt* (button enables, request fires, response element appears) and tolerate error responses from an unloaded model.

## Configuration

**`playwright.config.js`:**
- Browser: Chromium (headless by default, `HEADED=1` env for headed mode)
- `baseURL` from `BASE_URL` env
- `testDir: ./tests`, `testMatch: **/*.spec.js`
- `timeout: 30000` per test, `expect.timeout: 5000`
- Reporters: HTML (`results/report/`), JUnit XML (`results/junit.xml`)
- Screenshots on failure: `results/screenshots/`

**`package.json` scripts:**
```json
{
  "test":            "playwright test",
  "test:headed":     "HEADED=1 playwright test",
  "test:ui":         "playwright test --ui",
  "test:navigation": "playwright test tests/navigation.spec.js",
  "test:tts":        "playwright test tests/tts.spec.js",
  "test:classify":   "playwright test tests/classify.spec.js",
  "test:chat":       "playwright test tests/chat.spec.js",
  "test:dashboard":  "playwright test tests/dashboard.spec.js",
  "test:models":     "playwright test tests/models.spec.js",
  "test:playground": "playwright test tests/playground.spec.js"
}
```

**Dependencies:** `@playwright/test` only.

## Relationship to Jest Tests

- Jest tests: HTTP-level API tests, no browser
- Playwright tests: browser-level UI tests against the same server
- Both share the same `BASE_URL` convention via `.env.test`
- Both tolerate an unavailable/unloaded server gracefully
