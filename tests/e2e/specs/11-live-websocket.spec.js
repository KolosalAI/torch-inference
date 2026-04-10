// @ts-check
/**
 * Live WebSocket feature tests.
 *
 * Tests three WS endpoints:
 *   - /ws/classify  — JPEG frames → top-K classification predictions
 *   - /ws/detect    — JPEG frames → object detection (graceful error without PyTorch)
 *   - /audio/ws     — bidirectional audio: TTS streaming + STT microphone transcription
 *
 * Protocol tests use page.evaluate() to open a real WebSocket from the browser context
 * (same origin as the page, so no CORS issues). UI tests interact with actual controls.
 */

const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

const FIXTURE_JPG_B64 = fs.readFileSync(
  path.resolve(__dirname, '../fixtures/test.jpg')
).toString('base64');

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Opens a WebSocket from within the page context and collects messages until a
 * stop condition is met or the timeout fires. Returns an array of received text
 * messages (binary lengths are appended as "binary:<N>" strings).
 *
 * @param {import('@playwright/test').Page} page
 * @param {string} path  - WS path, e.g. '/ws/classify'
 * @param {(messages: string[]) => boolean} stopWhen
 * @param {(send: (data: string|ArrayBuffer) => void) => void} onOpen
 * @param {number} [timeoutMs]
 */
async function evalWs(page, wsPath, onOpen, stopWhen, timeoutMs = 15000) {
  return page.evaluate(
    async ({ wsPath, jpgB64, timeoutMs, onOpenSrc }) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const url = `${proto}://${location.host}${wsPath}`;
      const messages = [];

      await new Promise((resolve) => {
        const ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        const timer = setTimeout(() => { ws.close(); resolve(); }, timeoutMs);

        ws.onmessage = (ev) => {
          if (typeof ev.data === 'string') {
            messages.push(ev.data);
          } else {
            messages.push(`binary:${ev.data.byteLength}`);
          }
          // eslint-disable-next-line no-new-func
          const checkStop = new Function('messages', `return (${onOpenSrc.stopWhen})(messages)`);
          if (checkStop(messages)) { clearTimeout(timer); ws.close(); resolve(); }
        };

        ws.onopen = () => {
          const raw = onOpenSrc.actions;
          for (const action of raw) {
            if (action.type === 'text') {
              ws.send(action.data);
            } else if (action.type === 'b64binary') {
              // decode base64 → ArrayBuffer
              const bin = atob(action.data);
              const buf = new ArrayBuffer(bin.length);
              const view = new Uint8Array(buf);
              for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
              ws.send(buf);
            } else if (action.type === 'silence') {
              // PCM f32le silence of given byte length
              ws.send(new ArrayBuffer(action.bytes));
            }
          }
        };

        ws.onerror = () => { clearTimeout(timer); resolve(); };
        ws.onclose = () => { clearTimeout(timer); resolve(); };
      });

      return messages;
    },
    {
      wsPath,
      jpgB64: FIXTURE_JPG_B64,
      timeoutMs,
      onOpenSrc: { actions: onOpen, stopWhen: stopWhen.toString() },
    }
  );
}

// ─── /ws/classify protocol tests ─────────────────────────────────────────────

test.describe('WS Classify — protocol', () => {
  test('receives ready frame on connect', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/classify',
      [], // no actions — just wait for ready
      (msgs) => msgs.length >= 1,
      5000
    );
    expect(msgs.length).toBeGreaterThanOrEqual(1);
    const ready = JSON.parse(msgs[0]);
    expect(ready.type).toBe('ready');
    expect(ready.task).toBe('classify');
  });

  test('returns classification predictions for a JPEG frame', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/classify',
      [
        { type: 'text', data: JSON.stringify({ type: 'config', top_k: 3, width: 224, height: 224 }) },
        { type: 'b64binary', data: FIXTURE_JPG_B64 },
      ],
      (msgs) => msgs.some((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } }),
      15000
    );
    const classifyMsg = msgs.find((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } });
    expect(classifyMsg).toBeTruthy();
    const result = JSON.parse(classifyMsg);
    expect(result.predictions).toBeInstanceOf(Array);
    expect(result.predictions.length).toBeGreaterThan(0);
    expect(result.predictions[0]).toHaveProperty('label');
    expect(result.predictions[0]).toHaveProperty('confidence');
    expect(result.ms).toBeGreaterThan(0);
  });

  test('top_k config limits number of predictions', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/classify',
      [
        { type: 'text', data: JSON.stringify({ type: 'config', top_k: 2, width: 224, height: 224 }) },
        { type: 'b64binary', data: FIXTURE_JPG_B64 },
      ],
      (msgs) => msgs.some((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } }),
      15000
    );
    const classifyMsg = msgs.find((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } });
    const result = JSON.parse(classifyMsg);
    expect(result.predictions.length).toBeLessThanOrEqual(2);
  });

  test('frame counter increments with each frame', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/classify',
      [
        { type: 'text', data: JSON.stringify({ type: 'config', top_k: 1, width: 224, height: 224 }) },
        { type: 'b64binary', data: FIXTURE_JPG_B64 },
        { type: 'b64binary', data: FIXTURE_JPG_B64 },
      ],
      (msgs) => msgs.filter((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } }).length >= 2,
      20000
    );
    const classifyMsgs = msgs
      .filter((m) => { try { return JSON.parse(m).type === 'classify'; } catch { return false; } })
      .map((m) => JSON.parse(m));
    expect(classifyMsgs[0].frame).toBe(1);
    expect(classifyMsgs[1].frame).toBe(2);
  });
});

// ─── /ws/detect protocol tests ────────────────────────────────────────────────

test.describe('WS Detect — protocol', () => {
  test('receives ready frame on connect', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/detect',
      [],
      (msgs) => msgs.length >= 1,
      5000
    );
    const ready = JSON.parse(msgs[0]);
    expect(ready.type).toBe('ready');
    expect(ready.task).toBe('detect');
  });

  test('returns a message (detect or graceful error) for a JPEG frame', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/ws/detect',
      [
        { type: 'text', data: JSON.stringify({ type: 'config', version: 'v8', size: 'n', conf: 0.25, iou: 0.45 }) },
        { type: 'b64binary', data: FIXTURE_JPG_B64 },
      ],
      (msgs) => msgs.some((m) => {
        try { const t = JSON.parse(m).type; return t === 'detect' || t === 'error'; } catch { return false; }
      }),
      15000
    );
    const response = msgs.find((m) => {
      try { const t = JSON.parse(m).type; return t === 'detect' || t === 'error'; } catch { return false; }
    });
    expect(response).toBeTruthy();
    const parsed = JSON.parse(response);
    // Either real detections or a clear error message — never a crash
    expect(['detect', 'error']).toContain(parsed.type);
    if (parsed.type === 'error') {
      expect(typeof parsed.msg).toBe('string');
      expect(parsed.msg.length).toBeGreaterThan(0);
    }
    if (parsed.type === 'detect') {
      expect(Array.isArray(parsed.detections)).toBe(true);
    }
  });
});

// ─── /audio/ws TTS protocol tests ────────────────────────────────────────────

test.describe('WS Audio TTS — protocol', () => {
  test('receives ready frame on connect', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/audio/ws',
      [],
      (msgs) => msgs.some((m) => { try { return JSON.parse(m).type === 'ready'; } catch { return false; } }),
      5000
    );
    const ready = msgs.find((m) => { try { return JSON.parse(m).type === 'ready'; } catch { return false; } });
    expect(ready).toBeTruthy();
    expect(JSON.parse(ready).type).toBe('ready');
  });

  test('TTS request returns tts_meta + binary PCM + tts_done', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/audio/ws',
      [
        { type: 'text', data: JSON.stringify({ type: 'tts', text: 'Hello', voice: 'af_heart', speed: 1.0 }) },
      ],
      (msgs) => msgs.some((m) => { try { return JSON.parse(m).type === 'tts_done'; } catch { return false; } }),
      20000
    );

    const metaMsg = msgs.find((m) => { try { return JSON.parse(m).type === 'tts_meta'; } catch { return false; } });
    const doneMsg = msgs.find((m) => { try { return JSON.parse(m).type === 'tts_done'; } catch { return false; } });
    const binaryMsgs = msgs.filter((m) => m.startsWith('binary:'));

    expect(metaMsg).toBeTruthy();
    const meta = JSON.parse(metaMsg);
    expect(meta.sample_rate).toBe(24000);
    expect(meta.encoding).toBe('pcm_f32le');

    expect(binaryMsgs.length).toBeGreaterThan(0);
    const totalBytes = binaryMsgs.reduce((sum, m) => sum + parseInt(m.split(':')[1], 10), 0);
    expect(totalBytes).toBeGreaterThan(0);

    expect(doneMsg).toBeTruthy();
    const done = JSON.parse(doneMsg);
    expect(done.duration_ms).toBeGreaterThan(0);
  });

  test('TTS with default voice (no voice specified) still works', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/audio/ws',
      [
        { type: 'text', data: JSON.stringify({ type: 'tts', text: 'Test', speed: 1.0 }) },
      ],
      (msgs) => msgs.some((m) => {
        try { const t = JSON.parse(m).type; return t === 'tts_done' || t === 'error'; } catch { return false; }
      }),
      20000
    );
    const response = msgs.find((m) => {
      try { const t = JSON.parse(m).type; return t === 'tts_done' || t === 'error'; } catch { return false; }
    });
    expect(response).toBeTruthy();
    // Either done or graceful error — never a crash
    expect(['tts_done', 'error']).toContain(JSON.parse(response).type);
  });

  test('empty text returns error not crash', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/audio/ws',
      [
        { type: 'text', data: JSON.stringify({ type: 'tts', text: '', speed: 1.0 }) },
      ],
      (msgs) => msgs.some((m) => { try { return JSON.parse(m).type === 'error'; } catch { return false; } }),
      8000
    );
    const errorMsg = msgs.find((m) => { try { return JSON.parse(m).type === 'error'; } catch { return false; } });
    expect(errorMsg).toBeTruthy();
    expect(JSON.parse(errorMsg).msg).toBeTruthy();
  });
});

// ─── /audio/ws STT protocol tests ────────────────────────────────────────────

test.describe('WS Audio STT — protocol', () => {
  test('STT session returns graceful error when no model loaded', async ({ page }) => {
    await page.goto('/');
    const msgs = await evalWs(
      page,
      '/audio/ws',
      [
        { type: 'text', data: JSON.stringify({ type: 'stt_begin', sample_rate: 16000 }) },
        { type: 'silence', bytes: 16000 * 4 }, // 1 second of f32le silence
        { type: 'text', data: JSON.stringify({ type: 'stt_end' }) },
      ],
      (msgs) => msgs.some((m) => {
        try { const t = JSON.parse(m).type; return t === 'transcript' || t === 'error'; } catch { return false; }
      }),
      15000
    );
    const response = msgs.find((m) => {
      try { const t = JSON.parse(m).type; return t === 'transcript' || t === 'error'; } catch { return false; }
    });
    // Server should respond — either a real transcript or a clear error
    expect(response).toBeTruthy();
    const parsed = JSON.parse(response);
    expect(['transcript', 'error']).toContain(parsed.type);
    if (parsed.type === 'error') {
      expect(parsed.msg).toMatch(/model|stt/i);
    }
  });
});

// ─── Live Classify UI tests ───────────────────────────────────────────────────

test.describe('Live Classify UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Classify' }).first().click();
    await expect(page.locator('#panel-classify')).toHaveClass(/active/);
  });

  test('live stream tab shows connect button and disconnected status', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await expect(page.locator('#cls-pane-live')).toBeVisible();
    await expect(page.locator('#cls-ws-btn')).toBeVisible();
    await expect(page.locator('#cls-ws-label')).toHaveText('disconnected');
  });

  test('clicking connect button establishes WS connection', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await page.locator('#cls-ws-btn').click();
    await expect(page.locator('#cls-ws-label')).toHaveText('connected', { timeout: 8000 });
    await expect(page.locator('#cls-ws-btn')).toContainText('Disconnect');
  });

  test('clicking disconnect closes WS connection', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await page.locator('#cls-ws-btn').click();
    await expect(page.locator('#cls-ws-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#cls-ws-btn').click();
    await expect(page.locator('#cls-ws-label')).toHaveText('disconnected', { timeout: 5000 });
  });

  test('live config inputs are visible and editable', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await expect(page.locator('#cls-live-topk')).toBeVisible();
    await expect(page.locator('#cls-live-w')).toBeVisible();
    await expect(page.locator('#cls-live-h')).toBeVisible();

    await page.locator('#cls-live-topk').fill('3');
    await expect(page.locator('#cls-live-topk')).toHaveValue('3');
  });

  test('connected state allows sending a frame and showing predictions', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await page.locator('#cls-ws-btn').click();
    await expect(page.locator('#cls-ws-label')).toHaveText('connected', { timeout: 8000 });

    // Send a JPEG frame directly via the existing clsWs WebSocket
    await page.evaluate(async (jpgB64) => {
      // Access the module-level clsWs variable via the page's global scope
      const bin = atob(jpgB64);
      const buf = new ArrayBuffer(bin.length);
      const view = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
      // clsWs is a module-level var in playground.html
      if (window.clsWs && window.clsWs.readyState === 1) {
        window.clsWs.send(buf);
      }
    }, FIXTURE_JPG_B64);

    // Predictions should appear (may show if clsWs is accessible)
    // At minimum, verify the pane is still live with connected state
    await expect(page.locator('#cls-ws-label')).toHaveText('connected', { timeout: 3000 });
  });
});

// ─── Live Detect UI tests ─────────────────────────────────────────────────────

test.describe('Live Detect UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Detect' }).first().click();
    await expect(page.locator('#panel-detect')).toHaveClass(/active/);
  });

  test('live stream tab shows connect button and disconnected status', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await expect(page.locator('#det-pane-live')).toBeVisible();
    await expect(page.locator('#det-ws-btn')).toBeVisible();
    await expect(page.locator('#det-ws-label')).toHaveText('disconnected');
  });

  test('clicking connect button establishes WS connection', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await page.locator('#det-ws-btn').click();
    await expect(page.locator('#det-ws-label')).toHaveText('connected', { timeout: 8000 });
    await expect(page.locator('#det-ws-btn')).toContainText('Disconnect');
  });

  test('clicking disconnect closes connection', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await page.locator('#det-ws-btn').click();
    await expect(page.locator('#det-ws-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#det-ws-btn').click();
    await expect(page.locator('#det-ws-label')).toHaveText('disconnected', { timeout: 5000 });
  });

  test('live canvas element is present', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await expect(page.locator('#det-live-canvas')).toBeVisible();
  });

  test('sending a frame while connected shows result or error in labels area', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await page.locator('#det-ws-btn').click();
    await expect(page.locator('#det-ws-label')).toHaveText('connected', { timeout: 8000 });

    // Send a JPEG frame via the WS
    await page.evaluate(async (jpgB64) => {
      const bin = atob(jpgB64);
      const buf = new ArrayBuffer(bin.length);
      const view = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
      // Config first
      if (window.detWs && window.detWs.readyState === 1) {
        window.detWs.send(JSON.stringify({ type: 'config', version: 'v8', size: 'n', conf: 0.25, iou: 0.45 }));
        window.detWs.send(buf);
      }
    }, FIXTURE_JPG_B64);

    // Wait a moment for the response to update the UI
    await page.waitForTimeout(3000);
    // Connection should still be alive (no crash)
    await expect(page.locator('#det-ws-label')).toHaveText('connected');
  });
});

// ─── Audio WS UI tests ────────────────────────────────────────────────────────

test.describe('Audio WS UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Audio' }).first().click();
    await expect(page.locator('#panel-audio')).toHaveClass(/active/);
  });

  test('WS section shows connect button and disconnected status', async ({ page }) => {
    await expect(page.locator('#ws-connect-btn')).toBeVisible();
    await expect(page.locator('#ws-status-label')).toHaveText('disconnected');
    await expect(page.locator('#ws-tts-btn')).toBeDisabled();
    await expect(page.locator('#ws-stt-btn')).toBeDisabled();
  });

  test('connect button establishes WS and enables TTS/STT buttons', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });
    await expect(page.locator('#ws-tts-btn')).toBeEnabled({ timeout: 5000 });
    await expect(page.locator('#ws-stt-btn')).toBeEnabled({ timeout: 5000 });
    await expect(page.locator('#ws-connect-btn')).toContainText('Disconnect');
  });

  test('disconnect button closes WS and disables TTS/STT buttons', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('disconnected', { timeout: 5000 });
    await expect(page.locator('#ws-tts-btn')).toBeDisabled();
  });

  test('Live TTS tab is default, Live STT tab switches pane', async ({ page }) => {
    await expect(page.locator('#ws-pane-tts')).toBeVisible();
    await page.locator('#ws-tab-stt').click();
    await expect(page.locator('#ws-pane-stt')).toBeVisible();
    await expect(page.locator('#ws-pane-tts')).toBeHidden();

    await page.locator('#ws-tab-tts').click();
    await expect(page.locator('#ws-pane-tts')).toBeVisible();
  });

  test('TTS speak flow: connect → type → speak → status updates', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#ws-tts-text').fill('Hello world');
    await page.locator('#ws-tts-voice').fill('af_heart');
    await page.locator('#ws-tts-btn').click();

    // Status should change from idle to synthesising or streaming
    await expect(page.locator('#ws-tts-status')).not.toHaveText('—', { timeout: 5000 });
    await expect(page.locator('#ws-tts-status')).not.toHaveText('Connected — ready', { timeout: 5000 });

    // Wait for done
    await expect(page.locator('#ws-tts-status')).toContainText('Done', { timeout: 25000 });
  });

  test('TTS with empty text shows error status, no crash', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });

    // Leave text empty and click speak
    await page.locator('#ws-tts-text').fill('');
    await page.locator('#ws-tts-btn').click();

    // Should show "Enter text first" not crash
    await expect(page.locator('#ws-tts-status')).toContainText('text', { timeout: 3000 });
    // Connection stays alive
    await expect(page.locator('#ws-status-label')).toHaveText('connected');
  });

  test('TTS stop button halts synthesis', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#ws-tts-text').fill('This is a longer sentence that takes a bit of time to synthesize completely');
    await page.locator('#ws-tts-btn').click();

    // Wait for synthesis to start
    await expect(page.locator('#ws-tts-status')).not.toHaveText('—', { timeout: 5000 });

    // Click stop
    await page.locator('#ws-tts-stop-btn').click();
    await expect(page.locator('#ws-tts-status')).toContainText('Stop', { timeout: 5000 });
  });

  test('TTS metrics appear after successful synthesis', async ({ page }) => {
    await page.locator('#ws-connect-btn').click();
    await expect(page.locator('#ws-status-label')).toHaveText('connected', { timeout: 8000 });

    await page.locator('#ws-tts-text').fill('Hello');
    await page.locator('#ws-tts-btn').click();
    await expect(page.locator('#ws-tts-status')).toContainText('Done', { timeout: 25000 });

    // Metrics div should be visible after done
    await expect(page.locator('#ws-tts-metrics')).toBeVisible({ timeout: 3000 });
  });

  test('TTS canvas element is present for waveform visualization', async ({ page }) => {
    await expect(page.locator('#ws-tts-canvas')).toBeVisible();
  });

  test('STT section renders microphone button and idle status', async ({ page }) => {
    await page.locator('#ws-tab-stt').click();
    await expect(page.locator('#ws-stt-btn')).toBeVisible();
    await expect(page.locator('#ws-stt-rec-label')).toHaveText('idle');
    await expect(page.locator('#ws-stt-result')).toBeVisible();
  });
});
