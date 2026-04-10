// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('TTS Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'TTS' }).first().click();
    await expect(page.locator('#panel-tts')).toHaveClass(/active/);
  });

  test('TTS panel renders all controls', async ({ page }) => {
    await expect(page.locator('#tts-text')).toBeVisible();
    await expect(page.locator('#tts-engine')).toBeVisible();
    await expect(page.locator('#tts-voice')).toBeVisible();
    await expect(page.locator('#tts-speed')).toBeVisible();
    await expect(page.locator('#tts-btn')).toBeVisible();
  });

  test('engine dropdown is populated from /tts/engines', async ({ page }) => {
    // Wait for refreshTtsSelects() to add engine options
    const select = page.locator('#tts-engine');
    await page.waitForFunction(() => {
      const sel = document.getElementById('tts-engine');
      return sel && sel.options.length > 1;
    }, null, { timeout: 8_000 });
    const count = await select.locator('option').count();
    expect(count).toBeGreaterThan(1);
  });

  test('synthesise button runs TTS and shows result', async ({ page }) => {
    // Wait for engines to load, then ensure the default (empty) option is selected
    // so the server uses its default kokoro-onnx engine which has a model loaded
    await page.waitForFunction(() => {
      const sel = document.getElementById('tts-engine');
      return sel && sel.options.length > 1;
    }, null, { timeout: 8_000 });
    await page.locator('#tts-engine').selectOption('');

    // Use short text so it finishes quickly
    await page.locator('#tts-text').fill('Hello Kolosal.');
    await page.locator('#tts-speed').fill('1.0');
    await page.locator('#tts-btn').click();

    // Status should show streaming progress then "Done"
    await expect(page.locator('#tts-status')).toContainText('Done', { timeout: 45_000 });
    // Audio element should have a src set
    const audioSrc = await page.locator('#tts-audio').getAttribute('src');
    expect(audioSrc).toBeTruthy();
    expect(audioSrc).toContain('blob:');
  });

  test('synthesise button is re-enabled after completion', async ({ page }) => {
    await page.waitForFunction(() => {
      const sel = document.getElementById('tts-engine');
      return sel && sel.options.length > 1;
    }, null, { timeout: 8_000 });
    await page.locator('#tts-engine').selectOption('');
    await page.locator('#tts-text').fill('Hi.');
    await page.locator('#tts-btn').click();
    await expect(page.locator('#tts-status')).toContainText('Done', { timeout: 45_000 });
    await expect(page.locator('#tts-btn')).toBeEnabled();
  });

  test('metrics bar appears after synthesis', async ({ page }) => {
    await page.waitForFunction(() => {
      const sel = document.getElementById('tts-engine');
      return sel && sel.options.length > 1;
    }, null, { timeout: 8_000 });
    await page.locator('#tts-engine').selectOption('');
    await page.locator('#tts-text').fill('Test.');
    await page.locator('#tts-btn').click();
    await expect(page.locator('#tts-status')).toContainText('Done', { timeout: 45_000 });
    await expect(page.locator('#tts-metrics')).toBeVisible();
    const metricText = await page.locator('#tts-metric-time').textContent();
    expect(metricText).toMatch(/\d+\s*ms/);
  });

  test('empty text does not trigger request', async ({ page }) => {
    await page.locator('#tts-text').fill('');
    const [req] = await Promise.all([
      page.waitForRequest('**/tts/stream', { timeout: 2_000 }).catch(() => null),
      page.locator('#tts-btn').click(),
    ]);
    expect(req).toBeNull();
    await expect(page.locator('#tts-status')).toHaveText('waiting…');
  });

  test('code snippet tabs switch content', async ({ page }) => {
    // Map lang key → exact button label in the HTML
    const tabLabels = { curl: 'cURL', python: 'Python', js: 'JavaScript', go: 'Go', rust: 'Rust', node: 'Node.js' };
    for (const [lang, label] of Object.entries(tabLabels)) {
      await page.locator(`#tts-ep-tabs .ws-tab`, { hasText: label }).click();
      await expect(page.locator(`#tts-ep-${lang}`)).toBeVisible();
      const code = await page.locator(`#tts-ep-${lang}-code`).textContent();
      expect(code?.length).toBeGreaterThan(10);
    }
  });

  test('TTS health banner hidden when engines are loaded', async ({ page }) => {
    // 7 engines loaded per startup log — banner should stay hidden
    await expect(page.locator('#tts-no-engine-banner')).toBeHidden();
  });

  test('voice dropdown loads voices for selected engine', async ({ page }) => {
    // Wait for engines to load
    await page.waitForFunction(() => {
      const sel = document.getElementById('tts-engine');
      return sel && sel.options.length > 1;
    }, null, { timeout: 8_000 });
    const engineSel = page.locator('#tts-engine');
    // Select kokoro-onnx by value (it has voices loaded)
    const options = await engineSel.locator('option').allInnerTexts();
    const hasKokoro = options.some((o) => o.toLowerCase().includes('kokoro'));
    if (hasKokoro) {
      // Select by value (e.id = 'kokoro-onnx' or 'kokoro')
      const values = await engineSel.locator('option').evaluateAll((opts) => opts.map((o) => o.value));
      const kokoroVal = values.find((v) => v.includes('kokoro') && v !== '');
      if (kokoroVal) {
        await engineSel.selectOption(kokoroVal);
        await page.waitForFunction((voiceId) => {
          const sel = document.getElementById(voiceId);
          return sel && sel.options.length > 1;
        }, 'tts-voice', { timeout: 6_000 });
        const count = await page.locator('#tts-voice').locator('option').count();
        expect(count).toBeGreaterThan(1);
      }
    }
  });
});
