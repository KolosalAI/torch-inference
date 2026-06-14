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
    await expect(page.locator(S.panelDashboard)).toHaveClass(/active/);
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
