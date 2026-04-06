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
    await expect(page.locator(S.panelTTS).locator('button:has-text("Stop")')).toBeVisible();
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

  test('tts-voice is a <select> element', async ({ page }) => {
    await expect(page.locator('#tts-voice')).toHaveJSProperty('tagName', 'SELECT');
  });

  test('voice select repopulates when engine changes', async ({ page }) => {
    await page.route('/tts/engines/*/voices', async route => {
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
});
