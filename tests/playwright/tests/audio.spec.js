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
