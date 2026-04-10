// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

const FIXTURE_WAV = path.resolve(__dirname, '../fixtures/test.wav');

test.describe('Audio Transcription Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Audio' }).first().click();
    await expect(page.locator('#panel-audio')).toHaveClass(/active/);
  });

  test('audio panel renders all controls', async ({ page }) => {
    await expect(page.locator('#audio-dropzone')).toBeVisible();
    await expect(page.locator('#audio-model')).toBeVisible();
    await expect(page.locator('#audio-timestamps')).toBeVisible();
    await expect(page.locator('#audio-btn')).toBeVisible();
    await expect(page.locator('#audio-btn')).toBeDisabled();
  });

  test('health badge shows audio status', async ({ page }) => {
    const badge = page.locator('#audio-health-badge');
    await expect(badge).not.toHaveText('checking…', { timeout: 8_000 });
    const text = await badge.textContent();
    expect(text).toMatch(/audio/i);
  });

  test('file upload enables transcribe button', async ({ page }) => {
    const fileInput = page.locator('#audio-file');
    await fileInput.setInputFiles(FIXTURE_WAV);

    await expect(page.locator('#audio-btn')).toBeEnabled({ timeout: 5_000 });
    await expect(page.locator('#audio-file-name')).toBeVisible();
    await expect(page.locator('#audio-file-name')).toContainText('test.wav');
  });

  test('transcribe shows response (error or result) without crashing', async ({ page }) => {
    const fileInput = page.locator('#audio-file');
    await fileInput.setInputFiles(FIXTURE_WAV);
    await expect(page.locator('#audio-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#audio-btn').click();
    // Either a real transcript or a graceful error message — never empty/stuck
    await expect(page.locator('#audio-result-text')).not.toHaveText(
      'Upload a file to transcribe.',
      { timeout: 20_000 }
    );
    await expect(page.locator('#audio-result-text')).not.toHaveText('Transcribing…', {
      timeout: 20_000,
    });
    const resultText = await page.locator('#audio-result-text').textContent();
    expect(resultText?.length).toBeGreaterThan(0);
  });

  test('transcribe button re-enables after response', async ({ page }) => {
    const fileInput = page.locator('#audio-file');
    await fileInput.setInputFiles(FIXTURE_WAV);
    await expect(page.locator('#audio-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#audio-btn').click();
    await expect(page.locator('#audio-result-text')).not.toHaveText('Transcribing…', {
      timeout: 20_000,
    });
    await expect(page.locator('#audio-btn')).toBeEnabled();
  });

  test('WebSocket audio section renders with connect button', async ({ page }) => {
    await expect(page.locator('#ws-connect-btn')).toBeVisible();
    await expect(page.locator('#ws-status-label')).toHaveText('disconnected');
  });

  test('WebSocket Live TTS / Live STT tabs switch panes', async ({ page }) => {
    await page.locator('#ws-tab-stt').click();
    await expect(page.locator('#ws-pane-stt')).toBeVisible();
    await expect(page.locator('#ws-pane-tts')).toBeHidden();

    await page.locator('#ws-tab-tts').click();
    await expect(page.locator('#ws-pane-tts')).toBeVisible();
    await expect(page.locator('#ws-pane-stt')).toBeHidden();
  });

  test('dropzone has correct accept attribute', async ({ page }) => {
    const input = page.locator('#audio-file');
    const accept = await input.getAttribute('accept');
    expect(accept).toContain('audio');
  });
});
