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

  test('status element is present', async ({ page }) => {
    await expect(page.locator(S.ttsStatus)).toBeVisible();
  });

  test('Stop button is present', async ({ page }) => {
    await expect(page.locator(S.panelTTS).locator('button:has-text("Stop")')).toBeVisible();
  });

  test('clicking Speak changes status text', async ({ page }) => {
    await page.locator(S.ttsText).fill('Hello world.');
    const statusBefore = await page.locator(S.ttsStatus).textContent();
    await page.locator(S.ttsBtn).click();
    await expect(page.locator(S.ttsStatus)).not.toHaveText(statusBefore ?? '', { timeout: 10000 });
  });
});
