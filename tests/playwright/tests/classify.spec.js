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
