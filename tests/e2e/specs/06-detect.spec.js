// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

const FIXTURE_JPG = path.resolve(__dirname, '../fixtures/test.jpg');

test.describe('Object Detection Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Detect' }).first().click();
    await expect(page.locator('#panel-detect')).toHaveClass(/active/);
  });

  test('detection panel renders all controls', async ({ page }) => {
    await expect(page.locator('#detect-dropzone')).toBeVisible();
    await expect(page.locator('#detect-version')).toBeVisible();
    await expect(page.locator('#detect-size')).toBeVisible();
    await expect(page.locator('#detect-conf')).toBeVisible();
    await expect(page.locator('#detect-btn')).toBeVisible();
    await expect(page.locator('#detect-btn')).toBeDisabled();
  });

  test('model version dropdown has expected options', async ({ page }) => {
    const opts = await page.locator('#detect-version option').allTextContents();
    expect(opts).toContain('v8');
    expect(opts).toContain('v5');
    expect(opts).toContain('v11');
  });

  test('model size dropdown has expected options', async ({ page }) => {
    const opts = await page.locator('#detect-size option').allTextContents();
    expect(opts).toContain('nano');
    expect(opts).toContain('small');
    expect(opts).toContain('large');
  });

  test('confidence threshold input has correct default', async ({ page }) => {
    const val = await page.locator('#detect-conf').inputValue();
    expect(parseFloat(val)).toBe(0.5);
  });

  test('file upload enables detect button', async ({ page }) => {
    const fileInput = page.locator('#detect-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#detect-btn')).toBeEnabled({ timeout: 5_000 });
  });

  test('detect shows response after upload', async ({ page }) => {
    const fileInput = page.locator('#detect-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#detect-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#detect-btn').click();
    // Should show either detection results or an error — not the initial placeholder
    await expect(page.locator('#detect-out')).not.toHaveText(
      'Upload an image to detect objects.',
      { timeout: 20_000 }
    );
    const text = await page.locator('#detect-out').textContent();
    expect(text?.length).toBeGreaterThan(0);
  });

  test('detect button re-enables after response', async ({ page }) => {
    const fileInput = page.locator('#detect-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#detect-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#detect-btn').click();
    await expect(page.locator('#detect-out')).not.toHaveText(
      'Upload an image to detect objects.',
      { timeout: 20_000 }
    );
    await expect(page.locator('#detect-btn')).toBeEnabled();
  });

  test('tab switching shows live stream pane', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await expect(page.locator('#det-pane-live')).toBeVisible();
    await expect(page.locator('#det-pane-file')).toBeHidden();

    await page.locator('#det-tab-file').click();
    await expect(page.locator('#det-pane-file')).toBeVisible();
    await expect(page.locator('#det-pane-live')).toBeHidden();
  });

  test('live stream pane shows connect button and disconnected status', async ({ page }) => {
    await page.locator('#det-tab-live').click();
    await expect(page.locator('#det-ws-btn')).toBeVisible();
    await expect(page.locator('#det-ws-label')).toHaveText('disconnected');
  });
});
