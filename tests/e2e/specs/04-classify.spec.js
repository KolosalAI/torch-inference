// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

const FIXTURE_JPG = path.resolve(__dirname, '../fixtures/test.jpg');

test.describe('Image Classification Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Classify' }).first().click();
    await expect(page.locator('#panel-classify')).toHaveClass(/active/);
  });

  test('classification panel renders all controls', async ({ page }) => {
    await expect(page.locator('#dropzone')).toBeVisible();
    await expect(page.locator('#cls-topk')).toBeVisible();
    await expect(page.locator('#cls-w')).toBeVisible();
    await expect(page.locator('#cls-h')).toBeVisible();
    await expect(page.locator('#cls-btn')).toBeVisible();
    // Buttons disabled before file upload
    await expect(page.locator('#cls-btn')).toBeDisabled();
    await expect(page.locator('#cls-stream-btn')).toBeDisabled();
  });

  test('file upload enables classify button and shows preview', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);

    await expect(page.locator('#cls-btn')).toBeEnabled({ timeout: 5_000 });
    await expect(page.locator('#cls-stream-btn')).toBeEnabled();
    await expect(page.locator('#img-preview')).toBeVisible();
    await expect(page.locator('#cls-out')).toContainText('image(s) loaded');
  });

  test('classify returns predictions and displays them', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#cls-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#cls-btn').click();
    await expect(page.locator('#cls-out')).toContainText('.', { timeout: 20_000 });
    const out = await page.locator('#cls-out').textContent();
    // Should contain percentage scores like "xx.xx%"
    expect(out).toMatch(/\d+\.\d+%/);
  });

  test('classify uses custom top-k value', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#cls-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#cls-topk').fill('3');
    await page.locator('#cls-btn').click();
    await expect(page.locator('#cls-out')).toContainText('%', { timeout: 20_000 });
    const out = await page.locator('#cls-out').textContent() ?? '';
    // Count lines with percentages — should be ≤ 3
    const lines = out.split('\n').filter((l) => l.includes('%'));
    expect(lines.length).toBeLessThanOrEqual(3);
  });

  test('classify button re-enables after completion', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#cls-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#cls-btn').click();
    await expect(page.locator('#cls-out')).toContainText('%', { timeout: 20_000 });
    await expect(page.locator('#cls-btn')).toBeEnabled();
  });

  test('metrics bar appears after classification', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#cls-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#cls-btn').click();
    await expect(page.locator('#cls-out')).toContainText('%', { timeout: 20_000 });
    await expect(page.locator('#cls-metrics')).toBeVisible();
  });

  test('tab switching shows live stream pane', async ({ page }) => {
    await page.locator('#cls-tab-live').click();
    await expect(page.locator('#cls-pane-live')).toBeVisible();
    await expect(page.locator('#cls-pane-file')).toBeHidden();

    // Switch back
    await page.locator('#cls-tab-file').click();
    await expect(page.locator('#cls-pane-file')).toBeVisible();
    await expect(page.locator('#cls-pane-live')).toBeHidden();
  });

  test('streaming classify returns results', async ({ page }) => {
    const fileInput = page.locator('#img-file');
    await fileInput.setInputFiles(FIXTURE_JPG);
    await expect(page.locator('#cls-stream-btn')).toBeEnabled({ timeout: 5_000 });

    await page.locator('#cls-stream-btn').click();
    // Stream results show JSON with confidence/label (not % like batch results)
    await expect(page.locator('#cls-out')).not.toContainText('Streaming', { timeout: 20_000 });
    const out = await page.locator('#cls-out').textContent() ?? '';
    // Should contain either JSON predictions or a completion summary
    expect(out.length).toBeGreaterThan(10);
    expect(out).not.toMatch(/^Error:/);
  });
});
