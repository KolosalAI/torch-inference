// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Dashboard Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Dashboard' }).first().click();
    await expect(page.locator('#panel-dashboard')).toHaveClass(/active/);
  });

  test('dashboard renders tiles, sparkline and live badge', async ({ page }) => {
    await expect(page.locator('#d-uptime')).toBeVisible();
    await expect(page.locator('#d-total')).toBeVisible();
    await expect(page.locator('#d-active')).toBeVisible();
    await expect(page.locator('#d-latency')).toBeVisible();
    await expect(page.locator('#d-errors')).toBeVisible();
    await expect(page.locator('#d-rps')).toBeVisible();
    await expect(page.locator('#dash-live-badge')).toBeVisible();
    await expect(page.locator('#dash-spark')).toBeVisible();
  });

  test('SSE stream populates tiles within 5 s', async ({ page }) => {
    await expect(page.locator('#d-uptime')).not.toHaveText('—', { timeout: 8_000 });
    await expect(page.locator('#d-errors')).not.toHaveText('—');
  });

  test('resource bars are rendered', async ({ page }) => {
    await expect(page.locator('#bar-cpu')).toBeAttached();
    await expect(page.locator('#bar-ram')).toBeAttached();
    await expect(page.locator('#bar-gpu')).toBeAttached();
  });

  test('live badge becomes visible when SSE connects', async ({ page }) => {
    const badge = page.locator('#dash-live-badge');
    await expect(badge).toBeVisible();
    // Opacity should be 1 (not faded) after data arrives
    await page.waitForTimeout(5_000);
    const opacity = await badge.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
    expect(opacity).toBeCloseTo(1, 1);
  });

  test('download tasks list renders (empty or with items)', async ({ page }) => {
    // dl-tasks-list should exist in DOM
    await expect(page.locator('#dl-tasks-list')).toBeAttached();
  });
});
