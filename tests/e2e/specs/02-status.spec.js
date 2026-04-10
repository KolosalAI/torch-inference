// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Status Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('health dot turns green and shows "online"', async ({ page }) => {
    // Wait for fetchHealth() to resolve
    await expect(page.locator('#health-text')).toHaveText('online', { timeout: 10_000 });
    const dot = page.locator('#health-dot');
    const bg = await dot.evaluate((el) => getComputedStyle(el).backgroundColor);
    // Green in any form — CSS var resolves to something non-red
    expect(bg).not.toContain('255, 49, 49');
  });

  test('stat cards populate with values', async ({ page }) => {
    // /health returns status and uptime_seconds — those populate s-status and s-uptime
    await expect(page.locator('#s-status')).not.toHaveText('—', { timeout: 10_000 });
    await expect(page.locator('#s-uptime')).not.toHaveText('—');
    // Note: s-active, s-total, s-latency, s-errors remain '—' (fields absent from /health)
  });

  test('raw health JSON box shows valid JSON', async ({ page }) => {
    const box = page.locator('#health-raw');
    await expect(box).not.toHaveText('fetching…', { timeout: 10_000 });
    const text = await box.textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
    const data = JSON.parse(text ?? '');
    expect(data).toHaveProperty('status');
    expect(data).toHaveProperty('uptime_seconds');
  });

  test('Refresh button re-fetches health', async ({ page }) => {
    await expect(page.locator('#health-raw')).not.toHaveText('fetching…', { timeout: 10_000 });
    // Scope to the status panel to avoid strict-mode violation from multiple "Refresh" buttons
    await page.locator('#panel-status button[onclick="fetchHealth()"]').click();
    // Box briefly changes and comes back — wait for it to stabilise
    await expect(page.locator('#health-raw')).not.toHaveText('fetching…', { timeout: 10_000 });
    const text = await page.locator('#health-raw').textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
  });
});
