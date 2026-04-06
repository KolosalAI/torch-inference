const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Health badge and Status panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Status panel is active by default — no nav click needed
  });

  test('health badge is visible in topbar', async ({ page }) => {
    await expect(page.locator(S.healthText)).toBeVisible();
  });

  test('health badge leaves "checking" state within 5s', async ({ page }) => {
    await expect(page.locator(S.healthText)).not.toHaveText('checking', { timeout: 5000 });
  });

  test('status panel shows all 6 stat cards', async ({ page }) => {
    await expect(page.locator(S.sStatus)).toBeVisible();
    await expect(page.locator(S.sUptime)).toBeVisible();
    await expect(page.locator(S.sActive)).toBeVisible();
    await expect(page.locator(S.sTotal)).toBeVisible();
    await expect(page.locator(S.sLatency)).toBeVisible();
    await expect(page.locator(S.sErrors)).toBeVisible();
  });

  test('stat cards populate from "—" within 5s', async ({ page }) => {
    await expect(page.locator(S.sStatus)).not.toHaveText('—', { timeout: 5000 });
  });

  test('health-raw box shows valid JSON from /health', async ({ page }) => {
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    const text = await page.locator(S.healthRaw).textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
  });

  test('Refresh button re-fetches and populates health-raw', async ({ page }) => {
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    await page.locator('#panel-status button:has-text("Refresh")').click();
    await expect(page.locator(S.healthRaw)).not.toHaveText('fetching…', { timeout: 5000 });
    const text = await page.locator(S.healthRaw).textContent();
    expect(() => JSON.parse(text ?? '')).not.toThrow();
  });
});
