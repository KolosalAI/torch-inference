const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('System tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navSystem).click();
    await expect(page.locator(S.panelSystem)).toHaveClass(/active/);
  });

  test('System nav item is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(S.navSystem)).toBeVisible();
  });

  test('three sub-tabs are visible', async ({ page }) => {
    await expect(page.locator(S.sysTabInfo)).toBeVisible();
    await expect(page.locator(S.sysTabPerf)).toBeVisible();
    await expect(page.locator(S.sysTabConfig)).toBeVisible();
  });

  test('Info sub-tab shows system data', async ({ page }) => {
    await page.route('/system/info', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ hostname: 'testhost', os: 'macOS', arch: 'aarch64',
        cpu_count: 8, total_memory_mb: 16384, server_version: '0.5.0', uptime_seconds: 1200 }),
    }));
    await page.route('/system/gpu/stats', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify([]),
    }));
    await page.locator(S.sysTabInfo).click();
    await expect(page.locator(S.sysInfoCards)).toContainText('macOS', { timeout: 5000 });
  });

  test('Performance sub-tab loads metrics', async ({ page }) => {
    await page.route('/performance', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ avg_latency_ms: 14, p50_latency_ms: 12, p95_latency_ms: 38,
        p99_latency_ms: 62, total_requests: 500, requests_per_second: 142, error_rate: 0.01 }),
    }));
    await page.locator(S.sysTabPerf).click();
    await expect(page.locator(S.sysPerfTiles)).toContainText('14', { timeout: 5000 });
  });

  test('Optimization Tips button calls /performance/optimize', async ({ page }) => {
    await page.route('/performance', route => route.fulfill({
      status: 200, contentType: 'application/json', body: '{}',
    }));
    await page.route('/performance/optimize', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ suggestions: [
        { category: 'caching', description: 'Increase cache size', impact: 'high', action: 'Set cache_size=1000' },
      ]}),
    }));
    await page.locator(S.sysTabPerf).click();
    await page.locator(S.sysPerfPane + ' button:has-text("Optimization Tips")').click();
    await expect(page.locator(S.sysOptTips)).toContainText('Increase cache size', { timeout: 5000 });
  });

  test('Config sub-tab renders JSON from /system/config', async ({ page }) => {
    await page.route('/system/config', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ workers: 8, port: 8000, log_level: 'info' }),
    }));
    await page.locator(S.sysTabConfig).click();
    await expect(page.locator(S.sysConfigContent)).toContainText('"workers"', { timeout: 5000 });
  });
});
