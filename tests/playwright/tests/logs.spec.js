const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Logs', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navLogs).click();
    await expect(page.locator(S.panelLogs)).toHaveClass(/active/);
  });

  test('Logs nav item is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(S.navLogs)).toBeVisible();
  });

  test('log file list loads and shows files', async ({ page }) => {
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        available_log_files: [{ name: 'server.log', size_mb: 2.4, line_count: 18420, modified: '2026-04-05 10:00:00' }],
        log_directory: 'logs', log_level: 'info', total_log_size_mb: 2.4,
      }),
    }));
    await page.reload();
    await page.locator(S.navLogs).click();
    await expect(page.locator(S.logsFileList)).toContainText('server.log', { timeout: 5000 });
  });

  test('View button loads log content into viewer', async ({ page }) => {
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ available_log_files: [
        { name: 'server.log', size_mb: 0.1, line_count: 3, modified: '2026-04-05 10:00:00' },
      ], log_directory: 'logs', log_level: 'info', total_log_size_mb: 0.1 }),
    }));
    await page.route('/logs/server.log', route => route.fulfill({
      status: 200, contentType: 'text/plain', body: '[INFO] server started\n[INFO] listening on :8000',
    }));
    await page.reload();
    await page.locator(S.navLogs).click();
    await page.locator(S.logsFileList + ' button:has-text("View")').first().click();
    await expect(page.locator(S.logsViewerContent)).toContainText('server started', { timeout: 5000 });
  });

  test('Clear button calls DELETE /logs/{file} after confirmation', async ({ page }) => {
    let deleteCalled = '';
    await page.route('/logs', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ available_log_files: [
        { name: 'server.log', size_mb: 0.1, line_count: 3, modified: '2026-04-05 10:00:00' },
      ], log_directory: 'logs', log_level: 'info', total_log_size_mb: 0.1 }),
    }));
    await page.route('/logs/**', async route => {
      if (route.request().method() === 'DELETE') {
        deleteCalled = route.request().url();
        await route.fulfill({ status: 200 });
      } else { await route.continue(); }
    });
    await page.reload();
    await page.locator(S.navLogs).click();
    page.once('dialog', d => d.accept());
    await page.locator(S.logsFileList + ' button:has-text("Clear")').first().click();
    await expect.poll(() => deleteCalled, { timeout: 5000 }).toContain('/logs/server.log');
  });
});
