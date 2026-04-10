// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('System Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'System' }).first().click();
    await expect(page.locator('#panel-system')).toHaveClass(/active/);
  });

  test('system panel shows info tab with hardware details', async ({ page }) => {
    await page.waitForTimeout(2_000);
    const content = await page.locator('#panel-system').textContent();
    expect(content).toMatch(/cpu|memory|gpu|arch|os/i);
  });

  test('system info content is non-empty', async ({ page }) => {
    const infoArea = page.locator('#sys-info-content, #sys-config-content, .card').first();
    await expect(infoArea).toBeAttached();
    await page.waitForTimeout(2_000);
    const text = await page.locator('#panel-system').textContent();
    expect(text?.trim().length).toBeGreaterThan(20);
  });

  test('three sub-tabs are present: Info, Performance, Config', async ({ page }) => {
    await expect(page.locator('#sys-tab-info')).toBeVisible();
    await expect(page.locator('#sys-tab-perf')).toBeVisible();
    await expect(page.locator('#sys-tab-config')).toBeVisible();
  });

  test('Info sub-tab loads system info cards', async ({ page }) => {
    // Info tab is active by default
    await expect(page.locator('#sys-info-pane')).toBeVisible();
    await page.waitForTimeout(2_500);
    // sys-info-cards should be populated
    const cards = page.locator('#sys-info-cards');
    await expect(cards).toBeAttached();
    const text = await cards.textContent();
    expect(text?.length).toBeGreaterThan(5);
  });

  test('Performance sub-tab loads metrics tiles', async ({ page }) => {
    await page.locator('#sys-tab-perf').click();
    await expect(page.locator('#sys-perf-pane')).toBeVisible();
    await expect(page.locator('#sys-info-pane')).toBeHidden();

    // Wait for loadPerformance() to populate tiles
    await page.waitForTimeout(3_000);
    const tiles = page.locator('#sys-perf-tiles');
    await expect(tiles).toBeAttached();
    const text = await tiles.textContent();
    expect(text?.length).toBeGreaterThan(0);
  });

  test('Performance refresh button reloads data', async ({ page }) => {
    await page.locator('#sys-tab-perf').click();
    await page.waitForTimeout(2_000);
    const refreshBtn = page.locator('#sys-perf-pane button', { hasText: 'Refresh' });
    await expect(refreshBtn).toBeVisible();
    await refreshBtn.click();
    await page.waitForTimeout(2_000);
    const tiles = await page.locator('#sys-perf-tiles').textContent();
    expect(tiles?.length).toBeGreaterThan(0);
  });

  test('Optimization Tips button shows tips', async ({ page }) => {
    await page.locator('#sys-tab-perf').click();
    await page.waitForTimeout(2_000);
    const tipsBtn = page.locator('button', { hasText: 'Optimization Tips' });
    await expect(tipsBtn).toBeVisible();
    await tipsBtn.click();
    // Tips div should become visible
    await expect(page.locator('#sys-opt-tips')).toBeVisible({ timeout: 8_000 });
    const tipsText = await page.locator('#sys-opt-tips').textContent();
    expect(tipsText?.length).toBeGreaterThan(0);
  });

  test('Config sub-tab loads server configuration', async ({ page }) => {
    await page.locator('#sys-tab-config').click();
    await expect(page.locator('#sys-config-pane')).toBeVisible();
    await expect(page.locator('#sys-perf-pane')).toBeHidden();

    // sys-config-content starts as "Loading…" and gets filled
    const pre = page.locator('#sys-config-content');
    await expect(pre).not.toHaveText('Loading…', { timeout: 8_000 });
    const configText = await pre.textContent();
    expect(configText?.length).toBeGreaterThan(10);
  });

  test('Config Copy Config button is present', async ({ page }) => {
    await page.locator('#sys-tab-config').click();
    const copyBtn = page.locator('#sys-config-pane button', { hasText: 'Copy Config' });
    await expect(copyBtn).toBeVisible();
  });

  test('switching back to Info tab shows info pane', async ({ page }) => {
    await page.locator('#sys-tab-perf').click();
    await expect(page.locator('#sys-perf-pane')).toBeVisible();

    await page.locator('#sys-tab-info').click();
    await expect(page.locator('#sys-info-pane')).toBeVisible();
    await expect(page.locator('#sys-perf-pane')).toBeHidden();
  });
});

test.describe('Logs Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Logs' }).first().click();
    await expect(page.locator('#panel-logs')).toHaveClass(/active/);
  });

  test('logs panel renders and attempts to load logs', async ({ page }) => {
    await page.waitForTimeout(2_000);
    const content = await page.locator('#panel-logs').textContent();
    expect(content?.trim().length).toBeGreaterThan(0);
  });
});

test.describe('Endpoints Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.locator('.nav-item', { hasText: 'Endpoints' }).first().click();
    await expect(page.locator('#panel-endpoints')).toHaveClass(/active/);
  });

  test('shows TTS, Classification, and System sections', async ({ page }) => {
    const content = await page.locator('#panel-endpoints').textContent();
    expect(content).toContain('/tts/stream');
    expect(content).toContain('/classify/batch');
    expect(content).toContain('/health');
    expect(content).toContain('/system/info');
    expect(content).toContain('/performance');
  });

  test('POST and GET method badges are visible', async ({ page }) => {
    await expect(page.locator('.method.post').first()).toBeVisible();
    await expect(page.locator('.method.get').first()).toBeVisible();
  });

  test('endpoint paths are listed correctly', async ({ page }) => {
    const paths = await page.locator('.ep-path').allTextContents();
    expect(paths).toContain('/tts/stream');
    expect(paths).toContain('/tts/engines');
    expect(paths).toContain('/health');
  });
});
