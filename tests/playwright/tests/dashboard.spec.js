const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navDashboard).click();
    await expect(page.locator(S.panelDashboard)).toHaveClass(/active/);
  });

  test('Overview and Playground tabs render', async ({ page }) => {
    await expect(page.locator(S.dashTabOverview)).toBeVisible();
    await expect(page.locator(S.dashTabPlayground)).toBeVisible();
  });

  test('Overview tab is active by default', async ({ page }) => {
    await expect(page.locator(S.dashTabOverview)).toHaveClass(/active/);
    await expect(page.locator(S.dashOverview)).toBeVisible();
  });

  test('all 6 metrics tiles render', async ({ page }) => {
    await expect(page.locator(S.dUptime)).toBeVisible();
    await expect(page.locator(S.dTotal)).toBeVisible();
    await expect(page.locator(S.dActive)).toBeVisible();
    await expect(page.locator(S.dLatency)).toBeVisible();
    await expect(page.locator(S.dErrors)).toBeVisible();
    await expect(page.locator(S.dRps)).toBeVisible();
  });

  test('metrics tiles update from "—" within 5s via SSE', async ({ page }) => {
    await expect(page.locator(S.dUptime)).not.toHaveText('—', { timeout: 5000 });
  });

  test('spark canvas is rendered', async ({ page }) => {
    await expect(page.locator(S.dashSpark)).toBeVisible();
  });

  test('SSE live badge is visible', async ({ page }) => {
    await expect(page.locator(S.dashLiveBadge)).toBeVisible();
  });

  test('CPU, RAM, GPU resource bars render', async ({ page }) => {
    await expect(page.locator(S.barCpu)).toBeVisible();
    await expect(page.locator(S.barRam)).toBeVisible();
    await expect(page.locator(S.barGpu)).toBeVisible();
  });

  test('CPU, RAM, GPU value labels render', async ({ page }) => {
    await expect(page.locator(S.valCpu)).toBeVisible();
    await expect(page.locator(S.valRam)).toBeVisible();
    await expect(page.locator(S.valGpu)).toBeVisible();
  });

  test('GPU device info element is in the DOM', async ({ page }) => {
    await expect(page.locator(S.gpuDeviceInfo)).toBeAttached();
  });

  test('clicking Playground tab shows playground, hides overview', async ({ page }) => {
    await page.locator(S.dashTabPlayground).click();
    await expect(page.locator(S.dashPlayground)).toBeVisible();
    await expect(page.locator(S.dashOverview)).toBeHidden();
  });

  test('clicking Overview tab restores overview, hides playground', async ({ page }) => {
    await page.locator(S.dashTabPlayground).click();
    await page.locator(S.dashTabOverview).click();
    await expect(page.locator(S.dashOverview)).toBeVisible();
    await expect(page.locator(S.dashPlayground)).toBeHidden();
  });
});
