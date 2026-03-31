const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('page title contains Kolosal', async ({ page }) => {
    await expect(page).toHaveTitle(/Kolosal/);
  });

  test('sidebar renders 7 nav items', async ({ page }) => {
    await expect(page.locator('.nav-item')).toHaveCount(7);
  });

  test('Status is active and its panel visible by default', async ({ page }) => {
    await expect(page.locator(S.navStatus)).toHaveClass(/active/);
    await expect(page.locator(S.panelStatus)).toHaveClass(/active/);
  });

  test('clicking TTS Stream shows TTS panel', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator(S.panelTTS)).toHaveClass(/active/);
    await expect(page.locator(S.navTTS)).toHaveClass(/active/);
  });

  test('clicking Classify shows Classify panel', async ({ page }) => {
    await page.locator(S.navClassify).click();
    await expect(page.locator(S.panelClassify)).toHaveClass(/active/);
  });

  test('clicking LLM Chat shows LLM panel', async ({ page }) => {
    await page.locator(S.navLLM).click();
    await expect(page.locator(S.panelLLM)).toHaveClass(/active/);
  });

  test('clicking Completion shows Completion panel', async ({ page }) => {
    await page.locator(S.navCompletion).click();
    await expect(page.locator(S.panelCompletion)).toHaveClass(/active/);
  });

  test('clicking Dashboard shows Dashboard panel', async ({ page }) => {
    await page.locator(S.navDashboard).click();
    await expect(page.locator(S.panelDashboard)).toHaveClass(/active/);
  });

  test('clicking Endpoints shows Endpoints panel', async ({ page }) => {
    await page.locator(S.navEndpoints).click();
    await expect(page.locator(S.panelEndpoints)).toHaveClass(/active/);
  });

  test('only one panel is active at a time', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator('.panel.active')).toHaveCount(1);
    await page.locator(S.navClassify).click();
    await expect(page.locator('.panel.active')).toHaveCount(1);
  });

  test('only one nav item is active at a time', async ({ page }) => {
    await page.locator(S.navTTS).click();
    await expect(page.locator('.nav-item.active')).toHaveCount(1);
    await page.locator(S.navDashboard).click();
    await expect(page.locator('.nav-item.active')).toHaveCount(1);
  });

  test('Endpoints panel lists endpoint rows', async ({ page }) => {
    await page.locator(S.navEndpoints).click();
    await expect(
      page.locator(S.panelEndpoints).locator('.endpoint-row').first()
    ).toBeVisible();
  });
});
