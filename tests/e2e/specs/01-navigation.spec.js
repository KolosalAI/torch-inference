// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('page loads with correct title and branding', async ({ page }) => {
    await expect(page).toHaveTitle('Kolosal Inference');
    await expect(page.locator('.logo-name')).toContainText('Kolosal');
    await expect(page.locator('.topbar')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
  });

  test('Status panel is active by default', async ({ page }) => {
    await expect(page.locator('#panel-status')).toHaveClass(/active/);
    await expect(page.locator('.nav-item.active')).toContainText('Status');
  });

  const panels = [
    { label: 'TTS',        panelId: 'panel-tts' },
    { label: 'Classify',   panelId: 'panel-classify' },
    { label: 'Audio',      panelId: 'panel-audio' },
    { label: 'Detect',     panelId: 'panel-detect' },
    { label: 'Dashboard',  panelId: 'panel-dashboard' },
    { label: 'Models',     panelId: 'panel-models' },
    { label: 'Logs',       panelId: 'panel-logs' },
    { label: 'System',     panelId: 'panel-system' },
    { label: 'Endpoints',  panelId: 'panel-endpoints' },
  ];

  for (const { label, panelId } of panels) {
    test(`clicking "${label}" nav item shows correct panel`, async ({ page }) => {
      const btn = page.locator(`.nav-item`, { hasText: label }).first();
      await btn.click();
      await expect(page.locator(`#${panelId}`)).toHaveClass(/active/);
      await expect(btn).toHaveClass(/active/);
      // All other panels should be hidden
      await expect(page.locator('.panel.active')).toHaveCount(1);
    });
  }

  test('theme toggle switches between light and dark modes', async ({ page }) => {
    const html = page.locator('html');
    const toggle = page.locator('#theme-toggle');
    await expect(toggle).toBeVisible();

    // The initial attribute may be absent (null) or 'light' — normalise to 'light'
    const raw = await html.getAttribute('data-theme');
    const initialTheme = raw ?? 'light';

    await toggle.click();
    const newTheme = await html.getAttribute('data-theme');
    expect(newTheme).not.toBe(initialTheme);

    // Toggle back — attribute will be explicitly set to the initial value
    await toggle.click();
    const restoredTheme = await html.getAttribute('data-theme') ?? 'light';
    expect(restoredTheme).toBe(initialTheme);
  });
});
