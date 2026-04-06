/**
 * Endpoints panel UI tests — covers the "API Reference" panel
 * that lists all server routes grouped by category card.
 */
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

// Single shared beforeEach for the entire panel — all three describe blocks
// previously duplicated this navigation setup.
test.describe('Endpoints panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navEndpoints).click();
    await expect(page.locator(S.panelEndpoints)).toHaveClass(/active/);
  });

  // ── Navigation ────────────────────────────────────────────────────────────

  test.describe('navigation', () => {
    test('Endpoints nav item becomes active', async ({ page }) => {
      await expect(page.locator(S.navEndpoints)).toHaveClass(/active/);
    });

    test('only one panel is active', async ({ page }) => {
      await expect(page.locator('.panel.active')).toHaveCount(1);
    });

    test('switching away hides Endpoints panel', async ({ page }) => {
      await page.locator(S.navStatus).click();
      await expect(page.locator(S.panelEndpoints)).not.toHaveClass(/active/);
    });
  });

  // ── Category cards ────────────────────────────────────────────────────────

  test.describe('category cards', () => {
    test('panel has a visible title', async ({ page }) => {
      await expect(page.locator(`${S.panelEndpoints} .panel-title`)).toBeVisible();
    });

    test('panel contains at least one .card', async ({ page }) => {
      expect(await page.locator(`${S.panelEndpoints} .card`).count()).toBeGreaterThanOrEqual(1);
    });

    for (const title of ['TTS', 'Classification', 'LLM', 'System']) {
      test(`${title} card is present`, async ({ page }) => {
        await expect(
          page.locator(`${S.panelEndpoints} .card-title:has-text("${title}")`)
        ).toBeVisible();
      });
    }
  });

  // ── Endpoint rows ─────────────────────────────────────────────────────────

  test.describe('endpoint rows', () => {
    test('at least one .endpoint-row is visible', async ({ page }) => {
      await expect(page.locator(`${S.panelEndpoints} .endpoint-row`).first()).toBeVisible();
    });

    test('endpoint rows display HTTP method badges', async ({ page }) => {
      await expect(
        page.locator(`${S.panelEndpoints} .endpoint-row .method`).first()
      ).toBeVisible();
    });

    test('endpoint rows display path starting with /', async ({ page }) => {
      const text = await page.locator(`${S.panelEndpoints} .endpoint-row .ep-path`).first().textContent();
      expect(text).toMatch(/^\//);
    });

    test('endpoint rows include non-empty description', async ({ page }) => {
      const text = await page.locator(`${S.panelEndpoints} .endpoint-row .ep-desc`).first().textContent();
      expect(text?.length).toBeGreaterThan(0);
    });

    for (const method of ['get', 'post']) {
      test(`${method.toUpperCase()} badges are present`, async ({ page }) => {
        await expect(
          page.locator(`${S.panelEndpoints} .endpoint-row .method.${method}`).first()
        ).toBeVisible();
      });
    }

    for (const route of ['/health', '/tts/synthesize', '/v1/models', '/classify/batch', '/stats']) {
      test(`${route} row is listed`, async ({ page }) => {
        await expect(
          page.locator(`${S.panelEndpoints} .ep-path`).filter({ hasText: new RegExp(`^${route.replace(/\//g, '\\/')}$`) })
        ).toBeVisible();
      });
    }

    test('total endpoint row count is at least 10', async ({ page }) => {
      expect(
        await page.locator(`${S.panelEndpoints} .endpoint-row`).count()
      ).toBeGreaterThanOrEqual(10);
    });
  });
});
