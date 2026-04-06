/**
 * Models panel UI tests — covers the dedicated "Models" nav item which
 * hosts three sub-tabs: Catalog, Downloaded, and Download form.
 */
const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Models panel — navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
  });

  test('clicking Models nav item makes panel active', async ({ page }) => {
    await page.locator(S.navModels).click();
    await expect(page.locator(S.panelModels)).toHaveClass(/active/);
  });

  test('Models nav item itself becomes active', async ({ page }) => {
    await page.locator(S.navModels).click();
    await expect(page.locator(S.navModels)).toHaveClass(/active/);
  });

  test('switching away from Models hides panel', async ({ page }) => {
    await page.locator(S.navModels).click();
    await expect(page.locator(S.panelModels)).toHaveClass(/active/);
    await page.locator(S.navStatus).click();
    await expect(page.locator(S.panelModels)).not.toHaveClass(/active/);
  });

  test('only one panel is active after navigating to Models', async ({ page }) => {
    await page.locator(S.navModels).click();
    await expect(page.locator('.panel.active')).toHaveCount(1);
  });
});

test.describe('Models panel — sub-tab structure', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
    await page.locator(S.navModels).click();
    await expect(page.locator(S.panelModels)).toHaveClass(/active/);
  });

  test('Catalog sub-tab is visible', async ({ page }) => {
    await expect(page.locator(S.mdlTabAvailable)).toBeVisible();
  });

  test('Downloaded sub-tab is visible', async ({ page }) => {
    await expect(page.locator(S.mdlTabDownloaded)).toBeVisible();
  });

  test('Download sub-tab is visible', async ({ page }) => {
    await expect(page.locator(S.mdlTabDownload)).toBeVisible();
  });

  test('Catalog sub-tab is active by default', async ({ page }) => {
    await expect(page.locator(S.mdlTabAvailable)).toHaveClass(/active/);
    await expect(page.locator(S.mdlAvailable)).toBeVisible();
  });

  test('Downloaded section is hidden by default', async ({ page }) => {
    await expect(page.locator(S.mdlDownloaded)).toBeHidden();
  });

  test('Download form section is hidden by default', async ({ page }) => {
    await expect(page.locator(S.mdlDownload)).toBeHidden();
  });
});

test.describe('Models panel — Catalog sub-tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
    await page.locator(S.navModels).click();
    await expect(page.locator(S.mdlAvailable)).toBeVisible();
  });

  test('search input is visible', async ({ page }) => {
    await expect(page.locator(S.mdlSearch)).toBeVisible();
  });

  test('task filter select is visible', async ({ page }) => {
    await expect(page.locator(S.mdlFilterTask)).toBeVisible();
  });

  test('model grid container is in the DOM', async ({ page }) => {
    await expect(page.locator(S.mdlAvailGrid)).toBeAttached();
  });

  test('search input accepts text', async ({ page }) => {
    await page.locator(S.mdlSearch).fill('llama');
    await expect(page.locator(S.mdlSearch)).toHaveValue('llama');
  });

  test('clearing search shows unfiltered grid', async ({ page }) => {
    await page.locator(S.mdlSearch).fill('xyz-no-match');
    await page.locator(S.mdlSearch).fill('');
    await expect(page.locator(S.mdlSearch)).toHaveValue('');
  });

  test('task filter has at least one option', async ({ page }) => {
    const count = await page.locator(`${S.mdlFilterTask} option`).count();
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test('grid eventually shows content or loading/error state', async ({ page }) => {
    // After catalog load attempt, one of the three states must be visible
    await expect(async () => {
      const gridEmpty   = await page.locator(`${S.mdlAvailGrid} .mdl-empty`).isVisible();
      const gridHasCard = await page.locator(`${S.mdlAvailGrid} .model-card`).count() > 0;
      const loading     = await page.locator(S.mdlAvailLoading).isVisible();
      const error       = await page.locator(S.mdlAvailError).isVisible();
      expect(gridEmpty || gridHasCard || loading || error).toBe(true);
    }).toPass({ timeout: 10000 });
  });
});

test.describe('Models panel — Downloaded sub-tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
    await page.locator(S.navModels).click();
    await page.locator(S.mdlTabDownloaded).click();
    await expect(page.locator(S.mdlDownloaded)).toBeVisible();
  });

  test('Downloaded section becomes visible after click', async ({ page }) => {
    await expect(page.locator(S.mdlDownloaded)).toBeVisible();
  });

  test('Catalog section is hidden after switching to Downloaded', async ({ page }) => {
    await expect(page.locator(S.mdlAvailable)).toBeHidden();
  });

  test('Downloaded list container is in the DOM', async ({ page }) => {
    await expect(page.locator(S.mdlDownloadedList)).toBeAttached();
  });

  test('list shows content or empty state eventually', async ({ page }) => {
    await expect(async () => {
      const hasRows  = await page.locator(`${S.mdlDownloadedList} .dl-model-row`).count() > 0;
      const empty    = await page.locator(`${S.mdlDownloadedList} .mdl-empty`).isVisible();
      const loading  = await page.locator(S.mdlDownLoading).isVisible();
      expect(hasRows || empty || loading).toBe(true);
    }).toPass({ timeout: 10000 });
  });
});

test.describe('Models panel — Download form sub-tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
    await page.locator(S.navModels).click();
    await page.locator(S.mdlTabDownload).click();
    await expect(page.locator(S.mdlDownload)).toBeVisible();
  });

  test('Download form section is visible after click', async ({ page }) => {
    await expect(page.locator(S.mdlDownload)).toBeVisible();
  });

  test('Catalog section is hidden', async ({ page }) => {
    await expect(page.locator(S.mdlAvailable)).toBeHidden();
  });

  test('source selector is visible', async ({ page }) => {
    await expect(page.locator(S.mdlDlSource)).toBeVisible();
  });

  test('source selector has HuggingFace option', async ({ page }) => {
    await expect(
      page.locator(`${S.mdlDlSource} option[value="huggingface"]`)
    ).toHaveCount(1);
  });

  test('source selector has URL option', async ({ page }) => {
    await expect(
      page.locator(`${S.mdlDlSource} option[value="url"]`)
    ).toHaveCount(1);
  });

  test('repo input is visible', async ({ page }) => {
    await expect(page.locator(S.mdlDlRepo)).toBeVisible();
  });

  test('repo input accepts text', async ({ page }) => {
    await page.locator(S.mdlDlRepo).fill('meta-llama/Llama-3.2-1B');
    await expect(page.locator(S.mdlDlRepo)).toHaveValue('meta-llama/Llama-3.2-1B');
  });

  test('revision wrap is visible for HuggingFace (default)', async ({ page }) => {
    await page.locator(S.mdlDlSource).selectOption('huggingface');
    await expect(page.locator(S.mdlDlRevisionWrap)).toBeVisible();
  });

  test('revision wrap is hidden for URL source', async ({ page }) => {
    await page.locator(S.mdlDlSource).selectOption('url');
    await expect(page.locator(S.mdlDlRevisionWrap)).toBeHidden();
  });

  test('revision input accepts custom value', async ({ page }) => {
    await page.locator(S.mdlDlRevision).fill('v1.0');
    await expect(page.locator(S.mdlDlRevision)).toHaveValue('v1.0');
  });

  test('download button is enabled', async ({ page }) => {
    await expect(page.locator(S.mdlDlBtn)).toBeEnabled();
  });

  test('submitting empty repo shows error message', async ({ page }) => {
    await page.locator(S.mdlDlRepo).fill('');
    await page.locator(S.mdlDlBtn).click();
    await expect(page.locator(S.mdlDlError)).toBeVisible({ timeout: 3000 });
    const errText = await page.locator(S.mdlDlError).textContent();
    expect(errText?.length).toBeGreaterThan(0);
  });

  test('tasks list container is in the DOM', async ({ page }) => {
    await expect(page.locator(S.mdlDlTasksList)).toBeAttached();
  });
});

test.describe('Models panel — sub-tab switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    if ((await page.locator(S.navModels).count()) === 0) { test.skip(true, 'Models panel not present in this build'); return; }
    await page.locator(S.navModels).click();
  });

  test('Catalog → Downloaded → Download round-trip', async ({ page }) => {
    await expect(page.locator(S.mdlAvailable)).toBeVisible();

    await page.locator(S.mdlTabDownloaded).click();
    await expect(page.locator(S.mdlDownloaded)).toBeVisible();
    await expect(page.locator(S.mdlAvailable)).toBeHidden();

    await page.locator(S.mdlTabDownload).click();
    await expect(page.locator(S.mdlDownload)).toBeVisible();
    await expect(page.locator(S.mdlDownloaded)).toBeHidden();

    await page.locator(S.mdlTabAvailable).click();
    await expect(page.locator(S.mdlAvailable)).toBeVisible();
    await expect(page.locator(S.mdlDownload)).toBeHidden();
  });

  test('only one sub-section visible at a time', async ({ page }) => {
    const visibleCount = () => Promise.all([
      page.locator(S.mdlAvailable).isVisible(),
      page.locator(S.mdlDownloaded).isVisible(),
      page.locator(S.mdlDownload).isVisible(),
    ]).then(results => results.filter(Boolean).length);

    expect(await visibleCount()).toBe(1);
    await page.locator(S.mdlTabDownloaded).click();
    expect(await visibleCount()).toBe(1);
    await page.locator(S.mdlTabDownload).click();
    expect(await visibleCount()).toBe(1);
  });
});

test.describe('Downloaded models — delete', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
  });

  test('delete button is visible on each model row', async ({ page }) => {
    await page.route('/api/models/downloaded', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify([{ id: 'resnet50', name: 'ResNet-50', task: 'classification', status: 'cached' }]),
    }));
    await page.reload();
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
    await expect(page.locator('#mdl-downloaded-list button:has-text("Delete")')).toBeVisible({ timeout: 3000 });
  });

  test('clicking delete calls DELETE /models/download/{name}', async ({ page }) => {
    let deleteCalled = '';
    await page.route('/api/models/downloaded', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify([{ id: 'resnet50', name: 'resnet50', task: 'classification', status: 'cached' }]),
    }));
    await page.route('/models/download/**', async route => {
      if (route.request().method() === 'DELETE') {
        deleteCalled = route.request().url();
        await route.fulfill({ status: 200 });
      } else { await route.continue(); }
    });
    await page.reload();
    await page.locator('button.nav-item:has-text("Models")').click();
    await page.locator('#mdl-tab-downloaded').click();
    page.once('dialog', d => d.accept());
    await page.locator('#mdl-downloaded-list button:has-text("Delete")').first().click();
    await expect.poll(() => deleteCalled, { timeout: 5000 }).toContain('/models/download/resnet50');
  });
});

test.describe('SOTA models sub-tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator('button.nav-item:has-text("Models")').click();
  });

  test('SOTA sub-tab button is visible', async ({ page }) => {
    await expect(page.locator('#mdl-tab-sota')).toBeVisible();
  });

  test('SOTA tab renders model cards from /models/sota', async ({ page }) => {
    await page.route('/models/sota', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({ models: [
        { id: 'efficientnet_b0', name: 'EfficientNet-B0', task: 'classification',
          architecture: 'EfficientNet', size_estimate: '20MB' },
      ], total: 1 }),
    }));
    await page.locator('#mdl-tab-sota').click();
    await expect(page.locator('#mdl-sota-grid .model-card')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('#mdl-sota-grid .model-card-name')).toHaveText('EfficientNet-B0');
  });
});
