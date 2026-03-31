const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('Model Download Manager', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navDashboard).click();
    // Download manager lives inside the overview tab (visible by default)
    await expect(page.locator(S.dashOverview)).toBeVisible();
  });

  test('download form elements are visible', async ({ page }) => {
    await expect(page.locator(S.dlSource)).toBeVisible();
    await expect(page.locator(S.dlRepo)).toBeVisible();
    await expect(page.locator(S.dlRevision)).toBeVisible();
    await expect(page.locator(S.dlBtn)).toBeVisible();
  });

  test('source selector has HuggingFace and URL options', async ({ page }) => {
    await expect(
      page.locator(`${S.dlSource} option[value="huggingface"]`)
    ).toHaveCount(1);
    await expect(
      page.locator(`${S.dlSource} option[value="url"]`)
    ).toHaveCount(1);
  });

  test('"No downloads yet." shown initially', async ({ page }) => {
    await expect(page.locator(S.dlTasksEmpty)).toBeVisible();
    await expect(page.locator(S.dlTasksEmpty)).toHaveText('No downloads yet.');
  });

  test('repo input accepts text', async ({ page }) => {
    await page.locator(S.dlRepo).fill('meta-llama/Llama-3.2-1B');
    await expect(page.locator(S.dlRepo)).toHaveValue('meta-llama/Llama-3.2-1B');
  });

  test('revision field is visible for HuggingFace source', async ({ page }) => {
    await page.locator(S.dlSource).selectOption('huggingface');
    await expect(page.locator(S.dlRevisionWrap)).toBeVisible();
  });

  test('selecting URL source hides revision field', async ({ page }) => {
    await page.locator(S.dlSource).selectOption('url');
    await expect(page.locator(S.dlRevisionWrap)).toBeHidden();
  });

  test('Download button is enabled', async ({ page }) => {
    await expect(page.locator(S.dlBtn)).toBeEnabled();
  });

  test('revision input accepts custom value', async ({ page }) => {
    await page.locator(S.dlRevision).fill('v1.0');
    await expect(page.locator(S.dlRevision)).toHaveValue('v1.0');
  });
});
