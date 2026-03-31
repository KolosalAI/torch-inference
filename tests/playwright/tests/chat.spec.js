const { test, expect } = require('@playwright/test');
const S = require('../utils/selectors');

test.describe('LLM Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navLLM).click();
    await expect(page.locator(S.panelLLM)).toHaveClass(/active/);
  });

  test('chat input and Send button render', async ({ page }) => {
    await expect(page.locator(S.chatInput)).toBeVisible();
    await expect(page.locator(S.chatBtn)).toBeVisible();
  });

  test('chat history is empty on load', async ({ page }) => {
    await expect(page.locator(S.chatHistory)).toBeEmpty();
  });

  test('parameter fields have correct defaults', async ({ page }) => {
    await expect(page.locator(S.chatModel)).toHaveValue('default');
    await expect(page.locator(S.chatMaxtok)).toHaveValue('256');
    await expect(page.locator(S.chatTemp)).toHaveValue('0.7');
  });

  test('sending a message adds it to history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello there');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
    await expect(
      page.locator(S.chatHistory).locator('.msg.user .msg-body')
    ).toHaveText('Hello there');
  });

  test('Enter key (without Shift) sends the message', async ({ page }) => {
    await page.locator(S.chatInput).fill('Enter key test');
    await page.locator(S.chatInput).press('Enter');
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
  });

  test('after round-trip, Send button re-enables', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
  });

  test('history contains 2 messages after a round-trip', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
    await expect(page.locator(S.chatHistory).locator('.msg')).toHaveCount(2);
  });

  test('Clear chat button empties history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.msg.user')
    ).toHaveCount(1, { timeout: 5000 });
    await page.locator('button:has-text("Clear chat")').click();
    await expect(page.locator(S.chatHistory)).toBeEmpty();
  });
});

test.describe('Text Completion', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.locator(S.navCompletion).click();
    await expect(page.locator(S.panelCompletion)).toHaveClass(/active/);
  });

  test('prompt textarea renders with default content', async ({ page }) => {
    await expect(page.locator(S.cmpPrompt)).toBeVisible();
    const value = await page.locator(S.cmpPrompt).inputValue();
    expect(value.length).toBeGreaterThan(0);
  });

  test('parameter fields render with defaults', async ({ page }) => {
    await expect(page.locator(S.cmpModel)).toHaveValue('default');
    await expect(page.locator(S.cmpMaxtok)).toHaveValue('128');
    await expect(page.locator(S.cmpTemp)).toHaveValue('0.7');
    await expect(page.locator(S.cmpTopp)).toHaveValue('1.0');
  });

  test('Complete button is enabled', async ({ page }) => {
    await expect(page.locator(S.cmpBtn)).toBeEnabled();
  });

  test('output starts as "waiting…"', async ({ page }) => {
    await expect(page.locator(S.cmpOut)).toHaveText('waiting…');
  });

  test('prompt textarea accepts custom input', async ({ page }) => {
    await page.locator(S.cmpPrompt).fill('The quick brown fox');
    await expect(page.locator(S.cmpPrompt)).toHaveValue('The quick brown fox');
  });

  test('clicking Complete updates the output box', async ({ page }) => {
    await page.locator(S.cmpBtn).click();
    await expect(page.locator(S.cmpOut)).not.toHaveText('waiting…', { timeout: 30000 });
  });
});
