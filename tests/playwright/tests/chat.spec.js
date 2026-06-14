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

  test('chat history welcome screen is shown on load', async ({ page }) => {
    await expect(page.locator(S.chatHistory).locator('.chat-row.user')).toHaveCount(0);
    await expect(page.locator('#chat-welcome')).toBeVisible();
  });

  test('parameter fields have correct defaults', async ({ page }) => {
    await expect(page.locator(S.chatTemp)).toHaveValue('0.7');
    await expect(page.locator(S.chatMaxtok)).toHaveValue('1024');
  });

  test('sending a message adds it to history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello there');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.chat-row.user')
    ).toHaveCount(1, { timeout: 5000 });
    await expect(
      page.locator(S.chatHistory).locator('.chat-row.user .chat-bubble')
    ).toHaveText('Hello there');
  });

  test('Enter key (without Shift) sends the message', async ({ page }) => {
    await page.locator(S.chatInput).fill('Enter key test');
    await page.locator(S.chatInput).press('Enter');
    await expect(
      page.locator(S.chatHistory).locator('.chat-row.user')
    ).toHaveCount(1, { timeout: 5000 });
  });

  test('after round-trip, Send button re-enables', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
  });

  test('history gains user message and assistant response after a round-trip', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hi');
    await page.locator(S.chatBtn).click();
    await expect(page.locator(S.chatBtn)).toBeEnabled({ timeout: 30000 });
    await expect(page.locator(S.chatHistory).locator('.chat-row.user')).toHaveCount(1);
    await expect(page.locator(S.chatHistory).locator('.chat-row.asst')).toHaveCount(1);
  });

  test('New chat button clears user messages from history', async ({ page }) => {
    await page.locator(S.chatInput).fill('Hello');
    await page.locator(S.chatBtn).click();
    await expect(
      page.locator(S.chatHistory).locator('.chat-row.user')
    ).toHaveCount(1, { timeout: 5000 });
    await page.locator(S.panelLLM).locator('button:has-text("New chat")').click();
    await expect(page.locator(S.chatHistory).locator('.chat-row.user')).toHaveCount(0);
    await expect(page.locator('#chat-welcome')).toBeVisible();
  });
});
