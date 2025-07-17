import { test, expect } from '@playwright/test';
import { ChatHelper } from '../helpers/chat-helpers';

// Utility to print debug info on failure
async function printDebugInfo(page) {
  // eslint-disable-next-line no-console
  console.error('Page content at failure:', await page.content());
  await page.screenshot({ path: `debug-failure-${Date.now()}.png` });
}

test('BasicChat E2E: should load, send a message, and receive a response', async ({ page }) => {
  const chat = new ChatHelper(page);
  await chat.waitForAppLoad();
  const chatInput = page.getByPlaceholder('Type a message...');
  await expect(chatInput).toBeVisible();
  await chat.sendMessage('Hello, world!');
  await chat.waitForResponse();
  const lastResponse = await chat.getLastResponse();
  await expect(lastResponse).toBeVisible();
  // Optionally, check for a greeting in the response
  await expect(lastResponse).toContainText(/hello|hi|world/i);
});

test('BasicChat E2E: should load, send a message, and receive a streaming response', async ({ page }) => {
  const chat = new ChatHelper(page);
  await chat.waitForAppLoad();
  const chatInput = page.getByPlaceholder('Type a message...');
  await expect(chatInput).toBeVisible();
  await chat.sendMessage('Hello, world!');
  await page.waitForSelector('[data-testid="stChatMessage"]', { timeout: 30000 });
  // Wait for response to complete (no more streaming indicator)
  await page.waitForFunction(() => {
    const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
    const lastMessage = messages[messages.length - 1];
    return lastMessage && !lastMessage.textContent?.includes('▌');
  }, { timeout: 60000 });
  const lastResponse = await chat.getLastResponse();
  await expect(lastResponse).toBeVisible();
  await expect(lastResponse).toContainText(/hello|hi|world/i);
  const responseText = await lastResponse.textContent();
  expect(responseText?.length).toBeGreaterThan(10);
}); 