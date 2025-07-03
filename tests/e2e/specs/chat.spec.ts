import { test, expect } from '@playwright/test';
import { ChatHelper } from '../helpers/chat-helpers';

test.describe('Chat Interface', () => {
  let chat: ChatHelper;
  test.beforeEach(async ({ page }) => {
    chat = new ChatHelper(page);
    await chat.waitForAppLoad();
  });

  test('should display chat interface', async ({ page }) => {
    // Check for main chat elements
    await expect(page.locator('textarea[data-testid="stTextInput"]')).toBeVisible();
    await expect(page.locator('button:has-text("Send")')).toBeVisible();
  });

  test('should send and receive messages', async ({ page }) => {
    const testMessage = 'Hello, how are you?';
    await chat.sendMessage(testMessage);
    await chat.waitForResponse();
    await expect(await chat.getLastResponse()).toContainText(testMessage);
  });

  test('should handle reasoning modes', async ({ page }) => {
    // Test reasoning mode selector
    await chat.selectReasoningMode('Multi-Step Reasoning');
    
    // Verify mode change
    await expect(page.locator('text=Multi-Step Reasoning')).toBeVisible();
  });
}); 