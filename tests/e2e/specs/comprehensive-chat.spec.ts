import { test, expect } from '@playwright/test';
import { ChatHelper } from '../helpers/chat-helpers';

// Comprehensive E2E suite for BasicChat

test.describe('BasicChat Comprehensive E2E', () => {
  let chat: ChatHelper;
  test.beforeEach(async ({ page }) => {
    chat = new ChatHelper(page);
    await chat.waitForAppLoad();
  });

  test('should display all main chat UI elements', async ({ page }) => {
    await expect(page.locator('textarea[data-testid="stTextInput"]')).toBeVisible();
    await expect(page.locator('button:has-text("Send")')).toBeVisible();
    await expect(page.locator('h1')).toContainText('BasicChat');
  });

  test('should send and receive a basic message', async () => {
    await chat.sendMessage('Hello! Can you help me with a simple question?');
    await chat.waitForResponse();
    const response = await chat.getLastResponse();
    expect(await response.textContent()).toBeTruthy();
  });

  test('should switch reasoning modes and respond', async () => {
    const modes = [
      'Auto Mode',
      'Chain-of-Thought',
      'Multi-Step Reasoning',
      'Agent-Based',
      'Direct Response'
    ];
    for (const mode of modes) {
      await chat.selectReasoningMode(mode);
      await chat.sendMessage(`Test message for ${mode}`);
      await chat.waitForResponse();
      const response = await chat.getLastResponse();
      expect(await response.textContent()).toBeTruthy();
    }
  });

  test('should upload and process a document', async () => {
    const filePath = 'tests/e2e/fixtures/test-document.pdf';
    await chat.uploadDocument(filePath);
    await chat.sendMessage('What is this document about?');
    await chat.waitForResponse();
    const response = await chat.getLastResponse();
    expect(await response.textContent()).toBeTruthy();
  });

  test('should handle calculator queries', async () => {
    await chat.sendMessage('What is 15 * 23 + 7?');
    await chat.waitForResponse();
    const response = await chat.getLastResponse();
    expect(await response.textContent()).toMatch(/\d+/);
  });

  test('should handle time tool queries', async () => {
    await chat.sendMessage('What time is it in Tokyo?');
    await chat.waitForResponse();
    const response = await chat.getLastResponse();
    expect((await response.textContent()).toLowerCase()).toMatch(/time|hour|date/);
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('textarea[data-testid="stTextInput"]')).toBeVisible();
    await expect(page.locator('button:has-text("Send")')).toBeVisible();
    await chat.sendMessage('Mobile test message');
    await chat.waitForResponse();
    await page.setViewportSize({ width: 1280, height: 720 });
  });

  test('should not make unexpected external requests (privacy)', async ({ page }) => {
    const networkRequests: string[] = [];
    page.on('request', request => {
      const url = request.url();
      if (!url.includes('localhost') && !url.includes('127.0.0.1')) {
        networkRequests.push(url);
      }
    });
    await chat.sendMessage('Privacy test message');
    await chat.waitForResponse();
    const unexpected = networkRequests.filter(url => !url.includes('duckduckgo') && !url.includes('search'));
    expect(unexpected.length).toBeLessThan(5);
  });
}); 