import { test, expect } from '@playwright/test';
import path from 'path';
import { ChatHelper } from '../helpers/chat-helpers';

test.describe('File Upload', () => {
  let chat: ChatHelper;
  test.beforeEach(async ({ page }) => {
    chat = new ChatHelper(page);
    await chat.waitForAppLoad();
  });

  test('should upload and process a document', async ({ page }) => {
    const filePath = 'tests/e2e/fixtures/test-document.pdf';
    await chat.uploadDocument(filePath);
    await chat.sendMessage('What is this document about?');
    await chat.waitForResponse();
    const response = await chat.getLastResponse();
    expect(await response.textContent()).toBeTruthy();
  });

  test('should handle invalid file types', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/invalid.txt');
    
    await page.setInputFiles('input[type="file"]', filePath);
    
    // Should show error message
    await expect(page.locator('text=Unsupported file type')).toBeVisible();
  });
}); 