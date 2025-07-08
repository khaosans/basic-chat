import { Page, expect } from '@playwright/test';

export class ChatHelper {
  constructor(private page: Page) {}

  // Wait for the app to load and the chat input to appear, with robust error logging
  async waitForAppLoad() {
    try {
      await this.page.waitForSelector('text=BasicChat', { timeout: 40000 });
      await this.page.getByPlaceholder('Type a message...').waitFor({ timeout: 10000 });
    } catch (err) {
      if (!this.page.isClosed()) {
        // Save a screenshot for debugging
        await this.page.screenshot({ path: 'debug-failure.png' });
        // Log page content for inspection
        const content = await this.page.content();
        console.error('❌ waitForAppLoad failed. Page content at failure:', content);
      } else {
        console.error('❌ waitForAppLoad failed. Page was closed before error handling.');
      }
      console.error('❌ waitForAppLoad error:', err);
      throw err;
    }
  }

  // Send a message using the chat input and send button
  async sendMessage(message: string) {
    const chatInput = this.page.getByPlaceholder('Type a message...');
    await chatInput.waitFor({ timeout: 10000 });
    await chatInput.fill(message);
    await this.page.keyboard.press('Enter');
  }

  // Wait for a chat response to appear
  async waitForResponse(timeout = 30000) {
    await this.page.waitForSelector('[data-testid="stChatMessage"]', { timeout });
  }

  // Get the last chat response element
  async getLastResponse() {
    const responses = this.page.locator('[data-testid="stChatMessage"]');
    return responses.last();
  }

  // Switch reasoning mode (if selectbox is present)
  async selectReasoningMode(mode: string) {
    await this.page.selectOption('select[data-testid="stSelectbox"]', mode);
    await expect(this.page.locator(`text=${mode}`)).toBeVisible();
  }

  // Upload a document (if file input is present)
  async uploadDocument(filePath: string) {
    await this.page.setInputFiles('input[type="file"]', filePath);
    await this.page.waitForSelector('text=Processing document', { timeout: 30000 });
    await this.page.waitForSelector('text=Document processed successfully', { timeout: 60000 });
  }

  async isPageValid() {
    try {
      await this.page.evaluate(() => document.readyState);
      return true;
    } catch {
      return false;
    }
  }
} 