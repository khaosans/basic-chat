import { Page, expect } from '@playwright/test';

export class ChatHelper {
  constructor(private page: Page) {}

  async waitForAppLoad() {
    try {
      await this.page.getByPlaceholder('Type a message...').waitFor({ timeout: 30000 });
    } catch (err) {
      console.error('Page content at failure:', await this.page.content());
      await this.page.screenshot({ path: 'debug-failure.png' });
      throw err;
    }
  }

  async sendMessage(message: string) {
    const chatInput = this.page.getByPlaceholder('Type a message...');
    await chatInput.fill(message);
    await this.page.keyboard.press('Enter');
  }

  async waitForResponse(timeout = 30000) {
    await this.page.waitForSelector('[data-testid="stChatMessage"]', { timeout });
  }

  async getLastResponse() {
    const responses = this.page.locator('[data-testid="stChatMessage"]');
    return responses.last();
  }

  async selectReasoningMode(mode: string) {
    await this.page.selectOption('select[data-testid="stSelectbox"]', mode);
    await expect(this.page.locator(`text=${mode}`)).toBeVisible();
  }

  async uploadDocument(filePath: string) {
    await this.page.setInputFiles('input[type="file"]', filePath);
    await this.page.waitForSelector('text=Processing document', { timeout: 30000 });
    await this.page.waitForSelector('text=Document processed successfully', { timeout: 60000 });
  }
} 