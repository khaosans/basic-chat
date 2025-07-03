import { Page, expect } from '@playwright/test';

export class ChatHelper {
  constructor(private page: Page) {}

  async waitForAppLoad() {
    await this.page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
    await this.page.waitForSelector('textarea[data-testid="stTextInput"]', { timeout: 10000 });
  }

  async sendMessage(message: string) {
    await this.page.fill('textarea[data-testid="stTextInput"]', message);
    await this.page.click('button:has-text("Send")');
  }

  async waitForResponse(timeout = 30000) {
    await this.page.waitForSelector('.stMarkdown', { timeout });
  }

  async getLastResponse() {
    const responses = this.page.locator('.stMarkdown');
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