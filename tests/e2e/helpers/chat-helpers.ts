import { Page, expect } from '@playwright/test';

export class ChatHelper {
  constructor(private page: Page) {}

  async waitForAppLoad() {
    // Add a sleep before waiting for the input to ensure infra is up
    await new Promise((resolve) => setTimeout(resolve, 5000)); // 5 seconds
    let attempts = 0;
    const maxAttempts = 3;
    while (attempts < maxAttempts) {
      try {
        await this.page.getByPlaceholder('Type a message...').waitFor({ timeout: 20000 });
        return;
      } catch (err) {
        attempts++;
        if (attempts >= maxAttempts) {
          if (await this.page.isClosed()) {
            console.error('Page was closed before app loaded!');
          } else {
            try {
              console.error('Page content at failure:', await this.page.content());
            } catch (e) {
              console.error('Could not get page content:', e);
            }
          }
          await this.page.screenshot({ path: `debug-failure-${Date.now()}.png` });
          throw err;
        }
        await this.page.reload();
        await this.page.waitForLoadState('networkidle');
      }
    }
  }

  async sendMessage(message: string) {
    const chatInput = this.page.getByPlaceholder('Type a message...');
    await chatInput.waitFor({ timeout: 10000 });
    await chatInput.fill(message);
    await this.page.keyboard.press('Enter');
  }

  async waitForResponse(timeout = 60000) {
    await this.page.waitForSelector('[data-testid="stChatMessage"]', { timeout });
  }

  async getLastResponse() {
    const responses = this.page.locator('[data-testid="stChatMessage"]');
    return responses.last();
  }

  async selectReasoningMode(mode: string) {
    await this.page.selectOption('select[data-testid="stSelectbox"]', mode);
    await expect(this.page.locator(`text=${mode}`)).toBeVisible({ timeout: 10000 });
  }

  async uploadDocument(filePath: string) {
    await this.page.setInputFiles('input[type="file"]', filePath);
    await this.page.waitForSelector('text=Processing document', { timeout: 30000 });
    await this.page.waitForSelector('text=Document processed successfully', { timeout: 90000 });
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