/**
 * Basic E2E Tests for BasicChat
 *
 * 1. Sends a message and checks for a response.
 * 2. Switches reasoning mode and verifies the change.
 * 3. Uploads a document and checks for upload success.
 *
 * To run:
 *   npx playwright test tests/e2e/specs/basic-e2e.spec.ts --project=chromium
 */
import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('BasicChat E2E', () => {
  test('should send a message and receive a response', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByPlaceholder('Type a message...')).toBeVisible({ timeout: 15000 });
    await page.fill('textarea[placeholder="Type a message..."]', 'Hello, BasicChat!');
    await page.click('button:has-text("Send")');
    try {
      await expect(page.locator('.stMarkdown')).toContainText('Hello', { timeout: 20000 });
    } catch (err) {
      console.error('Page content at failure:', await page.content());
      throw err;
    }
  });

  test('should switch reasoning mode and verify', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByPlaceholder('Type a message...')).toBeVisible({ timeout: 15000 });
    // Find the reasoning mode select (first selectbox)
    const select = page.locator('select').first();
    await select.selectOption({ label: 'Multi-Step Reasoning' });
    await expect(page.locator('text=Multi-Step Reasoning')).toBeVisible({ timeout: 5000 });
  });

  test('should upload a document and see upload success', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByPlaceholder('Type a message...')).toBeVisible({ timeout: 15000 });
    const filePath = path.resolve('tests/e2e/fixtures/test-document.pdf');
    await page.setInputFiles('input[type="file"]', filePath);
    // Wait for upload or processing indicator
    try {
      await expect(page.locator('text=Processing document')).toBeVisible({ timeout: 10000 });
      await expect(page.locator('text=Document processed successfully')).toBeVisible({ timeout: 30000 });
    } catch (err) {
      console.error('Page content at failure:', await page.content());
      throw err;
    }
  });
}); 