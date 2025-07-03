/**
 * BasicChat E2E Test Suite
 *
 * This file contains end-to-end (E2E) tests for the BasicChat Streamlit application.
 *
 * ## What is Covered
 * - Sending a message and verifying a response
 * - Switching reasoning modes and verifying the UI
 * - Uploading a document and checking for upload success
 *
 * ## Best Practices
 * - Uses robust, always-present selectors (e.g., input placeholders)
 * - Waits for UI readiness before interacting
 * - Logs page content on failure for easier debugging
 * - Uses fixtures with real, extractable text for upload tests
 *
 * ## How to Run Locally
 *   npx playwright test tests/e2e/specs/basic-e2e.spec.ts --project=chromium
 *
 * ## How to Run in CI
 *   - The test will run automatically via GitHub Actions on PRs and pushes.
 *   - Reports are uploaded as artifacts for review.
 *
 * ## Troubleshooting
 * - If a test fails, open the Playwright HTML report for screenshots, video, and logs.
 * - Ensure the test PDF contains real text (not blank or image-only).
 * - If selectors change in the UI, update them here.
 *
 * ## Maintainers
 * - See progress.md for E2E test history and updates.
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