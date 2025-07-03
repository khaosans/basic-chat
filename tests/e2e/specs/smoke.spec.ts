/**
 * Smoke Test for BasicChat Streamlit App
 *
 * This test verifies that the main page loads and displays a unique, recognizable element.
 * - Waits for the app to load by checking for the message input placeholder.
 * - Logs the page content if the test fails for easier debugging.
 *
 * Best Practices:
 * - Use a unique, always-present selector or text.
 * - Provide clear error messages.
 * - Keep the test fast and reliable.
 *
 * To run:
 *   npx playwright test tests/e2e/specs/smoke.spec.ts --project=chromium
 */
import { test, expect } from '@playwright/test';

test.describe('Smoke Test', () => {
  test('should load the main page and display the message input', async ({ page }) => {
    await page.goto('/');
    try {
      // Wait for the message input placeholder (robust, always present)
      await expect(
        page.getByPlaceholder('Type a message...'),
        'Expected message input to be visible after app load.'
      ).toBeVisible({ timeout: 15000 });
    } catch (err) {
      // Log the page content for debugging
      // eslint-disable-next-line no-console
      console.error('Page content at failure:', await page.content());
      throw err;
    }
  });
});

test.skip('smoke test: Streamlit app loads', async ({ page }) => {
  await page.goto('http://localhost:8501');
  // Check for the Streamlit title in the page
  await expect(page).toHaveTitle(/Streamlit/i);
  // Optionally, check for any visible text or element
  await expect(page.locator('body')).toBeVisible();
}); 