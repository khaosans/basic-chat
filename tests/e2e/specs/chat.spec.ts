import { test, expect } from '@playwright/test';

test.describe('Chat Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for Streamlit to load
    await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
  });

  test('should display chat interface', async ({ page }) => {
    // Check for main chat elements
    await expect(page.locator('textarea[data-testid="stTextInput"]')).toBeVisible();
    await expect(page.locator('button:has-text("Send")')).toBeVisible();
  });

  test('should send and receive messages', async ({ page }) => {
    const testMessage = 'Hello, how are you?';
    
    // Type message
    await page.fill('textarea[data-testid="stTextInput"]', testMessage);
    
    // Send message
    await page.click('button:has-text("Send")');
    
    // Wait for response
    await page.waitForSelector('.stMarkdown', { timeout: 30000 });
    
    // Verify message appears
    await expect(page.locator('.stMarkdown')).toContainText(testMessage);
  });

  test('should handle reasoning modes', async ({ page }) => {
    // Test reasoning mode selector
    await page.selectOption('select[data-testid="stSelectbox"]', 'Multi-Step Reasoning');
    
    // Verify mode change
    await expect(page.locator('text=Multi-Step Reasoning')).toBeVisible();
  });
}); 