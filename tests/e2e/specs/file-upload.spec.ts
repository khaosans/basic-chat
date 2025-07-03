import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('File Upload', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
  });

  test('should upload PDF file', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    // Upload file
    await page.setInputFiles('input[type="file"]', filePath);
    
    // Wait for processing
    await page.waitForSelector('text=Processing complete', { timeout: 60000 });
    
    // Verify file was processed
    await expect(page.locator('text=Document processed successfully')).toBeVisible();
  });

  test('should handle invalid file types', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/invalid.txt');
    
    await page.setInputFiles('input[type="file"]', filePath);
    
    // Should show error message
    await expect(page.locator('text=Unsupported file type')).toBeVisible();
  });
}); 