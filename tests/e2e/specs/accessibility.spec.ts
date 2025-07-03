import { test, expect } from '@playwright/test';

test.describe('Accessibility', () => {
  test('should meet WCAG standards', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
    
    // Check for proper heading structure
    await expect(page.locator('h1')).toBeVisible();
    
    // Check for alt text on images
    const images = page.locator('img');
    for (let i = 0; i < await images.count(); i++) {
      const alt = await images.nth(i).getAttribute('alt');
      expect(alt).toBeTruthy();
    }
    
    // Check for proper form labels
    await expect(page.locator('label')).toBeVisible();
  });
}); 