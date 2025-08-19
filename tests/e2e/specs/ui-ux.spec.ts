/**
 * UI/UX Tests for BasicChat Streamlit App
 *
 * This test suite verifies that UI improvements work correctly:
 * - Dropdown menu visibility and styling
 * - Sidebar element contrast and readability
 * - Form element accessibility
 *
 * To run:
 *   npx playwright test tests/e2e/specs/ui-ux.spec.ts --project=chromium
 */
import { test, expect } from '@playwright/test';
import { ChatHelper } from '../helpers/chat-helpers';

test.describe('UI/UX Improvements', () => {
  let chatHelper: ChatHelper;

  test.beforeEach(async ({ page }) => {
    chatHelper = new ChatHelper(page);
    await page.goto('/');
    await chatHelper.waitForAppLoad();
  });

  test('should have visible dropdown menus with proper contrast', async ({ page }) => {
    // Test reasoning mode dropdown
    const reasoningDropdown = page.locator('select[data-testid="stSelectbox"]').first();
    await expect(reasoningDropdown).toBeVisible();
    
    // Check that dropdown has proper styling
    const dropdownStyles = await reasoningDropdown.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        backgroundColor: styles.backgroundColor,
        color: styles.color,
        borderColor: styles.borderColor,
        fontWeight: styles.fontWeight,
        fontSize: styles.fontSize
      };
    });

    // Verify dropdown has white background and dark text
    expect(dropdownStyles.backgroundColor).toMatch(/rgb\(255,\s*255,\s*255\)/);
    expect(dropdownStyles.color).toMatch(/rgb\(0,\s*0,\s*0\)/);
    expect(parseInt(dropdownStyles.fontWeight)).toBeGreaterThanOrEqual(600);
    expect(dropdownStyles.fontSize).toBe('14px');
  });

  test('should display selected dropdown values clearly', async ({ page }) => {
    // Get the reasoning mode dropdown
    const reasoningDropdown = page.locator('select[data-testid="stSelectbox"]').first();
    
    // Check initial selected value is visible
    const selectedValue = await reasoningDropdown.evaluate((el) => {
      const select = el as HTMLSelectElement;
      return select.options[select.selectedIndex]?.text || '';
    });
    
    expect(selectedValue).toBeTruthy();
    expect(selectedValue.length).toBeGreaterThan(0);
    
    // Verify the selected text is visible in the dropdown
    const dropdownText = await reasoningDropdown.textContent();
    expect(dropdownText).toContain(selectedValue);
  });

  test('should have proper sidebar styling and contrast', async ({ page }) => {
    // Check sidebar background
    const sidebar = page.locator('.css-1d391kg');
    await expect(sidebar).toBeVisible();
    
    const sidebarStyles = await sidebar.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        backgroundColor: styles.backgroundColor,
        borderRight: styles.borderRight
      };
    });

    // Verify sidebar has proper background and border
    expect(sidebarStyles.backgroundColor).toMatch(/rgb\(248,\s*249,\s*250\)/);
    expect(sidebarStyles.borderRight).toContain('1px solid');
  });

  test('should have visible form elements in sidebar', async ({ page }) => {
    // Check for reasoning mode label
    await expect(page.locator('text=Reasoning Mode')).toBeVisible();
    
    // Check for document upload area
    const fileUploader = page.locator('.stFileUploader');
    await expect(fileUploader).toBeVisible();
    
    // Check for AI validation section
    await expect(page.locator('text=AI Validation')).toBeVisible();
  });

  test('should maintain dropdown functionality while improving visibility', async ({ page }) => {
    const chatHelper = new ChatHelper(page);
    
    // Test changing reasoning mode
    const originalMode = await page.locator('select[data-testid="stSelectbox"]').first()
      .evaluate((el) => (el as HTMLSelectElement).value);
    
    // Change to a different mode
    await chatHelper.selectReasoningMode('Chain-of-Thought');
    
    // Verify the mode changed
    const newMode = await page.locator('select[data-testid="stSelectbox"]').first()
      .evaluate((el) => (el as HTMLSelectElement).value);
    
    expect(newMode).toBe('Chain-of-Thought');
    expect(newMode).not.toBe(originalMode);
  });

  test('should have proper contrast for all interactive elements', async ({ page }) => {
    // Check button styling
    const buttons = page.locator('.stButton button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      const firstButton = buttons.first();
      const buttonStyles = await firstButton.evaluate((el) => {
        const styles = window.getComputedStyle(el);
        return {
          backgroundColor: styles.backgroundColor,
          color: styles.color,
          border: styles.border
        };
      });

      // Verify button has proper contrast
      expect(buttonStyles.backgroundColor).toMatch(/rgb\(16,\s*163,\s*127\)/);
      expect(buttonStyles.color).toMatch(/rgb\(255,\s*255,\s*255\)/);
    }
  });

  test('should handle dropdown interactions without breaking', async ({ page }) => {
    // Test that dropdowns can be opened and closed
    const reasoningDropdown = page.locator('select[data-testid="stSelectbox"]').first();
    
    // Click on dropdown to open it
    await reasoningDropdown.click();
    
    // Verify dropdown options are visible
    const options = page.locator('select[data-testid="stSelectbox"] option');
    await expect(options.first()).toBeVisible();
    
    // Select an option
    await reasoningDropdown.selectOption('Multi-Step');
    
    // Verify selection worked
    const selectedValue = await reasoningDropdown.evaluate((el) => (el as HTMLSelectElement).value);
    expect(selectedValue).toBe('Multi-Step');
  });
});
