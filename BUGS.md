# Known Bugs and Issues

## PNG Processing Issues
**Status**: Open
**Priority**: Medium
**Reported**: 2024-03-24

### Description
Some PNG files are not being processed correctly by the OCR system. This appears to be related to:
- Image quality/contrast
- Text clarity
- PNG compression type
- Image size and resolution

### Steps to Reproduce
1. Upload a PNG file with text
2. Observe OCR results
3. Compare with same image in JPEG format

### Current Workaround
- Convert PNG to JPEG before uploading
- Use high-contrast images
- Ensure text is clearly visible

### Proposed Solution
1. Add pre-processing steps for PNG files:
   - Convert to grayscale
   - Adjust contrast
   - Apply thresholding
2. Try multiple OCR passes with different settings
3. Add fallback to alternative OCR engines

### Technical Notes
Related to Tesseract OCR limitations with certain PNG formats. 