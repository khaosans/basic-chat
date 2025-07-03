#!/usr/bin/env python3
"""
Generate lightweight test assets for integration tests
"""

import os
from pathlib import Path

def create_test_files():
    """Create minimal test files for integration tests"""
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test PDF
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF'
    
    with open(test_dir / "test_document.pdf", "wb") as f:
        f.write(pdf_content)
    
    # Create minimal test text file
    with open(test_dir / "test_document.txt", "w") as f:
        f.write("This is a test document for integration testing.")
    
    # Create minimal test image (1x1 pixel PNG)
    png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82'
    
    with open(test_dir / "test_image.png", "wb") as f:
        f.write(png_content)

if __name__ == "__main__":
    create_test_files()
    print("✅ Test assets generated successfully!") 