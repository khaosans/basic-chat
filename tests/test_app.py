import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app import DocumentProcessor

def test_document_processor_init():
    processor = DocumentProcessor()
    assert processor is not None
    assert processor.processed_files == []
    assert processor.processed_filenames == set()

def test_mime_type_detection():
    processor = DocumentProcessor()
    assert processor._get_mime_type("test.pdf") == "application/pdf"
    assert processor._get_mime_type("test.png") == "image/png"
    assert processor._get_mime_type("test.txt") == "text/plain" 