import logging
from app import DocumentProcessor
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_processing():
    processor = DocumentProcessor()
    
    # Test image upload
    with open("test_image.png", "rb") as f:
        file_content = f.read()
    
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        
        def read(self):
            return self._content
            
        def getvalue(self):
            return self._content
    
    mock_file = MockFile("test_image.png", file_content)
    
    try:
        result = processor.process_file(mock_file)
        logger.info(f"Processing result: {result}")
        
        # Test querying
        context = processor.get_relevant_context("What's in the image?")
        logger.info(f"Retrieved context: {context}")
        
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_image_processing()
    print(f"Test {'passed' if success else 'failed'}") 