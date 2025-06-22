#!/usr/bin/env python3
"""
Test script for document upload functionality
This script helps diagnose upload issues by testing the document processor directly
"""

import os
import sys
import logging
import tempfile
import pytest
from io import BytesIO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_processor import DocumentProcessor
from config import EMBEDDING_MODEL, VISION_MODEL

class MockUploadedFile:
    """Mock uploaded file for testing"""
    def __init__(self, name, content, file_type):
        self.name = name
        self._content = content
        self.type = file_type
        self.file_id = f"test_{name}_{hash(content)}"
    
    def getvalue(self):
        return self._content

@pytest.mark.integration
def test_text_upload():
    """Test text file upload"""
    logger.info("=== Testing Text File Upload ===")
    
    try:
        # Create document processor
        doc_processor = DocumentProcessor()
        logger.info("Document processor created successfully")
        
        # Create mock text file
        text_content = b"This is a test document for upload functionality. It contains some sample text to verify that the document processing works correctly."
        mock_file = MockUploadedFile("test.txt", text_content, "text/plain")
        
        logger.info(f"Created mock text file: {mock_file.name} ({len(text_content)} bytes)")
        
        # Process the file
        doc_processor.process_file(mock_file)
        logger.info("Text file processed successfully")
        
        # Check if file was added
        processed_files = doc_processor.get_processed_files()
        logger.info(f"Processed files: {processed_files}")
        
        # Test search functionality
        search_results = doc_processor.search_documents("test document")
        logger.info(f"Search results: {len(search_results)} found")
        
        assert len(processed_files) > 0, "No files were processed"
        return True
        
    except Exception as e:
        logger.error(f"Text upload test failed: {e}")
        pytest.fail(f"Text upload test failed: {e}")

@pytest.mark.integration
def test_pdf_upload():
    """Test PDF file upload with a real PDF file"""
    logger.info("=== Testing PDF File Upload ===")
    
    try:
        # Create document processor
        doc_processor = DocumentProcessor()
        logger.info("Document processor created successfully")
        
        # Use the generated real PDF file
        pdf_path = Path(__file__).parent / "test_sample.pdf"
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        mock_file = MockUploadedFile("test_sample.pdf", pdf_content, "application/pdf")
        logger.info(f"Loaded real PDF file: {mock_file.name} ({len(pdf_content)} bytes)")
        
        # Process the file
        doc_processor.process_file(mock_file)
        logger.info("PDF file processed successfully")
        
        # Check if file was added
        processed_files = doc_processor.get_processed_files()
        logger.info(f"Processed files: {processed_files}")
        
        assert len(processed_files) > 0, "No files were processed"
        return True
        
    except Exception as e:
        logger.error(f"PDF upload test failed: {e}")
        pytest.fail(f"PDF upload test failed: {e}")

@pytest.mark.integration
def test_image_upload():
    """Test image file upload with a real PNG file"""
    logger.info("=== Testing Image File Upload ===")
    
    try:
        # Create document processor
        doc_processor = DocumentProcessor()
        logger.info("Document processor created successfully")
        
        # Use the generated real PNG file
        image_path = Path(__file__).parent.parent / "assets" / "problem.png"
        with open(image_path, "rb") as f:
            image_content = f.read()
        mock_file = MockUploadedFile("problem.png", image_content, "image/png")
        logger.info(f"Loaded real image file: {mock_file.name} ({len(image_content)} bytes)")
        
        # Process the file
        doc_processor.process_file(mock_file)
        logger.info("Image file processed successfully")
        
        # Check if file was added
        processed_files = doc_processor.get_processed_files()
        logger.info(f"Processed files: {processed_files}")
        
        assert len(processed_files) > 0, "No files were processed"
        return True
        
    except Exception as e:
        logger.error(f"Image upload test failed: {e}")
        pytest.fail(f"Image upload test failed: {e}")

@pytest.mark.integration
def test_image_qa_flow():
    """Test UX flow: upload image, then ask a question about it"""
    logger.info("=== Testing Image Upload + QA Flow ===")
    try:
        doc_processor = DocumentProcessor()
        logger.info("Document processor created successfully")
        # Use the generated real PNG file
        image_path = Path(__file__).parent.parent / "assets" / "problem.png"
        with open(image_path, "rb") as f:
            image_content = f.read()
        mock_file = MockUploadedFile("problem.png", image_content, "image/png")
        logger.info(f"Loaded real image file: {mock_file.name} ({len(image_content)} bytes)")
        # Process the file
        doc_processor.process_file(mock_file)
        logger.info("Image file processed successfully")
        # Simulate asking a question about the image
        from reasoning_engine import ReasoningEngine
        engine = ReasoningEngine()
        question = "What is the polynomial in the image?"
        result = engine.run(question, mode="Agent", document_processor=doc_processor)
        logger.info(f"QA result: success={result.success}, content={result.content[:100]}...")
        assert result.success, f"Reasoning failed: {result.error}"
        return True
    except Exception as e:
        logger.error(f"Image QA flow test failed: {e}")
        pytest.fail(f"Image QA flow test failed: {e}")

@pytest.mark.integration
def test_chromadb_connection():
    """Test ChromaDB connection and basic operations"""
    logger.info("=== Testing ChromaDB Connection ===")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Test ChromaDB settings
        settings = Settings(
            persist_directory="./test_chroma_db",
            anonymized_telemetry=False
        )
        logger.info("ChromaDB settings created")
        
        # Test client creation
        client = chromadb.Client(settings)
        logger.info("ChromaDB client created successfully")
        
        # Test collection creation
        collection = client.create_collection(name="test_collection")
        logger.info("Test collection created successfully")
        
        # Test basic operations
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["1"]
        )
        logger.info("Document added to collection")
        
        # Test query
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        logger.info(f"Query successful: {len(results['documents'][0])} results")
        
        # Clean up
        client.delete_collection("test_collection")
        if os.path.exists("./test_chroma_db"):
            import shutil
            shutil.rmtree("./test_chroma_db")
        
        assert len(results['documents'][0]) > 0, "No documents found in query"
        return True
        
    except Exception as e:
        logger.error(f"ChromaDB connection test failed: {e}")
        pytest.fail(f"ChromaDB connection test failed: {e}")

@pytest.mark.integration
def test_embeddings():
    """Test embedding model functionality"""
    logger.info("=== Testing Embedding Model ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test embedding model loading
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        logger.info(f"Generated embedding: {len(embedding)} dimensions")
        
        assert len(embedding) > 0, "Embedding should have dimensions"
        assert isinstance(embedding, (list, tuple, type(embedding))), "Embedding should be a vector"
        
        return True
        
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        pytest.fail(f"Embedding test failed: {e}")

@pytest.mark.integration
def test_vision_model():
    """Test vision model functionality"""
    logger.info("=== Testing Vision Model ===")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Test vision model loading
        processor = AutoProcessor.from_pretrained(VISION_MODEL)
        model = AutoModelForVision2Seq.from_pretrained(VISION_MODEL)
        logger.info(f"Vision model loaded: {VISION_MODEL}")
        
        assert processor is not None, "Processor should be loaded"
        assert model is not None, "Model should be loaded"
        
        return True
        
    except Exception as e:
        logger.error(f"Vision model test failed: {e}")
        pytest.fail(f"Vision model test failed: {e}")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 