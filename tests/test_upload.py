#!/usr/bin/env python3
"""
Test script for document upload functionality
This script helps diagnose upload issues by testing the document processor directly
"""

import os
import sys
import logging
import tempfile
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def test_text_upload():
    """Test text file upload"""
    logger.info("=== Testing Text File Upload ===")
    
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
    assert len(processed_files) > 0, "No files were processed"
    
    # Test search functionality
    search_results = doc_processor.search_documents("test document")
    logger.info(f"Search results: {len(search_results)} found")
    assert len(search_results) > 0, "No search results found"

def test_pdf_upload():
    """Test PDF file upload with a real PDF file"""
    logger.info("=== Testing PDF File Upload ===")
    
    # Create document processor
    doc_processor = DocumentProcessor()
    logger.info("Document processor created successfully")
    
    # Use the generated real PDF file
    pdf_path = "tests/data/test_sample.pdf"
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

def test_image_upload():
    """Test image file upload with a real PNG file"""
    logger.info("=== Testing Image File Upload ===")
    
    # Create document processor
    doc_processor = DocumentProcessor()
    logger.info("Document processor created successfully")
    
    # Use the generated real PNG file
    image_path = "tests/data/test_sample.png"
    with open(image_path, "rb") as f:
        image_content = f.read()
    mock_file = MockUploadedFile("test_sample.png", image_content, "image/png")
    logger.info(f"Loaded real image file: {mock_file.name} ({len(image_content)} bytes)")
    
    # Process the file
    doc_processor.process_file(mock_file)
    logger.info("Image file processed successfully")
    
    # Check if file was added
    processed_files = doc_processor.get_processed_files()
    logger.info(f"Processed files: {processed_files}")
    assert len(processed_files) > 0, "No files were processed"

def test_image_qa_flow():
    """Test UX flow: upload image, then ask a question about it"""
    logger.info("=== Testing Image Upload + QA Flow ===")
    
    doc_processor = DocumentProcessor()
    logger.info("Document processor created successfully")
    # Use the generated real PNG file
    image_path = "tests/data/test_sample.png"
    with open(image_path, "rb") as f:
        image_content = f.read()
    mock_file = MockUploadedFile("test_sample.png", image_content, "image/png")
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

def test_chromadb_connection():
    """Test ChromaDB connection and basic operations"""
    logger.info("=== Testing ChromaDB Connection ===")
    
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
    assert len(results['documents'][0]) > 0, "No query results found"
    
    # Clean up
    client.delete_collection("test_collection")
    if os.path.exists("./test_chroma_db"):
        import shutil
        shutil.rmtree("./test_chroma_db")

def test_embeddings():
    """Test embeddings functionality"""
    logger.info("=== Testing Embeddings ===")
    
    from langchain_ollama import OllamaEmbeddings
    
    # Test embeddings initialization
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )
    logger.info(f"Embeddings initialized with model: {EMBEDDING_MODEL}")
    
    # Test embedding generation
    test_text = "This is a test sentence for embeddings."
    embedding = embeddings.embed_query(test_text)
    logger.info(f"Embedding generated successfully, dimension: {len(embedding)}")
    assert len(embedding) > 0, "Embedding should have non-zero dimension"

def test_vision_model():
    """Test vision model functionality"""
    logger.info("=== Testing Vision Model ===")
    
    from langchain_ollama import ChatOllama
    
    # Test vision model initialization
    vision_model = ChatOllama(model=VISION_MODEL)
    logger.info(f"Vision model initialized: {VISION_MODEL}")
    assert vision_model is not None, "Vision model should be initialized"

def main():
    """Run all tests"""
    logger.info("Starting upload functionality tests")
    
    # Clean up any existing ChromaDB directories first
    logger.info("Cleaning up existing ChromaDB directories")
    DocumentProcessor.cleanup_all_chroma_directories()
    
    tests = [
        ("ChromaDB Connection", test_chromadb_connection),
        ("Embeddings", test_embeddings),
        ("Vision Model", test_vision_model),
        ("Text Upload", test_text_upload),
        ("PDF Upload", test_pdf_upload),
        ("Image Upload", test_image_upload),
        ("Image QA Flow", test_image_qa_flow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Clean up after tests
    logger.info("Cleaning up after tests")
    DocumentProcessor.cleanup_all_chroma_directories()
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Upload functionality should work correctly.")
    else:
        logger.error("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 