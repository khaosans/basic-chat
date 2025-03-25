#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
import shutil

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_chatbot.core.document_engine import DocumentEngine
from rag_chatbot.core.vector_engine import VectorEngine
from rag_chatbot.core.rag_engine import RAGEngine

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('setup.log'),
            logging.StreamHandler()
        ]
    )

def clean_chroma():
    """Clean existing ChromaDB data"""
    try:
        if Path("./chroma_db").exists():
            shutil.rmtree("./chroma_db")
            logging.info("✅ Cleaned existing ChromaDB data")
    except Exception as e:
        logging.error(f"Failed to clean ChromaDB: {e}")

def verify_components():
    """Verify all components are working"""
    try:
        # Clean existing ChromaDB data
        clean_chroma()
        
        # Initialize components
        doc_engine = DocumentEngine()
        vector_engine = VectorEngine()
        rag_engine = RAGEngine()
        
        # Test document engine
        assert doc_engine.processor is not None, "Document processor not initialized"
        
        # Test vector engine
        assert vector_engine.store is not None, "Vector store not initialized"
        assert vector_engine.collection is not None, "ChromaDB collection not initialized"
        
        # Test RAG engine
        assert rag_engine.llm is not None, "LLM not initialized"
        
        # Test basic vector store operation
        test_doc = "This is a test document"
        vector_engine.store.add_texts([test_doc])
        results = vector_engine.store.similarity_search("test", k=1)
        assert len(results) > 0, "Vector store search failed"
        
        return True
    except Exception as e:
        logging.error(f"Component verification failed: {e}")
        return False

def main():
    setup_logging()
    if verify_components():
        logging.info("✅ All components verified successfully")
    else:
        logging.error("❌ Component verification failed")

if __name__ == "__main__":
    main() 