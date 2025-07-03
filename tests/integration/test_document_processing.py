#!/usr/bin/env python3
"""
Test script to verify document processing and vector search functionality
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from document_processor import DocumentProcessor
from app import DocumentSummaryTool

def create_test_document():
    """Create a simple test document"""
    content = """
    This is a test document about artificial intelligence.
    
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Some of the activities 
    computers with artificial intelligence are designed for include:
    
    1. Speech recognition
    2. Learning
    3. Planning
    4. Problem solving
    
    Machine learning is a subset of AI that enables computers to learn and improve 
    from experience without being explicitly programmed.
    """
    
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name

def test_document_processing(mock_external_services):
    """Test the document processing functionality"""
    print("🧪 Testing Document Processing...")
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    print("✅ DocumentProcessor initialized successfully")
    
    # Create test document
    test_file_path = create_test_document()
    print(f"✅ Test document created: {test_file_path}")
    
    # Create a mock uploaded file object
    class MockUploadedFile:
        def __init__(self, file_path):
            self.name = os.path.basename(file_path)
            self.type = "text/plain"
            
        def getvalue(self):
            with open(test_file_path, 'rb') as f:
                return f.read()
    
    mock_file = MockUploadedFile(test_file_path)
    
    # Process the document
    result = doc_processor.process_file(mock_file)
    print(f"✅ Document processed: {result}")
    
    # Check processed files
    processed_files = doc_processor.get_processed_files()
    print(f"✅ Processed files: {len(processed_files)}")
    assert len(processed_files) > 0, "No files were processed"
    for file_data in processed_files:
        print(f"   - {file_data['name']} ({file_data['type']})")
    
    # Test document search
    print("\n🔍 Testing Document Search...")
    query = "What is artificial intelligence?"
    context = doc_processor.get_relevant_context(query, k=2)
    print(f"✅ Search query: '{query}'")
    print(f"✅ Found context: {len(context)} characters")
    if context:
        print(f"   Preview: {context[:200]}...")
    else:
        print("   ⚠️ No context found")
    
    # Test document summary
    print("\n📋 Testing Document Summary...")
    summary_tool = DocumentSummaryTool(doc_processor)
    summary_result = summary_tool.execute("summarize document")
    print(f"✅ Summary result: {summary_result.success}")
    assert summary_result.success, f"Summary failed: {summary_result.error}"
    if summary_result.success:
        print(f"   Content: {summary_result.content[:200]}...")
    
    # Cleanup
    os.unlink(test_file_path)
    print(f"✅ Cleanup completed")
    
    print("\n🎉 All tests passed! Document processing is working correctly.")

if __name__ == "__main__":
    # Import the DocumentSummaryTool from app.py
    import sys
    sys.path.append('.')
    
    # We need to import the Tool classes
    from abc import ABC, abstractmethod
    from typing import Optional, Dict, List
    from dataclasses import dataclass
    
    @dataclass
    class ToolResponse:
        content: str
        success: bool = True
        error: Optional[str] = None

    class Tool(ABC):
        @abstractmethod
        def name(self) -> str:
            pass

        @abstractmethod
        def description(self) -> str:
            pass

        @abstractmethod
        def triggers(self) -> List[str]:
            pass

        @abstractmethod
        def execute(self, input_text: str) -> ToolResponse:
            pass

    class DocumentSummaryTool(Tool):
        def __init__(self, doc_processor):
            self.doc_processor = doc_processor

        def name(self) -> str:
            return "Document Summary"

        def description(self) -> str:
            return "Summarizes uploaded documents."

        def triggers(self) -> List[str]:
            return ["summarize document", "summarize the document", "give me a summary"]

        def execute(self, input_text: str) -> ToolResponse:
            try:
                processed_files = self.doc_processor.get_processed_files()
                if not processed_files:
                    return ToolResponse(content="No documents have been uploaded yet.", success=False)

                summary = ""
                for file_data in processed_files:
                    summary += f"📄 **{file_data['name']}** ({file_data['type']})\n"
                    summary += f"Size: {file_data['size']} bytes\n"
                    summary += "✅ Document processed and available for search\n\n"

                return ToolResponse(content=summary)
            except Exception as e:
                return ToolResponse(content=f"Error summarizing document: {e}", success=False, error=str(e))
    
    test_document_processing() 
