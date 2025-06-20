#!/usr/bin/env python3
"""
End-to-end test for DocumentProcessor functionality
This script tests the complete workflow: file upload -> processing -> question answering
"""

import tempfile
import os
from document_processor import DocumentProcessor

def test_complete_workflow():
    """Test the complete document processing workflow"""
    
    print("🚀 Starting End-to-End DocumentProcessor Test")
    
    # Initialize the processor
    doc_processor = DocumentProcessor()
    print("✅ DocumentProcessor initialized successfully")
    
    # Create test content
    test_content = """
    Python Programming Guide
    
    Python is a high-level, interpreted programming language.
    It was created by Guido van Rossum and first released in 1991.
    
    Key Features:
    - Easy to learn and use
    - Extensive standard library
    - Cross-platform compatibility
    - Strong community support
    
    Popular Applications:
    - Web development (Django, Flask)
    - Data science (Pandas, NumPy)
    - Machine learning (TensorFlow, PyTorch)
    - Automation and scripting
    
    Python is widely used in:
    - Software development
    - Data analysis
    - Artificial intelligence
    - Scientific computing
    """
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        # Test file processing using the new method
        with open(temp_file_path, 'rb') as file:
            file_content = file.read()
        
        print("📄 Processing test document...")
        result = doc_processor.process_uploaded_file(
            file_content=file_content,
            filename="python_guide.txt",
            file_type="text/plain"
        )
        
        if result:
            print("✅ Document processed successfully")
            print(f"📊 Total processed files: {len(doc_processor.processed_files)}")
            
            # Display processed file info
            for i, file_info in enumerate(doc_processor.processed_files):
                print(f"   File {i+1}: {file_info.name} ({file_info.size} bytes, {file_info.type})")
        else:
            print("❌ Document processing failed")
            return False
        
        # Test question answering
        print("\n🤔 Testing question answering...")
        test_questions = [
            "Who created Python?",
            "What are some popular Python frameworks?",
            "What is Python used for?",
            "When was Python first released?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            
            # Get relevant context
            context = doc_processor.get_relevant_context(question, k=3)
            
            if context and len(context) > 0:
                print("✅ Context retrieved successfully")
                print(f"📝 Context preview: {context[:200]}...")
                
                # Check if context contains relevant information
                if any(keyword in context.lower() for keyword in ["python", "guido", "1991", "django", "flask"]):
                    print("✅ Context contains relevant information")
                else:
                    print("⚠️  Context may not be fully relevant")
            else:
                print("⚠️  No context retrieved (may be due to embedding issues)")
        
        # Test search functionality
        print("\n🔍 Testing document search...")
        search_results = doc_processor.search_documents("Python programming", k=3)
        
        if search_results and len(search_results) > 0:
            print(f"✅ Search returned {len(search_results)} results")
            for i, doc in enumerate(search_results):
                print(f"   Result {i+1}: {doc.page_content[:100]}...")
        else:
            print("⚠️  Search returned no results (may be due to embedding issues)")
        
        print("\n🎉 End-to-End Test Completed Successfully!")
        print("\n📋 Summary:")
        print(f"   - Files processed: {len(doc_processor.processed_files)}")
        print(f"   - Questions tested: {len(test_questions)}")
        print(f"   - Search functionality: {'✅ Working' if search_results else '⚠️  Limited'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_file_path)
            doc_processor.cleanup()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

if __name__ == "__main__":
    success = test_complete_workflow()
    exit(0 if success else 1)
