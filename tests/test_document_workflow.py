"""
Comprehensive tests for document processing workflow
Tests the complete flow: upload -> embed -> query with RAG
"""

import pytest
import tempfile
import os
import io
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import base64

from document_processor import DocumentProcessor, ProcessedFile
from reasoning_engine import MultiStepReasoning, ReasoningResult
from langchain_core.documents import Document


class TestDocumentWorkflow:
    """Test the complete document processing workflow"""
    
    @pytest.fixture
    def doc_processor(self):
        """Create a document processor instance"""
        return DocumentProcessor()
    
    @pytest.fixture
    def mock_uploaded_file(self):
        """Create a mock uploaded file"""
        file_mock = Mock()
        file_mock.name = "test_document.txt"
        file_mock.type = "text/plain"
        file_mock.getvalue.return_value = b"This is a test document content for testing purposes."
        return file_mock
    
    @pytest.fixture
    def mock_pdf_file(self):
        """Create a mock PDF file"""
        file_mock = Mock()
        file_mock.name = "test_document.pdf"
        file_mock.type = "application/pdf"
        file_mock.getvalue.return_value = b"%PDF-1.4\nTest PDF content"
        return file_mock
    
    @pytest.fixture
    def mock_image_file(self):
        """Create a mock image file"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        file_mock = Mock()
        file_mock.name = "test_image.png"
        file_mock.type = "image/png"
        file_mock.getvalue.return_value = img_bytes.getvalue()
        return file_mock
    
    def test_collection_name_sanitization(self, doc_processor):
        """Test that collection names are properly sanitized"""
        # Test with spaces and special characters
        test_cases = [
            ("Screenshot 2025-06-20 at 12_35_06 PM.png", "collection_Screenshot_2025-06-20_at_12_35_06_PM_png"),
            ("document with spaces.pdf", "collection_document_with_spaces_pdf"),
            ("file@#$%^&*().txt", "collection_file_______txt"),
            ("normal-file.txt", "collection_normal-file_txt"),
            ("", "collection_document"),  # Empty name
            ("...", "collection_document"),  # Only special chars
        ]
        
        for input_name, expected_name in test_cases:
            file_mock = Mock()
            file_mock.name = input_name
            file_mock.type = "text/plain"
            file_mock.getvalue.return_value = b"test content"
            
            # Mock the processing to avoid actual ChromaDB operations
            with patch.object(doc_processor, 'client') as mock_client:
                mock_collection = Mock()
                mock_client.get_or_create_collection.return_value = mock_collection
                
                # Mock document loading
                with patch('document_processor.PyPDFLoader') as mock_loader:
                    mock_loader.return_value.load.return_value = [Document(page_content="test")]
                    
                    doc_processor.process_file(file_mock)
                    
                    # Check that collection was created with sanitized name
                    mock_client.get_or_create_collection.assert_called_once()
                    call_args = mock_client.get_or_create_collection.call_args
                    assert call_args[1]['name'] == expected_name
    
    @patch('document_processor.PyPDFLoader')
    @patch('document_processor.UnstructuredImageLoader')
    def test_document_processing_flow(self, mock_image_loader, mock_pdf_loader, doc_processor, mock_pdf_file):
        """Test the complete document processing flow"""
        # Mock the document loaders
        mock_pdf_loader.return_value.load.return_value = [
            Document(page_content="This is PDF content about AI and machine learning.")
        ]
        
        # Mock ChromaDB operations
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock vectorstore
            with patch('document_processor.Chroma') as mock_chroma:
                mock_vectorstore = Mock()
                mock_chroma.return_value = mock_vectorstore
                
                # Process the document
                doc_processor.process_file(mock_pdf_file)
                
                # Verify the flow
                assert len(doc_processor.processed_files) == 1
                processed_file = doc_processor.processed_files[0]
                assert processed_file.name == "test_document.pdf"
                assert processed_file.type == "application/pdf"
                assert processed_file.collection_name.startswith("collection_")
                
                # Verify ChromaDB was called
                mock_client.get_or_create_collection.assert_called_once()
                mock_vectorstore.add_documents.assert_called_once()
    
    @patch('document_processor.UnstructuredImageLoader')
    def test_image_processing_flow(self, mock_image_loader, doc_processor, mock_image_file):
        """Test image processing flow"""
        # Mock the image loader
        mock_image_loader.return_value.load.return_value = [
            Document(page_content="This is an image containing text about computer vision.")
        ]
        
        # Mock ChromaDB operations
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock vectorstore
            with patch('document_processor.Chroma') as mock_chroma:
                mock_vectorstore = Mock()
                mock_chroma.return_value = mock_vectorstore
                
                # Process the image
                doc_processor.process_file(mock_image_file)
                
                # Verify the flow
                assert len(doc_processor.processed_files) == 1
                processed_file = doc_processor.processed_files[0]
                assert processed_file.name == "test_image.png"
                assert processed_file.type == "image/png"
    
    def test_document_search_functionality(self, doc_processor):
        """Test document search functionality"""
        # Add some mock processed files
        doc_processor.processed_files = [
            ProcessedFile("doc1.pdf", 1000, "application/pdf", "collection_doc1_pdf"),
            ProcessedFile("doc2.txt", 500, "text/plain", "collection_doc2_txt"),
        ]
        
        # Mock ChromaDB search
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection1 = Mock()
            mock_collection2 = Mock()
            
            # Mock search results
            mock_collection1.query.return_value = {
                'documents': [['AI and machine learning concepts']]
            }
            mock_collection2.query.return_value = {
                'documents': [['Data science and analytics']]
            }
            
            mock_client.get_collection.side_effect = [mock_collection1, mock_collection2]
            
            # Test search
            results = doc_processor.search_documents("AI", k=2)
            
            # Verify results
            assert len(results) == 2
            assert "AI and machine learning concepts" in results[0]
            assert "Data science and analytics" in results[1]
    
    def test_get_relevant_context(self, doc_processor):
        """Test getting relevant context from documents"""
        # Add mock processed files
        doc_processor.processed_files = [
            ProcessedFile("doc1.pdf", 1000, "application/pdf", "collection_doc1_pdf"),
        ]
        
        # Mock ChromaDB query
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['This document contains information about neural networks and deep learning.']],
                'distances': [[0.1]],  # High relevance (low distance)
                'metadatas': [[{'source': 'doc1.pdf'}]]
            }
            mock_client.get_collection.return_value = mock_collection
            
            # Test context retrieval
            context = doc_processor.get_relevant_context("neural networks", k=1)
            
            # Verify context contains relevant information
            assert "neural networks" in context.lower()
            assert "doc1.pdf" in context
            assert "relevance" in context.lower()
    
    @patch('reasoning_engine.ChatOllama')
    def test_multi_step_reasoning_with_documents(self, mock_chat_ollama):
        """Test multi-step reasoning with document context"""
        # Mock the LLM
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_llm.invoke.return_value.content = "Based on the document, the answer is..."
        
        # Create document processor with mock data
        doc_processor = DocumentProcessor()
        doc_processor.processed_files = [
            ProcessedFile("test.pdf", 1000, "application/pdf", "collection_test_pdf"),
        ]
        
        # Mock document search
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['The document discusses machine learning algorithms and their applications.']],
                'distances': [[0.2]],
                'metadatas': [[{'source': 'test.pdf'}]]
            }
            mock_client.get_collection.return_value = mock_collection
            
            # Create multi-step reasoning with proper mocking
            with patch.object(MultiStepReasoning, '__init__', return_value=None):
                reasoning = MultiStepReasoning(doc_processor)
                reasoning.doc_processor = doc_processor
                reasoning.llm = mock_llm
                
                # Mock the step_by_step_reasoning method
                with patch.object(reasoning, 'step_by_step_reasoning') as mock_reasoning:
                    mock_reasoning.return_value = ReasoningResult(
                        content="Based on the document, machine learning is discussed extensively.",
                        reasoning_steps=["Step 1: Analyze document", "Step 2: Extract ML content"],
                        confidence=0.9,
                        sources=["test.pdf"],
                        success=True
                    )
                    
                    # Test reasoning with document context
                    result = reasoning.step_by_step_reasoning("What does the document say about machine learning?")
                    
                    # Verify the result
                    assert isinstance(result, ReasoningResult)
                    assert result.success
                    assert "machine learning" in result.content.lower()
    
    def test_document_removal(self, doc_processor):
        """Test removing documents from the processor"""
        # Add a mock processed file
        doc_processor.processed_files = [
            ProcessedFile("test.pdf", 1000, "application/pdf", "collection_test_pdf"),
        ]
        
        # Mock ChromaDB deletion
        with patch.object(doc_processor, 'client') as mock_client:
            # Remove the document
            doc_processor.remove_file("test.pdf")
            
            # Verify collection was deleted
            mock_client.delete_collection.assert_called_once_with("collection_test_pdf")
            
            # Verify file was removed from processed files
            assert len(doc_processor.processed_files) == 0
    
    def test_reset_state(self, doc_processor):
        """Test resetting the processor state"""
        # Add mock processed files
        doc_processor.processed_files = [
            ProcessedFile("doc1.pdf", 1000, "application/pdf", "collection_doc1_pdf"),
            ProcessedFile("doc2.txt", 500, "text/plain", "collection_doc2_txt"),
        ]
        
        # Mock ChromaDB operations
        with patch.object(doc_processor, 'client') as mock_client:
            # Mock directory operations
            with patch('os.path.exists') as mock_exists, \
                 patch('shutil.rmtree') as mock_rmtree, \
                 patch('os.makedirs') as mock_makedirs:
                
                mock_exists.return_value = True
                
                # Reset state
                doc_processor.reset_state()
                
                # Verify all collections were deleted
                assert mock_client.delete_collection.call_count == 2
                
                # Verify processed files were cleared
                assert len(doc_processor.processed_files) == 0
                
                # Verify directory was reset
                mock_rmtree.assert_called_once()
                mock_makedirs.assert_called_once()
    
    def test_error_handling_invalid_file_type(self, doc_processor):
        """Test error handling for invalid file types"""
        # Create a file with unsupported type
        file_mock = Mock()
        file_mock.name = "test.xyz"
        file_mock.type = "application/unknown"
        file_mock.getvalue.return_value = b"test content"
        
        # Test that processing raises an error
        with pytest.raises(Exception) as exc_info:
            doc_processor.process_file(file_mock)
        
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_error_handling_processing_failure(self, doc_processor):
        """Test error handling when document processing fails"""
        file_mock = Mock()
        file_mock.name = "test.pdf"
        file_mock.type = "application/pdf"
        file_mock.getvalue.return_value = b"invalid pdf content"
        
        # Mock PyPDFLoader to raise an exception
        with patch('document_processor.PyPDFLoader') as mock_loader:
            mock_loader.side_effect = Exception("PDF processing failed")
            
            # Test that processing raises an error
            with pytest.raises(Exception) as exc_info:
                doc_processor.process_file(file_mock)
            
            assert "Error processing file" in str(exc_info.value)


class TestIntegrationWorkflow:
    """Integration tests for the complete workflow"""
    
    @patch('reasoning_engine.ChatOllama')
    @patch('document_processor.PyPDFLoader')
    def test_complete_workflow(self, mock_pdf_loader, mock_chat_ollama):
        """Test the complete workflow: upload -> embed -> query"""
        # Mock the LLM
        mock_llm = Mock()
        mock_chat_ollama.return_value = mock_llm
        mock_llm.invoke.return_value.content = "Based on the document content, the answer is..."
        
        # Create document processor
        doc_processor = DocumentProcessor()
        
        # Create mock PDF file
        file_mock = Mock()
        file_mock.name = "AI_Research.pdf"
        file_mock.type = "application/pdf"
        file_mock.getvalue.return_value = b"%PDF-1.4\nAI research content"
        
        # Mock document loading
        mock_pdf_loader.return_value.load.return_value = [
            Document(page_content="This document discusses artificial intelligence, machine learning, and neural networks.")
        ]
        
        # Mock ChromaDB operations
        with patch.object(doc_processor, 'client') as mock_client:
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock vectorstore
            with patch('document_processor.Chroma') as mock_chroma:
                mock_vectorstore = Mock()
                mock_chroma.return_value = mock_vectorstore
                
                # Step 1: Upload and process document
                doc_processor.process_file(file_mock)
                
                # Verify document was processed
                assert len(doc_processor.processed_files) == 1
                assert doc_processor.processed_files[0].name == "AI_Research.pdf"
                
                # Step 2: Mock search results for query
                mock_collection.query.return_value = {
                    'documents': [['This document discusses artificial intelligence, machine learning, and neural networks.']],
                    'distances': [[0.1]],
                    'metadatas': [[{'source': 'AI_Research.pdf'}]]
                }
                
                # Step 3: Create reasoning engine and query
                with patch.object(MultiStepReasoning, '__init__', return_value=None):
                    reasoning = MultiStepReasoning(doc_processor)
                    reasoning.doc_processor = doc_processor
                    reasoning.llm = mock_llm
                    
                    # Mock the step_by_step_reasoning method
                    with patch.object(reasoning, 'step_by_step_reasoning') as mock_reasoning:
                        mock_reasoning.return_value = ReasoningResult(
                            content="Based on the document content, the answer is that AI and machine learning are discussed extensively.",
                            reasoning_steps=["Step 1: Analyze document", "Step 2: Extract AI content"],
                            confidence=0.9,
                            sources=["AI_Research.pdf"],
                            success=True
                        )
                        
                        result = reasoning.step_by_step_reasoning("What does the document say about AI?")
                        
                        # Verify the complete workflow worked
                        assert isinstance(result, ReasoningResult)
                        assert result.success
                        assert "AI" in result.content or "artificial intelligence" in result.content.lower()
                        
                        # Verify all components were called
                        mock_client.get_or_create_collection.assert_called_once()
                        mock_vectorstore.add_documents.assert_called_once()
                        mock_llm.invoke.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 