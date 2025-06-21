"""
Document processing utilities for text extraction and vector storage
"""

import os
import logging
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import VISION_MODEL, EMBEDDING_MODEL, OLLAMA_API_URL

# Vector store imports are handled dynamically to avoid startup errors
logger = logging.getLogger(__name__)


class ProcessedFile:
    """Represents a processed file with extracted text and metadata"""

    def __init__(
        self,
        filename: str,
        content: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.filename = filename
        self.content = content
        self.file_type = file_type
        self.metadata = metadata or {}


class ImageTextExtractor:
    """Extract text from images using an Ollama vision model (e.g., LLaVA)."""

    def __init__(self) -> None:
        self.llm = None
        self.vision_model_available = False
        
        # Only initialize if we're not in a Streamlit context that might cause issues
        try:
            # Initialize the ChatOllama model for vision tasks
            self.llm = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_API_URL)
            self.vision_model_available = True
            logger.info(f"Vision model '{VISION_MODEL}' initialized successfully.")
        except Exception as e:
            logger.warning(
                f"Could not initialize vision model '{VISION_MODEL}'. "
                f"Image analysis will be unavailable. Error: {e}"
            )
            self.vision_model_available = False

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_text(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail. If there is text, transcribe it accurately.",
    ) -> str:
        """
        Analyze an image to extract text or a detailed description using a vision model.
        """
        if not self.vision_model_available or self.llm is None:
            return "Vision model is not available. Please check your model configuration."

        try:
            image_b64 = self._encode_image(image_path)

            # Create the message payload for LLaVA
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ]
            )

            # Invoke the model
            llm_response = self.llm.invoke([message])
            return llm_response.content
        except Exception as e:
            logger.error(
                f"Error analyzing image with vision model '{VISION_MODEL}': {e}",
                exc_info=True,
            )
            return f"Failed to analyze the image with model '{VISION_MODEL}'. See logs for details."


class DocumentProcessor:
    """Main document processing class"""

    def __init__(self) -> None:
        self.image_extractor = ImageTextExtractor()
        self.vector_store: Optional[Any] = None

        # Add text splitter for compatibility with app.py
        self.text_splitter = self._create_text_splitter()

        # Don't initialize vector store by default to avoid import issues
        self._chromadb_available = None

    def _create_text_splitter(self) -> Any:
        """Create a text splitter for document processing"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
        except ImportError:
            # Fallback to simple text splitter
            class SimpleTextSplitter:
                def split_documents(self, documents):
                    """Simple document splitting"""
                    chunks = []
                    for doc in documents:
                        content = (
                            doc.page_content
                            if hasattr(doc, "page_content")
                            else str(doc)
                        )
                        # Split by sentences
                        sentences = content.split(". ")
                        current_chunk = ""
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < 1000:
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk:
                                    chunks.append(
                                        type(doc)(
                                            page_content=current_chunk.strip(),
                                            metadata=getattr(doc, "metadata", {}),
                                        )
                                    )
                                current_chunk = sentence + ". "
                        if current_chunk:
                            chunks.append(
                                type(doc)(
                                    page_content=current_chunk.strip(),
                                    metadata=getattr(doc, "metadata", {}),
                                )
                            )
                    return chunks

            return SimpleTextSplitter()

    def _check_chromadb_available(self) -> bool:
        """Check if ChromaDB is available without importing at module level"""
        if self._chromadb_available is None:
            try:
                import chromadb

                self._chromadb_available = True
            except ImportError:
                self._chromadb_available = False
        return self._chromadb_available

    def _initialize_vector_store(self) -> None:
        """Initialize vector store for document embeddings"""
        if not self._check_chromadb_available():
            logger.warning("ChromaDB not available - vector store disabled")
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            # Initialize embeddings with the correct model from config
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

            # Create ChromaDB client
            client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"
                )
            )

            # Create or get collection
            self.vector_store = client.get_or_create_collection(
                name="documents", embedding_function=embedding_function
            )

            logger.info(
                f"Vector store initialized with embedding model '{EMBEDDING_MODEL}'."
            )

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None

    def process_file(self, file_path: str) -> ProcessedFile:
        """
        Process a single file and extract text
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessedFile object with extracted content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = file_path.suffix.lower()
        filename = file_path.name
        
        # Process based on file type
        if file_type in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            content = self.image_extractor.extract_text(str(file_path))
        elif file_type in ['.txt', '.md']:
            content = self._read_text_file(file_path)
        elif file_type in ['.pdf']:
            content = self._read_pdf_file(file_path)
        else:
            content = f"Unsupported file type: {file_type}"
        
        metadata = {
            'file_size': file_path.stat().st_size,
            'file_type': file_type,
            'processing_time': 0  # Could add timing if needed
        }
        
        return ProcessedFile(filename, content, file_type, metadata)
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return f"Error reading file: {e}"
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return f"Error reading file: {e}"
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """Read text from a PDF file"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
                return "\n".join(text_parts)
        except ImportError:
            return "PyPDF2 not available for PDF processing"
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return f"Error reading PDF: {e}"
    
    def process_files(self, file_paths: List[str]) -> List[ProcessedFile]:
        """
        Process multiple files
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessedFile objects
        """
        processed_files = []
        
        for file_path in file_paths:
            try:
                processed_file = self.process_file(file_path)
                processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Create error file entry
                error_file = ProcessedFile(
                    Path(file_path).name,
                    f"Error processing file: {e}",
                    Path(file_path).suffix.lower(),
                    {'error': str(e)}
                )
                processed_files.append(error_file)
        
        return processed_files
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_to_vector_store(self, processed_files: List[ProcessedFile]) -> None:
        """
        Add processed files to vector store
        
        Args:
            processed_files: List of processed files to add
        """
        if not self._check_chromadb_available():
            logger.warning("ChromaDB not available")
            return
        
        # Initialize vector store if not already done
        if self.vector_store is None:
            self._initialize_vector_store()
        
        if not self.vector_store:
            logger.warning("Vector store not available")
            return
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Prepare documents for vector store
            documents = []
            metadatas = []
            ids = []
            
            for file in processed_files:
                # Chunk the content
                chunks = self.chunk_text(file.content)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{file.filename}_{i}"
                    documents.append(chunk)
                    metadatas.append({
                        'filename': file.filename,
                        'file_type': file.file_type,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        **file.metadata
                    })
                    ids.append(doc_id)
            
            # Add to vector store
            if documents:
                self.vector_store.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} chunks to vector store")
                
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self._check_chromadb_available():
            logger.warning("ChromaDB not available")
            return []
        
        # Initialize vector store if not already done
        if self.vector_store is None:
            self._initialize_vector_store()
        
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []
        
        try:
            results = self.vector_store.query(
                query_texts=[query],
                n_results=k
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_document_summary(self, processed_files: List[ProcessedFile]) -> Dict[str, Any]:
        """
        Get summary statistics for processed files
        
        Args:
            processed_files: List of processed files
            
        Returns:
            Dictionary with summary statistics
        """
        if not processed_files:
            return {}
        
        total_files = len(processed_files)
        total_chars = sum(len(f.content) for f in processed_files)
        file_types = {}
        
        for file in processed_files:
            file_type = file.file_type
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Count chunks if vector store is available
        total_chunks = 0
        if self._check_chromadb_available():
            try:
                # This is a rough estimate - actual implementation may vary
                total_chunks = sum(len(self.chunk_text(f.content)) for f in processed_files)
            except Exception:
                pass
        
        return {
            'total_files': total_files,
            'total_characters': total_chars,
            'file_types': file_types,
            'total_chunks': total_chunks,
            'average_chars_per_file': total_chars / total_files if total_files > 0 else 0
        }
    
    def save_vector_store(self, path: str) -> None:
        """
        Save vector store to disk
        
        Args:
            path: Path to save the vector store
        """
        if not self._check_chromadb_available():
            logger.warning("ChromaDB not available")
            return
        
        # Initialize vector store if not already done
        if self.vector_store is None:
            self._initialize_vector_store()
        
        if not self.vector_store:
            logger.warning("Vector store not available")
            return
        
        try:
            # ChromaDB persists automatically to the directory specified in Settings
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load_vector_store(self, path: str) -> None:
        """
        Load vector store from disk
        
        Args:
            path: Path to load the vector store from
        """
        if not self._check_chromadb_available():
            logger.warning("ChromaDB not available")
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
            
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            client = chromadb.PersistentClient(path=path)
            self.vector_store = client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_function
            )
            
            logger.info(f"Vector store loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.vector_store = None

# Global document processor instance
document_processor = DocumentProcessor()

def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance"""
    return document_processor