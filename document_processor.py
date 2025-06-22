from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import tempfile
import os
import importlib.util
from PIL import Image
from io import BytesIO
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import re
import base64
import logging
import traceback
import uuid
import atexit
import signal
import weakref

from config import EMBEDDING_MODEL, VISION_MODEL

# Configure logging for document processor
logger = logging.getLogger(__name__)

# Global registry for cleanup tracking
_chroma_instances = weakref.WeakSet()

def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

@dataclass
class ProcessedFile:
    name: str
    size: int
    type: str
    collection_name: str

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with Chroma DB"""
        logger.info("Initializing DocumentProcessor")
        
        # Register this instance for cleanup
        _chroma_instances.add(self)
        
        try:
            logger.info(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://localhost:11434"
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
        
        try:
            logger.info(f"Initializing vision model: {VISION_MODEL}")
            self.vision_model = ChatOllama(model=VISION_MODEL)
            logger.info("Vision model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise
        
        # Initialize Chroma settings with unique persistent directory
        try:
            logger.info("Initializing ChromaDB settings")
            # Use a unique directory for this instance to avoid conflicts
            unique_id = str(uuid.uuid4())[:8]
            persist_dir = f"./chroma_db_{unique_id}"
            
            # Create persistent directory first
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
                logger.info(f"Created persistent directory: {persist_dir}")
            
            # Use PersistentClient instead of Client to avoid ephemeral conflicts
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.persist_directory = persist_dir
            logger.info(f"ChromaDB persistent client initialized successfully with directory: {persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        # Initialize processed files list
        self.processed_files: List[ProcessedFile] = []
        
        # Initialize text splitter
        try:
            logger.info("Initializing text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            logger.info("Text splitter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            raise
        
        logger.info("DocumentProcessor initialized successfully")

    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected"""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during automatic cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()

    def _get_image_description(self, uploaded_file) -> str:
        """Generates a detailed description for an image file using a vision model."""
        logger.info(f"Processing image: {uploaded_file.name}")
        
        try:
            image_bytes = uploaded_file.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            logger.info(f"Image encoded to base64, size: {len(image_base64)} characters")
            
            msg = self.vision_model.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in detail, paying close attention to any text, user interface elements, and the overall context. Be as descriptive as possible."},
                            {
                                "type": "image_url",
                                "image_url": f"data:{uploaded_file.type};base64,{image_base64}",
                            },
                        ]
                    )
                ]
            )
            
            description = msg.content
            logger.info(f"Image description generated, length: {len(description)} characters")
            return description
            
        except Exception as e:
            logger.error(f"Failed to process image {uploaded_file.name}: {e}")
            logger.error(f"Image processing traceback: {traceback.format_exc()}")
            raise

    def process_file(self, uploaded_file) -> None:
        """Process an uploaded file and store it in Chroma DB"""
        logger.info(f"Starting file processing: {uploaded_file.name} (type: {uploaded_file.type})")
        
        try:
            documents = []
            file_path = None
            
            # Use a temporary file for loaders that require a path
            if not uploaded_file.type.startswith("image/"):
                logger.info("Creating temporary file for non-image document")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name
                logger.info(f"Temporary file created: {file_path}")

            # Determine file type and load accordingly
            file_type = uploaded_file.type
            logger.info(f"Processing file type: {file_type}")
            
            if file_type == "application/pdf":
                logger.info("Loading PDF document")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                logger.info(f"PDF loaded successfully, {len(documents)} pages")
                
            elif file_type.startswith("image/"):
                logger.info("Processing image document")
                description = self._get_image_description(uploaded_file)
                documents = [Document(page_content=description, metadata={"source": uploaded_file.name})]
                logger.info("Image document processed successfully")
                
            elif file_type in ["text/plain", "text/markdown"]:
                logger.info("Loading text document")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
                logger.info(f"Text document loaded successfully, {len(content)} characters")
                
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Clean up temporary file
            if file_path:
                try:
                    os.unlink(file_path)
                    logger.info(f"Temporary file cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

            if not documents:
                error_msg = "Could not extract any content from the document."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create collection name from file name (sanitized for ChromaDB)
            sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', uploaded_file.name)
            sanitized_name = sanitized_name.strip('._-')  # Remove leading/trailing special chars
            if not sanitized_name:
                sanitized_name = "document"
            collection_name = f"collection_{sanitized_name}"
            logger.info(f"Collection name: {collection_name}")

            # Create or get collection
            try:
                logger.info(f"Creating/getting ChromaDB collection: {collection_name}")
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_functions.OllamaEmbeddingFunction(
                        model_name=EMBEDDING_MODEL
                    )
                )
                logger.info(f"Collection {collection_name} ready")
            except Exception as e:
                logger.error(f"Failed to create/get collection {collection_name}: {e}")
                raise

            # Create Chroma vectorstore
            try:
                logger.info("Creating Chroma vectorstore")
                vectorstore = Chroma(
                    client=self.client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info("Chroma vectorstore created successfully")
            except Exception as e:
                logger.error(f"Failed to create Chroma vectorstore: {e}")
                raise

            # Load and split documents
            try:
                logger.info("Splitting documents")
                if file_type == "application/pdf" or file_type.startswith("image/"):
                    splits = self.text_splitter.split_documents(documents)
                else:
                    splits = self.text_splitter.split_documents(documents)
                logger.info(f"Documents split into {len(splits)} chunks")
            except Exception as e:
                logger.error(f"Failed to split documents: {e}")
                raise

            # Add documents to vectorstore
            try:
                logger.info("Adding documents to vectorstore")
                vectorstore.add_documents(splits)
                logger.info("Documents added to vectorstore successfully")
            except Exception as e:
                logger.error(f"Failed to add documents to vectorstore: {e}")
                raise

            # Store file info
            processed_file = ProcessedFile(
                name=uploaded_file.name,
                size=len(uploaded_file.getvalue()),
                type=file_type,
                collection_name=collection_name
            )
            self.processed_files.append(processed_file)
            logger.info(f"File processing completed successfully: {uploaded_file.name}")

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up temp file on error
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    logger.info(f"Cleaned up temporary file on error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file on error: {cleanup_error}")
            
            raise Exception(f"Error processing file: {str(e)}")

    def get_processed_files(self) -> List[Dict]:
        """Get list of processed files"""
        logger.debug(f"Getting processed files, count: {len(self.processed_files)}")
        return [
            {"name": f.name, "size": f.size, "type": f.type}
            for f in self.processed_files
        ]

    def get_available_documents(self) -> List[str]:
        """Get a list of the names of available documents."""
        return [f.name for f in self.processed_files]

    def remove_file(self, file_name: str) -> None:
        """Remove a file from the processor and its collection from Chroma"""
        logger.info(f"Removing file: {file_name}")
        
        try:
            # Find the file in processed files
            file_to_remove = next(
                (f for f in self.processed_files if f.name == file_name),
                None
            )
            
            if file_to_remove:
                logger.info(f"Found file to remove: {file_to_remove.name}")
                
                # Delete collection from Chroma
                try:
                    logger.info(f"Deleting ChromaDB collection: {file_to_remove.collection_name}")
                    self.client.delete_collection(file_to_remove.collection_name)
                    logger.info(f"Collection {file_to_remove.collection_name} deleted successfully")
                except Exception as e:
                    logger.error(f"Failed to delete collection {file_to_remove.collection_name}: {e}")
                    raise
                
                # Remove from processed files list
                self.processed_files = [
                    f for f in self.processed_files if f.name != file_name
                ]
                logger.info(f"File {file_name} removed from processed files list")
            else:
                error_msg = f"File {file_name} not found"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"Error removing file {file_name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Error removing file: {str(e)}")

    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search across all collections for relevant documents."""
        logger.info(f"Searching documents for query: '{query}' (k={k})")
        
        results = []
        for processed_file in self.processed_files:
            try:
                logger.debug(f"Searching collection: {processed_file.collection_name}")
                collection = self.client.get_collection(processed_file.collection_name)
                query_results = collection.query(
                    query_texts=[query],
                    n_results=k,
                    include=["metadatas", "documents"]
                )
                
                for i, doc_content in enumerate(query_results['documents'][0]):
                    metadata = query_results['metadatas'][0][i]
                    results.append(Document(page_content=doc_content, metadata=metadata))

                logger.debug(f"Found {len(query_results['documents'][0])} results in {processed_file.collection_name}")
            except Exception as e:
                logger.error(f"Error searching collection {processed_file.collection_name}: {e}")
                continue
        
        logger.info(f"Search completed, total results: {len(results)}")
        return results

    def reset_state(self) -> None:
        """Reset the processor state and clear all collections"""
        logger.info("Resetting document processor state")
        
        try:
            # Delete all collections
            for processed_file in self.processed_files:
                try:
                    logger.info(f"Deleting collection: {processed_file.collection_name}")
                    self.client.delete_collection(processed_file.collection_name)
                except Exception as e:
                    logger.warning(f"Failed to delete collection {processed_file.collection_name}: {e}")
            
            # Clear processed files list
            self.processed_files.clear()
            logger.info("Processed files list cleared")
            
            # Delete persistence directory
            if os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                    os.makedirs(self.persist_directory)
                    logger.info("ChromaDB persistence directory reset")
                except Exception as e:
                    logger.error(f"Failed to reset ChromaDB directory: {e}")
                    raise

        except Exception as e:
            logger.error(f"Error resetting state: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f"Error resetting state: {str(e)}")

    def _clear_vector_store(self):
        """Safely clear the vector store and its directory"""
        logger.info("Clearing vector store")
        
        try:
            if hasattr(self, 'persist_directory') and os.path.exists(self.persist_directory):
                # Force remove readonly files
                def handle_error(func, path, exc_info):
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                shutil.rmtree(self.persist_directory, onerror=handle_error)
                logger.info("Vector store cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear vector store: {str(e)}")

    def get_relevant_context(self, query, k=3):
        """Get relevant context from documents using ChromaDB collections."""
        logger.info(f"Getting relevant context for query: '{query}' (k={k})")
        
        if not self.processed_files:
            logger.info("No documents have been processed yet")
            return ""
        
        try:
            logger.info(f"Searching for: {query}")
            logger.info(f"Available documents: {[f.name for f in self.processed_files]}")
            
            all_results = []
            
            # Search across all collections
            for processed_file in self.processed_files:
                try:
                    logger.debug(f"Searching collection: {processed_file.collection_name}")
                    collection = self.client.get_collection(processed_file.collection_name)
                    query_results = collection.query(
                        query_texts=[query],
                        n_results=k,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if query_results['documents'] and query_results['documents'][0]:
                        for i, doc in enumerate(query_results['documents'][0]):
                            distance = query_results['distances'][0][i] if query_results['distances'] else 0
                            relevance = round((1 - distance) * 100, 2)
                            
                            logger.info(f"Document: {processed_file.name}, Relevance: {relevance}%")
                            logger.debug(f"Content length: {len(doc)} characters")
                            
                            context = f"Source: {processed_file.name} (Type: {processed_file.type}, Relevance: {relevance}%)"
                            if processed_file.type.startswith('image'):
                                context += f"\nImage Analysis:\n{doc}"
                            else:
                                context += f"\nContent: {doc}"
                            all_results.append((relevance, context))
                            
                except Exception as e:
                    logger.error(f"Error searching collection {processed_file.collection_name}: {str(e)}")
                    continue
            
            if not all_results:
                logger.info("No relevant documents found")
                return ""
            
            # Sort by relevance and take top k
            all_results.sort(key=lambda x: x[0], reverse=True)
            top_results = all_results[:k]
            
            logger.info(f"Found {len(top_results)} relevant document sections")
            
            return "\n\n---\n\n".join([result[1] for result in top_results])
            
        except Exception as e:
            logger.error(f"Error during document search: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error searching documents: {str(e)}"

    def cleanup(self):
        """Clean up resources and delete stored data"""
        logger.info("Cleaning up document processor resources")
        
        try:
            # Delete all collections
            for processed_file in self.processed_files:
                try:
                    logger.info(f"Deleting collection: {processed_file.collection_name}")
                    self.client.delete_collection(processed_file.collection_name)
                except Exception as e:
                    logger.warning(f"Failed to delete collection {processed_file.collection_name}: {str(e)}")
            
            self._clear_vector_store()
            self.processed_files = []
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Try force cleanup
            try:
                if hasattr(self, 'persist_directory') and os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory, ignore_errors=True)
                    logger.info("Force cleanup completed")
            except Exception as force_error:
                logger.error(f"Force cleanup failed: {force_error}")

    @staticmethod
    def cleanup_all_chroma_directories():
        """Clean up all ChromaDB directories created by this processor"""
        logger.info("Cleaning up all ChromaDB directories")
        
        try:
            import glob
            # Find all chroma_db_* directories
            chroma_dirs = glob.glob("./chroma_db_*")
            
            for chroma_dir in chroma_dirs:
                try:
                    if os.path.exists(chroma_dir):
                        shutil.rmtree(chroma_dir, ignore_errors=True)
                        logger.info(f"Cleaned up directory: {chroma_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up directory {chroma_dir}: {e}")
            
            # Also clean up the original chroma_db directory if it exists
            if os.path.exists("./chroma_db"):
                try:
                    shutil.rmtree("./chroma_db", ignore_errors=True)
                    logger.info("Cleaned up original chroma_db directory")
                except Exception as e:
                    logger.warning(f"Failed to clean up original chroma_db directory: {e}")
            
            # Clean up test directories
            test_dirs = ["./test_chroma_db", "./temp_chroma_db"]
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    try:
                        shutil.rmtree(test_dir, ignore_errors=True)
                        logger.info(f"Cleaned up test directory: {test_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up test directory {test_dir}: {e}")
                    
            logger.info("All ChromaDB directories cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to clean up ChromaDB directories: {e}")

    @staticmethod
    def cleanup_all_instances():
        """Clean up all DocumentProcessor instances"""
        logger.info("Cleaning up all DocumentProcessor instances")
        
        try:
            # Clean up all registered instances
            for instance in list(_chroma_instances):
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup instance: {e}")
            
            # Clear the registry
            _chroma_instances.clear()
            
            # Clean up all directories
            DocumentProcessor.cleanup_all_chroma_directories()
            
            logger.info("All DocumentProcessor instances cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup all instances: {e}")

    @staticmethod
    def get_chroma_directories():
        """Get a list of all ChromaDB directories"""
        import glob
        chroma_dirs = glob.glob("./chroma_db*")
        return chroma_dirs

    @staticmethod
    def cleanup_old_directories(max_age_hours=24):
        """Clean up ChromaDB directories older than specified age"""
        import time
        import glob
        
        logger.info(f"Cleaning up ChromaDB directories older than {max_age_hours} hours")
        
        try:
            chroma_dirs = glob.glob("./chroma_db*")
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for chroma_dir in chroma_dirs:
                try:
                    if os.path.exists(chroma_dir):
                        dir_age = current_time - os.path.getmtime(chroma_dir)
                        if dir_age > max_age_seconds:
                            shutil.rmtree(chroma_dir, ignore_errors=True)
                            logger.info(f"Cleaned up old directory: {chroma_dir} (age: {dir_age/3600:.1f}h)")
                except Exception as e:
                    logger.warning(f"Failed to check/clean directory {chroma_dir}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old directories: {e}")

# Global cleanup functions
def cleanup_on_exit():
    """Cleanup function to be called on application exit"""
    logger.info("Application exit cleanup initiated")
    DocumentProcessor.cleanup_all_instances()

def cleanup_on_signal(signum, frame):
    """Cleanup function to be called on signal"""
    logger.info(f"Signal {signum} received, cleaning up")
    DocumentProcessor.cleanup_all_instances()
    exit(0)

# Register cleanup handlers
atexit.register(cleanup_on_exit)

# Only register signal handlers if we're in the main thread
try:
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, cleanup_on_signal)
        signal.signal(signal.SIGTERM, cleanup_on_signal)
    else:
        logger.info("Not in main thread, skipping signal handler registration")
except Exception as e:
    logger.warning(f"Could not register signal handlers: {e}")
    # Continue without signal handlers - this is fine for Streamlit