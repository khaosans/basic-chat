from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
import tempfile
import os
import importlib.util
from PIL import Image
from langchain.docstore.document import Document
from ollama_api import get_available_models
import base64
from io import BytesIO
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import time

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
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Initialize Chroma settings
        self.chroma_settings = Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        )
        
        # Create Chroma client with error handling
        try:
            self.client = chromadb.Client(self.chroma_settings)
            # Test the client connection
            self.client.heartbeat()
        except Exception as e:
            print(f"Warning: Could not initialize Chroma client: {e}")
            # Create a fallback client
            self.client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False,
                is_persistent=True
            ))
        
        # Initialize processed files list
        self.processed_files: List[ProcessedFile] = []
        
        # Text splitter for documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create persistent directory if it doesn't exist
        if not os.path.exists("./chroma_db"):
            os.makedirs("./chroma_db")
        
        # Load existing processed files
        self._load_existing_files()

    def _load_existing_files(self):
        """Load existing processed files from Chroma collections"""
        try:
            collections = self.client.list_collections()
            for collection in collections:
                # Extract file name from collection name
                if collection.name.startswith("collection_"):
                    file_name = collection.name.replace("collection_", "").replace("_", ".")
                    # Try to reconstruct the original file name
                    # This is a simplified approach - in production you might want to store metadata
                    processed_file = ProcessedFile(
                        name=file_name,
                        size=0,  # We don't have the original size
                        type="unknown",
                        collection_name=collection.name
                    )
                    self.processed_files.append(processed_file)
        except Exception as e:
            print(f"Warning: Could not load existing files: {e}")

    def _safe_delete_collection(self, collection_name: str):
        """Safely delete a Chroma collection with error handling"""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Warning: Could not delete collection {collection_name}: {e}")
            return False

    def _safe_get_collection(self, collection_name: str):
        """Safely get a Chroma collection with error handling"""
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not get collection {collection_name}: {e}")
            return None

    def process_file(self, uploaded_file) -> None:
        """Process an uploaded file and store it in Chroma DB"""
        try:
            # Create temporary file to process
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            # Determine file type and load accordingly
            file_type = uploaded_file.type
            documents = []
            if file_type == "application/pdf":
                if is_package_installed("pypdf"):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                else:
                    # Fallback for PDF without pypdf
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                    documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
            elif file_type.startswith("image/"):
                if is_package_installed("unstructured"):
                    from langchain_community.document_loaders import UnstructuredImageLoader
                    loader = UnstructuredImageLoader(file_path)
                    documents = loader.load()
                else:
                    # Fallback for images - just store filename and type
                    documents = [Document(
                        page_content=f"Image file: {uploaded_file.name}",
                        metadata={"source": uploaded_file.name, "type": "image"}
                    )]
            elif file_type.startswith("text/") or file_type == "text/plain":
                # Handle text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
            else:
                # Try to read as text for unknown types
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]
                except:
                    raise ValueError(f"Unsupported file type: {file_type}")

            # Load and split documents
            splits = self.text_splitter.split_documents(documents)

            # Create collection name from file name (sanitized)
            safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('-', '_', '.'))
            collection_name = f"collection_{safe_name}_{int(time.time())}"

            # Create or get collection with error handling
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
            except Exception as e:
                print(f"Warning: Could not create collection: {e}")
                # Fallback to basic collection
                collection = self.client.get_or_create_collection(name=collection_name)

            # Create Chroma vectorstore
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )

            # Add documents to vectorstore with error handling
            try:
                vectorstore.add_documents(splits)
            except Exception as e:
                print(f"Warning: Could not add documents to vectorstore: {e}")
                # Try alternative approach
                try:
                    # Add documents directly to collection
                    texts = [doc.page_content for doc in splits]
                    metadatas = [doc.metadata for doc in splits]
                    collection.add(
                        documents=texts,
                        metadatas=metadatas
                    )
                except Exception as e2:
                    raise Exception(f"Failed to add documents: {e2}")

            # Store file info
            processed_file = ProcessedFile(
                name=uploaded_file.name,
                size=len(uploaded_file.getvalue()),
                type=file_type,
                collection_name=collection_name
            )
            self.processed_files.append(processed_file)

            # Cleanup temporary file
            try:
                os.unlink(file_path)
            except:
                pass  # Ignore cleanup errors

        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def get_processed_files(self) -> List[Dict]:
        """Get list of processed files"""
        return [
            {"name": f.name, "size": f.size, "type": f.type}
            for f in self.processed_files
        ]

    def remove_file(self, file_name: str) -> None:
        """Remove a file from the processor and its collection from Chroma"""
        try:
            # Find the file in processed files
            file_to_remove = next(
                (f for f in self.processed_files if f.name == file_name),
                None
            )
            
            if file_to_remove:
                # Delete collection from Chroma using safe method
                if self._safe_delete_collection(file_to_remove.collection_name):
                    # Remove from processed files list
                    self.processed_files = [
                        f for f in self.processed_files if f.name != file_name
                    ]
                else:
                    raise Exception(f"Failed to delete collection for {file_name}")
            else:
                raise ValueError(f"File {file_name} not found")

        except Exception as e:
            raise Exception(f"Error removing file: {str(e)}")

    def search_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Search across all collections for relevant documents"""
        results = []
        for processed_file in self.processed_files:
            collection = self._safe_get_collection(processed_file.collection_name)
            if collection:
                try:
                    query_results = collection.query(
                        query_texts=[query],
                        n_results=k
                    )
                    if query_results and query_results['documents']:
                        results.extend(query_results['documents'][0])
                except Exception as e:
                    print(f"Warning: Could not query collection {processed_file.collection_name}: {e}")
        return results

    def reset_state(self) -> None:
        """Reset the processor state and clear all collections"""
        try:
            # Delete all collections using safe method
            for processed_file in self.processed_files:
                self._safe_delete_collection(processed_file.collection_name)
            
            # Clear processed files list
            self.processed_files.clear()
            
            # Delete persistence directory with error handling
            if os.path.exists("./chroma_db"):
                try:
                    import shutil
                    shutil.rmtree("./chroma_db")
                    os.makedirs("./chroma_db")
                except Exception as e:
                    print(f"Warning: Could not reset chroma_db directory: {e}")
                    # Try to recreate the client
                    try:
                        self.client = chromadb.Client(self.chroma_settings)
                    except:
                        pass

        except Exception as e:
            raise Exception(f"Error resetting state: {str(e)}")

    def get_relevant_context(self, query, k=3):
        """Get relevant context from documents with enhanced debug logging."""
        if not self.processed_files:
            print("No documents have been processed yet")
            return ""
        
        try:
            print(f"Searching for: {query}")
            print(f"Available documents: {[f.name for f in self.processed_files]}")
            
            # Search across all collections
            all_results = []
            for processed_file in self.processed_files:
                collection = self._safe_get_collection(processed_file.collection_name)
                if collection:
                    try:
                        # Query the collection
                        query_results = collection.query(
                            query_texts=[query],
                            n_results=k,
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        if query_results and query_results['documents']:
                            for i, doc in enumerate(query_results['documents'][0]):
                                distance = query_results['distances'][0][i] if query_results['distances'] else 0
                                metadata = query_results['metadatas'][0][i] if query_results['metadatas'] else {}
                                
                                relevance = round((1 - distance) * 100, 2) if distance <= 1 else 0
                                
                                print(f"Document: {processed_file.name}, Relevance: {relevance}%")
                                print(f"Content length: {len(doc)} characters")
                                
                                context = f"Source: {processed_file.name} (Type: {processed_file.type}, Relevance: {relevance}%)"
                                if processed_file.type.startswith('image'):
                                    context += f"\nImage Analysis:\n{doc}"
                                else:
                                    context += f"\nContent: {doc}"
                                
                                all_results.append((context, relevance))
                    except Exception as e:
                        print(f"Error querying collection {processed_file.collection_name}: {e}")
            
            # Sort by relevance and take top k
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_results = all_results[:k]
            
            if not top_results:
                print("No relevant documents found")
                return ""
            
            print(f"Found {len(top_results)} relevant documents")
            
            # Return formatted results
            return "\n\n---\n\n".join([result[0] for result in top_results])
            
        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Error searching documents: {str(e)}"

    def cleanup(self):
        """Clean up resources and delete stored data"""
        try:
            # Delete all collections
            for processed_file in self.processed_files:
                self._safe_delete_collection(processed_file.collection_name)
            
            # Clear processed files list
            self.processed_files.clear()
            
            # Clear the chroma_db directory
            if os.path.exists("./chroma_db"):
                try:
                    import shutil
                    shutil.rmtree("./chroma_db", ignore_errors=True)
                    os.makedirs("./chroma_db")
                except Exception as e:
                    print(f"Warning: Could not clear chroma_db directory: {e}")
            
            print("Cleanup completed successfully")
            
        except Exception as e:
            print(f"Warning: Failed to cleanup: {str(e)}")
            # Try force cleanup
            try:
                if os.path.exists("./chroma_db"):
                    import shutil
                    shutil.rmtree("./chroma_db", ignore_errors=True)
            except:
                pass

    def process_uploaded_file(self, file_content: bytes, filename: str, file_type: str) -> bool:
        """Process uploaded file content and store it in Chroma DB"""
        try:
            # Create temporary file to process
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_content)
                file_path = tmp_file.name

            # Determine file type and load accordingly
            documents = []
            if file_type == "application/pdf":
                if is_package_installed("pypdf"):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                else:
                    # Fallback for PDF without pypdf
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                    documents = [Document(page_content=content, metadata={"source": filename})]
            elif file_type.startswith("image/"):
                # Handle image files
                if is_package_installed("unstructured"):
                    from langchain_community.document_loaders import UnstructuredImageLoader
                    loader = UnstructuredImageLoader(file_path)
                    documents = loader.load()
                else:
                    # Fallback for images - just store filename and type
                    documents = [Document(
                        page_content=f"Image file: {filename}",
                        metadata={"source": filename, "type": "image"}
                    )]
            elif file_type.startswith("text/") or file_type == "text/plain":
                # Handle text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                documents = [Document(page_content=content, metadata={"source": filename})]
            else:
                # Try to read as text for unknown types
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    documents = [Document(page_content=content, metadata={"source": filename})]
                except:
                    raise ValueError(f"Unsupported file type: {file_type}")

            # Split documents
            splits = self.text_splitter.split_documents(documents)

            # Create collection name from file name (sanitized)
            collection_name = self._generate_collection_name(filename)

            # Create or get collection with error handling
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
            except Exception as e:
                print(f"Warning: Could not create collection: {e}")
                # Fallback to basic collection
                collection = self.client.get_or_create_collection(name=collection_name)

            # Create Chroma vectorstore
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )

            # Add documents to vectorstore with error handling
            try:
                vectorstore.add_documents(splits)
            except Exception as e:
                print(f"Warning: Could not add documents to vectorstore: {e}")
                # Try alternative approach
                try:
                    # Add documents directly to collection
                    texts = [doc.page_content for doc in splits]
                    metadatas = [doc.metadata for doc in splits]
                    ids = [f"{collection_name}_{i}" for i in range(len(texts))]
                    collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e2:
                    raise Exception(f"Failed to add documents: {e2}")

            # Store file info
            processed_file = ProcessedFile(
                name=filename,
                size=len(file_content),
                type=file_type,
                collection_name=collection_name
            )
            self.processed_files.append(processed_file)

            # Cleanup temporary file
            try:
                os.unlink(file_path)
            except:
                pass  # Ignore cleanup errors

            return True

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False

    def _generate_collection_name(self, filename: str) -> str:
        """Generate a safe collection name from filename"""
        # Remove extension and sanitize
        base_name = os.path.splitext(filename)[0]
        safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
        # Ensure it starts with a letter and add timestamp for uniqueness
        collection_name = f"collection_{safe_name}_{int(time.time())}"
        return collection_name

    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search across all collections for relevant documents and return Document objects"""
        results = []
        for processed_file in self.processed_files:
            collection = self._safe_get_collection(processed_file.collection_name)
            if collection:
                try:
                    query_results = collection.query(
                        query_texts=[query],
                        n_results=k,
                        include=['documents', 'metadatas']
                    )
                    if query_results and query_results['documents']:
                        for i, doc_content in enumerate(query_results['documents'][0]):
                            metadata = query_results['metadatas'][0][i] if query_results['metadatas'] else {}
                            doc = Document(page_content=doc_content, metadata=metadata)
                            results.append(doc)
                except Exception as e:
                    print(f"Warning: Could not query collection {processed_file.collection_name}: {e}")
        return results[:k]  # Return top k results