from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
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
from typing import Dict, Any

def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

class DocumentProcessor:
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        """Reset all internal state and storage"""
        self._clear_vector_store()
        self.processed_files = {}
        try:
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,  # Use fixed embedding model
                base_url="http://localhost:11434"
            )
            
            os.makedirs("./chroma_db", exist_ok=True)
            
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
                collection_name="document_store"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize vector store: {str(e)}")
            self.vector_store = None
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def _clear_vector_store(self):
        """Safely clear the vector store and its directory"""
        try:
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                try:
                    self.vector_store.delete_collection()
                except:
                    pass
            if os.path.exists("./chroma_db"):
                # Force remove readonly files
                def handle_error(func, path, exc_info):
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                shutil.rmtree("./chroma_db", onerror=handle_error)
        except Exception as e:
            print(f"Warning: Could not clear vector store: {str(e)}")

    def _load_processed_files(self):
        """Load processed files from vector store metadata"""
        if self.vector_store is None:
            return {}
        
        processed_files = {}
        try:
            # Get all documents from the collection
            docs = self.vector_store.get()
            if docs and docs['metadatas']:
                for metadata in docs['metadatas']:
                    if metadata and 'source' in metadata:
                        processed_files[metadata['source']] = {
                            'name': metadata['source'],
                            'type': metadata.get('type', 'document'),
                            'size': metadata.get('size', 0)
                        }
        except Exception as e:
            print(f"Warning: Failed to load processed files: {str(e)}")
        return processed_files

    def process_file(self, uploaded_file):
        """Process an uploaded file with improved duplicate handling"""
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if exact file already exists
        if file_id in self.processed_files:
            print(f"Identical file '{uploaded_file.name}' already processed")
            return True

        # Remove any older version of the file if it exists
        if uploaded_file.name in self.processed_files:
            print(f"Removing older version of '{uploaded_file.name}'")
            self.remove_file(uploaded_file.name)

        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            # Process different file types
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                if not is_package_installed('PIL'):
                    raise ImportError("The 'pillow' package is required for processing images. Please install it with 'pip install pillow'.")
                
                # Use fixed image model
                llava = ChatOllama(
                    model=IMAGE_MODEL,  # Use fixed image model
                    base_url="http://localhost:11434"
                )
                
                # Use llava model for image understanding
                image = Image.open(file_path)
                
                # Convert image to base64
                buffered = BytesIO()
                image.save(buffered, format=image.format)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Get detailed image analysis from llava with multiple prompts
                prompts = [
                    "Describe this image in detail, focusing on the main subjects and their arrangement.",
                    "What text or written content can you see in this image? Please be specific.",
                    "What objects, shapes, or notable visual elements can you identify?",
                    "Describe the colors, lighting, and overall visual composition of this image."
                ]
                
                analysis_results = []
                for i, prompt in enumerate(prompts):
                    try:
                        print(f"Processing image analysis step {i+1}/4: {prompt.split('.')[0]}...")
                        response_stream = llava.stream(f"<image>{img_str}</image>{prompt}")
                        result = ""
                        for chunk in response_stream:
                            if hasattr(chunk, 'content'):
                                result += chunk.content
                                print(f"\rProgress: {len(result)} characters processed", end="")
                        analysis_results.append(result)
                        print("\nStep completed successfully.")
                    except Exception as e:
                        print(f"\nWarning: Failed to analyze image with prompt '{prompt}': {str(e)}")
                        analysis_results.append("Analysis failed for this aspect")
                
                # Combine all analysis results
                full_analysis = {
                    "general_description": analysis_results[0],
                    "text_content": analysis_results[1],
                    "objects_and_elements": analysis_results[2],
                    "visual_composition": analysis_results[3]
                }
                
                # Create a Document object with comprehensive image content and metadata
                documents = [Document(
                    page_content=f"Image: {uploaded_file.name}\n" +
                                f"Size: {image.size}\n" +
                                f"Mode: {image.mode}\n" +
                                f"General Description: {full_analysis['general_description']}\n" +
                                f"Text Content: {full_analysis['text_content']}\n" +
                                f"Objects and Elements: {full_analysis['objects_and_elements']}\n" +
                                f"Visual Composition: {full_analysis['visual_composition']}",
                    metadata={
                        "source": uploaded_file.name,
                        "type": "image",
                        "description": full_analysis['general_description'],
                        "text_content": full_analysis['text_content'],
                        "objects": full_analysis['objects_and_elements'],
                        "composition": full_analysis['visual_composition'],
                        "size": uploaded_file.size
                    }
                )]
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")

            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)

            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db",
                    collection_name="document_store"
                )
            else:
                self.vector_store.add_documents(splits)
                self.vector_store.persist()
            
            # Store file information
            self.processed_files[uploaded_file.name] = {
                'name': uploaded_file.name,
                'type': uploaded_file.type,
                'size': uploaded_file.size
            }
            return True
        except Exception as e:
            raise e
        finally:
            # Clean up temporary file
            os.unlink(file_path)

    def get_relevant_context(self, query, k=3):
        """Get relevant context from documents with enhanced debug logging."""
        if self.vector_store is None:
            print("Warning: Vector store is not initialized")
            return ""
        
        if not self.processed_files:
            print("No documents have been processed yet")
            return ""
        
        try:
            print(f"Searching for: {query}")
            print(f"Available documents: {list(self.processed_files.keys())}")
            
            # Search for relevant documents with distance score
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not docs_and_scores:
                print("No relevant documents found")
                return ""
            
            print(f"Found {len(docs_and_scores)} relevant documents")
            
            # Format documents with metadata and relevance info
            formatted_contexts = []
            for doc, score in docs_and_scores:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'document')
                relevance = round((1 - score) * 100, 2)
                
                print(f"Document: {source}, Type: {doc_type}, Score: {relevance}%")
                print(f"Content length: {len(doc.page_content)} characters")
                
                context = f"Source: {source} (Type: {doc_type}, Relevance: {relevance}%)"
                if doc_type == 'image':
                    context += f"\nImage Analysis:\n{doc.page_content}"
                else:
                    context += f"\nContent: {doc.page_content}"
                formatted_contexts.append(context)
            
            return "\n\n---\n\n".join(formatted_contexts)
            
        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return f"Error searching documents: {str(e)}"

    def remove_file(self, filename: str) -> bool:
        """Remove a file and ensure state consistency"""
        if filename in self.processed_files:
            try:
                # Remove from vector store
                if self.vector_store is not None:
                    docs = self.vector_store.get(where={"source": filename})
                    if docs and docs['ids']:
                        try:
                            self.vector_store.delete(ids=docs['ids'])
                            self.vector_store.persist()
                        except Exception as e:
                            print(f"Error removing documents: {str(e)}")
                            # If vector store is corrupted, reset everything
                            self.reset_state()
                            return True
                
                # Remove from processed files
                del self.processed_files[filename]
                return True
            except Exception as e:
                print(f"Warning: Failed to remove file: {str(e)}")
                # Attempt recovery by resetting state
                self.reset_state()
                return True
        return False

    def get_processed_files(self):
        return list(self.processed_files.values())

    def cleanup(self):
        """Clean up resources and delete stored data"""
        try:
            if self.vector_store is not None:
                try:
                    self.vector_store.delete_collection()
                except Exception as e:
                    print(f"Warning: Failed to delete collection: {str(e)}")
                finally:
                    self.vector_store = None
            
            self._clear_vector_store()
            self.processed_files = {}
            print("Cleanup completed successfully")
            
        except Exception as e:
            print(f"Warning: Failed to cleanup: {str(e)}")
            # Try force cleanup
            try:
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db", ignore_errors=True)
            except:
                pass