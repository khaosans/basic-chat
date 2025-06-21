import streamlit as st
# Must be first Streamlit command
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import time
import requests
import json
import datetime
import pytz
import asyncio
import logging
import traceback
from typing import Optional, Dict, List, Union, Any, AsyncGenerator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain.docstore.document import Document
import tempfile
from gtts import gTTS
import hashlib
import base64
from PIL import Image
import io
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import httpx
from pathlib import Path

# Adjust the path to import from parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our new reasoning engine
from reasoning_engine import (
    ReasoningEngine,
    ReasoningResult
)

# Import new async components
from config import config
from utils.async_ollama import AsyncOllamaClient, get_async_client
from utils.caching import cache_manager

# Import our modules
from reasoning_engine import ReasoningEngine, ReasoningResult
from utils.async_ollama import AsyncOllamaClient
from utils.enhanced_tools import EnhancedCalculator, EnhancedTimeTools
from web_search import WebSearch, search_web
from document_processor import DocumentProcessor, ProcessedFile

# Load configuration from config.py
from config import (
    CHAT_MODEL,
    REASONING_MODEL,
    VISION_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_API_URL,
    TIMEZONE,
)

load_dotenv(".env.local")  # Load environment variables from .env.local

# Use Ollama model instead of Hugging Face
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

# Add a system prompt definition
SYSTEM_PROMPT = """
You are a helpful and knowledgeable AI assistant with advanced reasoning capabilities. You can:
1. Answer questions about a wide range of topics using logical reasoning
2. Summarize documents that have been uploaded with detailed analysis
3. Have natural, friendly conversations with enhanced understanding
4. Break down complex problems into manageable steps
5. Provide well-reasoned explanations for your answers

Please be concise, accurate, and helpful in your responses. 
If you don't know something, just say so instead of making up information.
Always show your reasoning process when appropriate.
"""

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

class OllamaChat:
    """Enhanced Ollama chat with async support and caching"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CHAT_MODEL
        self.api_url = f"{OLLAMA_API_URL}/api/generate"
        self.system_prompt = SYSTEM_PROMPT
        
        # Initialize async chat client
        self.async_chat: Optional[AsyncOllamaClient] = None  # Will be initialized when needed
        
        # Fallback to sync implementation if needed
        self._use_sync_fallback = False

    def query(self, payload: Dict) -> Optional[str]:
        """Query the Ollama API with async support and fallback"""
        if not self._use_sync_fallback:
            try:
                # Try async implementation with proper event loop handling
                import asyncio
                try:
                    # Check if there's already a running event loop
                    asyncio.get_running_loop()
                    # If we're in a running loop, use sync fallback
                    self._use_sync_fallback = True
                except RuntimeError:
                    # No running event loop, safe to use asyncio.run
                    return asyncio.run(self._query_async(payload))
            except Exception as e:
                st.warning(f"Async query failed, falling back to sync: {e}")
                self._use_sync_fallback = True
        
        # Fallback to original sync implementation
        return self._query_sync(payload)
    
    async def _query_async(self, payload: Dict) -> Optional[str]:
        """Async query implementation"""
        try:
            if self.async_chat is None:
                self.async_chat = await get_async_client()
            return await self.async_chat.generate(payload.get('prompt', ''), self.model_name)
        except Exception as e:
            st.error(f"Async query error: {e}")
            return None
    
    def _query_sync(self, payload: Dict) -> Optional[str]:
        """Original sync query implementation as fallback"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        # Format the request for Ollama
        user_input = payload.get("inputs", "")
        ollama_payload = {
            "model": self.model_name,
            "prompt": user_input,
            "system": self.system_prompt,
            "stream": True  # Enable streaming
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=ollama_payload, stream=True)
                response.raise_for_status()
                
                full_response = ""
                for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                    if chunk:
                        try:
                            chunk_data = json.loads(chunk.strip())
                            response_text = chunk_data.get("response", "")
                            full_response += response_text
                        except json.JSONDecodeError:
                            print(f"JSONDecodeError: {chunk}")  # Debugging
                            continue
                return full_response
            
            except requests.exceptions.RequestException as e:
                st.error(f"Ollama API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return None
            except Exception as e:
                st.error(f"Error processing Ollama response: {e}")
                return None
        return None
    
    async def query_stream(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream query with async support"""
        if not self._use_sync_fallback:
            try:
                if self.async_chat is None:
                    self.async_chat = await get_async_client()
                async for chunk in self.async_chat.generate_stream(payload.get('prompt', ''), self.model_name):
                    yield chunk
                return
            except Exception as e:
                st.warning(f"Async stream failed, falling back to sync: {e}")
                self._use_sync_fallback = True
        
        # Fallback to sync implementation
        for chunk in self._query_stream_sync(payload):
            yield chunk
    
    def _query_stream_sync(self, payload: Dict):
        """Sync stream implementation as fallback"""
        user_input = payload.get("inputs", "")
        ollama_payload = {
            "model": self.model_name,
            "prompt": user_input,
            "system": self.system_prompt,
            "stream": True
        }
        
        try:
            response = requests.post(self.api_url, json=ollama_payload, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                if chunk:
                    try:
                        chunk_data = json.loads(chunk.strip())
                        response_text = chunk_data.get("response", "")
                        if response_text:
                            yield response_text
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.error(f"Error in stream query: {e}")
            yield f"Error: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            if self.async_chat is None:
                self.async_chat = await get_async_client()
            return await self.async_chat.health_check()
        except Exception:
            return False
    
    def health_check_sync(self) -> bool:
        """Synchronous health check fallback"""
        try:
            response = requests.get(f"{OLLAMA_API_URL.replace('/api', '')}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return cache_manager.get_stats()

def process_file_with_document_processor(file, doc_processor: DocumentProcessor) -> str:
    """Process a file using the document processor with enhanced error handling"""
    try:
        logger.info(f"Starting to process file: {file.name} (type: {file.type}, size: {len(file.getvalue())} bytes)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(file.getvalue())
        temp_path = tmp_file.name
        logger.info(f"Created temporary file: {temp_path}")
        
        try:
            # Process based on file type
            if file.type.startswith("image/"):
                logger.info(f"Processing image file: {file.type}")
                # Extract text from image
                extracted_text = doc_processor.image_extractor.extract_text(temp_path)
                logger.info(f"Successfully extracted text from image: {len(extracted_text)} characters")
                # Create a simple document structure
                from dataclasses import dataclass
                @dataclass
                class SimpleDocument:
                    page_content: str
                    metadata: dict
                
                documents = [SimpleDocument(page_content=extracted_text, metadata={"source": file.name, "type": "image"})]
                
            elif file.type == "application/pdf":
                logger.info(f"Processing PDF file: {file.name}")
                # For PDF, we'll use the existing PDF processing
                extracted_text = doc_processor._read_pdf_file(Path(temp_path))
                documents = [SimpleDocument(page_content=extracted_text, metadata={"source": file.name, "type": "pdf"})]
                
            else:
                logger.info(f"Processing text file: {file.name}")
                # For text files, read directly
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [SimpleDocument(page_content=content, metadata={"source": file.name, "type": "text"})]

            # Split documents into chunks
            logger.info("Splitting documents into chunks")
            chunks = doc_processor.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Store file info in session state for context retrieval
            file_info = {
                "name": file.name,
                "size": len(file.getvalue()),
                "type": file.type,
                "chunks": len(chunks),
                "content": [chunk.page_content for chunk in chunks],  # Store document content
                "upload_time": datetime.datetime.now().isoformat()
            }
            
            # Check if file already exists
            existing_files = [doc["name"] for doc in st.session_state.uploaded_documents]
            if file.name in existing_files:
                logger.info(f"Updating existing file: {file.name}")
                # Update existing file
                for i, doc in enumerate(st.session_state.uploaded_documents):
                    if doc["name"] == file.name:
                        st.session_state.uploaded_documents[i] = file_info
                        break
            else:
                logger.info(f"Adding new file: {file.name}")
                # Add new file
                st.session_state.uploaded_documents.append(file_info)
            
            # Add to vector store if available
            if hasattr(doc_processor, 'vector_store') and doc_processor.vector_store is not None:
                try:
                    # Convert chunks to ProcessedFile format for vector store
                    processed_files = []
                    for i, chunk in enumerate(chunks):
                        processed_file = ProcessedFile(
                            filename=f"{file.name}_chunk_{i}",
                            content=chunk.page_content,
                            file_type=file.type,
                            metadata=chunk.metadata
                        )
                        processed_files.append(processed_file)
                    
                    doc_processor.add_to_vector_store(processed_files)
                    logger.info("Successfully added to vector store")
                except Exception as e:
                    logger.warning(f"Failed to add to vector store: {e}")
            else:
                logger.info("Vector store not available, skipping vector storage")
            
            # Return summary
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            return f"Successfully processed {file.name}. Extracted {len(chunks)} chunks with {total_chars} total characters."
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
                
    except Exception as e:
        error_msg = f"Failed to process file {file.name}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return error_msg

def get_relevant_context_from_session(query: str, k: int = 3) -> str:
    """
    Get relevant context from uploaded documents in session state.
    If the query seems generic, it returns the full content of the most recent document.
    """
    if not st.session_state.uploaded_documents:
        return ""
    
    try:
        query_lower = query.lower()
        # More specific triggers for generic queries about the whole document
        generic_triggers = [
            "this document", "the document", "the image", "this image",
            "summarize", "describe", "what is in", "what's in",
            "read the", "answer the"
        ]
        
        is_generic_query = any(trigger in query_lower for trigger in generic_triggers)
        
        # If query is generic and there are documents, return content of latest document
        if is_generic_query and st.session_state.uploaded_documents:
            latest_doc = st.session_state.uploaded_documents[-1]
            return "\n\n".join(latest_doc['content'])

        # Otherwise, perform keyword-based search across all documents
        relevant_content = []
        for doc in st.session_state.uploaded_documents:
            for content_chunk in doc["content"]:
                if query_lower in content_chunk.lower():
                    relevant_content.append(content_chunk)
            # Limit to k chunks to avoid overly long context
            if len(relevant_content) >= k:
                break
        
        return "\n\n".join(relevant_content[:k])
    except Exception as e:
        logger.error(f"Error getting context from session: {e}")
        return ""

def get_uploaded_documents_from_session() -> List[Dict]:
    """Get list of uploaded documents from session state"""
    return st.session_state.uploaded_documents

def remove_document_from_session(filename: str) -> bool:
    """Remove a document from the uploaded documents in session state"""
    try:
        st.session_state.uploaded_documents = [
            doc for doc in st.session_state.uploaded_documents 
            if doc["name"] != filename
        ]
        return True
    except Exception as e:
        print(f"Error removing document: {e}")
        return False

def clear_all_documents_from_session():
    """Clear all uploaded documents from session state"""
    st.session_state.uploaded_documents = []

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

class DocumentSummaryTool(Tool):
    def __init__(self, doc_processor):
        self.doc_processor = doc_processor

    def name(self) -> str:
        return "Document Summary"

    def description(self) -> str:
        return "Summarizes uploaded documents."

    def triggers(self) -> List[str]:
        return ["summarize document", "summarize the document", "give me a summary", "summarize documents", "document summary"]

    def execute(self, input_text: str) -> ToolResponse:
        try:
            documents = get_uploaded_documents_from_session()
            if not documents:
                return ToolResponse(content="No documents have been uploaded yet. Please upload a document first.", success=False)

            summary = "## 📚 Document Summary\n\n"
            
            for i, file_data in enumerate(documents, 1):
                summary += f"### {i}. {file_data['name']}\n"
                summary += f"- **Type:** {file_data['type']}\n"
                summary += f"- **Size:** {format_file_size(file_data['size'])}\n"
                summary += f"- **Chunks:** {file_data['chunks']}\n"
                summary += f"- **Uploaded:** {file_data['upload_time'][:19]}\n"
                
                # Add a brief content preview
                if file_data['content']:
                    first_content = file_data['content'][0]
                    preview = first_content[:200] + "..." if len(first_content) > 200 else first_content
                    summary += f"- **Preview:** {preview}\n"
                
                summary += "\n"

            return ToolResponse(content=summary)
        except Exception as e:
            return ToolResponse(content=f"Error summarizing documents: {e}", success=False, error=str(e))

class DateApiTool(Tool):
    def name(self) -> str:
        return "Date API"

    def description(self) -> str:
        return "Provides the current date."

    def triggers(self) -> List[str]:
        return ["current date", "what is the date", "today's date"]

    def execute(self, input_text: str) -> ToolResponse:
        try:
            today = datetime.date.today()
            date_str = today.strftime("%Y-%m-%d")
            return ToolResponse(content=f"Today's date is: {date_str}")
        except Exception as e:
            return ToolResponse(content=f"Error getting date: {e}", success=False)

class TimeTool(Tool):
    def name(self) -> str:
        return "Current Time"

    def description(self) -> str:
        return "Provides the current time and timezone."

    def triggers(self) -> List[str]:
        return ["what is the time", "current time", "what time is it", "what is today"]

    def execute(self, input_text: str) -> ToolResponse:
        timezone_str = os.environ.get("TIMEZONE", "UTC")  # Default to UTC
        try:
            timezone = pytz.timezone(timezone_str)
            now = datetime.datetime.now(pytz.utc).astimezone(timezone)
            time_str = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
            return ToolResponse(content=f"The current time is: {time_str}")
        except pytz.exceptions.UnknownTimeZoneError:
            return ToolResponse(content="Invalid timezone specified. Please set the TIMEZONE environment variable to a valid timezone.", success=False)

class DocumentQueryTool(Tool):
    def __init__(self, doc_processor, reasoning_engine):
        self.doc_processor = doc_processor
        self.reasoning_engine = reasoning_engine

    def name(self) -> str:
        return "Document Query"

    def description(self) -> str:
        return "Ask questions about uploaded documents."

    def triggers(self) -> List[str]:
        # Triggers should be more specific to avoid false positives
        return ["in the document", "about the document", "does the document say", "what's in the image", "describe the image", "answer the image", "answer whats in the image"]

    def execute(self, input_text: str) -> ToolResponse:
        try:
            documents = get_uploaded_documents_from_session()
            logger.info(f"DocumentQueryTool: Found {len(documents)} documents in session")
            
            if not documents:
                return ToolResponse(content="No documents have been uploaded yet. Please upload a document first.", success=False)

            # Get relevant context from documents
            context = get_relevant_context_from_session(input_text, k=3)
            logger.info(f"DocumentQueryTool: Retrieved context length: {len(context)} characters")
            
            if not context:
                return ToolResponse(content=f"I couldn't find any content matching your query: '{input_text}' in the uploaded documents.", success=False)

            # We have context, now let's use the reasoning engine to answer
            enhanced_prompt = f"""You are an expert at answering questions based on provided text.
Using ONLY the context below, please answer the user's question.
If the answer is not in the context, say that you cannot answer based on the provided information.

Context:
---
{context}
---
User question: {input_text}
"""
            logger.info(f"DocumentQueryTool: Sending prompt to reasoning engine")
            # Use the reasoning engine to get an answer
            result = self.reasoning_engine.reason(enhanced_prompt, context="")
            
            logger.info(f"DocumentQueryTool: Received response from reasoning engine")
            return ToolResponse(content=result.final_answer)
        except Exception as e:
            logger.error(f"Error querying documents: {e}", exc_info=True)
            return ToolResponse(content=f"Error querying documents: {e}", success=False, error=str(e))

class ToolRegistry:
    def __init__(self, doc_processor, reasoning_engine):
        self.tools: List[Tool] = [
            DocumentSummaryTool(doc_processor),
            TimeTool(),
            DateApiTool(),
            DocumentQueryTool(doc_processor, reasoning_engine)
        ]

    def get_tool(self, input_text: str) -> Optional[Tool]:
        for tool in self.tools:
            if any(trigger in input_text.lower() for trigger in tool.triggers()):
                return tool
        return None

def text_to_speech(text):
    """Convert text to speech and return the audio file path"""
    # Handle empty or None text
    if not text or text.strip() == "":
        return None
    
    try:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        audio_file = f"temp_{text_hash}.mp3"
        
        # Check if file already exists
        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
            return audio_file
        
        # Generate new audio file with timeout and error handling
        import threading
        import time
        
        # Flag to track if generation completed
        generation_completed = threading.Event()
        generation_error = None
        result_file = None
        
        def generate_audio():
            nonlocal generation_error, result_file
            try:
                # Set a shorter timeout for gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(audio_file)
                
                # Verify the file was created successfully
                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                    result_file = audio_file
                else:
                    generation_error = Exception("Audio file was not created successfully")
                    
            except Exception as e:
                generation_error = e
            finally:
                generation_completed.set()
        
        # Start audio generation in a separate thread
        audio_thread = threading.Thread(target=generate_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Wait for completion with timeout (15 seconds)
        if generation_completed.wait(timeout=15):
            if generation_error:
                raise generation_error
            return result_file
        else:
            # Timeout occurred
            raise Exception("Audio generation timed out after 15 seconds")
            
    except Exception as e:
        # Clean up any partial files
        try:
            if 'audio_file' in locals() and os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass
        raise Exception(f"Failed to generate audio: {str(e)}")

def get_audio_html(file_path: str) -> str:
    """
    Generate simple audio player HTML for testing compatibility.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        HTML string for the audio player
    """
    if not file_path or not os.path.exists(file_path):
        return '<p>No audio available</p>'
    
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
            html = f"""
            <audio controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            return html
    except Exception as e:
        return f'<p>Error loading audio: {str(e)}</p>'

def get_professional_audio_html(file_path: str) -> str:
    """
    Generate professional, minimal audio player HTML.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        HTML string for the audio player
    """
    if not file_path:
        return '<p style="color: #4a5568; font-style: italic; text-align: center; margin: 8px 0;">No audio available</p>'
    
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
            # Professional, minimal audio player
            html = f"""
            <div style="
                background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            ">
                <audio 
                    controls 
                    style="
                        width: 100%;
                        height: 40px;
                        border-radius: 8px;
                        background: white;
                        border: 1px solid #e2e8f0;
                        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                    "
                    preload="metadata"
                    aria-label="Audio playback controls"
                >
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            return html
            
    except FileNotFoundError:
        return '<p style="color: #e53e3e; font-style: italic; text-align: center; margin: 8px 0;">Audio file not found</p>'
    except Exception as e:
        return f'<p style="color: #e53e3e; font-style: italic; text-align: center; margin: 8px 0;">Error loading audio</p>'

def create_enhanced_audio_button(content: str, message_key: str):
    """
    Create a professional, streamlined audio button with clean UX patterns.
    
    Args:
        content: The text content to convert to speech
        message_key: Unique key for this message's audio state
    """
    # Initialize session state for this message's audio
    audio_state_key = f"audio_state_{message_key}"
    if audio_state_key not in st.session_state:
        st.session_state[audio_state_key] = {
            "status": "idle",  # idle, loading, ready, error
            "audio_file": None,
            "error_message": None,
            "had_error": False  # Track if there was a previous error
        }
    
    audio_state = st.session_state[audio_state_key]
    
    # Create a clean container with consistent spacing
    with st.container():
        # Subtle divider for audio section
        st.markdown("<hr style='margin: 16px 0 8px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        # Audio section header
        st.markdown(
            """
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
                font-size: 14px;
                color: #4a5568;
                font-weight: 500;
            ">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C13.1 2 14 2.9 14 4V12C14 13.1 13.1 14 12 14S10 13.1 10 12V4C10 2.9 10.9 2 12 2M18.5 12C18.5 15.6 15.6 18.5 12 18.5S5.5 15.6 5.5 12H7C7 14.5 9 16.5 11.5 16.5S16 14.5 16 12H18.5M12 20C16.4 20 20 16.4 20 12H22C22 17.5 17.5 22 12 22S2 17.5 2 12H4C4 16.4 7.6 20 12 20Z"/>
                </svg>
                Audio
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Handle different states with clean, minimal UI
        if audio_state["status"] == "idle":
            # Clean, professional generate button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🎵 Generate Audio",
                    key=f"audio_btn_{message_key}",
                    help="Click to generate audio version of this message",
                    use_container_width=True
                ):
                    # Generate audio immediately with spinner
                    try:
                        with st.spinner("Generating audio..."):
                            audio_file = text_to_speech(content)
                            if audio_file:
                                audio_state["audio_file"] = audio_file
                                audio_state["status"] = "ready"
                                audio_state["had_error"] = False  # Clear error flag on success
                            else:
                                audio_state["status"] = "error"
                                audio_state["error_message"] = "No content available for voice generation"
                                audio_state["had_error"] = True  # Set error flag
                    except Exception as e:
                        audio_state["status"] = "error"
                        audio_state["error_message"] = f"Failed to generate audio: {str(e)}"
                        audio_state["had_error"] = True  # Set error flag
                    
                    st.rerun()
        
        elif audio_state["status"] == "ready":
            # Clean audio player with minimal controls
            audio_html = get_professional_audio_html(audio_state["audio_file"])
            st.markdown(audio_html, unsafe_allow_html=True)
            
            # Only show regenerate if there was a previous error
            if hasattr(audio_state, "had_error") and audio_state.get("had_error", False):
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    if st.button(
                        "Regenerate Audio",
                        key=f"regenerate_{message_key}",
                        help="Generate new audio version",
                        use_container_width=True
                    ):
                        audio_state["status"] = "idle"
                        audio_state["audio_file"] = None
                        audio_state["had_error"] = False
                        # Clean up old file
                        try:
                            if audio_state["audio_file"] and os.path.exists(audio_state["audio_file"]):
                                os.remove(audio_state["audio_file"])
                        except:
                            pass
                        st.rerun()
        
        elif audio_state["status"] == "error":
            # Clean error state
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(
                    f"""
                    <div style="
                        padding: 12px;
                        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
                        border: 1px solid #fc8181;
                        border-radius: 8px;
                        color: #c53030;
                        font-size: 14px;
                        text-align: center;
                        box-shadow: 0 1px 2px rgba(197, 48, 48, 0.1);
                    ">
                        {audio_state['error_message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if st.button(
                    "Try Again",
                    key=f"retry_{message_key}",
                    help="Retry audio generation",
                    use_container_width=True
                ):
                    audio_state["status"] = "idle"
                    audio_state["error_message"] = None
                    audio_state["had_error"] = False  # Clear error flag on retry
                    st.rerun()

def cleanup_audio_files() -> None:
    """Clean up temporary audio files from session state"""
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("audio_state_"):
            audio_state = st.session_state[key]
            if audio_state.get("audio_file") and os.path.exists(audio_state["audio_file"]):
                try:
                    os.remove(audio_state["audio_file"])
                except:
                    pass

def get_audio_file_size(file_path: str) -> str:
    """Get human-readable file size for audio files"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except:
        return "Unknown size"

def display_reasoning_result(result: ReasoningResult):
    """Display reasoning result with enhanced formatting"""
    if not result.success:
        st.error(f"Reasoning failed: {result.error}")
        return
    
    # Display main content
    st.write(result.content)
    
    # Display reasoning steps if available
    if result.reasoning_steps:
        with st.expander("🔍 Reasoning Steps", expanded=True):
            for i, step in enumerate(result.reasoning_steps, 1):
                # Add visual indicators for different step types
                if step.startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)', '10)')):
                    st.markdown(f"**Step {i}:** {step}")
                elif step.startswith(('Step', 'STEP')):
                    st.markdown(f"**{step}**")
                else:
                    st.markdown(f"• {step}")
    
    # Display confidence and sources
    col1, col2 = st.columns(2)
    with col1:
        # Color code confidence levels
        if result.confidence >= 0.8:
            st.metric("Confidence", f"{result.confidence:.1%}", delta="High")
        elif result.confidence >= 0.6:
            st.metric("Confidence", f"{result.confidence:.1%}", delta="Medium")
        else:
            st.metric("Confidence", f"{result.confidence:.1%}", delta="Low")
    with col2:
        st.write("**Sources:**", ", ".join(result.sources))

def enhanced_chat_interface(doc_processor, reasoning_engine):
    """Main chat interface with enhanced features"""
    st.header("💬 Chat with AI")
    
    # Initialize chat components
    if "ollama_chat" not in st.session_state:
        st.session_state.ollama_chat = OllamaChat(model_name=CHAT_MODEL)
    
    ollama_chat = st.session_state.ollama_chat
    tool_registry = ToolRegistry(doc_processor, reasoning_engine)

    # Initialize welcome message if needed
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hello! I'm your AI assistant with enhanced reasoning capabilities. I'm currently using **Chain-of-Thought** reasoning with the **Mistral** model by default. You can change these settings in the sidebar if needed. Upload documents to analyze them, or start asking questions!"
        }]

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                # Create a unique key using message index and content hash
                unique_key = f"{i}_{hash(msg['content'])}_{len(msg['content'])}"
                create_enhanced_audio_button(msg["content"], unique_key)

    # Chat input
    if prompt := st.chat_input("Type a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process response based on reasoning mode
        with st.chat_message("assistant"):
            # First check if it's a tool-based query
            tool = tool_registry.get_tool(prompt)
            if tool:
                with st.spinner(f"Using {tool.name()}..."):
                    response = tool.execute(prompt)
                    if response.success:
                        st.write(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    else:
                        st.error(response.content)
            else:
                # Get relevant document context if available
                document_context = get_relevant_context_from_session(prompt, k=2)
                
                # Show document context indicator if available
                if document_context:
                    with st.expander("📄 Using Document Context", expanded=False):
                        st.info("Found relevant content in uploaded documents:")
                        st.text_area("Document Context:", document_context, height=100, disabled=True)
                
                # Enhance the prompt with document context if available
                enhanced_prompt = prompt
                if document_context:
                    enhanced_prompt = f"""Context from uploaded documents:
{document_context}

User question: {prompt}

Please answer the user's question, referencing the document context when relevant."""
                
                # Use reasoning modes with separated thought process and final output
                with st.spinner(f"Processing with Chain-of-Thought reasoning..."):
                    try:
                        result = reasoning_engine.reason(enhanced_prompt, context="")
                        # Stream the thought process
                        thought_placeholder = st.empty()
                        for step in result.reasoning_steps:
                            thought_placeholder.markdown(f"- {step}")
                            time.sleep(0.5)  # Simulate streaming for smooth UX
                        # Show final answer separately
                        st.markdown("### 📝 Final Answer")
                        st.markdown(result.final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": result.final_answer})
                    except Exception as e:
                        st.error(f"Error in Chain-of-Thought mode: {str(e)}")
                        # Fallback to standard mode
                        if response := ollama_chat.query({"inputs": enhanced_prompt}):
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Health Check Function ---
def check_ollama_server_status() -> bool:
    """Check if Ollama server is running"""
    try:
        # Use sync health check to avoid event loop issues
        chat = OllamaChat()
        return chat.health_check_sync()
    except Exception as e:
        logger.error(f"Error checking Ollama server status: {e}")
        return False

def handle_file_upload():
    """Callback function to handle file uploads."""
    uploaded_file = st.session_state.get("doc_uploader")
    if uploaded_file is None:
        return

    doc_processor = st.session_state.doc_processor
    try:
        # Check if this file has been processed already to avoid loops
        if "last_processed_filename" not in st.session_state or \
           st.session_state.last_processed_filename != uploaded_file.name:
            
            result = process_file_with_document_processor(uploaded_file, doc_processor)
            st.success(result)
            st.session_state.last_processed_filename = uploaded_file.name
                                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

# --- Main Application ---
def main():
    """Main application entry point."""
    
    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an AI assistant. Upload a document or ask me anything."}]
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if "reasoning_engine" not in st.session_state:
        st.session_state.reasoning_engine = ReasoningEngine()
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = check_ollama_server_status()

    doc_processor = st.session_state.doc_processor
    reasoning_engine = st.session_state.reasoning_engine

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("⚙️ System Status")
        if st.session_state.ollama_status:
            st.success("✅ Ollama Server is Online")
        else:
            st.error("❌ Ollama Server is Offline")
        
        if st.button("🔄 Refresh Status"):
            st.session_state.ollama_status = check_ollama_server_status()
        st.rerun()

        st.header("📚 Document Management")
        # ... (rest of sidebar UI) ...
    
    # Clean up audio files on app start
    if "audio_cleanup_done" not in st.session_state:
        cleanup_audio_files()
        st.session_state.audio_cleanup_done = True

    # Enhanced chat interface
    enhanced_chat_interface(doc_processor, reasoning_engine)

    with st.sidebar:
        st.header("📚 Documents")
        
        # Add reset button at the top
        if st.button("🔄 Reset All Data", help="Clear all messages and documents"):
            st.session_state.messages = []
            clear_all_documents_from_session()
            st.success("All data has been reset!")
            st.rerun()
        
        st.markdown("---")
        
        # Document upload section
        st.subheader("📤 Upload Documents")
        st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload documents to analyze",
            key="doc_uploader",
            on_change=handle_file_upload
        )
        
        st.markdown("---")
        
        # Document management section
        st.subheader("📋 Uploaded Documents")
        documents = get_uploaded_documents_from_session()
        
        if not documents:
            st.info("No documents uploaded yet.")
        else:
            st.write(f"**Total documents:** {len(documents)}")
            
            for i, doc in enumerate(documents):
                with st.expander(f"📄 {doc['name']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Type:** {doc['type']}")
                        st.write(f"**Size:** {format_file_size(doc['size'])}")
                        st.write(f"**Chunks:** {doc['chunks']}")
                        st.write(f"**Uploaded:** {doc['upload_time'][:19]}")
                    
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            if remove_document_from_session(doc['name']):
                                st.success(f"Removed {doc['name']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to remove {doc['name']}")
                    
                    # Show content preview
                    if doc['content']:
                        st.write("**Content Preview:**")
                        preview = doc['content'][0][:300] + "..." if len(doc['content'][0]) > 300 else doc['content'][0]
                        st.text(preview)
        
        st.markdown("---")
        
        # Document search section
        st.subheader("🔍 Document Search")
        search_query = st.text_input(
            "Search in documents", 
            key="doc_search_query",
            help="Enter keywords to search for in the document text."
        )
        
        if st.button("Search", key="doc_search_button"):
            if search_query:
                with st.spinner("Searching..."):
                    search_results = get_relevant_context_from_session(search_query, k=5)
                    st.session_state.search_results = search_results
            else:
                st.session_state.search_results = ""

        if "search_results" in st.session_state and st.session_state.search_results:
            st.text_area("Search Results", st.session_state.search_results, height=150)

        st.markdown("---")
        
        # --- Advanced Options ---
        st.subheader("🛠️ Advanced Options")
        
        # Add a checkbox to show debug info
        st.checkbox("Show Debug Info", key="show_debug_info")

        if st.session_state.get("show_debug_info"):
            st.subheader("🐛 Debug Information")
            
            # Display model configuration
            st.write("**Model Configuration:**")
            st.json({
                "Chat Model": CHAT_MODEL,
                "Reasoning Model": REASONING_MODEL,
                "Vision Model": VISION_MODEL,
                "Embedding Model": EMBEDDING_MODEL,
            })

            # Session State
            st.write("**Session State:**")
            st.json(st.session_state.to_dict(), expanded=False)
            
            # Cache Stats
            st.subheader("Cache Stats")
            try:
                if ollama_chat := st.session_state.get("ollama_chat"):
                    stats = ollama_chat.get_cache_stats()
                    st.json(stats)
                else:
                    st.info("Chat not initialized yet.")
            except Exception as e:
                st.error(f"Could not retrieve cache stats: {e}")

            # Display Logs
            st.subheader("Application Logs")
            st.info("Check the terminal/console for detailed logs")

if __name__ == "__main__":
    main()