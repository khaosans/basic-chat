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
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
import tempfile
from gtts import gTTS
import hashlib
import base64
from PIL import Image
import io

# Import our new reasoning engine
from reasoning_engine import (
    ReasoningAgent, 
    ReasoningChain, 
    MultiStepReasoning, 
    ReasoningDocumentProcessor,
    ReasoningResult
)

# Import new async components
from config import config
from utils.async_ollama import AsyncOllamaChat, async_chat
from utils.caching import response_cache

# Import session management
from session_manager import session_manager, ChatSession

load_dotenv(".env.local")  # Load environment variables from .env.local

# Use Ollama model instead of Hugging Face
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
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

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict]]  # Can be text or list of content blocks
    timestamp: Optional[datetime.datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()

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
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or OLLAMA_MODEL
        self.api_url = f"{OLLAMA_API_URL}/generate"
        self.system_prompt = SYSTEM_PROMPT
        
        # Initialize async chat client
        self.async_chat = AsyncOllamaChat(self.model_name)
        
        # Fallback to sync implementation if needed
        self._use_sync_fallback = False

    def query(self, payload: Dict) -> Optional[str]:
        """Query the Ollama API with async support and fallback"""
        if not self._use_sync_fallback:
            try:
                # Try async implementation
                return asyncio.run(self._query_async(payload))
            except Exception as e:
                st.warning(f"Async query failed, falling back to sync: {e}")
                self._use_sync_fallback = True
        
        # Fallback to original sync implementation
        return self._query_sync(payload)
    
    def query_with_image(self, text: str, image_data: Dict) -> Optional[str]:
        """Query the Ollama API with image input"""
        try:
            # Check if model supports images (like llava)
            if not self._is_multimodal_model():
                return "This model doesn't support image processing. Please use a multimodal model like 'llava'."
            
            # Prepare multimodal payload
            ollama_payload = {
                "model": self.model_name,
                "prompt": text,
                "system": self.system_prompt,
                "images": [image_data["data"]],  # Base64 encoded image
                "stream": False
            }
            
            response = requests.post(self.api_url, json=ollama_payload)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data.get("response", "")
            
        except Exception as e:
            st.error(f"Error processing image query: {e}")
            return None
    
    def _is_multimodal_model(self) -> bool:
        """Check if the current model supports multimodal inputs"""
        multimodal_models = ['llava', 'llava:7b', 'llava:13b', 'llava:34b', 'bakllava', 'llava-llama3']
        return any(model in self.model_name.lower() for model in multimodal_models)
    
    async def _query_async(self, payload: Dict) -> Optional[str]:
        """Async query implementation"""
        try:
            return await self.async_chat.query(payload)
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
    
    async def query_stream(self, payload: Dict):
        """Stream query with async support"""
        if not self._use_sync_fallback:
            try:
                async for chunk in self.async_chat.query_stream(payload):
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
            return await self.async_chat.health_check()
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return response_cache.get_stats()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.processed_files = []
        
        # Initialize vectorstore if exists
        self.vectorstore = None

    def process_file(self, file) -> None:
        """Process and store file with proper chunking and embedding"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name

            # Load documents based on file type
            if file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file.type.startswith("image/"):
                try:
                    loader = UnstructuredImageLoader(file_path)
                    documents = loader.load()
                except Exception as e:
                    st.error(f"Failed to load image: {str(e)}")
                    return
            else:
                raise ValueError(f"Unsupported file type: {file.type}")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Store file info
            self.processed_files.append({
                "name": file.name,
                "size": len(file.getvalue()),
                "type": file.type,
                "chunks": len(chunks)
            })

            # Cleanup
            os.unlink(file_path)
            
            return f"✅ Processed {file.name} into {len(chunks)} chunks"

        except Exception as e:
            raise Exception(f"Failed to process file: {str(e)}")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for a query"""
        if not self.vectorstore:
            return ""
        
        try:
            return ""
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""

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
            if not self.doc_processor.processed_files:
                return ToolResponse(content="No documents have been uploaded yet.", success=False)

            summary = ""
            for file_data in self.doc_processor.processed_files:
                summary += f"Summary of {file_data['name']}:\n"
                # In a real implementation, you would summarize the document content here
                # For now, just return the document name
                summary += "This feature is not yet implemented.\n"

            return ToolResponse(content=summary)
        except Exception as e:
            return ToolResponse(content=f"Error summarizing document: {e}", success=False, error=str(e))

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

class ToolRegistry:
    def __init__(self, doc_processor):
        self.tools: List[Tool] = [
            DocumentSummaryTool(doc_processor),
            TimeTool(),  # Add the TimeTool to the registry
            DateApiTool()
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
                        "🔄 Regenerate Audio",
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

def cleanup_audio_files():
    """Clean up temporary audio files from session state"""
    for key in list(st.session_state.keys()):
        if key.startswith("audio_state_"):
            audio_state = st.session_state[key]
            if audio_state.get("audio_file") and os.path.exists(audio_state["audio_file"]):
                try:
                    os.remove(audio_state["audio_file"])
                except:
                    pass

def get_audio_file_size(file_path: str) -> str:
    """Get human-readable file size"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def process_uploaded_image(image_file) -> Dict:
    """Process uploaded image and return metadata and base64 data"""
    try:
        # Check file size (max 10MB)
        image_data = image_file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise Exception("Image file is too large. Please use an image smaller than 10MB.")
        
        image_file.seek(0)  # Reset file pointer
        
        # Open with PIL for processing
        image = Image.open(io.BytesIO(image_data))
        
        # Check if image is valid
        if image.size[0] == 0 or image.size[1] == 0:
            raise Exception("Invalid image file.")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 1024px on longest side)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Get image info
        file_size = len(image_data)
        file_size_str = get_audio_file_size_from_bytes(file_size)
        
        return {
            "type": "image",
            "data": img_base64,
            "mime_type": "image/jpeg",
            "file_name": image_file.name,
            "file_size": file_size_str,
            "dimensions": image.size,
            "original_size": file_size
        }
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def get_audio_file_size_from_bytes(size_bytes: int) -> str:
    """Get human-readable file size from bytes"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def create_image_message_content(text: str, image_data: Dict) -> List[Dict]:
    """Create a message content list with both text and image"""
    content = []
    
    if text.strip():
        content.append({
            "type": "text",
            "text": text
        })
    
    content.append({
        "type": "image",
        "image_data": image_data
    })
    
    return content

def display_message_content(content: Union[str, List[Dict]]):
    """Display message content (text and/or images)"""
    if isinstance(content, str):
        st.write(content)
    elif isinstance(content, list):
        for block in content:
            if block["type"] == "text":
                st.write(block["text"])
            elif block["type"] == "image":
                image_data = block["image_data"]
                st.image(
                    base64.b64decode(image_data["data"]),
                    caption=f"📷 {image_data['file_name']} ({image_data['file_size']})",
                    use_column_width=True
                )

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

# Session Management Functions
def create_new_session(title: str, model: str, reasoning_mode: str) -> ChatSession:
    """Create a new chat session"""
    try:
        session = session_manager.create_session(
            title=title,
            model=model,
            reasoning_mode=reasoning_mode,
            user_id="default"
        )
        
        # Clear current messages and set new session
        st.session_state.messages = []
        st.session_state.current_session = session
        
        st.success(f"Created new session: {title}")
        return session
        
    except Exception as e:
        st.error(f"Failed to create session: {e}")
        return None

def load_session(session_id: str) -> bool:
    """Load a chat session"""
    try:
        session = session_manager.load_session(session_id)
        if session:
            st.session_state.messages = session.messages
            st.session_state.current_session = session
            st.success(f"Loaded session: {session.title}")
            return True
        else:
            st.error("Session not found")
            return False
            
    except Exception as e:
        st.error(f"Failed to load session: {e}")
        return False

def save_current_session() -> bool:
    """Save the current session"""
    try:
        if "current_session" not in st.session_state:
            # Create new session if none exists
            title = st.text_input("Enter session title:", value="New Chat Session")
            if title:
                session = create_new_session(title, st.session_state.get("selected_model", "mistral"), 
                                           st.session_state.get("reasoning_mode", "Standard"))
                if not session:
                    return False
            else:
                return False
        
        session = st.session_state.current_session
        session.messages = st.session_state.messages
        session.updated_at = datetime.datetime.now()
        
        success = session_manager.save_session(session)
        if success:
            st.success(f"Session saved: {session.title}")
        else:
            st.error("Failed to save session")
        return success
        
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False

def auto_save_current_session():
    """Auto-save current session if enabled"""
    if (config.session.enable_auto_save and 
        "current_session" in st.session_state and 
        "messages" in st.session_state):
        
        # Only auto-save if we have messages and it's been a while
        if len(st.session_state.messages) > 0:
            session = st.session_state.current_session
            session_manager.auto_save_session(session.id, st.session_state.messages)

def render_session_management_sidebar():
    """Render session management UI in sidebar"""
    with st.sidebar:
        st.header("💾 Chat Sessions")
        
        # Session controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session", help="Save current conversation"):
                save_current_session()
        
        with col2:
            if st.button("🔄 New Session", help="Start fresh conversation"):
                title = st.text_input("Session title:", value="New Chat Session")
                if title:
                    create_new_session(title, st.session_state.get("selected_model", "mistral"), 
                                     st.session_state.get("reasoning_mode", "Standard"))
        
        # Session search
        if config.session.enable_session_search:
            search_query = st.text_input("🔍 Search sessions", placeholder="Search by title or content...")
            
            # Get sessions
            if search_query:
                sessions = session_manager.search_sessions(search_query)
            else:
                sessions = session_manager.list_sessions(limit=10)
            
            # Display sessions
            if sessions:
                st.subheader("📝 Recent Sessions")
                for session in sessions:
                    with st.expander(f"📝 {session.title}", expanded=False):
                        st.caption(f"Model: {session.model_used} | Mode: {session.reasoning_mode}")
                        st.caption(f"Updated: {session.updated_at.strftime('%Y-%m-%d %H:%M')}")
                        st.caption(f"Messages: {len(session.messages)}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("📂 Load", key=f"load_{session.id}"):
                                load_session(session.id)
                        with col2:
                            if st.button("✏️ Edit", key=f"edit_{session.id}"):
                                new_title = st.text_input("New title:", value=session.title, key=f"edit_title_{session.id}")
                                if new_title and new_title != session.title:
                                    session_manager.update_session(session.id, {"title": new_title})
                                    st.rerun()
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{session.id}"):
                                if session_manager.delete_session(session.id):
                                    st.success("Session deleted")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete session")
            else:
                st.info("No sessions found")
        
        # Export/Import section
        with st.expander("📤 Export/Import"):
            if "current_session" in st.session_state:
                session = st.session_state.current_session
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📄 Export JSON", key="export_json"):
                        json_data = session_manager.export_session(session.id, "json")
                        if json_data:
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"{session.title}.json",
                                mime="application/json"
                            )
                
                with col2:
                    if st.button("📝 Export Markdown", key="export_md"):
                        md_data = session_manager.export_session(session.id, "markdown")
                        if md_data:
                            st.download_button(
                                label="Download Markdown",
                                data=md_data,
                                file_name=f"{session.title}.md",
                                mime="text/markdown"
                            )
            
            # Import session
            uploaded_file = st.file_uploader("Import session", type=["json"])
            if uploaded_file:
                try:
                    session_data = uploaded_file.read().decode()
                    imported_session = session_manager.import_session(session_data, "json")
                    if imported_session:
                        st.success(f"Imported session: {imported_session.title}")
                        st.rerun()
                    else:
                        st.error("Failed to import session")
                except Exception as e:
                    st.error(f"Error importing session: {e}")

def display_current_session_info():
    """Display current session information"""
    if "current_session" in st.session_state:
        session = st.session_state.current_session
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        ">
            <strong>📝 Session:</strong> {session.title} | 
            <strong>🤖 Model:</strong> {session.model_used} | 
            <strong>🧠 Mode:</strong> {session.reasoning_mode}
            <br>
            <small>Last updated: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')} | 
            Messages: {len(session.messages)}</small>
        </div>
        """, unsafe_allow_html=True)

def analyze_image_with_reasoning(image_data: Dict, text_prompt: str, reasoning_mode: str, ollama_chat, reasoning_chain, multi_step, reasoning_agent):
    """Analyze image using different reasoning modes"""
    try:
        if reasoning_mode == "Chain-of-Thought":
            # For images, we'll use a simplified chain-of-thought approach
            prompt_with_context = f"""
            Analyze this image step by step:
            1. First, describe what you see in the image
            2. Then, answer the user's question: "{text_prompt}"
            
            Please show your reasoning process.
            """
            response = ollama_chat.query_with_image(prompt_with_context, image_data)
            return response
            
        elif reasoning_mode == "Multi-Step":
            # Multi-step analysis for images
            steps = [
                "Describe the visual elements in the image",
                "Identify any text, objects, or people",
                "Analyze the context and setting",
                f"Answer the specific question: {text_prompt}"
            ]
            
            full_response = ""
            for i, step in enumerate(steps, 1):
                step_prompt = f"Step {i}: {step}"
                step_response = ollama_chat.query_with_image(step_prompt, image_data)
                if step_response:
                    full_response += f"\n\n**Step {i}:** {step_response}"
            
            return full_response.strip()
            
        elif reasoning_mode == "Agent-Based":
            # Agent-based image analysis
            agent_prompt = f"""
            As an image analysis agent, please:
            1. Examine the image carefully
            2. Identify key elements and details
            3. Provide a comprehensive analysis
            4. Answer the user's question: "{text_prompt}"
            
            Show your analysis process.
            """
            response = ollama_chat.query_with_image(agent_prompt, image_data)
            return response
            
        else:  # Standard mode
            # Simple image analysis
            response = ollama_chat.query_with_image(text_prompt, image_data)
            return response
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def enhanced_chat_interface(doc_processor):
    """Enhanced chat interface with reasoning capabilities"""
    # Professional CSS with clean audio styling
    st.markdown(
        """
        <style>
            /* Main container */
            .main {
                max-width: 800px !important;
                padding: 1rem !important;
            }
            
            /* Message containers */
            .stChatMessage {
                padding: 0.5rem 0 !important;
            }
            
            /* User messages */
            [data-testid="chat-message-user"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border-radius: 12px 12px 4px 12px !important;
                padding: 0.75rem 1rem !important;
                margin: 0.25rem 0 !important;
                margin-left: auto !important;
                max-width: 80% !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2) !important;
            }
            
            /* Assistant messages */
            [data-testid="chat-message-assistant"] {
                background: #ffffff !important;
                color: #2d3748 !important;
                border-radius: 12px 12px 12px 4px !important;
                padding: 0.75rem 1rem !important;
                margin: 0.25rem 0 !important;
                margin-right: auto !important;
                max-width: 80% !important;
                border: 1px solid #e2e8f0 !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            }

            /* Professional audio button styling */
            .stButton button {
                background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%) !important;
                color: #4a5568 !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 8px !important;
                padding: 10px 20px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
            }

            .stButton button:hover:not(:disabled) {
                background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%) !important;
                border-color: #cbd5e0 !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
                color: #2d3748 !important;
            }

            .stButton button:disabled {
                background: #f7fafc !important;
                color: #a0aec0 !important;
                border-color: #e2e8f0 !important;
                cursor: not-allowed !important;
                opacity: 0.6 !important;
                transform: none !important;
                box-shadow: none !important;
            }

            /* Loading spinner animation */
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading-spinner {
                animation: spin 1s linear infinite;
            }
            
            /* Reasoning mode styling */
            .reasoning-mode {
                background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
                border: 1px solid #90cdf4;
                border-radius: 8px;
                padding: 0.5rem;
                margin: 0.5rem 0;
            }

            /* Focus indicators for accessibility */
            .stButton button:focus {
                outline: 2px solid #4299e1 !important;
                outline-offset: 2px !important;
            }

            /* High contrast mode support */
            @media (prefers-contrast: high) {
                .stButton button {
                    border: 2px solid #000 !important;
                }
            }

            /* Reduced motion support */
            @media (prefers-reduced-motion: reduce) {
                .stButton button,
                .loading-spinner {
                    transition: none !important;
                    animation: none !important;
                }
            }
            
            /* Image upload section styling */
            .image-upload-section {
                background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
                border: 2px dashed #cbd5e0;
                border-radius: 12px;
                padding: 1rem;
                margin: 1rem 0;
                transition: all 0.2s ease;
            }
            
            .image-upload-section:hover {
                border-color: #4299e1;
                background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
            }
            
            /* Image preview styling */
            .image-preview {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            /* Chat message with image styling */
            .chat-message-image {
                border-radius: 8px;
                margin: 0.5rem 0;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add reasoning mode selection in sidebar
    with st.sidebar:
        st.header("🧠 Reasoning Mode")
        reasoning_mode = st.selectbox(
            "Select Reasoning Mode",
            ["Standard", "Chain-of-Thought", "Multi-Step", "Agent-Based"],
            help="Choose how the AI should process your questions"
        )
        
        # Enhanced mode descriptions with detailed explanations
        mode_descriptions = {
            "Standard": {
                "short": "Basic chat with simple responses",
                "detailed": """
                - Direct question-answer format
                - No visible reasoning steps
                - Fastest response time
                - Best for simple queries
                """
            },
            "Chain-of-Thought": {
                "short": "Shows step-by-step reasoning process",
                "detailed": """
                - Breaks down complex problems
                - Shows each step of thinking
                - Explains assumptions and logic
                - Best for understanding 'why'
                """
            },
            "Multi-Step": {
                "short": "Breaks complex questions into multiple steps",
                "detailed": """
                - Analyzes question components
                - Gathers relevant context
                - Builds structured solution
                - Best for complex problems
                """
            },
            "Agent-Based": {
                "short": "Uses tools and agents for enhanced capabilities",
                "detailed": """
                - Accesses external tools
                - Uses multiple specialized agents
                - Combines different capabilities
                - Best for tasks requiring tools
                """
            }
        }
        
        # Show short description in info box
        st.info(f"**{reasoning_mode}**: {mode_descriptions[reasoning_mode]['short']}")
        
        # Show detailed explanation in an expander
        with st.expander("See detailed explanation"):
            st.markdown(mode_descriptions[reasoning_mode]['detailed'])
        
        # Add model selection with detailed descriptions
        st.header("🤖 Model Selection")
        
        # Define model descriptions and use cases
        model_descriptions = {
            "mistral": {
                "short": "Powerful general-purpose model with strong reasoning",
                "detailed": """
                - Best for: Complex reasoning and analysis
                - Strengths:
                  • Strong logical reasoning capabilities
                  • Excellent at step-by-step problem solving
                  • Good balance of speed and accuracy
                  • Efficient context handling
                - Ideal for:
                  • Academic and technical writing
                  • Mathematical problem solving
                  • Code analysis and explanation
                  • Complex multi-step tasks
                """
            },
            "llava": {
                "short": "Multimodal model for text and image processing",
                "detailed": """
                - Best for: Image understanding and visual tasks
                - Strengths:
                  • Can analyze images and provide descriptions
                  • Understands visual context and details
                  • Can answer questions about images
                  • Combines visual and textual reasoning
                - Ideal for:
                  • Image analysis tasks
                  • Visual question answering
                  • Document analysis with images
                  • Visual content description
                """
            },
            "codellama": {
                "short": "Specialized model for programming tasks",
                "detailed": """
                - Best for: Code-related tasks and development
                - Strengths:
                  • Strong code understanding
                  • Excellent at code generation
                  • Bug detection and fixing
                  • Technical documentation
                - Ideal for:
                  • Programming assistance
                  • Code review and analysis
                  • Algorithm implementation
                  • Technical problem solving
                """
            },
            "llama2": {
                "short": "Versatile base model for general tasks",
                "detailed": """
                - Best for: General-purpose applications
                - Strengths:
                  • Well-rounded capabilities
                  • Good at general conversation
                  • Decent reasoning abilities
                  • Broad knowledge base
                - Ideal for:
                  • General chat applications
                  • Basic content generation
                  • Simple analysis tasks
                  • Everyday queries
                """
            },
            "nomic-embed-text": {
                "short": "Specialized model for text embeddings",
                "detailed": """
                - Best for: Text analysis and similarity tasks
                - Strengths:
                  • High-quality text embeddings
                  • Semantic search capabilities
                  • Document comparison
                  • Content organization
                - Ideal for:
                  • Document retrieval
                  • Similarity matching
                  • Content classification
                  • Search functionality
                """
            }
        }
        
        try:
            from ollama_api import get_available_models
            available_models = get_available_models()
            
            # Initialize session state for model selection if not exists
            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = OLLAMA_MODEL
            
            # Create columns for model selection and quick info button
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_model = st.selectbox(
                    "Choose Model",
                    available_models,
                    index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
                    key='model_selector',
                    help="Select the Ollama model to use for reasoning"
                )
            
            # Update session state
            st.session_state.selected_model = selected_model
            
            # Show model information based on selection
            if selected_model.lower() in model_descriptions:
                model_info = model_descriptions[selected_model.lower()]
                
                # Show short description in info box
                st.info(f"**{selected_model}**: {model_info['short']}")
                
                # Show detailed description in expander
                with st.expander("See model capabilities and best uses"):
                    st.markdown(model_info['detailed'])
                    
                    # Add performance note specific to the model
                    st.markdown("---")
                    st.markdown("**💻 Performance Note:**")
                    if selected_model.lower() in ['llava', 'codellama']:
                        st.markdown("This model may require more computational resources.")
                    elif selected_model.lower() == 'mistral':
                        st.markdown("Offers good performance with moderate resource requirements.")
                    else:
                        st.markdown("Standard resource requirements apply.")
            else:
                # Generic description for unknown models
                st.info(f"**{selected_model}**: Custom or specialized Ollama model")
                with st.expander("About this model"):
                    st.markdown("""
                    This appears to be a custom or specialized model. Consider the following:
                    - Capabilities will depend on the model's training
                    - Refer to the model's documentation for specific use cases
                    - Performance characteristics may vary
                    - Test the model with your specific use case
                    """)
                    
        except Exception as e:
            st.warning(f"Could not fetch available models: {e}")
            selected_model = OLLAMA_MODEL
            if selected_model in model_descriptions:
                st.info(f"**{selected_model}**: {model_descriptions[selected_model]['short']}")
                with st.expander("See model details"):
                    st.markdown(model_descriptions[selected_model]['detailed'])

        # Add a general note about model selection
        with st.expander("📝 Tips for choosing models"):
            st.markdown("""
            **When selecting a model, consider:**
            - Task complexity and specific requirements
            - Available system resources (RAM, CPU/GPU)
            - Speed vs accuracy trade-offs
            - Whether you need specialized capabilities (code, images, etc.)
            
            **Quick Guide:**
            - Use **Mistral** for general reasoning and complex tasks
            - Use **LLaVA** for image-related tasks
            - Use **CodeLlama** for programming tasks
            - Use **Llama2** for general conversation
            - Use **Nomic-Embed** for text embedding and search
            """)

    # Initialize reasoning components with the selected model
    reasoning_agent = ReasoningAgent(selected_model)
    reasoning_chain = ReasoningChain(selected_model)
    multi_step = MultiStepReasoning(doc_processor, selected_model)
    
    # Create chat instances
    ollama_chat = OllamaChat(selected_model)
    tool_registry = ToolRegistry(doc_processor)

    # Display current session info if exists
    display_current_session_info()

    # Initialize welcome message if needed
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hello! I'm your AI assistant with enhanced reasoning capabilities. Choose a reasoning mode from the sidebar and let's start exploring! You can also upload images to discuss them with me!"
        }]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            display_message_content(msg["content"])
            if msg["role"] == "assistant":
                # Extract text content for audio button
                text_content = msg["content"]
                if isinstance(msg["content"], list):
                    text_blocks = [block["text"] for block in msg["content"] if block["type"] == "text"]
                    text_content = " ".join(text_blocks)
                if text_content:
                    create_enhanced_audio_button(text_content, hash(text_content))

    # Image upload section
    st.markdown("---")
    
    # Add helpful information about image upload
    with st.expander("📷 Image Upload Help", expanded=False):
        st.markdown("""
        **How to use image upload:**
        1. Upload an image using the file uploader below
        2. Type your question about the image in the chat input
        3. The AI will analyze the image and respond to your question
        
        **Supported formats:** PNG, JPG, JPEG, GIF, BMP, WebP
        
        **Best results with:** Multimodal models like 'llava', 'llava:7b', 'llava:13b'
        
        **Example prompts:**
        - "What's in this image?"
        - "Describe this photo in detail"
        - "What objects can you see?"
        - "Is there any text in this image?"
        - "What emotions does this image convey?"
        - "Analyze the composition of this photo"
        - "What's happening in this scene?"
        
        **Tips:**
        - Images are automatically resized for optimal processing
        - Clear, high-quality images work best
        - Be specific in your questions for better results
        - Use different reasoning modes for different types of analysis
        """)
        
        # Add quick prompt buttons
        st.markdown("**Quick prompts:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("What's in this image?", key="prompt1"):
                st.session_state.quick_prompt = "What's in this image?"
        with col2:
            if st.button("Describe this photo", key="prompt2"):
                st.session_state.quick_prompt = "Describe this photo in detail"
        with col3:
            if st.button("Analyze objects", key="prompt3"):
                st.session_state.quick_prompt = "What objects can you see in this image?"

    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "📷 Upload an image to discuss",
            type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
            help="Upload an image to discuss with the AI. Works best with multimodal models like 'llava'.",
            key="chat_image_uploader"
        )
    
    with col2:
        if uploaded_image:
            st.markdown("**Preview:**")
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            try:
                image_data = process_uploaded_image(uploaded_image)
                st.info(f"📊 **Image Info:**\n- Size: {image_data['file_size']}\n- Dimensions: {image_data['dimensions'][0]}×{image_data['dimensions'][1]}px")
                
                # Check if current model supports images
                if not ollama_chat._is_multimodal_model():
                    st.warning("""
                    ⚠️ **Model Recommendation:**
                    
                    Your current model doesn't support image processing. 
                    For best results with images, switch to a multimodal model like:
                    - **llava** (recommended)
                    - **llava:7b** (faster)
                    - **llava:13b** (more accurate)
                    """)
                else:
                    st.success("✅ Current model supports image processing!")
                    
            except:
                pass

    # Chat input
    # Check if there's a quick prompt set
    if hasattr(st.session_state, 'quick_prompt') and st.session_state.quick_prompt:
        # Show the quick prompt as a button or info
        st.info(f"💡 Quick prompt ready: **{st.session_state.quick_prompt}**")
        if st.button("Use this prompt", key="use_quick_prompt"):
            st.session_state.pending_prompt = st.session_state.quick_prompt
            st.session_state.quick_prompt = ""
            st.rerun()
    
    if prompt := st.chat_input("Type a message..."):
        # Check if there's a pending prompt from quick prompt
        if hasattr(st.session_state, 'pending_prompt') and st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = ""
        
        # Handle image + text message
        if uploaded_image:
            try:
                # Process the uploaded image
                image_data = process_uploaded_image(uploaded_image)
                
                # Create message content with both text and image
                message_content = create_image_message_content(prompt, image_data)
                
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user", 
                    "content": message_content
                })
                
                # Clear the uploaded image
                st.session_state.chat_image_uploader = None
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                # Fall back to text-only message
                st.session_state.messages.append({"role": "user", "content": prompt})
        else:
            # Text-only message
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Auto-save session if enabled and we have a current session
        if "current_session" in st.session_state:
            auto_save_current_session()
        
        # Process response based on reasoning mode
        with st.chat_message("assistant"):
            # Check if this is an image message
            last_message = st.session_state.messages[-1]
            has_image = isinstance(last_message["content"], list) and any(
                block["type"] == "image" for block in last_message["content"]
            )
            
            if has_image:
                # Handle image + text query
                image_block = next(block for block in last_message["content"] if block["type"] == "image")
                text_blocks = [block["text"] for block in last_message["content"] if block["type"] == "text"]
                text_prompt = " ".join(text_blocks) if text_blocks else "What's in this image?"
                
                # Show image being analyzed
                st.markdown("**📷 Analyzing this image:**")
                st.image(
                    base64.b64decode(image_block["image_data"]["data"]),
                    caption=f"Analyzing: {image_block['image_data']['file_name']}",
                    use_column_width=True
                )
                
                with st.spinner(f"🔍 Analyzing image with {reasoning_mode} reasoning..."):
                    response = analyze_image_with_reasoning(image_block["image_data"], text_prompt, reasoning_mode, ollama_chat, reasoning_chain, multi_step, reasoning_agent)
                    if response:
                        st.markdown("### 📝 Analysis Result")
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to analyze image. Please try again or check if your model supports image processing.")
            else:
                # Handle text-only query (existing logic)
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
                    # Use reasoning modes with separated thought process and final output
                    with st.spinner(f"Processing with {reasoning_mode} reasoning..."):
                        try:
                            if reasoning_mode == "Chain-of-Thought":
                                with st.expander("💭 Thought Process", expanded=True):
                                    result = reasoning_chain.execute_reasoning(prompt)
                                    # Stream the thought process
                                    thought_placeholder = st.empty()
                                    for step in result.reasoning_steps:
                                        thought_placeholder.markdown(f"- {step}")
                                        time.sleep(0.5)  # Simulate streaming for smooth UX
                                
                                # Show final answer separately
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.content)
                                st.session_state.messages.append({"role": "assistant", "content": result.content})
                                
                            elif reasoning_mode == "Multi-Step":
                                with st.expander("🔍 Analysis & Planning", expanded=True):
                                    result = multi_step.step_by_step_reasoning(prompt)
                                    # Stream the analysis phase
                                    analysis_placeholder = st.empty()
                                    for step in result.reasoning_steps:
                                        analysis_placeholder.markdown(f"- {step}")
                                        time.sleep(0.5)
                                
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.content)
                                st.session_state.messages.append({"role": "assistant", "content": result.content})
                                
                            elif reasoning_mode == "Agent-Based":
                                with st.expander("🤖 Agent Actions", expanded=True):
                                    result = reasoning_agent.run(prompt)
                                    # Stream agent actions
                                    action_placeholder = st.empty()
                                    for step in result.reasoning_steps:
                                        action_placeholder.markdown(f"- {step}")
                                        time.sleep(0.5)
                                
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.content)
                                st.session_state.messages.append({"role": "assistant", "content": result.content})
                                
                            else:  # Standard mode
                                if response := ollama_chat.query({"inputs": prompt}):
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                else:
                                    st.error("Failed to get response")
                                    
                        except Exception as e:
                            st.error(f"Error in {reasoning_mode} mode: {str(e)}")
                            # Fallback to standard mode
                            if response := ollama_chat.query({"inputs": prompt}):
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-save session after response
        if "current_session" in st.session_state:
            auto_save_current_session()

        st.rerun()

# Main Function
def main():
    """Main application entry point"""
    
    # Clean up audio files on app start
    if "audio_cleanup_done" not in st.session_state:
        cleanup_audio_files()
        st.session_state.audio_cleanup_done = True

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize document processor
    doc_processor = DocumentProcessor()

    # Enhanced chat interface
    enhanced_chat_interface(doc_processor)

    with st.sidebar:
        st.header("📚 Documents")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload documents to analyze",
        )
        if uploaded_file:
            try:
                result = doc_processor.process_file(uploaded_file)
                st.success(f"Document uploaded successfully! {result}")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Render session management sidebar
    render_session_management_sidebar()

if __name__ == "__main__":
    main()