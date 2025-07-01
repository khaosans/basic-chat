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
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
import tempfile
from gtts import gTTS
import hashlib
import base64

# Import our new reasoning engine
from reasoning_engine import (
    ReasoningAgent, 
    ReasoningChain, 
    MultiStepReasoning, 
    AutoReasoning,
    ReasoningResult
)

# Import new async components
from config import config
from utils.async_ollama import AsyncOllamaChat, async_chat
from utils.caching import response_cache

# Import the proper DocumentProcessor with vector database support
from document_processor import DocumentProcessor, ProcessedFile

# Import task management components
from task_manager import TaskManager
from task_ui import (
    display_task_status, 
    create_task_message, 
    display_task_result,
    display_task_metrics,
    display_active_tasks,
    should_use_background_task,
    create_deep_research_message
)

# Import configuration constants
from config import (
    APP_TITLE,
    FAVICON_PATH,
    DEFAULT_MODEL,
    VISION_MODEL,
    REASONING_MODES,
    DEFAULT_REASONING_MODE
)

# Import Ollama API functions
from ollama_api import get_available_models

# Import enhanced tools
from utils.enhanced_tools import text_to_speech, get_professional_audio_html, get_audio_file_size, cleanup_audio_files

load_dotenv(".env.local")  # Load environment variables from .env.local

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

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
                logger.warning(f"Async query failed, falling back to sync: {e}")
                self._use_sync_fallback = True
        
        # Fallback to original sync implementation
        return self._query_sync(payload)
    
    async def _query_async(self, payload: Dict) -> Optional[str]:
        """Async query implementation"""
        try:
            return await self.async_chat.query(payload)
        except Exception as e:
            logger.error(f"Async query error: {e}")
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
                logger.debug(f"Making Ollama API request (attempt {attempt + 1}/{max_retries})")
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
                            logger.debug(f"JSONDecodeError: {chunk}")
                            continue
                return full_response
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return None
            except Exception as e:
                logger.error(f"Error processing Ollama response: {e}")
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
                logger.warning(f"Async stream failed, falling back to sync: {e}")
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
            logger.error(f"Error in stream query: {e}")
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
            # Small button positioned towards the right
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col3:
                if st.button(
                    "🔊",
                    key=f"audio_btn_{message_key}",
                    help="Click to generate audio version of this message",
                    use_container_width=False
                ):
                    # Set loading state immediately
                    audio_state["status"] = "loading"
                    st.rerun()
        
        elif audio_state["status"] == "loading":
            # Show loading state with disabled button
            col1, col2, col3 = st.columns([3, 1, 0.5])
            with col3:
                # Disabled button with loading indicator
                st.button(
                    "⏳",
                    key=f"audio_btn_{message_key}",
                    help="Generating audio...",
                    use_container_width=False,
                    disabled=True
                )
            
            # Generate audio in the background
            try:
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

def enhanced_chat_interface(doc_processor):
    """Enhanced chat interface with reasoning modes and document processing"""
    
    # Initialize session state for reasoning mode if not exists
    if "reasoning_mode" not in st.session_state:
        st.session_state.reasoning_mode = "Auto"
    
    # Initialize deep research mode
    if "deep_research_mode" not in st.session_state:
        st.session_state.deep_research_mode = False
    
    # Initialize last refresh time
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = 0
    
    # Auto-refresh for active tasks (every 3 seconds)
    import time
    current_time = time.time()
    active_tasks = st.session_state.task_manager.get_active_tasks()
    running_tasks = [task for task in active_tasks if task.status in ["pending", "running"]]
    
    if running_tasks and (current_time - st.session_state.last_refresh_time) > 3:
        st.session_state.last_refresh_time = current_time
        st.rerun()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("✨ Configuration")
        
        # Reasoning Mode Selection
        reasoning_mode = st.selectbox(
            "🧠 Reasoning Mode",
            options=REASONING_MODES,
            index=REASONING_MODES.index(st.session_state.reasoning_mode),
            help="Choose how the AI should approach your questions"
        )
        
        # Update session state if mode changed
        if reasoning_mode != st.session_state.reasoning_mode:
            st.session_state.reasoning_mode = reasoning_mode
            st.rerun()
        
        st.info(f"""
        - **Active Model**: `{st.session_state.selected_model}`
        - **Reasoning Mode**: `{st.session_state.reasoning_mode}`
        """)

        st.markdown("---")
        
        # --- Task Management ---
        if config.enable_background_tasks:
            display_task_metrics(st.session_state.task_manager)
            display_active_tasks(st.session_state.task_manager)
            st.markdown("---")
        
        # --- Document Management ---
        st.header("📚 Documents")
        
        uploaded_file = st.file_uploader(
            "Upload a document to analyze",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload a document to chat with it.",
            key="document_uploader"
        )

        # Handle file upload processing
        if uploaded_file and uploaded_file.file_id != st.session_state.get("processed_file_id"):
            logger.info(f"Processing new document: {uploaded_file.name}")
            
            # Check if this should be a background task
            if config.enable_background_tasks and uploaded_file.size > 1024 * 1024:  # > 1MB
                import tempfile, os
                # Save uploaded file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                # Submit as background task
                task_id = st.session_state.task_manager.submit_task(
                    "document_processing",
                    file_path=temp_file_path,
                    file_type=uploaded_file.type,
                    file_size=uploaded_file.size
                )
                # Add task message
                task_message = create_task_message(task_id, "Document Processing", 
                                                 file_name=uploaded_file.name)
                st.session_state.messages.append(task_message)
                # Update session state to mark as processed
                st.session_state.processed_file_id = uploaded_file.file_id
                st.success(f"🚀 Document '{uploaded_file.name}' submitted for background processing!")
                st.rerun()
            else:
                # Process immediately
                try:
                    # Process the uploaded file
                    doc_processor.process_file(uploaded_file)
                    
                    # Update session state to mark as processed
                    st.session_state.processed_file_id = uploaded_file.file_id
                    
                    # Show success message
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    
                except Exception as e:
                    logger.error(f"Error processing document '{uploaded_file.name}': {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    logger.error(f"File details - Name: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {len(uploaded_file.getvalue())} bytes")
                    
                    # Log additional diagnostic information
                    try:
                        logger.info(f"Document processor state: {len(doc_processor.processed_files)} processed files")
                        logger.info(f"ChromaDB client status: {doc_processor.client is not None}")
                        logger.info(f"Embeddings model: {doc_processor.embeddings.model}")
                    except Exception as diag_error:
                        logger.error(f"Error during diagnostics: {diag_error}")
                    
                    st.error(f"❌ Error processing document: {str(e)}")
                    # Also mark as processed on error to prevent reprocessing loop
                    st.session_state.processed_file_id = uploaded_file.file_id

        processed_files = doc_processor.get_processed_files()
        if processed_files:
            st.subheader("📋 Processed Documents")
            for file_data in processed_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"• {file_data['name']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{file_data['name']}", help="Remove document"):
                        doc_processor.remove_file(file_data['name'])
                        st.rerun()
        else:
            st.info("No documents uploaded yet.")

    # Initialize reasoning components with the selected model from session state
    selected_model = st.session_state.selected_model
    
    # Create chat instances
    ollama_chat = OllamaChat(selected_model)
    tool_registry = ToolRegistry(doc_processor)
    
    # Initialize reasoning engines
    reasoning_chain = ReasoningChain(selected_model)
    multi_step = MultiStepReasoning(selected_model)
    reasoning_agent = ReasoningAgent(selected_model)
    
    # Initialize welcome message if needed
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hello! I'm your AI assistant with enhanced reasoning capabilities. Choose a reasoning mode from the sidebar and let's start exploring!"
        }]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Handle task messages
            if msg.get("is_task"):
                task_id = msg.get("task_id")
                if task_id:
                    task_status = st.session_state.task_manager.get_task_status(task_id)
                    if task_status:
                        if task_status.status == "completed":
                            # Display task result
                            display_task_result(task_status)
                        elif task_status.status == "failed":
                            st.error(f"Task failed: {task_status.error}")
                        else:
                            # Show task status
                            display_task_status(task_id, st.session_state.task_manager, "message_loop")
            
            # Add audio button for assistant messages
            if msg["role"] == "assistant" and not msg.get("is_task"):
                create_enhanced_audio_button(msg["content"], hash(msg['content']))

    # Chat input with deep research toggle
    st.markdown("---")
    
    # Deep Research Toggle (ChatGPT-style)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        deep_research_toggle = st.toggle(
            "🔬 Deep Research Mode",
            value=st.session_state.deep_research_mode,
            help="Enable comprehensive research with multiple sources and detailed analysis"
        )
        
        # Update session state if toggle changed
        if deep_research_toggle != st.session_state.deep_research_mode:
            st.session_state.deep_research_mode = deep_research_toggle
            if deep_research_toggle:
                st.info("🔬 Deep Research Mode enabled! Your queries will now trigger comprehensive research with multiple sources.")
            else:
                st.info("✅ Standard mode enabled. Switch back to deep research for comprehensive analysis.")
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type a message..."):
        # Determine if this should be a deep research task
        if st.session_state.deep_research_mode:
            # Always use deep research for complex queries in research mode
            should_be_research_task = True
        else:
            # Check if this should be a long-running task
            should_be_long_task = should_use_background_task(prompt, st.session_state.reasoning_mode, config)
            should_be_research_task = False
        
        if should_be_research_task:
            # Submit as deep research task
            task_id = st.session_state.task_manager.submit_task(
                "deep_research",
                query=prompt,
                research_depth="comprehensive"
            )
            
            # Add task message to chat
            task_message = create_deep_research_message(task_id, prompt)
            st.session_state.messages.append(task_message)
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message immediately
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display task message
            with st.chat_message("assistant"):
                st.write(task_message["content"])
                display_task_status(task_id, st.session_state.task_manager, "new_task")
            
            st.rerun()
        elif should_be_long_task:
            # Submit as background task (existing logic)
            task_id = st.session_state.task_manager.submit_task(
                "reasoning",
                query=prompt,
                mode=st.session_state.reasoning_mode
            )
            
            # Add task message to chat
            task_message = create_task_message(task_id, "Reasoning", query=prompt)
            st.session_state.messages.append(task_message)
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message immediately
            with st.chat_message("user"):
                st.write(prompt)
            
            # Display task message
            with st.chat_message("assistant"):
                st.write(task_message["content"])
                display_task_status(task_id, st.session_state.task_manager, "new_task")
            
            st.rerun()
        else:
            # Process normally (existing code)
            # Add user message to session state immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message immediately
            with st.chat_message("user"):
                st.write(prompt)
            
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
                    # Use reasoning modes with separated thought process and final output
                    with st.spinner(f"Processing with {st.session_state.reasoning_mode} reasoning..."):
                        try:
                            # Get relevant document context first
                            context = doc_processor.get_relevant_context(prompt) if doc_processor else ""
                            
                            # Add context to the prompt if available
                            enhanced_prompt = prompt
                            if context:
                                enhanced_prompt = f"Context from uploaded documents:\n{context}\n\nQuestion: {prompt}"
                            
                            if st.session_state.reasoning_mode == "Chain-of-Thought":
                                result = reasoning_chain.execute_reasoning(question=prompt, context=context)
                                
                                with st.expander("💭 Thought Process", expanded=False):
                                    # Display the thought process
                                    st.markdown(result.thought_process)
                                
                                # Show final answer separately
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.final_answer)
                                st.session_state.messages.append({"role": "assistant", "content": result.final_answer})
                                
                            elif st.session_state.reasoning_mode == "Multi-Step":
                                result = multi_step.step_by_step_reasoning(query=prompt, context=context)
                                
                                with st.expander("🔍 Analysis & Planning", expanded=False):
                                    # Display the analysis phase
                                    st.markdown(result.thought_process)
                                
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.final_answer)
                                st.session_state.messages.append({"role": "assistant", "content": result.final_answer})
                                
                            elif st.session_state.reasoning_mode == "Agent-Based":
                                result = reasoning_agent.run(query=prompt, context=context)
                                
                                with st.expander("🤖 Agent Actions", expanded=False):
                                    # Display agent actions
                                    st.markdown(result.thought_process)
                                
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.final_answer)
                                st.session_state.messages.append({"role": "assistant", "content": result.final_answer})
                                
                            elif st.session_state.reasoning_mode == "Auto":
                                auto_reasoning = AutoReasoning(selected_model)
                                result = auto_reasoning.auto_reason(query=prompt, context=context)
                                
                                # Show which mode was auto-selected
                                st.info(f"🤖 Auto-selected: **{result.reasoning_mode}** reasoning")
                                
                                with st.expander("💭 Thought Process", expanded=False):
                                    # Display the thought process
                                    st.markdown(result.thought_process)
                                
                                st.markdown("### 📝 Final Answer")
                                st.markdown(result.final_answer)
                                st.session_state.messages.append({"role": "assistant", "content": result.final_answer})
                                
                            else:  # Standard mode
                                # Note: The standard mode now also benefits from context
                                if response := ollama_chat.query({"inputs": enhanced_prompt}):
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                else:
                                    st.error("Failed to get response")
                                    
                        except Exception as e:
                            logger.error(f"Error in {st.session_state.reasoning_mode} mode: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            st.error(f"Error in {st.session_state.reasoning_mode} mode: {str(e)}")
                            # Fallback to standard mode
                            if response := ollama_chat.query({"inputs": prompt}):
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Add audio button for the assistant's response
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                create_enhanced_audio_button(st.session_state.messages[-1]["content"], hash(st.session_state.messages[-1]["content"]))

# Main Function
def main():
    """Main application entry point"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=FAVICON_PATH,
        layout="wide"
    )

    # Clean up audio files on app start
    if "audio_cleanup_done" not in st.session_state:
        cleanup_audio_files()
        st.session_state.audio_cleanup_done = True

    # Clean up old ChromaDB directories on app start
    if "chroma_cleanup_done" not in st.session_state:
        try:
            from document_processor import DocumentProcessor
            DocumentProcessor.cleanup_old_directories(max_age_hours=1)  # Clean up directories older than 1 hour
            st.session_state.chroma_cleanup_done = True
        except Exception as e:
            logger.warning(f"Failed to cleanup old ChromaDB directories: {e}")

    # Initialize document processor and session state variables
    if "doc_processor" not in st.session_state:
        logger.info("Initializing document processor")
        st.session_state.doc_processor = DocumentProcessor()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "reasoning_mode" not in st.session_state:
        st.session_state.reasoning_mode = DEFAULT_REASONING_MODE
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None
    
    # Initialize task manager if background tasks are enabled
    if config.enable_background_tasks and "task_manager" not in st.session_state:
        logger.info("Initializing task manager")
        st.session_state.task_manager = TaskManager()
        
        # Clean up old tasks periodically
        if "task_cleanup_done" not in st.session_state:
            try:
                st.session_state.task_manager.cleanup_old_tasks(max_age_hours=24)
                st.session_state.task_cleanup_done = True
            except Exception as e:
                logger.warning(f"Failed to cleanup old tasks: {e}")
        
    doc_processor = st.session_state.doc_processor

    # Enhanced chat interface
    enhanced_chat_interface(doc_processor)

    # Add cleanup buttons in sidebar for development
    with st.sidebar:
        st.markdown("---")
        st.header("🧹 Development Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗄️ Cleanup ChromaDB", help="Clean up all ChromaDB directories"):
                try:
                    from document_processor import DocumentProcessor
                    DocumentProcessor.cleanup_all_chroma_directories()
                    st.success("ChromaDB cleanup completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
        
        with col2:
            if st.button("📋 Cleanup Tasks", help="Clean up old completed tasks"):
                try:
                    if "task_manager" in st.session_state:
                        st.session_state.task_manager.cleanup_old_tasks(max_age_hours=1)
                        st.success("Task cleanup completed!")
                        st.rerun()
                    else:
                        st.warning("No task manager available")
                except Exception as e:
                    st.error(f"Task cleanup failed: {e}")

if __name__ == "__main__":
    main()
