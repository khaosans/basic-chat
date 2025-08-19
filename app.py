import streamlit as st
from config import (
    APP_TITLE,
    FAVICON_PATH,
    DEFAULT_MODEL,
    VISION_MODEL,
    REASONING_MODES,
    DEFAULT_REASONING_MODE
)
# Must be first Streamlit command
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=FAVICON_PATH,
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

# Import Ollama API functions
from ollama_api import get_available_models

# Import enhanced tools
from utils.enhanced_tools import text_to_speech, get_professional_audio_html, get_audio_file_size, cleanup_audio_files

# Import AI validation system
from ai_validator import AIValidator, ValidationLevel, ValidationMode, ValidationResult

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

def display_message_content(content: str, max_chunk_size: int = 8000):
    """
    Display message content in chunks to prevent truncation.
    Uses best practices for handling large text content in Streamlit.
    """
    if not content:
        return
    
    # Clean the content
    content = content.strip()
    
    # If content is small enough, display normally
    if len(content) <= max_chunk_size:
        try:
            st.markdown(content, unsafe_allow_html=False)
        except Exception as e:
            # Fallback to text display
            st.text(content)
        return
    
    # For large content, split into manageable chunks
    try:
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, display current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                st.markdown(current_chunk, unsafe_allow_html=False)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Display remaining content
        if current_chunk:
            st.markdown(current_chunk, unsafe_allow_html=False)
            
    except Exception as e:
        # Ultimate fallback - display as text in chunks
        st.error(f"Error displaying content: {e}")
        for i in range(0, len(content), max_chunk_size):
            chunk = content[i:i + max_chunk_size]
            st.text(chunk)
            if i + max_chunk_size < len(content):
                st.markdown("---")

def display_reasoning_process(thought_process: str, max_chunk_size: int = 6000):
    """
    Display reasoning process with proper formatting and chunking.
    """
    if not thought_process or not thought_process.strip():
        return
    
    try:
        # Clean and format the thought process
        cleaned_process = thought_process.strip()
        
        # If it's small enough, display in expander
        if len(cleaned_process) <= max_chunk_size:
            with st.expander("💭 Reasoning Process", expanded=False):
                st.markdown(cleaned_process, unsafe_allow_html=False)
        else:
            # For large reasoning processes, show in multiple expanders
            paragraphs = cleaned_process.split('\n\n')
            current_chunk = ""
            chunk_count = 1
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    with st.expander(f"💭 Reasoning Process (Part {chunk_count})", expanded=False):
                        st.markdown(current_chunk, unsafe_allow_html=False)
                    current_chunk = paragraph
                    chunk_count += 1
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Display remaining content
            if current_chunk:
                with st.expander(f"💭 Reasoning Process (Part {chunk_count})", expanded=False):
                    st.markdown(current_chunk, unsafe_allow_html=False)
                    
    except Exception as e:
        st.error(f"Error displaying reasoning process: {e}")
        with st.expander("💭 Reasoning Process (Raw)", expanded=False):
            st.text(thought_process)

def display_validation_result(validation_result: ValidationResult, message_id: str):
    """
    Display AI validation results with interactive options.
    """
    if not validation_result:
        return
    
    # Create expander for validation details
    with st.expander(f"🔍 AI Self-Check (Quality: {validation_result.quality_score:.1%})", expanded=False):
        # Quality score with color coding
        col1, col2 = st.columns([1, 3])
        with col1:
            if validation_result.quality_score >= 0.8:
                st.success(f"Quality: {validation_result.quality_score:.1%}")
            elif validation_result.quality_score >= 0.6:
                st.warning(f"Quality: {validation_result.quality_score:.1%}")
            else:
                st.error(f"Quality: {validation_result.quality_score:.1%}")
        
        with col2:
            st.caption(validation_result.validation_notes)
        
        # Display issues if any
        if validation_result.issues:
            st.markdown("**Issues Detected:**")
            for issue in validation_result.issues:
                severity_color = {
                    "critical": "🚨",
                    "high": "⚠️", 
                    "medium": "📝",
                    "low": "ℹ️"
                }
                icon = severity_color.get(issue.severity, "📝")
                
                with st.container():
                    st.markdown(f"{icon} **{issue.issue_type.value.replace('_', ' ').title()}** ({issue.severity})")
                    st.caption(f"Location: {issue.location}")
                    st.write(issue.description)
                    if issue.suggested_fix:
                        st.info(f"💡 Suggested fix: {issue.suggested_fix}")
                    st.divider()
        
        # Show improved output if available
        if validation_result.improved_output and validation_result.improved_output != validation_result.original_output:
            st.markdown("**✨ Improved Version Available**")
            
            # Option to use improved version
            if st.button(f"Use Improved Version", key=f"use_improved_{message_id}"):
                # Find and update the message in session state
                for i, msg in enumerate(st.session_state.messages):
                    if msg.get("role") == "assistant" and hash(msg.get("content", "")) == int(message_id):
                        st.session_state.messages[i]["content"] = validation_result.improved_output
                        st.session_state.messages[i]["was_improved"] = True
                        st.rerun()
                        break
            
            # Option to compare versions
            if st.checkbox(f"Compare Versions", key=f"compare_{message_id}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.text_area("original", validation_result.original_output, height=200, disabled=True, label_visibility="collapsed")
                with col2:
                    st.markdown("**Improved:**")
                    st.text_area("improved", validation_result.improved_output, height=200, disabled=True, label_visibility="collapsed")
        
        # Performance metrics
        st.caption(f"Validation completed in {validation_result.processing_time:.2f}s using {validation_result.validation_level.value} level")

def apply_ai_validation(content: str, question: str, context: str) -> ValidationResult:
    """Apply AI validation to content if enabled"""
    if not st.session_state.validation_enabled:
        return None
    
    try:
        validator = st.session_state.ai_validator
        return validator.validate_output(
            output=content,
            original_question=question,
            context=context,
            validation_level=st.session_state.validation_level
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None

def enhanced_chat_interface(doc_processor):
    """Enhanced chat interface with reasoning modes and document processing"""
    
    # Initialize session state for reasoning mode if not exists
    if "reasoning_mode" not in st.session_state:
        st.session_state.reasoning_mode = "Auto"
    
    # Initialize conversation context
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    
    def build_conversation_context(messages, max_messages=10):
        """Build conversation context from recent messages"""
        if not messages:
            return ""
        
        # Get recent messages (excluding the current user message)
        recent_messages = messages[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            if msg.get("role") == "user":
                context_parts.append(f"User: {msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                # For assistant messages, include the main content
                content = msg.get('content', '')
                if msg.get("message_type") == "reasoning":
                    # For reasoning messages, include the reasoning mode info
                    reasoning_mode = msg.get("reasoning_mode", "")
                    if reasoning_mode:
                        context_parts.append(f"Assistant ({reasoning_mode}): {content}")
                    else:
                        context_parts.append(f"Assistant: {content}")
                else:
                    context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)
    
    # Initialize deep research mode
    if "deep_research_mode" not in st.session_state:
        st.session_state.deep_research_mode = False
    
    # Initialize AI validation settings
    if "validation_enabled" not in st.session_state:
        st.session_state.validation_enabled = True
    if "validation_level" not in st.session_state:
        st.session_state.validation_level = ValidationLevel.STANDARD
    if "validation_mode" not in st.session_state:
        st.session_state.validation_mode = ValidationMode.ADVISORY
    # Initialize AI validator (will be created when selected_model is available)
    
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
    
    # Sidebar Configuration - ChatGPT-style Clean Design
    with st.sidebar:
        # App Header - Modern and Clean
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <h1 style="color: #1f2937; margin: 0; font-size: 1.5rem;">🤖 BasicChat</h1>
            <p style="color: #6b7280; margin: 0.25rem 0 0 0; font-size: 0.875rem;">AI Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Status - Compact
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Model:** `{st.session_state.selected_model}`")
            with col2:
                st.markdown(f"**Mode:** `{st.session_state.reasoning_mode}`")
        
        st.divider()
        
        # Reasoning Mode - Clean Dropdown
        st.markdown("**🧠 Reasoning Mode**")
        reasoning_mode = st.selectbox(
            "reasoning_mode",
            options=REASONING_MODES,
            index=REASONING_MODES.index(st.session_state.reasoning_mode),
            help="Choose reasoning approach",
            label_visibility="collapsed"
        )
        
        # Update session state if mode changed
        if reasoning_mode != st.session_state.reasoning_mode:
            st.session_state.reasoning_mode = reasoning_mode
            st.rerun()
        
        # Compact mode info
        mode_info = {
            "Auto": "Automatically selects the best approach",
            "Standard": "Direct conversation",
            "Chain-of-Thought": "Step-by-step reasoning",
            "Multi-Step": "Complex problem solving",
            "Agent-Based": "Tool-using assistant"
        }
        
        st.caption(mode_info.get(reasoning_mode, "Standard mode"))
        
        st.divider()
        
        # Task Status - Ultra Compact
        if config.enable_background_tasks:
            st.markdown("**📊 Tasks**")
            metrics = st.session_state.task_manager.get_task_metrics()
            
            # Single line metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active", metrics.get("active", 0), label_visibility="collapsed")
            with col2:
                st.metric("Done", metrics.get("completed", 0), label_visibility="collapsed")
            with col3:
                st.metric("Total", metrics.get("total", 0), label_visibility="collapsed")
            
            # Active tasks - very compact
            active_tasks = st.session_state.task_manager.get_active_tasks()
            if active_tasks:
                st.caption("🔄 Running tasks")
                for task in active_tasks[:2]:
                    # Handle different task status attributes safely
                    task_type = getattr(task, 'task_type', getattr(task, 'type', 'task'))
                    st.caption(f"• {task_type}")
            
            st.divider()
        
        # Document Upload - Clean
        st.markdown("**📚 Documents**")
        uploaded_file = st.file_uploader(
            "document_upload",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload document to analyze",
            label_visibility="collapsed"
        )

        # Handle file upload processing (keeping existing logic)
        if uploaded_file and uploaded_file.file_id != st.session_state.get("processed_file_id"):
            logger.info(f"Processing new document: {uploaded_file.name}")
            
            if config.enable_background_tasks and uploaded_file.size > 1024 * 1024:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                task_id = st.session_state.task_manager.submit_task(
                    "document_processing",
                    file_path=temp_file_path,
                    file_type=uploaded_file.type,
                    file_size=uploaded_file.size
                )
                task_message = create_task_message(task_id, "Document Processing", 
                                                 file_name=uploaded_file.name)
                st.session_state.messages.append(task_message)
                st.session_state.processed_file_id = uploaded_file.file_id
                st.success(f"🚀 Processing {uploaded_file.name}...")
                st.rerun()
            else:
                try:
                    doc_processor.process_file(uploaded_file)
                    st.session_state.processed_file_id = uploaded_file.file_id
                    st.success(f"✅ {uploaded_file.name} processed!")
                except Exception as e:
                    logger.error(f"Error processing document '{uploaded_file.name}': {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    logger.error(f"File details - Name: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {len(uploaded_file.getvalue())} bytes")
                    
                    try:
                        logger.info(f"Document processor state: {len(doc_processor.processed_files)} processed files")
                        logger.info(f"ChromaDB client status: {doc_processor.client is not None}")
                        logger.info(f"Embeddings model: {doc_processor.embeddings.model}")
                    except Exception as diag_error:
                        logger.error(f"Error during diagnostics: {diag_error}")
                    
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processed_file_id = uploaded_file.file_id

        # Show processed files - compact
        processed_files = doc_processor.get_processed_files()
        if processed_files:
            for file_data in processed_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📄 {file_data['name']}")
                with col2:
                    if st.button("×", key=f"delete_{file_data['name']}", help="Remove", use_container_width=True):
                        doc_processor.remove_file(file_data['name'])
                        st.rerun()
        
        st.divider()
        
        # AI Validation Settings
        st.markdown("**🔍 AI Validation**")
        
        # Validation toggle
        validation_enabled = st.toggle(
            "Enable AI Self-Check",
            value=st.session_state.validation_enabled,
            help="AI will validate and potentially improve its own responses"
        )
        if validation_enabled != st.session_state.validation_enabled:
            st.session_state.validation_enabled = validation_enabled
            st.rerun()
        
        if st.session_state.validation_enabled:
            # Validation level
            validation_level = st.selectbox(
                "Validation Level",
                options=[ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE],
                index=1,  # Default to STANDARD
                format_func=lambda x: {
                    ValidationLevel.BASIC: "Basic",
                    ValidationLevel.STANDARD: "Standard", 
                    ValidationLevel.COMPREHENSIVE: "Comprehensive"
                }[x],
                help="How thorough the validation should be"
            )
            if validation_level != st.session_state.validation_level:
                st.session_state.validation_level = validation_level
                st.rerun()
            
            # Validation mode
            validation_mode = st.selectbox(
                "Validation Mode",
                options=[ValidationMode.ADVISORY, ValidationMode.AUTO_FIX],
                index=0,  # Default to ADVISORY
                format_func=lambda x: {
                    ValidationMode.ADVISORY: "Advisory (Show Issues)",
                    ValidationMode.AUTO_FIX: "Auto-Fix (Use Improved)"
                }[x],
                help="How to handle validation results"
            )
            if validation_mode != st.session_state.validation_mode:
                st.session_state.validation_mode = validation_mode
                st.rerun()
        
        st.divider()
        
        # Development Tools - Minimal
        if st.button("🗄️ Reset", help="Clear all data", use_container_width=True):
            try:
                from document_processor import DocumentProcessor
                DocumentProcessor.cleanup_all_chroma_directories()
                if "task_manager" in st.session_state:
                    st.session_state.task_manager.cleanup_old_tasks(max_age_hours=1)
                st.success("✅ Reset complete!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Initialize reasoning components
    selected_model = st.session_state.selected_model
    ollama_chat = OllamaChat(selected_model)
    tool_registry = ToolRegistry(doc_processor)
    reasoning_chain = ReasoningChain(selected_model)
    multi_step = MultiStepReasoning(selected_model)
    reasoning_agent = ReasoningAgent(selected_model)
    
    # Initialize AI validator with the selected model
    if "ai_validator" not in st.session_state:
        st.session_state.ai_validator = AIValidator(selected_model)
    
    # Initialize welcome message if needed
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your AI assistant with enhanced reasoning capabilities. How can I help you today?",
            "message_type": "welcome"
        }]

    # Main Chat Area - ChatGPT Style with Design Rules
    st.markdown("""
    <style>
    /* Global ChatGPT-style theme with improved contrast */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Message styling */
    .message-container {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    .user-message {
        background-color: #f7f7f8;
        margin-left: 2rem;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background-color: #ffffff;
        margin-right: 2rem;
        max-width: 70%;
        float: left;
        clear: both;
        border: 1px solid #e5e5e5;
    }
    .message-avatar {
        width: 30px;
        height: 30px;
        border-radius: 2px;
        margin-right: 12px;
        float: left;
    }
    .message-content {
        overflow: hidden;
        padding: 12px 16px;
        line-height: 1.5;
        color: #1f2937;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 4px;
    }
    
    /* Improved contrast for buttons and UI elements */
    .stButton > button {
        background-color: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover {
        background-color: #0d8f6c !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Toggle styling - Enhanced for better visibility */
    .stCheckbox > label {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    /* Toggle switch styling */
    .stToggle > label {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    /* File uploader styling in sidebar */
    .css-1d391kg .stFileUploader > div {
        background-color: #ffffff !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .css-1d391kg .stFileUploader > div:hover {
        border-color: #10a37f !important;
        background-color: #f0f9ff !important;
    }
    
    /* Metrics styling in sidebar */
    .css-1d391kg .stMetric {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
    }
    
    .css-1d391kg .stMetric > div {
        color: #1f2937 !important;
    }
    
    /* Selectbox styling - Enhanced for better visibility */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        min-height: 40px !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #10a37f !important;
        box-shadow: 0 2px 6px rgba(16, 163, 127, 0.2) !important;
    }
    
    /* Comprehensive dropdown text visibility fix */
    .stSelectbox * {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Target all possible dropdown text elements */
    .stSelectbox [data-baseweb="select"] *,
    .stSelectbox [data-testid="stSelectbox"] *,
    .stSelectbox [role="combobox"] *,
    .stSelectbox [role="listbox"] *,
    .stSelectbox [role="option"] * {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Specific targeting for the selected value display */
    .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
    .stSelectbox [data-baseweb="select"] [role="combobox"],
    .stSelectbox [data-baseweb="select"] [role="listbox"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
    }
    
    /* Target all text elements within dropdowns */
    .stSelectbox span,
    .stSelectbox div,
    .stSelectbox p {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Dropdown options styling */
    .stSelectbox [data-baseweb="select"] [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid #f3f4f6 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    .stSelectbox [data-baseweb="select"] [role="option"]:hover {
        background-color: #f0f9ff !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .stSelectbox [data-baseweb="select"] [role="option"][aria-selected="true"] {
        background-color: #10a37f !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Dropdown container */
    .stSelectbox [data-baseweb="popover"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Dropdown arrow icon */
    .stSelectbox [data-baseweb="select"] svg {
        color: #6b7280 !important;
    }
    
    /* Force all text in selectboxes to be visible */
    .stSelectbox {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #f7f7f8 !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
    }
    
    /* Audio player styling */
    .stAudio {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    
    /* Sidebar styling - Enhanced for better visibility */
    .css-1d391kg {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg .stMarkdown {
        color: #1f2937 !important;
    }
    
    .css-1d391kg .stMarkdown strong {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar dividers */
    .css-1d391kg hr {
        border-color: #d1d5db !important;
        margin: 1rem 0 !important;
    }
    
    /* Sidebar captions */
    .css-1d391kg .stCaption {
        color: #6b7280 !important;
        font-size: 0.875rem !important;
    }
    
    /* Sidebar buttons */
    .css-1d391kg .stButton > button {
        background-color: #ef4444 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-size: 0.75rem !important;
        padding: 2px 6px !important;
        min-height: auto !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background-color: #dc2626 !important;
    }
    
    /* Success/Info/Error messages */
    .stSuccess {
        background-color: #d1e7dd !important;
        border: 1px solid #badbcc !important;
        color: #0f5132 !important;
    }
    
    .stInfo {
        background-color: #cff4fc !important;
        border: 1px solid #b6effb !important;
        color: #055160 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border: 1px solid #f5c2c7 !important;
        color: #842029 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat Messages Container - ChatGPT Style
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages with ChatGPT styling
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                # User message - right aligned, blue background
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 1rem 0; clear: both;">
                    <div style="background: #007AFF; color: white; padding: 12px 16px; border-radius: 18px; max-width: 70%; margin-left: 4rem; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant message - left aligned with avatar
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("""
                        <div style="width: 30px; height: 30px; background: #10a37f; border-radius: 2px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                            G
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        # Robust message display with chunking to prevent truncation
                        try:
                            # Always display the main content first using chunking
                            if msg.get("content"):
                                display_message_content(msg["content"])
                            
                            # Add optional reasoning info if available
                            if msg.get("reasoning_mode"):
                                st.caption(f"🤖 Reasoning: {msg['reasoning_mode']}")
                            
                            # Add optional tool info if available  
                            if msg.get("tool_name"):
                                st.caption(f"🛠️ Tool: {msg['tool_name']}")
                                
                            # Add expandable reasoning process if available using chunking
                            if msg.get("thought_process") and msg["thought_process"].strip():
                                display_reasoning_process(msg["thought_process"])
                            
                            # Add validation results if available
                            if msg.get("validation_result"):
                                display_validation_result(msg["validation_result"], str(hash(msg.get("content", ""))))
                        except Exception as e:
                            # Fallback display if anything fails
                            st.error(f"Error displaying message: {e}")
                            st.text(f"Raw content: {msg.get('content', 'No content')}")
                
                # Handle task messages
                if msg.get("is_task"):
                    task_id = msg.get("task_id")
                    if task_id:
                        task_status = st.session_state.task_manager.get_task_status(task_id)
                        if task_status:
                            if task_status.status == "completed":
                                display_task_result(task_status)
                            elif task_status.status == "failed":
                                st.error(f"Task failed: {task_status.error}")
                            else:
                                display_task_status(task_id, st.session_state.task_manager, "message_loop")
                
                # Add audio button for assistant messages
                if not msg.get("is_task"):
                    create_enhanced_audio_button(msg["content"], hash(msg['content']))

    # Chat Input - ChatGPT Style
    st.markdown("""
    <style>
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 900px;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e5e5e5;
        z-index: 1000;
    }
    .input-wrapper {
        display: flex;
        align-items: center;
        background: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 24px;
        padding: 8px 16px;
        margin: 0 1rem;
    }
    .input-field {
        flex: 1;
        border: none;
        background: transparent;
        outline: none;
        font-size: 16px;
        line-height: 1.5;
        padding: 8px 0;
    }
    .send-button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask anything..."):
        # Add user message to session state with standardized schema
        user_message = {
            "role": "user",
            "content": prompt,
            "message_type": "user"
        }
        st.session_state.messages.append(user_message)
        
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
            
            # User message already added above
            
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
            
            # User message already added above
            
            # Display task message
            with st.chat_message("assistant"):
                st.write(task_message["content"])
                display_task_status(task_id, st.session_state.task_manager, "new_task")
            
            st.rerun()
        else:
            # Process normally with enhanced UI
            # User message already added above
            
            with st.chat_message("assistant"):
                tool = tool_registry.get_tool(prompt)
                if tool:
                    with st.spinner(f"Using {tool.name()}..."):
                        response = tool.execute(prompt)
                        if response.success:
                            # Add standardized message
                            message = {
                                "role": "assistant", 
                                "content": response.content,
                                "message_type": "tool",
                                "tool_name": tool.name()
                            }
                            st.session_state.messages.append(message)
                            st.rerun()
                else:
                    with st.spinner(f"Thinking with {st.session_state.reasoning_mode} reasoning..."):
                        try:
                            context = doc_processor.get_relevant_context(prompt) if doc_processor else ""
                            enhanced_prompt = prompt
                            if context:
                                enhanced_prompt = f"Context from uploaded documents:\n{context}\n\nQuestion: {prompt}"
                            
                            if st.session_state.reasoning_mode == "Chain-of-Thought":
                                try:
                                    # Build conversation context
                                    conversation_context = build_conversation_context(st.session_state.messages)
                                    # Combine contexts safely
                                    if context and conversation_context:
                                        full_context = f"Document Context:\n{context}\n\nConversation History:\n{conversation_context}"
                                    elif context:
                                        full_context = context
                                    elif conversation_context:
                                        full_context = conversation_context
                                    else:
                                        full_context = ""
                                    
                                    result = reasoning_chain.execute_reasoning(question=prompt, context=full_context)
                                    
                                    # Apply AI validation if enabled
                                    content_to_use = result.final_answer or "No response generated"
                                    validation_result = apply_ai_validation(content_to_use, prompt, full_context)
                                    
                                    # Use improved content if auto-fix mode and improvement available
                                    if (validation_result and 
                                        st.session_state.validation_mode == ValidationMode.AUTO_FIX and 
                                        validation_result.improved_output):
                                        content_to_use = validation_result.improved_output
                                    
                                    # Create robust message
                                    message = {
                                        "role": "assistant", 
                                        "content": content_to_use,
                                        "reasoning_mode": getattr(result, 'reasoning_mode', 'Chain-of-Thought'),
                                        "thought_process": getattr(result, 'thought_process', ''),
                                        "message_type": "reasoning",
                                        "validation_result": validation_result
                                    }
                                    st.session_state.messages.append(message)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Chain-of-Thought reasoning failed: {e}")
                                    # Fallback to simple response
                                    fallback_message = {
                                        "role": "assistant",
                                        "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                                        "message_type": "error"
                                    }
                                    st.session_state.messages.append(fallback_message)
                                    st.rerun()
                                
                            elif st.session_state.reasoning_mode == "Multi-Step":
                                try:
                                    conversation_context = build_conversation_context(st.session_state.messages)
                                    full_context = context + "\n" + conversation_context if context or conversation_context else ""
                                    
                                    result = multi_step.step_by_step_reasoning(query=prompt, context=full_context)
                                    
                                    # Apply AI validation if enabled
                                    content_to_use = result.final_answer or "No response generated"
                                    validation_result = apply_ai_validation(content_to_use, prompt, full_context)
                                    
                                    # Use improved content if auto-fix mode and improvement available
                                    if (validation_result and 
                                        st.session_state.validation_mode == ValidationMode.AUTO_FIX and 
                                        validation_result.improved_output):
                                        content_to_use = validation_result.improved_output
                                    
                                    message = {
                                        "role": "assistant", 
                                        "content": content_to_use,
                                        "reasoning_mode": getattr(result, 'reasoning_mode', 'Multi-Step'),
                                        "thought_process": getattr(result, 'thought_process', ''),
                                        "message_type": "reasoning",
                                        "validation_result": validation_result
                                    }
                                    st.session_state.messages.append(message)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Multi-Step reasoning failed: {e}")
                                    fallback_message = {
                                        "role": "assistant",
                                        "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                                        "message_type": "error"
                                    }
                                    st.session_state.messages.append(fallback_message)
                                    st.rerun()
                                
                            elif st.session_state.reasoning_mode == "Agent-Based":
                                try:
                                    conversation_context = build_conversation_context(st.session_state.messages)
                                    full_context = context + "\n" + conversation_context if context or conversation_context else ""
                                    
                                    result = reasoning_agent.run(query=prompt, context=full_context)
                                    
                                    # Apply AI validation if enabled
                                    content_to_use = result.final_answer or "No response generated"
                                    validation_result = apply_ai_validation(content_to_use, prompt, full_context)
                                    
                                    # Use improved content if auto-fix mode and improvement available
                                    if (validation_result and 
                                        st.session_state.validation_mode == ValidationMode.AUTO_FIX and 
                                        validation_result.improved_output):
                                        content_to_use = validation_result.improved_output
                                    
                                    message = {
                                        "role": "assistant", 
                                        "content": content_to_use,
                                        "reasoning_mode": getattr(result, 'reasoning_mode', 'Agent-Based'),
                                        "thought_process": getattr(result, 'thought_process', ''),
                                        "message_type": "reasoning",
                                        "validation_result": validation_result
                                    }
                                    st.session_state.messages.append(message)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Agent-Based reasoning failed: {e}")
                                    fallback_message = {
                                        "role": "assistant",
                                        "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                                        "message_type": "error"
                                    }
                                    st.session_state.messages.append(fallback_message)
                                    st.rerun()
                                
                            elif st.session_state.reasoning_mode == "Auto":
                                try:
                                    auto_reasoning = AutoReasoning(selected_model)
                                    conversation_context = build_conversation_context(st.session_state.messages)
                                    full_context = context + "\n" + conversation_context if context or conversation_context else ""
                                    
                                    result = auto_reasoning.auto_reason(query=prompt, context=full_context)
                                    
                                    # Apply AI validation if enabled
                                    content_to_use = result.final_answer or "No response generated"
                                    validation_result = apply_ai_validation(content_to_use, prompt, full_context)
                                    
                                    # Use improved content if auto-fix mode and improvement available
                                    if (validation_result and 
                                        st.session_state.validation_mode == ValidationMode.AUTO_FIX and 
                                        validation_result.improved_output):
                                        content_to_use = validation_result.improved_output
                                    
                                    message = {
                                        "role": "assistant", 
                                        "content": content_to_use,
                                        "reasoning_mode": getattr(result, 'reasoning_mode', 'Auto'),
                                        "thought_process": getattr(result, 'thought_process', ''),
                                        "message_type": "reasoning",
                                        "validation_result": validation_result
                                    }
                                    st.session_state.messages.append(message)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Auto reasoning failed: {e}")
                                    fallback_message = {
                                        "role": "assistant",
                                        "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                                        "message_type": "error"
                                    }
                                    st.session_state.messages.append(fallback_message)
                                    st.rerun()
                                
                            else:  # Standard mode
                                try:
                                    conversation_context = build_conversation_context(st.session_state.messages)
                                    enhanced_prompt_with_context = f"{enhanced_prompt}\n\nConversation History:\n{conversation_context}"
                                    
                                    response = ollama_chat.query({"inputs": enhanced_prompt_with_context})
                                    
                                    if response and response.strip():
                                        # Apply AI validation if enabled
                                        content_to_use = response.strip()
                                        validation_result = apply_ai_validation(content_to_use, prompt, enhanced_prompt_with_context)
                                        
                                        # Use improved content if auto-fix mode and improvement available
                                        if (validation_result and 
                                            st.session_state.validation_mode == ValidationMode.AUTO_FIX and 
                                            validation_result.improved_output):
                                            content_to_use = validation_result.improved_output
                                        
                                        message = {
                                            "role": "assistant", 
                                            "content": content_to_use,
                                            "message_type": "standard",
                                            "validation_result": validation_result
                                        }
                                        st.session_state.messages.append(message)
                                        st.rerun()
                                    else:
                                        st.error("Failed to get response from the model")
                                except Exception as e:
                                    st.error(f"Standard mode failed: {e}")
                                    fallback_message = {
                                        "role": "assistant",
                                        "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                                        "message_type": "error"
                                    }
                                    st.session_state.messages.append(fallback_message)
                                    st.rerun()
                                    
                        except Exception as e:
                            logger.error(f"Error in {st.session_state.reasoning_mode} mode: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            st.error(f"Error in {st.session_state.reasoning_mode} mode: {str(e)}")
                            if response := ollama_chat.query({"inputs": prompt}):
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Audio buttons are automatically created for all assistant messages in the message display loop

    # Deep Research Mode Toggle - Below chat input modal
    st.markdown("---")
    
    # Center the toggle below the chat input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        deep_research_toggle = st.toggle(
            "🔬 Deep Research Mode",
            value=st.session_state.deep_research_mode,
            help="Enable comprehensive research with multiple sources"
        )
        
        if deep_research_toggle != st.session_state.deep_research_mode:
            st.session_state.deep_research_mode = deep_research_toggle
            if deep_research_toggle:
                st.success("🔬 Deep Research enabled")
            else:
                st.info("💬 Standard mode")
            st.rerun()

# Main Function
def main():
    """Main application entry point"""
    # st.set_page_config(  # <-- REMOVE THIS BLOCK
    #     page_title=APP_TITLE,
    #     page_icon=FAVICON_PATH,
    #     layout="wide"
    # )

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

if __name__ == "__main__":
    main()
