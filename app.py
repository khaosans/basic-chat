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
from typing import Optional, Dict, List
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
    ReasoningDocumentProcessor,
    ReasoningResult
)

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
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = f"{OLLAMA_API_URL}/generate"
        self.system_prompt = SYSTEM_PROMPT

    def query(self, payload: Dict) -> Optional[str]:
        """Query the Ollama API with retry logic"""
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
    text_hash = hashlib.md5(text.encode()).hexdigest()
    audio_file = f"temp_{text_hash}.mp3"
    if not os.path.exists(audio_file):
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
    return audio_file

def autoplay_audio(file_path):
    """Autoplay audio file"""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

def get_audio_html(file_path):
    """Generate HTML for audio player with controls"""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls style="width: 100%; margin-top: 10px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md

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
    """Enhanced chat interface with reasoning capabilities"""
    # Custom CSS for chat layout
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
                background: #007bff !important;
                color: white !important;
                border-radius: 8px 8px 0 8px !important;
                padding: 0.75rem 1rem !important;
                margin: 0.25rem 0 !important;
                margin-left: auto !important;
                max-width: 80% !important;
            }
            
            /* Assistant messages */
            [data-testid="chat-message-assistant"] {
                background: #f0f2f6 !important;
                color: #333 !important;
                border-radius: 8px 8px 8px 0 !important;
                padding: 0.75rem 1rem !important;
                margin: 0.25rem 0 !important;
                margin-right: auto !important;
                max-width: 80% !important;
                border: 1px solid #e1e4e8 !important;
            }

            /* Voice button styling */
            .stButton button {
                width: auto;
                margin: 0.5rem 0 0 0;
                background: #f0f2f6;
                color: #333;
                border: 1px solid #e1e4e8;
            }

            .stButton button:hover {
                background: #e1e4e8;
                border: 1px solid #d0d7de;
            }
            
            /* Reasoning mode styling */
            .reasoning-mode {
                background: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                padding: 0.5rem;
                margin: 0.5rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize reasoning components
    reasoning_agent = ReasoningAgent(OLLAMA_MODEL)
    reasoning_chain = ReasoningChain(OLLAMA_MODEL)
    multi_step = MultiStepReasoning(doc_processor, OLLAMA_MODEL)
    
    # Create chat instances
    ollama_chat = OllamaChat(OLLAMA_MODEL)
    tool_registry = ToolRegistry(doc_processor)

    # Add reasoning mode selection in sidebar
    with st.sidebar:
        st.header("🧠 Reasoning Mode")
        reasoning_mode = st.selectbox(
            "Select Reasoning Mode",
            ["Standard", "Chain-of-Thought", "Multi-Step", "Agent-Based"],
            help="Choose how the AI should process your questions"
        )
        
        # Add model selection
        st.header("🤖 Model Selection")
        try:
            from ollama_api import get_available_models
            available_models = get_available_models()
            selected_model = st.selectbox(
                "Choose Model",
                available_models,
                index=available_models.index(OLLAMA_MODEL) if OLLAMA_MODEL in available_models else 0,
                help="Select the Ollama model to use for reasoning"
            )
            # Update the model if changed
            if selected_model != OLLAMA_MODEL:
                OLLAMA_MODEL = selected_model
        except Exception as e:
            st.warning(f"Could not fetch available models: {e}")
            selected_model = OLLAMA_MODEL
        
        # Display mode description
        mode_descriptions = {
            "Standard": "Basic chat with simple responses",
            "Chain-of-Thought": "Shows step-by-step reasoning process",
            "Multi-Step": "Breaks complex questions into multiple steps",
            "Agent-Based": "Uses tools and agents for enhanced capabilities"
        }
        
        st.info(f"**{reasoning_mode}**: {mode_descriptions[reasoning_mode]}")
        st.info(f"**Model**: {selected_model}")

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
            if msg["role"] == "assistant":
                if st.button("🔊 Play Voice", key=f"audio_{hash(msg['content'])}"):
                    audio_file = text_to_speech(msg["content"])
                    st.markdown(get_audio_html(audio_file), unsafe_allow_html=True)

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
                # Use reasoning modes
                with st.spinner(f"Processing with {reasoning_mode} reasoning..."):
                    try:
                        if reasoning_mode == "Chain-of-Thought":
                            result = reasoning_chain.execute_reasoning(prompt)
                            display_reasoning_result(result)
                            st.session_state.messages.append({"role": "assistant", "content": result.content})
                            
                        elif reasoning_mode == "Multi-Step":
                            result = multi_step.step_by_step_reasoning(prompt)
                            display_reasoning_result(result)
                            st.session_state.messages.append({"role": "assistant", "content": result.content})
                            
                        elif reasoning_mode == "Agent-Based":
                            result = reasoning_agent.run(prompt)
                            display_reasoning_result(result)
                            st.session_state.messages.append({"role": "assistant", "content": result.content})
                            
                        else:  # Standard mode
                            if response := ollama_chat.query({"inputs": prompt}):
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.error("Failed to get response")
                                
                    except Exception as e:
                        st.error(f"Error in {reasoning_mode} mode: {str(e)}")
                        # Fallback to standard mode
                        if response := ollama_chat.query({"inputs": prompt}):
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()

# Main Function
def main():
    """Main application entry point"""

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

if __name__ == "__main__":
    main()