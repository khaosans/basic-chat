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

load_dotenv(".env.local")  # Load environment variables from .env.local

# Use Ollama model instead of Hugging Face
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")

# Add a system prompt definition
SYSTEM_PROMPT = """
You are a helpful and knowledgeable AI assistant. You can:
1. Answer questions about a wide range of topics
2. Summarize documents that have been uploaded
3. Have natural, friendly conversations

Please be concise, accurate, and helpful in your responses. 
If you don't know something, just say so instead of making up information.
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

def chat_interface(doc_processor):
    """Chat interface using Ollama with tools"""
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create chat instances
    ollama_chat = OllamaChat(OLLAMA_MODEL)
    tool_registry = ToolRegistry(doc_processor)

    # Initialize welcome message if needed
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hello! I'm your AI assistant. How can I help you today?"
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
        
        # Process response
        with st.chat_message("assistant"):
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
                with st.spinner("Thinking..."):
                    if response := ollama_chat.query({"inputs": prompt}):
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get response")

        st.rerun()

# Main Function
def main():
    """Main application entry point"""

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_interface(None)  # Pass the document processor if needed

    with st.sidebar:
        st.header("📚 Documents")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload documents to analyze",
        )
        if uploaded_file:
            st.success("Document uploaded successfully!")

if __name__ == "__main__":
    main()