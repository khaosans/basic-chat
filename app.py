import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_structured_chat_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import requests
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
import tempfile
from typing import Optional, Dict, Any, Type, List
from pydantic import BaseModel, Field
import subprocess
from datetime import datetime, timedelta

# Simple utility functions
def clear_temp_audio():
    """Remove temporary audio files created by TTS"""
    import glob, os
    for f in glob.glob("/tmp/audio_*.mp3"):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing temp audio file {f}: {e}")

def clear_session():
    """Clear session state and stored data, including temporary TTS files"""
    if 'messages' in st.session_state:
        st.session_state.messages.clear()
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")
    clear_temp_audio()

def ensure_models_available(models: List[str]) -> bool:
    """Ensure required models are available in Ollama"""
    try:
        for model in models:
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True
            )
            if model not in result.stdout:
                st.info(f"Pulling {model} model... This may take a few minutes.")
                subprocess.run(['ollama', 'pull', model], check=True)
        return True
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return False

# Update to use only llama2
REQUIRED_MODELS = ["llama2"]

# Must be first Streamlit command
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check models availability
if not ensure_models_available(REQUIRED_MODELS):
    st.error("Failed to initialize required models. Please ensure Ollama is running and try again.")
    st.stop()

# Initialize LLM globally with error handling
try:
    llm = ChatOllama(
        model="llama2",
        temperature=0.7,
        base_url="http://localhost:11434",
        model_kwargs={
            "num_ctx": 2048,  # Reduced context window
            "num_gpu": 1,     # Use single GPU
            "num_thread": 6   # Adjust based on CPU cores
        }
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 Hi! I'm your document-aware assistant. I can help you with:
            - 📄 Analyzing uploaded documents
            - 💬 General questions and discussions
            
            Feel free to upload a document or ask me anything!"""
        }
    ]

# Add constants for supported sports and default scope
SUPPORTED_SPORTS = {
    "soccer": ["EPL", "La Liga", "Champions League", "Serie A"],
    "basketball": ["NBA", "EuroLeague"],
    "tennis": ["ATP", "WTA", "Grand Slams"]
}

DEFAULT_SPORTS = ["soccer", "basketball", "tennis"]

# Tool definition
class SportsDataTool(BaseTool):
    name: str = Field(default="sports_data", description="Tool name")
    description: str = Field(
        default="Get real-time sports data for soccer, basketball, and tennis games.",
        description="Tool description"
    )
    
    def _run(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Run sports data query with filtering by sport and date."""
        try:
            params = self._parse_query(query)
            response = requests.get(
                "https://overtimemarketsv2.xyz/overtime-v2/games-info",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return self._format_games(data)
        except Exception as e:
            return {"error": f"Failed to fetch sports data: {str(e)}"}

    def _parse_query(self, query: Optional[str]) -> Dict[str, Any]:
        """Parse query for date and sport filters."""
        params = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sports": DEFAULT_SPORTS
        }
        
        if query:
            if "yesterday" in query.lower():
                params["date"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            elif "tomorrow" in query.lower():
                params["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            for sport in SUPPORTED_SPORTS.keys():
                if sport in query.lower():
                    params["sports"] = [sport]
                    break
        
        return params

    def _format_games(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format games data for display."""
        formatted = {
            "games": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sports_included": []
        }
        
        for game_id, game in data.items():
            game_info = {
                "sport": game.get("sport", "Unknown"),
                "league": game.get("tournamentName", "Unknown"),
                "status": game["gameStatus"],
                "teams": [
                    {
                        "name": team["name"],
                        "score": team.get("score", "N/A"),  # Use get to handle missing 'score'
                        "is_home": team["isHome"]
                    }
                    for team in game["teams"]
                ]
            }
            formatted["games"].append(game_info)
            if game.get("sport") not in formatted["sports_included"]:
                formatted["sports_included"].append(game.get("sport"))
                
        return formatted

# Update document processing
class DocumentProcessor:
    def __init__(self):
        try:
            self.llm = llm
            self.tools = [SportsDataTool()]
            
            # Fixed prompt template with properly escaped JSON example
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant who can:
                1. Answer questions about uploaded documents
                2. Get sports data when asked
                3. Help with general questions

                When using tools, follow this format:
                Action: tool_name
                Input: {{"param": "value"}}
                
                Available tools: {tools}
                Tool names: {tool_names}"""),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            # Simpler memory configuration
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Updated agent configuration
            self.agent = create_structured_chat_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt,
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=2,
                early_stopping_method="generate"
            )
            
            self.processed_files = []
            
            # Initialize vectorstore if exists
            if os.path.exists("./chroma_db"):
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
            else:
                self.vectorstore = None
        except Exception as e:
            st.error(f"Failed to initialize Document Processor: {str(e)}")
            raise

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
                loader = UnstructuredImageLoader(file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file.type}")

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create or update vectorstore
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db"
                )
            else:
                self.vectorstore.add_documents(chunks)

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
            docs = self.vectorstore.similarity_search(query, k=k)
            contexts = []
            
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown source')
                contexts.append(f"From {source}:\n{doc.page_content}")
            
            return "\n\n".join(contexts)
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""

    def process_message(self, prompt: str) -> str:
        """Process user messages and handle sports queries."""
        try:
            prompt_lower = prompt.lower().strip()
            
            if "sports" in prompt_lower or "games" in prompt_lower:
                sports_tool = SportsDataTool()
                sports_data = sports_tool._run(prompt_lower)
                
                if "error" in sports_data:
                    return f"Sorry, I couldn't fetch sports data: {sports_data['error']}"
                
                response = ["🎮 Available Games Today:"]
                for game in sports_data["games"]:
                    home_team = next(t for t in game["teams"] if t["is_home"])
                    away_team = next(t for t in game["teams"] if not t["is_home"])
                    
                    game_str = (
                        f"\n{game['sport']} - {game['league']}\n"
                        f"{home_team['name']} {home_team['score']} vs "
                        f"{away_team['name']} {away_team['score']}\n"
                        f"Status: {game['status']}"
                    )
                    response.append(game_str)
                
                return "\n".join(response)
            
            # Handle other queries through the agent
            response = self.agent_executor.invoke({"input": prompt})
            return response.get("output", "I couldn't process that request. Please try again.")
        except Exception as e:
            return f"I'm here to help! Please try asking your question in a different way. Error: {str(e)}"

def process_message(prompt: str) -> str:
    """Process chat message with RAG integration"""
    try:
        # Get document processor instance
        doc_processor = st.session_state.doc_processor
        
        # Use the agent to process the message
        response = doc_processor.process_message(prompt)
        return response

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try again."

# Replace the text_to_speech function with the following implementation using gTTS
def text_to_speech(text):
    """Convert text to speech using gTTS and return the audio file path."""
    from gtts import gTTS
    import os
    import time

    audio_path = f"/tmp/audio_{hash(text)}.mp3"
    
    # Check if the audio file already exists
    if not os.path.exists(audio_path):
        try:
            # Generate the speech using gTTS
            tts = gTTS(text=text, lang='en')
            tts.save(audio_path)
            
            # Verify that the file was created and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise Exception(f"gTTS failed to generate audio file: {audio_path}")
        
        except Exception as e:
            st.error(f"gTTS error: {e}")
            return None  # Return None to indicate failure
    
    return audio_path

def get_audio_html(audio_file):
    """Generate HTML for audio playback"""
    return f"""
    <audio controls>
        <source src="{audio_file}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """

def render_message(message, idx):
    """Render a single message with voice playback for all assistant responses"""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if st.button("🔊 Play Voice", key=f"audio_{idx}"):
                audio_file = text_to_speech(message["content"])
                if audio_file:  # Check if audio_file is not None
                    # Use st.audio to render the audio player instead of st.markdown
                    st.audio(audio_file, format="audio/mpeg")
                else:
                    st.error("Failed to generate audio for this message.")

def render_chat():
    """Render chat interface"""
    # Display chat messages with unique keys for voice button playback
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, idx)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message and render it
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message({"role": "user", "content": prompt}, len(st.session_state.messages)-1)
        
        # Get and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
        st.rerun()

def main():
    """Main application with RAG integration"""
    st.title("💬 Document-Aware Chat")
    
    # Initialize document processor
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Documents")
        
        # Model status indicators
        st.write("### 🤖 Model Status")
        for model in REQUIRED_MODELS:
            try:
                subprocess.run(['ollama', 'list'], capture_output=True, check=True)
                st.success(f"✅ {model}")
            except:
                st.error(f"❌ {model}")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            help="Upload documents to analyze"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing document..."):
                    result = st.session_state.doc_processor.process_file(uploaded_file)
                    st.success(result)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Show processed files
        if st.session_state.doc_processor.processed_files:
            st.markdown("### 📑 Processed Documents")
            for doc in st.session_state.doc_processor.processed_files:
                st.markdown(f"""
                <div style='background-color: #25262B; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0;'>
                    📄 {doc['name']} ({doc['chunks']} chunks)
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All"):
            clear_session()
            st.rerun()
    
    # Main chat interface
    render_chat()

# Add this CSS for better styling
st.markdown("""
<style>
    /* Chat container */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0.5rem 0;
    }
    
    /* Message bubbles */
    .user-message {
        background-color: #2C5364;
        margin-left: 20%;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .assistant-message {
        background-color: #203A43;
        margin-right: 20%;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Document cards */
    .document-card {
        background-color: #25262B;
        border: 1px solid #2C2D32;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()