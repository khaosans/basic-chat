import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
import tempfile

# Must be first Streamlit command
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize LLM
llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Update document processing
class DocumentProcessor:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
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

def process_message(prompt: str) -> str:
    """Process chat message with RAG integration"""
    try:
        # Get document context if available
        context = ""
        if hasattr(st.session_state, 'doc_processor'):
            context = st.session_state.doc_processor.get_relevant_context(prompt)

        # Prepare messages with context if available
        if context:
            messages = [
                SystemMessage(content="""You are a helpful AI assistant. When answering:
1. Use the provided context to give accurate information
2. Cite specific parts of the context when relevant
3. If the context doesn't fully answer the question, say so
4. Be clear about what information comes from the context vs. your general knowledge"""),
                HumanMessage(content=f"""Context information:
{context}

User question: {prompt}

Please provide a detailed answer based on the context above and your knowledge.""")
            ]
        else:
            messages = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=prompt)
            ]

        # Get response from the LLM
        response = llm.invoke(messages)
 
        # Format and return the assistant's response
        if context:
            return f"{response.content}\n\n_Response based on available document context_"
        return response.content

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