import streamlit as st
from langchain_community.chat_models import ChatOllama
from ollama_api import get_available_models, check_ollama_server
from document_processor import DocumentProcessor
import shutil, os
import time

def clear_session_state():
    """Clear all session state and stored data"""
    try:
        # Clear document processor state first
        if 'doc_processor' in st.session_state:
            st.session_state.doc_processor.reset_state()
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
    except Exception as e:
        print(f"Error clearing session state: {e}")
        # Force clear everything
        st.session_state.clear()
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db", ignore_errors=True)

def check_rate_limit():
    """Rate limiting function to prevent too many requests"""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    if st.session_state.chat_params['last_request_time']:
        time_diff = now - st.session_state.chat_params['last_request_time']
        if time_diff < timedelta(minutes=1):
            if st.session_state.chat_params['request_count'] >= st.session_state.chat_params['max_requests_per_minute']:
                return False
        else:
            # Reset counter after 1 minute
            st.session_state.chat_params['request_count'] = 0
    
    # Update last request time and count
    st.session_state.chat_params['last_request_time'] = now
    st.session_state.chat_params['request_count'] += 1
    return True

# Initialize Streamlit app
st.set_page_config(page_title="Streamlit Chatbot", layout="wide")

# Add reset button in sidebar
with st.sidebar:
    if st.button("🔄 Reset All Data", help="Clear all chat history and documents"):
        clear_session_state()
        st.rerun()

# Check Ollama server status
if not check_ollama_server():
    st.error("Cannot connect to Ollama server. Please ensure it's running on http://localhost:11434")
    st.stop()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """👋 Hi! I'm your document-aware assistant. I can help you with:
        - 📄 Analyzing uploaded documents
        - 🖼️ Understanding images
        - 💬 General questions and discussions
        
        Feel free to upload a document or ask me anything!"""
    }]
if 'error' not in st.session_state:
    st.session_state.error = None

# Initialize document processor
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

# Define fixed models for different tasks
CHAT_MODEL = "mistral"  # Main chat model
EMBEDDING_MODEL = "nomic-embed-text"  # For document embeddings
IMAGE_MODEL = "llava"  # For image analysis

# Remove model selection from session state since we're using fixed models
if 'chat_params' not in st.session_state:
    st.session_state.chat_params = {
        'last_request_time': None,
        'request_count': 0,
        'max_requests_per_minute': 20,
        'system_prompt': """You are a helpful AI assistant that can:
        1. Answer questions about uploaded documents
        2. Handle general queries
        3. Analyze images
        4. Provide clear and concise responses"""
    }

# Initialize ChatOllama with fixed model
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Add custom CSS for better UI
st.markdown("""
<style>
    /* Responsive container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Better chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #f5f5f5;
        margin-right: 20%;
    }
    
    /* Loading animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Better file upload area */
    .uploadedFile {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #2196F3;
    }
    
    /* Improved buttons */
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #2196F3;
        transition: all 0.3s ease;
    }
    
    /* Responsive layout */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            margin-left: 5%;
            margin-right: 5%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Update the chat interface
def create_message_container(role: str, content: str):
    """Create a styled message container"""
    class_name = "user-message" if role == "user" else "assistant-message"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {class_name}">
            {content}
        </div>
        """, unsafe_allow_html=True)

# Update the main layout
st.title("📚 Document-Aware Chatbot")

# Improved sidebar layout
with st.sidebar:
    st.title("💡 Assistant Settings")
    
    # System info with better styling
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px;'>
        <h4>🤖 System Configuration</h4>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li>📝 Chat: Mistral</li>
            <li>🔍 Embeddings: Nomic</li>
            <li>🖼️ Images: Llava</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <h4>📎 Upload Documents</h4>
        <p>Supported formats:</p>
        <ul>
            <li>PDF documents</li>
            <li>Images (PNG, JPG)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Update file upload handling
def show_upload_progress(uploaded_file):
    """Show upload progress with better UI"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Processing {uploaded_file.name}... {i+1}%")
        time.sleep(0.01)
    
    status_text.success(f"✅ {uploaded_file.name} processed successfully!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

# Update chat interface
def display_chat_interface():
    """Display chat interface with improved styling"""
    # Display chat messages
    for message in st.session_state.messages:
        create_message_container(message["role"], message["content"])
    
    # Chat input with better UX
    if prompt := st.chat_input("💭 Ask me anything...", key="chat_input"):
        # Show typing indicator
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Thinking..._")
            
            try:
                # Process message
                response = process_chat_message(prompt)
                
                # Update UI
                typing_placeholder.empty()
                create_message_container("assistant", response)
                
            except Exception as e:
                typing_placeholder.error(f"Error: {str(e)}")

# Add loading states and error handling
def show_loading_state():
    """Show loading state with animation"""
    with st.spinner("Loading..."):
        time.sleep(0.5)  # Simulate loading
        
def show_error(message: str):
    """Show error message with style"""
    st.error(f"❌ {message}")

# Update the document display
def display_documents():
    """Display processed documents with better UI"""
    if processed_files := st.session_state.doc_processor.get_processed_files():
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <h4>📑 Processed Documents</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for file in processed_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div style='padding: 0.5rem; border-radius: 4px; background-color: #fff;'>
                    📄 {file['name']} ({file['size']} bytes)
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"remove_{file['name']}"):
                    with st.spinner(f"Removing {file['name']}..."):
                        st.session_state.doc_processor.remove_file(file['name'])
                        st.rerun()

# Main app layout
def main():
    """Main application layout"""
    try:
        show_loading_state()
        display_chat_interface()
        display_documents()
    except Exception as e:
        show_error(str(e))

if __name__ == "__main__":
    main()

# Update the chat handling section
def process_chat_message(prompt: str, llm: ChatOllama) -> str:
    """Process chat messages with better error handling and model fallbacks"""
    try:
        # First try with selected model
        response = llm.invoke(prompt)
        return response.content
    except Exception as primary_error:
        print(f"Primary model error: {primary_error}")
        try:
            # Fallback to llama2 if primary model fails
            fallback_llm = ChatOllama(
                model="llama2",
                temperature=0.7,
                base_url="http://localhost:11434"
            )
            response = fallback_llm.invoke(prompt)
            return response.content + "\n\n(Response generated using fallback model)"
        except Exception as fallback_error:
            raise Exception(f"Chat failed: {str(fallback_error)}")