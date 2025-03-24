import streamlit as st
from langchain_community.chat_models import ChatOllama
from ollama_api import get_available_models, check_ollama_server
from document_processor import DocumentProcessor
import shutil, os

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
    st.session_state.messages = []
if 'error' not in st.session_state:
    st.session_state.error = None

# Initialize document processor
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

# Initialize session state for model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'llama2'

# Initialize chat parameters
if 'chat_params' not in st.session_state:
    st.session_state.chat_params = {
        'last_request_time': None,
        'request_count': 0,
        'max_requests_per_minute': 20
    }

# Sidebar for model selection and file upload
with st.sidebar:
    st.title("Settings")
    
    # Model selection with additional info
    st.subheader("Model Settings")
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Choose the AI model to use for chat responses"
    )
    st.session_state.selected_model = selected_model
    
    # Add model information
    st.info(f"""
    Currently using: {selected_model}
    - Temperature: 0.7
    - Server: localhost:11434
    """)
    
    # File upload section with better instructions
    st.subheader("Document Upload")
    st.markdown("""
    Supported files:
    - PDF documents
    - Images (PNG, JPG)
    
    Documents will be processed and used as context for answers.
    """)
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Files will be processed and used as context for the chatbot"
    )
    
    if uploaded_file is not None:
        try:
            # Create placeholder for progress updates
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Create a progress bar
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.text("Starting document processing...")
            
            # Redirect stdout to capture processing updates
            import sys
            from io import StringIO
            import re
            
            class StreamCapture:
                def __init__(self, progress_bar, status_text):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                
                def write(self, text):
                    if "Processing image analysis step" in text:
                        step = int(re.search(r'step (\d+)/4', text).group(1))
                        self.progress_bar.progress(step * 0.25)
                        self.status_text.text(text.strip())
                    elif "Progress:" in text:
                        self.status_text.text(text.strip())
                    sys.__stdout__.write(text)
                
                def flush(self):
                    pass
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StreamCapture(progress_bar, status_text)
            
            try:
                st.session_state.doc_processor.process_file(uploaded_file)
                progress_bar.progress(1.0)
                status_text.success(f"Document '{uploaded_file.name}' processed successfully!")
            finally:
                sys.stdout = old_stdout
                
            # Clear placeholders after success
            progress_placeholder.empty()
            status_placeholder.empty()
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
    
    # Display processed files
    st.subheader("Processed Documents")
    processed_files = st.session_state.doc_processor.get_processed_files()
    
    if not processed_files:
        st.info("No documents uploaded yet.")
    else:
        for file in processed_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {file['name']} ({file['size']} bytes)")
            with col2:
                if st.button("🗑️", key=f"remove_{file['name']}"):
                    try:
                        if st.session_state.doc_processor.remove_file(file['name']):
                            # Verify document removal
                            processed_files = st.session_state.doc_processor.get_processed_files()
                            if file['name'] not in [f['name'] for f in processed_files]:
                                st.success(f"Removed {file['name']}")
                            else:
                                st.warning("Document removal may not be complete. Try resetting all data.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing document: {str(e)}")
                        # Offer manual reset
                        if st.button("Reset All Data"):
                            clear_session_state()
                            st.rerun()

# Initialize ChatOllama with selected model
llm = ChatOllama(
    model=st.session_state.selected_model,
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Chat interface with welcome message
st.title("📚 Document-Aware Chatbot")
st.caption(f"Using model: {st.session_state.selected_model}")

# Add welcome message if no messages exist
if not st.session_state.messages:
    st.info("""
    👋 Welcome! I can help you with:
    - Answering questions about uploaded documents
    - General knowledge queries
    - Document analysis and understanding
    
    Try uploading a document and asking questions about it!
    """)

# Display error message if exists
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    if not check_rate_limit():
        st.error("Too many requests. Please wait a moment before sending more messages.")
        st.stop()
        
    if not prompt.strip():
        st.warning("Please enter a non-empty message.")
        st.stop()
    
    # Check if any documents are loaded
    processed_files = st.session_state.doc_processor.get_processed_files()
    if not processed_files and any(word in prompt.lower() for word in ['summarize', 'summary', 'pdf', 'document', 'image']):
        st.warning("No documents have been uploaded yet. Please upload a document first.")
        st.stop()
        
    try:
        # Validate model connection before processing
        llm.invoke("test")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get relevant context with status indicator and debug info
            with st.spinner("Searching documents..."):
                doc_context = st.session_state.doc_processor.get_relevant_context(prompt)
                if doc_context:
                    st.info("📄 Found relevant content in documents")
                    print(f"Context length: {len(doc_context)} characters")
                else:
                    print("No relevant document context found")
            
            # Enhanced prompt handling
            if doc_context:
                enhanced_prompt = f"""You are a helpful assistant with access to document content. Here is the relevant context:

{doc_context}

Using this context, please answer: {prompt}

Requirements:
1. If the query asks for a summary, provide a clear and concise summary
2. Always mention which documents or images you're referencing
3. Include relevance scores when available
4. If you can't find relevant information in the context, clearly state that"""
            else:
                if any(word in prompt.lower() for word in ['summarize', 'summary', 'pdf', 'document', 'image']):
                    enhanced_prompt = f"I apologize, but I don't have access to any relevant document content for: {prompt}"
                else:
                    enhanced_prompt = f"You are a helpful assistant. Please answer: {prompt}"
            
            # Display response
            with st.spinner("Thinking..."):
                try:
                    for chunk in llm.stream(enhanced_prompt):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower():
            st.error("Lost connection to Ollama server. Please check if it's running.")
        else:
            st.error(f"Error processing message: {error_msg}")
        st.stop()