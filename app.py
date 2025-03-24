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
def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except:
        return False

def clear_session():
    """Clear session state and stored data"""
    if 'messages' in st.session_state:
        st.session_state.messages.clear()
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")

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

# Update chat processing
def process_message(prompt: str) -> str:
    """Process chat message with RAG integration"""
    try:
        # Get document context if available
        context = ""
        if hasattr(st.session_state, 'doc_processor'):
            context = st.session_state.doc_processor.get_relevant_context(prompt)

        # Prepare messages with context
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

        # Get response
        response = llm.invoke(messages)
        
        # Format response
        if context:
            return f"{response.content}\n\n_Response based on available document context_"
        return response.content

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try again."

def render_message(message):
    """Render a single message"""
    with st.chat_message(message["role"]):
        st.write(message["content"])

def render_chat():
    """Render chat interface"""
    # Display chat messages
    for message in st.session_state.messages:
        render_message(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message({"role": "user", "content": prompt})
        
        # Get and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_message(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

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