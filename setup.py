import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "streamlit",
        "langchain-community",
        "chromadb",
        "pillow",
        "python-magic",
        "pypdf",
        "unstructured",
        "sentence-transformers",
    ]
    
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)

def setup_directories():
    """Create necessary directories"""
    directories = [
        "./chroma_db",
        "./temp",
        "./uploads"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def check_ollama():
    """Check if Ollama is installed and running"""
    import requests
    
    print("🤖 Checking Ollama installation...")
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        print("""❌ Ollama is not running. Please:
        1. Install Ollama from https://ollama.ai
        2. Run: ollama pull mistral
        3. Run: ollama pull nomic-embed-text
        4. Run: ollama pull llava""")
        return False

def main():
    """Main setup function"""
    print("🚀 Starting setup...")
    
    install_requirements()
    setup_directories()
    check_ollama()
    
    print("✨ Setup complete! Run 'streamlit run app.py' to start the application")

if __name__ == "__main__":
    main() 
