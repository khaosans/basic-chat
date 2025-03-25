from typing import List, Dict
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self):
        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Initialize ChromaDB with proper settings
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.store = Chroma(
                client=self.client,
                collection_name="documents",
                embedding_function=self.embedder
            )
            logger.info("✅ Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise 