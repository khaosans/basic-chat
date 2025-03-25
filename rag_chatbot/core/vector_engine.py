from typing import List, Dict
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self):
        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # Initialize ChromaDB with consistent settings
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="documents")
            logger.info("✅ Retrieved existing collection")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ Created new collection")
        
        # Initialize store
        self.store = Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embedder,
            persist_directory="./chroma_db"
        )
        logger.info("✅ Vector store initialized successfully")

    def reset_store(self):
        """Reset the vector store if needed"""
        try:
            # Delete existing collection
            self.client.delete_collection(name="documents")
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Reinitialize store
            self.store = Chroma(
                client=self.client,
                collection_name="documents",
                embedding_function=self.embedder,
                persist_directory="./chroma_db"
            )
            logger.info("✅ Vector store reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            raise

    async def vectorize(self, batch: Dict) -> bool:
        """Vectorize and store documents"""
        try:
            if not batch.get('documents'):
                return False
                
            self.store.add_texts(
                texts=batch['documents'],
                metadatas=batch.get('metadatas', []),
                ids=batch.get('ids', [])
            )
            return True
        except Exception as e:
            logger.error(f"Vectorization error: {e}")
            return False 