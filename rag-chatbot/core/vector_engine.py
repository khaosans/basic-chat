from typing import List, Dict
import numpy as np

class VectorEngine:
    def __init__(self):
        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        self.store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embedder
        )
        
    async def vectorize(self, batch: DocumentBatch) -> Dict:
        try:
            embeddings = await self.generate_embeddings(batch.chunks)
            return await self.store.add_documents(
                documents=batch.chunks,
                embeddings=embeddings,
                metadatas=[batch.metadata] * len(batch.chunks)
            )
        except Exception as e:
            logger.error(f"Vectorization error: {e}")
            raise

    async def search(self, query: str, k: int = 3) -> List[Dict]:
        return await self.store.similarity_search_with_score(query, k=k) 