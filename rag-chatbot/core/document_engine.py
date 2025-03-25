from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentBatch:
    chunks: List[str]
    metadata: dict
    source: str

class DocumentProcessor:
    def __init__(self):
        self.supported_types = {
            'application/pdf': 'pdf',
            'text/plain': 'text',
            'image/png': 'image',
            'image/jpeg': 'image'
        }
        
    async def process(self, file) -> DocumentBatch:
        try:
            mime_type = self._get_mime_type(file.name)
            return await self._process_by_type(file, mime_type)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    async def chunk(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

class DocumentEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = TextChunker()
        self.metadata_manager = MetadataManager()
    
    async def process_document(self, file) -> DocumentBatch:
        doc = await self.processor.process(file)
        chunks = await self.chunker.chunk(doc.text)
        metadata = await self.metadata_manager.extract(doc)
        return DocumentBatch(chunks, metadata, file.name) 