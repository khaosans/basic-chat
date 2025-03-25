from dataclasses import dataclass
from typing import List, Optional
import logging
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

@dataclass
class DocumentBatch:
    chunks: List[str]
    metadata: dict
    source: str

class MetadataManager:
    def __init__(self):
        pass
    
    async def extract(self, doc) -> dict:
        return {"source": doc.source}

class DocumentProcessor:
    def __init__(self):
        self.supported_types = {
            'application/pdf': 'pdf',
            'text/plain': 'text',
            'image/png': 'image',
            'image/jpeg': 'image'
        }
    
    def _get_mime_type(self, filename: str) -> str:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type
        
    async def process(self, file) -> DocumentBatch:
        try:
            mime_type = self._get_mime_type(file.name)
            return DocumentBatch(
                chunks=[],
                metadata={"type": mime_type},
                source=file.name
            )
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

class DocumentEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.metadata_manager = MetadataManager()
    
    async def process_document(self, file) -> DocumentBatch:
        """Process document asynchronously"""
        try:
            # Process the document
            doc = await self.processor.process(file)
            chunks = self.chunker.split_text(doc.text)
            metadata = await self.metadata_manager.extract(doc)
            
            return DocumentBatch(
                chunks=chunks,
                metadata=metadata,
                source=file.name
            )
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise 