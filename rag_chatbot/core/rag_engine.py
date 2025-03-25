from typing import List, Dict
import logging
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

class QueryProcessor:
    async def process(self, query: str) -> str:
        return query

class ContextRetriever:
    async def get_context(self, query: str) -> str:
        return ""

class PromptBuilder:
    async def build(self, query: str, context: str, history: List[Dict] = None) -> str:
        return query

class RAGEngine:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.context_retriever = ContextRetriever()
        self.prompt_builder = PromptBuilder()
        self.llm = ChatOllama(
            model="mistral",
            temperature=0.7,
            base_url="http://localhost:11434"
        ) 