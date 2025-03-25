from typing import List, Dict

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
    
    async def process_query(self, query: str, history: List[Dict] = None) -> str:
        processed_query = await self.query_processor.process(query)
        context = await self.context_retriever.get_context(processed_query)
        prompt = await self.prompt_builder.build(processed_query, context, history)
        return await self.llm.agenerate([prompt]) 