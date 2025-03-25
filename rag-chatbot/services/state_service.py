class StateService:
    def __init__(self):
        self.document_states = {}
        self.chat_history = []
        
    async def update_document_state(self, doc_id: str, state: str):
        self.document_states[doc_id] = state
        
    async def add_to_history(self, message: Dict):
        self.chat_history.append(message) 