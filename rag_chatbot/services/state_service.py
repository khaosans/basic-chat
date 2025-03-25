from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StateService:
    def __init__(self):
        self.document_states: Dict[str, str] = {}
        self.chat_history: List[Dict[str, str]] = []
        self.processing_state: Dict[str, Any] = {}
        logger.info("✅ State service initialized")
    
    def update_document_state(self, doc_id: str, state: str) -> None:
        """Update document processing state"""
        self.document_states[doc_id] = state
        logger.info(f"Document {doc_id} state updated to: {state}")
    
    def get_document_state(self, doc_id: str) -> Optional[str]:
        """Get document state"""
        return self.document_states.get(doc_id)
    
    def add_to_history(self, message: Dict[str, str]) -> None:
        """Add message to chat history"""
        self.chat_history.append(message)
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get chat history"""
        return self.chat_history
    
    def clear_history(self) -> None:
        """Clear chat history"""
        self.chat_history.clear()
        logger.info("Chat history cleared")
    
    def set_processing_state(self, key: str, value: Any) -> None:
        """Set processing state"""
        self.processing_state[key] = value
    
    def get_processing_state(self, key: str) -> Optional[Any]:
        """Get processing state"""
        return self.processing_state.get(key)
    
    def clear_all(self) -> None:
        """Clear all state"""
        self.document_states.clear()
        self.chat_history.clear()
        self.processing_state.clear()
        logger.info("All state cleared") 