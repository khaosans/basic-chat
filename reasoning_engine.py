"""
Unified Reasoning Engine for Enhanced AI Capabilities
"""

import os
import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass
import httpx
import logging

# Suppress ALL warnings
warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from web_search import search_web
from utils.enhanced_tools import EnhancedCalculator, EnhancedTimeTools
from config import REASONING_MODEL, OLLAMA_API_URL

# Configuration
logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """Result from reasoning operations"""
    final_answer: str
    reasoning_steps: List[str]
    confidence: float
    sources: List[str]
    success: bool = True
    error: Optional[str] = None

class ReasoningEngine:
    """A modern reasoning engine using only current LangChain APIs"""

    def __init__(self, model_name: str = REASONING_MODEL):
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL
        )
        
        # Initialize enhanced tools
        self.calculator = EnhancedCalculator()
        self.time_tools = EnhancedTimeTools()
        
        # Simple conversation history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _web_search(self, query: str) -> str:
        """Wrapper for web search tool"""
        try:
            results = search_web(query)
            if isinstance(results, list):
                return "\n".join(
                    f"Title: {res.title}\nLink: {res.url}\nSnippet: {res.snippet}"
                    for res in results
                )
            return "No search results available"
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "Web search is currently unavailable"

    def _create_enhanced_prompt(self, input_text: str, context: str) -> str:
        """Create an enhanced prompt with tools and context"""
        tools_info = """
Available tools you can use:
1. Calculator: For mathematical calculations
2. Time tools: For getting current time and date
3. Web search: For searching current information

If you need to use a tool, mention it in your response.
"""
        
        enhanced_prompt = f"""You are an AI assistant with access to various tools. 

{tools_info}

Context: {context}

User question: {input_text}

Please provide a helpful and accurate response. If you need to use any tools, mention which ones you would use and why.
"""
        return enhanced_prompt

    def reason(self, input_text: str, context: str, mode: str = "auto") -> ReasoningResult:
        """Run the reasoning process using direct LLM calls"""
        try:
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(input_text, context)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": input_text})
            
            # Create messages for the LLM using modern API
            messages = [
                SystemMessage(content="You are a helpful AI assistant with reasoning capabilities."),
                HumanMessage(content=enhanced_prompt)
            ]
            
            # Get response from LLM using modern invoke method
            response = self.llm.invoke(messages)
            
            # Extract the response content
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": final_answer})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return ReasoningResult(
                final_answer=str(final_answer),
                reasoning_steps=["Processed with modern reasoning engine"],
                confidence=0.9,
                sources=["llm_direct"],
                success=True
            )
            
        except httpx.HTTPStatusError as e:
            error_message = f"Ollama API Error: {e.response.status_code}. Make sure the Ollama server is running and the model '{self.llm.model}' is available."
            logger.error(f"{error_message} - Response: {e.response.text}")
            return ReasoningResult(
                final_answer="I'm having trouble connecting to the AI model. Please check if Ollama is running.",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=error_message
            )
        except Exception as e:
            # Check for specific ollama errors
            if "does not support chat" in str(e):
                error_message = f"Model '{self.llm.model}' does not support chat. Please configure a different REASONING_MODEL in your settings."
                logger.error(error_message)
                return ReasoningResult(
                    final_answer=error_message,
                    reasoning_steps=[],
                    confidence=0.0,
                    sources=[],
                    success=False,
                    error=error_message
                )

            logger.error(f"An unexpected error occurred in ReasoningEngine: {e}", exc_info=True)
            return ReasoningResult(
                final_answer="An unexpected error occurred while processing your request.",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )

    def get_cache_stats(self) -> Dict:
        """Returns empty dict, kept for compatibility."""
        return {} 