"""
Reasoning Engine for Enhanced AI Capabilities
Provides chain-of-thought, multi-step reasoning, and agent-based processing
"""

import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import streamlit as st

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

@dataclass
class ReasoningResult:
    """Result from reasoning operations"""
    content: str
    reasoning_steps: List[str]
    confidence: float
    sources: List[str]
    success: bool = True
    error: Optional[str] = None

class ReasoningAgent:
    """Enhanced agent with reasoning capabilities"""
    
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agent with better configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="calculator",
                func=self._calculate,
                description="Perform mathematical calculations and solve equations. Input should be a mathematical expression like '2 + 2' or '10 * 5'."
            ),
            Tool(
                name="current_time",
                func=self._get_current_time,
                description="Get the current date and time. No input needed."
            ),
            Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for current information (placeholder). Input should be a search query."
            )
        ]
    
    def _calculate(self, expression: str) -> str:
        """Safe mathematical calculation"""
        try:
            # Basic safety check
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def _get_current_time(self, query: str = "") -> str:
        """Get current time"""
        import datetime
        return f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _web_search(self, query: str) -> str:
        """Placeholder for web search functionality"""
        return f"Web search for '{query}' would be implemented here"
    
    def run(self, query: str) -> ReasoningResult:
        """Run the agent with reasoning"""
        try:
            response = self.agent.invoke({"input": query})
            return ReasoningResult(
                content=response.get("output", "No response generated"),
                reasoning_steps=["Agent processed query using available tools"],
                confidence=0.8,
                sources=["agent_tools"]
            )
        except Exception as e:
            return ReasoningResult(
                content="",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )

class ReasoningChain:
    """Chain-of-thought reasoning implementation"""
    
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        
        # Use ChatPromptTemplate for better chat model compatibility
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that excels at step-by-step reasoning. 
            When given a question, always break it down into clear steps and show your thinking process.
            Be thorough but concise in your reasoning."""),
            ("human", """Please answer this question using step-by-step reasoning:

Question: {question}

Let me think through this step by step:

1) First, I need to understand what's being asked
2) Then, I'll identify what information I need
3) I'll gather the necessary information
4) Finally, I'll provide a reasoned answer

Please provide your step-by-step reasoning and final answer:""")
        ])
        
        # Use the newer RunnableSequence approach
        self.chain = self.reasoning_prompt | self.llm
    
    def execute_reasoning(self, question: str) -> ReasoningResult:
        """Execute chain-of-thought reasoning"""
        try:
            response = self.chain.invoke({"question": question})
            
            # Extract content from the response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Parse the response to extract reasoning steps
            reasoning_steps = self._extract_reasoning_steps(content)
            
            return ReasoningResult(
                content=content,
                reasoning_steps=reasoning_steps,
                confidence=0.7,
                sources=["chain_of_thought"]
            )
        except Exception as e:
            return ReasoningResult(
                content="",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        steps = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)', '10)', '-', '•')):
                steps.append(line)
            elif line.startswith(('Step', 'STEP')):
                steps.append(line)
        return steps if steps else ["Chain-of-thought reasoning applied"]

class MultiStepReasoning:
    """Multi-step reasoning with document context"""
    
    def __init__(self, doc_processor, model_name: str = OLLAMA_MODEL):
        self.doc_processor = doc_processor
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that analyzes questions and breaks them down into steps."),
            ("human", """Analyze this query and break it down into steps:
Query: {query}

What are the key components and what information do I need?
List the steps needed to answer this question:""")
        ])
        
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that provides step-by-step reasoning based on analysis and context."),
            ("human", """Based on the analysis and context, let me reason through this:

Analysis: {analysis}
Context: {context}
Original Query: {query}

Let me think step by step:
1) What does the query ask for?
2) What relevant information do I have?
3) How do I connect the information to answer the query?
4) What is my final reasoned answer?

Step-by-step reasoning:""")
        ])
        
        # Create chains using RunnableSequence
        self.analysis_chain = self.analysis_prompt | self.llm
        self.reasoning_chain = self.reasoning_prompt | self.llm
    
    def step_by_step_reasoning(self, query: str) -> ReasoningResult:
        """Execute multi-step reasoning"""
        try:
            # Step 1: Analyze the query
            analysis_response = self.analysis_chain.invoke({"query": query})
            analysis = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
            
            # Step 2: Gather relevant context
            context = self.doc_processor.get_relevant_context(query) if self.doc_processor else ""
            
            # Step 3: Reason through the information
            reasoning_response = self.reasoning_chain.invoke({
                "analysis": analysis,
                "context": context,
                "query": query
            })
            reasoning = reasoning_response.content if hasattr(reasoning_response, 'content') else str(reasoning_response)
            
            # Combine analysis and reasoning
            full_response = f"Analysis:\n{analysis}\n\nReasoning:\n{reasoning}"
            
            return ReasoningResult(
                content=full_response,
                reasoning_steps=[
                    "Query analysis completed",
                    "Context gathered",
                    "Multi-step reasoning applied"
                ],
                confidence=0.9,
                sources=["multi_step_reasoning", "document_context"] if context else ["multi_step_reasoning"]
            )
        except Exception as e:
            return ReasoningResult(
                content="",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )

class ReasoningDocumentProcessor:
    """Enhanced document processor with reasoning capabilities"""
    
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_API_URL.replace("/api", "")
        )
        self.vectorstore = None
        self.processed_files = []
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that analyzes documents and extracts key information."),
            ("human", """Analyze this document and extract:
1. Key topics and themes
2. Important facts and data
3. Relationships between concepts
4. Potential questions this document could answer

Document: {document_text}

Provide a structured analysis:""")
        ])
        
        # Create chain using RunnableSequence
        self.analysis_chain = self.analysis_prompt | self.llm
    
    def analyze_document_content(self, document_text: str) -> Dict[str, str]:
        """Analyze document content for reasoning capabilities"""
        try:
            response = self.analysis_chain.invoke({"document_text": document_text[:2000]})  # Limit for analysis
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": analysis,
                "content": document_text,
                "key_topics": self._extract_topics(analysis),
                "reasoning_potential": self._assess_reasoning_potential(analysis)
            }
        except Exception as e:
            return {
                "analysis": f"Error analyzing document: {str(e)}",
                "content": document_text,
                "key_topics": [],
                "reasoning_potential": "low"
            }
    
    def _extract_topics(self, analysis: str) -> List[str]:
        """Extract key topics from analysis"""
        # Simple topic extraction - could be enhanced
        topics = []
        lines = analysis.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.')):
                topics.append(line.strip())
        return topics
    
    def _assess_reasoning_potential(self, analysis: str) -> str:
        """Assess the reasoning potential of the document"""
        if any(word in analysis.lower() for word in ['data', 'facts', 'relationships', 'analysis']):
            return "high"
        elif any(word in analysis.lower() for word in ['information', 'details', 'content']):
            return "medium"
        else:
            return "low"
    
    def create_reasoning_context(self, query: str) -> str:
        """Create context optimized for reasoning"""
        if not self.vectorstore:
            return "No documents available for context"
        
        try:
            relevant_docs = self.vectorstore.similarity_search(query, k=3)
            
            reasoning_context_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant that creates reasoning frameworks."),
                ("human", """Given this query: {query}

And these relevant documents: {relevant_docs}

Create a reasoning framework that:
1. Identifies the key information needed
2. Shows how different pieces connect
3. Provides a logical structure for answering

Format this as a reasoning template.""")
            ])
            
            context_chain = reasoning_context_prompt | self.llm
            response = context_chain.invoke({
                "query": query,
                "relevant_docs": "\n".join([doc.page_content for doc in relevant_docs])
            })
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error creating reasoning context: {str(e)}"
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from documents"""
        if not self.vectorstore:
            return ""
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error retrieving context: {str(e)}" 