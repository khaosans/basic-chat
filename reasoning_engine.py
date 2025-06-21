"""
Reasoning Engine for Enhanced AI Capabilities
Provides chain-of-thought, multi-step reasoning, and agent-based processing
"""

import os
import time
import logging
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import streamlit as st
import datetime

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from document_processor import DocumentProcessor
from web_search import search_web
from utils.enhanced_tools import EnhancedCalculator, EnhancedTimeTools
from config import DEFAULT_MODEL, OLLAMA_API_URL
from langchain_core.output_parsers import StrOutputParser

# Configure logging for reasoning engine
logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """Result from reasoning operations"""
    content: str  # Final answer/conclusion
    reasoning_steps: List[str]  # Step-by-step thought process
    confidence: float
    sources: List[str]
    intermediate_thoughts: List[str] = None  # For storing intermediate reasoning steps
    success: bool = True
    error: Optional[str] = None

class ReasoningAgent:
    """Enhanced agent with reasoning capabilities"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing ReasoningAgent with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_API_URL.replace("/api", "")
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
        
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input"
            )
            logger.info("Memory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise
        
        # Initialize enhanced tools
        try:
            self.calculator = EnhancedCalculator()
            self.time_tools = EnhancedTimeTools()
            logger.info("Enhanced tools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced tools: {e}")
            raise
        
        # Initialize tools
        try:
            self.tools = self._create_tools()
            logger.info(f"Created {len(self.tools)} tools")
        except Exception as e:
            logger.error(f"Failed to create tools: {e}")
            raise
        
        # System message to guide the agent
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant that can use tools to answer questions. "
                "If relevant, use the following context from documents to answer the user's question. "
                "If the context does not contain the answer, use your own knowledge or tools. "
                "Context: {context}"
            )
        )

        # Initialize agent with better configuration
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={
                    "system_message": system_message.content,
                    "input_variables": ["input", "agent_scratchpad", "chat_history", "context"]
                },
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def _create_tools(self) -> List[Tool]:
        """Create enhanced tools for the agent"""
        logger.debug("Creating enhanced tools")
        return [
            Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for current information. Use this for questions about recent events, current prices, weather, or any information that might change over time."
            ),
            Tool(
                name="enhanced_calculator",
                func=self._enhanced_calculate,
                description="Perform advanced mathematical calculations including trigonometry, logarithms, factorials, and more. Input should be a mathematical expression like '2 + 2', 'sqrt(16)', 'sin(pi/2)', 'factorial(5)', or 'gcd(12, 18)'."
            ),
            Tool(
                name="get_current_time",
                func=self._get_current_time,
                description="Get the current date and time with timezone support. Use this when asked about the current time or date. You can specify a timezone like 'UTC', 'EST', 'PST', 'GMT', 'JST', etc."
            ),
            Tool(
                name="time_conversion",
                func=self._convert_time,
                description="Convert time between different timezones. Input format: 'time_string, from_timezone, to_timezone' (e.g., '2024-01-01 12:00:00, UTC, EST')."
            ),
            Tool(
                name="time_difference",
                func=self._calculate_time_difference,
                description="Calculate the difference between two times. Input format: 'time1, time2, timezone' (e.g., '2024-01-01 12:00:00, 2024-01-01 14:00:00, UTC')."
            ),
            Tool(
                name="time_info",
                func=self._get_time_info,
                description="Get comprehensive time information including weekday, month, day of year, etc. Input should be a timezone name (e.g., 'UTC', 'EST')."
            )
        ]
    
    def _enhanced_calculate(self, expression: str) -> str:
        """Enhanced calculator function with detailed output"""
        logger.debug(f"Calculating expression: {expression}")
        try:
            result = self.calculator.calculate(expression)
            
            if result.success:
                # Format the output with steps
                output = f"✅ Calculation Result: {result.result}\n"
                output += f"📝 Expression: {result.expression}\n"
                output += "🔢 Steps:\n"
                for step in result.steps:
                    output += f"  {step}\n"
                logger.debug(f"Calculation successful: {result.result}")
                return output
            else:
                logger.warning(f"Calculation failed: {result.error}")
                return f"❌ Calculation Error: {result.error}\n📝 Expression: {result.expression}"
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"❌ Calculator Error: {str(e)}"
    
    def _get_current_time(self, timezone: str = "UTC") -> str:
        """Get current time with enhanced formatting"""
        logger.debug(f"Getting current time for timezone: {timezone}")
        try:
            result = self.time_tools.get_current_time(timezone)
            
            if result.success:
                output = f"🕐 Current Time: {result.current_time}\n"
                output += f"🌍 Timezone: {result.timezone}\n"
                output += f"📅 Unix Timestamp: {result.unix_timestamp:.0f}\n"
                
                # Add additional time info
                time_info = self.time_tools.get_time_info(timezone)
                if time_info["success"]:
                    output += f"📊 Day of Week: {time_info['weekday']}\n"
                    output += f"📊 Month: {time_info['month_name']}\n"
                    output += f"📊 Day of Year: {time_info['day_of_year']}\n"
                    output += f"📊 Business Day: {'Yes' if time_info['is_business_day'] else 'No'}\n"
                
                logger.debug(f"Time retrieved successfully for {timezone}")
                return output
            else:
                logger.warning(f"Time retrieval failed: {result.error}")
                return f"❌ Time Error: {result.error}"
        except Exception as e:
            logger.error(f"Time tool error: {e}")
            return f"❌ Time Error: {str(e)}"
    
    def _convert_time(self, input_str: str) -> str:
        """Convert time between timezones"""
        logger.debug(f"Converting time: {input_str}")
        try:
            # Parse input: "time_string, from_timezone, to_timezone"
            parts = [part.strip() for part in input_str.split(',')]
            if len(parts) != 3:
                return "❌ Invalid format. Use: 'time_string, from_timezone, to_timezone'"
            
            time_str, from_tz, to_tz = parts
            result = self.time_tools.convert_time(time_str, from_tz, to_tz)
            
            if result.success:
                output = f"🔄 Time Conversion:\n"
                output += f"📅 From: {time_str} ({from_tz})\n"
                output += f"📅 To: {result.current_time}\n"
                output += f"🌍 Target Timezone: {result.timezone}\n"
                output += f"📅 Unix Timestamp: {result.unix_timestamp:.0f}\n"
                logger.debug(f"Time conversion successful: {time_str} -> {result.current_time}")
                return output
            else:
                logger.warning(f"Time conversion failed: {result.error}")
                return f"❌ Conversion Error: {result.error}"
        except Exception as e:
            logger.error(f"Time conversion error: {e}")
            return f"❌ Conversion Error: {str(e)}"
    
    def _calculate_time_difference(self, input_str: str) -> str:
        """Calculate time difference between two times"""
        logger.debug(f"Calculating time difference: {input_str}")
        try:
            # Parse input: "time1, time2, timezone"
            parts = [part.strip() for part in input_str.split(',')]
            if len(parts) != 3:
                return "❌ Invalid format. Use: 'time1, time2, timezone'"
            
            time1, time2, timezone = parts
            result = self.time_tools.get_time_difference(time1, time2, timezone)
            
            if result["success"]:
                output = f"⏱️ Time Difference:\n"
                output += f"📅 Time 1: {time1}\n"
                output += f"📅 Time 2: {time2}\n"
                output += f"🌍 Timezone: {timezone}\n"
                output += f"⏰ Difference: {result['formatted_difference']}\n"
                output += f"📊 In seconds: {result['difference_seconds']:.0f}\n"
                output += f"📊 In minutes: {result['difference_minutes']:.1f}\n"
                output += f"📊 In hours: {result['difference_hours']:.2f}\n"
                output += f"📊 In days: {result['difference_days']}\n"
                logger.debug(f"Time difference calculated: {result['formatted_difference']}")
                return output
            else:
                logger.warning(f"Time difference calculation failed: {result.get('error', 'Unknown error')}")
                return f"❌ Difference Calculation Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Time difference calculation error: {e}")
            return f"❌ Difference Calculation Error: {str(e)}"
    
    def _get_time_info(self, timezone: str = "UTC") -> str:
        """Get comprehensive time information"""
        logger.debug(f"Getting time info for timezone: {timezone}")
        try:
            result = self.time_tools.get_time_info(timezone)
            
            if result["success"]:
                output = f"📅 Comprehensive Time Info ({timezone}):\n"
                output += f"🕐 Current Time: {result['current_time']}\n"
                output += f"📊 Year: {result['year']}\n"
                output += f"📊 Month: {result['month_name']} ({result['month']})\n"
                output += f"📊 Day: {result['day']}\n"
                output += f"📊 Time: {result['hour']:02d}:{result['minute']:02d}:{result['second']:02d}\n"
                output += f"📊 Day of Week: {result['weekday']}\n"
                output += f"📊 Day of Year: {result['day_of_year']}\n"
                output += f"🏖️ Weekend: {'Yes' if result['is_weekend'] else 'No'}\n"
                output += f"📊 Business Day: {'Yes' if result['is_business_day'] else 'No'}\n"
                output += f"📅 Unix Timestamp: {result['unix_timestamp']:.0f}\n"
                logger.debug(f"Time info retrieved successfully for {timezone}")
                return output
            else:
                logger.warning(f"Time info retrieval failed: {result.get('error', 'Unknown error')}")
                return f"❌ Time Info Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Time info error: {e}")
            return f"❌ Time Info Error: {str(e)}"
    
    def _web_search(self, query: str) -> str:
        """Perform web search with improved error handling"""
        logger.debug(f"Performing web search: {query}")
        try:
            results = search_web(query, max_results=3)
            
            # Check if we got meaningful results
            if "Unable to perform real-time search" in results or "Search Temporarily Unavailable" in results:
                logger.warning(f"Web search unavailable for query: {query}")
                return f"Web search is currently experiencing high traffic. For '{query}', please try again in a few minutes or visit a search engine directly."
            
            logger.debug(f"Web search completed successfully for: {query}")
            return results
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return f"Web search failed: {str(e)}. Please try again later."
    
    def run(self, query: str, context: str = "") -> ReasoningResult:
        """Execute agent-based reasoning"""
        logger.info(f"Running agent-based reasoning for query: '{query}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            # Run the agent with the query and context
            response = self.agent.invoke({
                "input": query,
                "context": context,
                "chat_history": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            })
            
            # Extract steps from the agent's thought process
            steps = []
            if hasattr(response, 'intermediate_steps'):
                for step in response.intermediate_steps:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        steps.append(f"🤔 Thought: {action.log}")
                        steps.append(f"🔍 Action: Used {action.tool}")
                        steps.append(f"📝 Result: {observation}")
            
            # If no intermediate steps, try to extract from the output
            if not steps:
                output = response["output"] if isinstance(response, dict) else str(response)
                # Try to parse the output for any structured information
                if "Thought:" in output:
                    lines = output.split('\n')
                    for line in lines:
                        if line.strip() and any(keyword in line for keyword in ['Thought:', 'Action:', 'Result:', 'Observation:']):
                            steps.append(line.strip())
            
            # If still no steps, create a basic step from the output
            if not steps:
                output = response["output"] if isinstance(response, dict) else str(response)
                steps = [f"🤔 Processed query: {query}", f"📝 Generated response: {output[:100]}..."]
            
            logger.info("Agent-based reasoning completed successfully")
            return ReasoningResult(
                content=response["output"] if isinstance(response, dict) else str(response),
                reasoning_steps=steps,
                confidence=0.9,
                sources=["agent_based_reasoning"],
                intermediate_thoughts=steps
            )
        except Exception as e:
            logger.error(f"Agent-based reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content="",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )

class ReasoningChain:
    """Chain-of-Thought reasoning using a RunnableSequence."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing ReasoningChain with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_API_URL.replace("/api", "")
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
        
        # Updated prompt template format for RunnableSequence
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert assistant. Your goal is to provide accurate and "
                "well-reasoned answers. If context from documents is provided, use it "
                "to form your answer. If not, use your general knowledge. "
                "Think step by step to arrive at the best conclusion.\n\n"
                "Available Document Context:\n{context}\n"
            )),
            ("human", "{question}")
        ])
        
        try:
            # Refactored to use RunnableSequence as recommended
            self.chain = self.prompt_template | self.llm | StrOutputParser()
            logger.info("RunnableSequence (Chain) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RunnableSequence: {e}")
            raise
    
    def execute_reasoning(self, question: str, context: str) -> ReasoningResult:
        """Execute chain-of-thought reasoning"""
        logger.info(f"Executing chain-of-thought reasoning for question: '{question}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            # Start timer for execution
            start_time = time.time()
            
            # Run the chain with the question and context
            response_content = self.chain.invoke({
                "question": question,
                "context": context
            })
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # The output from the new chain is a string directly
            content = response_content.strip()
            
            # For CoT, the reasoning steps are the thought process of the model
            # which is implicitly part of the final output.
            steps = [f"Input: {question}", f"Thought Process:\n{content}"]
            
            # Simulate confidence (can be improved with more advanced methods)
            confidence = 0.85
            
            logger.info(f"Chain-of-thought reasoning completed in {execution_time:.2f}s")
            return ReasoningResult(
                content=content,
                reasoning_steps=steps,
                confidence=confidence,
                sources=["LLM"],
                success=True
            )
        except Exception as e:
            logger.error(f"Chain-of-thought reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content=f"An error occurred: {e}",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e)
            )

class MultiStepReasoning:
    """Performs multi-step reasoning by breaking down a query"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing MultiStepReasoning with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_API_URL.replace("/api", "")
            )
            self.model_name = model_name
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _generate_sub_queries(self, query: str, context: str) -> List[str]:
        """Generate sub-queries to break down the main query."""
        logger.debug(f"Generating sub-queries for: '{query}'")
        
        prompt = f"""
        Given the following user query and context, please break it down into smaller, logical sub-queries that need to be answered to fully address the original query.
        Each sub-query should be a self-contained question.

        CONTEXT:
        {context}

        ORIGINAL QUERY:
        "{query}"

        SUB-QUERIES (provide as a numbered list):
        1. ...
        2. ...
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            sub_queries = []
            for line in response.strip().split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                    sub_queries.append(line.strip().split('.', 1)[1].strip())
            
            if not sub_queries:
                sub_queries.append(query) # Fallback to original query
                
            logger.info(f"Generated {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Failed to generate sub-queries: {e}")
            return [query]  # Fallback to original query

    def _answer_sub_query(self, sub_query: str, context: str) -> str:
        """Answer a single sub-query using the provided context."""
        logger.debug(f"Answering sub-query: '{sub_query}'")
        
        prompt = f"""
        Please answer the following question based *only* on the provided context.
        If the context does not contain the answer, state that the information is not available in the provided documents.

        CONTEXT:
        {context}

        QUESTION:
        "{sub_query}"

        ANSWER:
        """
        
        try:
            answer = self.llm.invoke(prompt).strip()
            logger.debug(f"Sub-query answered, length: {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Failed to answer sub-query '{sub_query}': {e}")
            return f"Error answering sub-query: {str(e)}"

    def _synthesize_final_answer(self, original_query: str, sub_answers: List[Dict[str, str]]) -> str:
        """Synthesize the final answer from the sub-answers."""
        logger.debug(f"Synthesizing final answer for: '{original_query}'")
        
        formatted_sub_answers = "\n\n".join(
            [f"Sub-Question: {item['question']}\nAnswer: {item['answer']}" for item in sub_answers]
        )

        prompt = f"""
        Given the original query and the following sub-question answers, please synthesize a comprehensive final answer.
        Ensure the final answer is coherent, well-structured, and directly addresses the original query.

        ORIGINAL QUERY:
        "{original_query}"

        SUB-ANSWERS:
        {formatted_sub_answers}

        FINAL ANSWER:
        """
        
        try:
            final_answer = self.llm.invoke(prompt).strip()
            logger.info(f"Final answer synthesized, length: {len(final_answer)} characters")
            return final_answer
        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            return f"Error synthesizing final answer: {str(e)}"

    def step_by_step_reasoning(self, query: str, context: str) -> ReasoningResult:
        """Execute step-by-step reasoning by decomposing the query."""
        logger.info(f"Executing multi-step reasoning for query: '{query}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            intermediate_thoughts = []
            
            # 1. Generate sub-queries
            logger.info("Step 1: Generating sub-queries")
            sub_queries = self._generate_sub_queries(query, context)
            intermediate_thoughts.append(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
            
            # 2. Answer each sub-query
            logger.info("Step 2: Answering sub-queries")
            sub_answers = []
            for i, sub_q in enumerate(sub_queries, 1):
                logger.debug(f"Answering sub-query {i}/{len(sub_queries)}: '{sub_q}'")
                answer = self._answer_sub_query(sub_q, context)
                sub_answers.append({"question": sub_q, "answer": answer})
                intermediate_thoughts.append(f"Answered '{sub_q}': '{answer[:100]}...'")

            # 3. Synthesize the final answer
            logger.info("Step 3: Synthesizing final answer")
            final_answer = self._synthesize_final_answer(query, sub_answers)
            intermediate_thoughts.append("Synthesized final answer.")
            
            logger.info("Multi-step reasoning completed successfully")
            return ReasoningResult(
                content=final_answer,
                reasoning_steps=[f"Sub-query: {q}" for q in sub_queries] + [f"Final Answer: {final_answer}"],
                confidence=0.9,
                sources=["Document Analysis"],
                intermediate_thoughts=intermediate_thoughts
            )
        except Exception as e:
            logger.error(f"Multi-step reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content=f"An error occurred during multi-step reasoning: {e}",
                reasoning_steps=[],
                confidence=0.0,
                sources=[],
                success=False,
                error=str(e),
                intermediate_thoughts=intermediate_thoughts
            )

class ReasoningEngine:
    """Main reasoning engine"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing ReasoningEngine with model: {model_name}")
        self.model_name = model_name
        self.agent_reasoner = None
        self.chain_reasoner = None
        self.multi_step_reasoner = None
    
    def _retrieve_and_format_context(self, query: str, doc_processor: Optional[DocumentProcessor]) -> str:
        """Retrieve context from documents and format it for the prompt."""
        logger.info(f"Retrieving context for query: '{query}'")
        
        if not doc_processor:
            logger.info("No document processor available, using general knowledge")
            return "No document processor available. Relying on general knowledge."
        
        available_docs = doc_processor.get_available_documents()
        if not available_docs:
            logger.info("No documents available, using general knowledge")
            return "No documents have been uploaded or processed. Relying on general knowledge."

        logger.info(f"Searching for: {query}")
        logger.info(f"Available documents: {available_docs}")

        try:
            relevant_docs = doc_processor.search_documents(query)
            
            if not relevant_docs:
                logger.info("No relevant documents found")
                return "Could not find relevant information in the uploaded documents for your query."
            
            context_parts = []
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'Unknown')
                relevance = doc.metadata.get('relevance_score', 'N/A')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                context_parts.append(
                    f"Source: {source} (Relevance: {relevance})\nContent: {content_preview}\n---"
                )
            
            context = "\n".join(context_parts)
            logger.info(f"Context retrieved, length: {len(context)} characters")
            return context
        except Exception as e:
            logger.error(f"Error during document search: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return "An error occurred while searching the documents."

    def run(self, query: str, mode: str, document_processor: Optional[DocumentProcessor] = None) -> ReasoningResult:
        """Run reasoning with selected mode"""
        logger.info(f"Running reasoning with mode: {mode} for query: '{query}'")
        
        context = self._retrieve_and_format_context(query, document_processor)

        if mode == "Agent":
            if self.agent_reasoner is None:
                logger.info("Initializing agent reasoner")
                self.agent_reasoner = ReasoningAgent(model_name=self.model_name)
            return self.agent_reasoner.run(query, context=context)
        
        elif mode == "Chain-of-Thought":
            if self.chain_reasoner is None:
                logger.info("Initializing chain reasoner")
                self.chain_reasoner = ReasoningChain(model_name=self.model_name)
            return self.chain_reasoner.execute_reasoning(question=query, context=context)
            
        elif mode == "Multi-Step":
            if self.multi_step_reasoner is None:
                logger.info("Initializing multi-step reasoner")
                self.multi_step_reasoner = MultiStepReasoning(model_name=self.model_name)
            return self.multi_step_reasoner.step_by_step_reasoning(query, context)
            
        else:
            error_msg = f"Unknown reasoning mode: {mode}"
            logger.error(error_msg)
            raise ValueError(error_msg)
