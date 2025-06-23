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
    """Result from reasoning operations with clear separation of thought process and final answer"""
    content: str  # Final answer/conclusion (clean, direct response)
    reasoning_steps: List[str]  # Step-by-step thought process (internal reasoning)
    thought_process: str  # Formatted thought process for display
    final_answer: str  # Clean final answer for display
    confidence: float
    sources: List[str]
    reasoning_mode: str  # Which reasoning mode was used
    intermediate_thoughts: List[str] = None  # For storing intermediate reasoning steps
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0  # Time taken for reasoning

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
    
    def run(self, query: str, context: str = "", stream_callback=None) -> ReasoningResult:
        """Execute agent-based reasoning"""
        logger.info(f"Running agent-based reasoning for query: '{query}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            if stream_callback:
                stream_callback("🤖 **Agent-Based Reasoning**\nInitializing agent with tools and memory...\n")
            
            # Run the agent with the query and context
            response = self.agent.invoke({
                "input": query,
                "context": context,
                "chat_history": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            })
            
            if stream_callback:
                stream_callback("🔍 **Agent Execution Complete**\nAnalyzing agent's thought process and actions...\n")
            
            # Extract steps from the agent's thought process
            steps = []
            if hasattr(response, 'intermediate_steps'):
                if stream_callback:
                    stream_callback(f"📊 **Found {len(response.intermediate_steps)} intermediate steps**\n")
                
                for i, step in enumerate(response.intermediate_steps, 1):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        step_text = f"🤔 Thought: {action.log}"
                        steps.append(step_text)
                        
                        if stream_callback:
                            stream_callback(f"**Step {i}:** {step_text}\n")
                        
                        action_text = f"🔍 Action: Used {action.tool}"
                        steps.append(action_text)
                        
                        if stream_callback:
                            stream_callback(f"**Action {i}:** {action_text}\n")
                        
                        result_text = f"📝 Result: {observation[:200]}..." if len(observation) > 200 else f"📝 Result: {observation}"
                        steps.append(result_text)
                        
                        if stream_callback:
                            stream_callback(f"**Result {i}:** {result_text}\n\n")
            
            # If no intermediate steps, try to extract from the output
            if not steps:
                if stream_callback:
                    stream_callback("⚠️ **No intermediate steps found**\nExtracting information from agent output...\n")
                
                output = response["output"] if isinstance(response, dict) else str(response)
                # Try to parse the output for any structured information
                if "Thought:" in output:
                    lines = output.split('\n')
                    for line in lines:
                        if line.strip() and any(keyword in line for keyword in ['Thought:', 'Action:', 'Result:', 'Observation:']):
                            steps.append(line.strip())
                            if stream_callback:
                                stream_callback(f"📝 {line.strip()}\n")
            
            # If still no steps, create a basic step from the output
            if not steps:
                if stream_callback:
                    stream_callback("📝 **Creating basic step from output**\n")
                
                output = response["output"] if isinstance(response, dict) else str(response)
                steps = [f"🤔 Processed query: {query}", f"📝 Generated response: {output[:100]}..."]
                
                if stream_callback:
                    stream_callback(f"🤔 **Processed Query:** {query}\n")
                    stream_callback(f"📝 **Generated Response:** {output[:200]}...\n")
            
            if stream_callback:
                stream_callback("✅ **Agent Reasoning Complete!**\n")
            
            logger.info("Agent-based reasoning completed successfully")
            return ReasoningResult(
                content=response["output"] if isinstance(response, dict) else str(response),
                reasoning_steps=steps,
                thought_process="\n".join(steps),
                final_answer=response["output"] if isinstance(response, dict) else str(response),
                confidence=0.9,
                sources=["agent_based_reasoning"],
                reasoning_mode="Agent",
                intermediate_thoughts=steps
            )
        except Exception as e:
            logger.error(f"Agent-based reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            if stream_callback:
                stream_callback(f"❌ **Agent Error:** {str(e)}\n")
            
            return ReasoningResult(
                content="",
                reasoning_steps=[],
                thought_process="",
                final_answer="",
                confidence=0.0,
                sources=[],
                reasoning_mode="Agent",
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
    
    def execute_reasoning(self, question: str, context: str, stream_callback=None) -> ReasoningResult:
        """Execute chain-of-thought reasoning with clear separation of thought process and final answer"""
        execution_time = 0.0  # Ensure always defined
        logger.info(f"Executing chain-of-thought reasoning for question: '{question}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            # Start timer for execution
            start_time = time.time()
            
            if stream_callback:
                stream_callback("🧠 **Starting Chain-of-Thought Reasoning**\nAnalyzing the question step by step...\n")
            
            # Improved prompt that clearly separates thought process from final answer
            improved_prompt = f"""
            You are a helpful AI assistant. Use chain-of-thought reasoning to answer the following question.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            INSTRUCTIONS:
            1. First, think through the problem step by step. Show your reasoning process.
            2. Then, provide a clear, concise final answer.
            3. Clearly separate your thought process from your final answer.
            
            THOUGHT PROCESS:
            (Think through the problem step by step here...)
            
            FINAL ANSWER:
            (Provide your final, direct answer here...)
            """
            
            if stream_callback:
                stream_callback("💭 **Generating Thought Process**\nWorking through the problem systematically...\n")
            
            # Run the chain with the improved prompt
            response_content = self.llm.invoke(improved_prompt)
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Parse the response to separate thought process from final answer
            # Extract content from AIMessage object
            content = response_content.content if hasattr(response_content, 'content') else str(response_content)
            content = content.strip()
            
            if stream_callback:
                stream_callback("✅ **Thought Process Generated**\nParsing and organizing the response...\n")
            
            # Extract thought process and final answer
            thought_process, final_answer = self._parse_chain_of_thought_response(content)
            
            # Create reasoning steps for display
            reasoning_steps = [
                f"Question: {question}",
                f"Context Analysis: {len(context)} characters of context provided",
                f"Thought Process: {thought_process[:200]}..." if len(thought_process) > 200 else f"Thought Process: {thought_process}",
                f"Final Answer: {final_answer}"
            ]
            
            # Simulate confidence (can be improved with more advanced methods)
            confidence = 0.85
            
            if stream_callback:
                stream_callback("🎯 **Reasoning Complete!**\n")
            
            logger.info(f"Chain-of-thought reasoning completed in {execution_time:.2f}s")
            return ReasoningResult(
                content=final_answer,  # Use final answer as main content
                reasoning_steps=reasoning_steps,
                thought_process=thought_process,
                final_answer=final_answer,
                confidence=confidence,
                sources=["LLM"],
                reasoning_mode="Chain-of-Thought",
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Chain-of-thought reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content=f"An error occurred: {e}",
                reasoning_steps=[],
                thought_process="",
                final_answer="",
                confidence=0.0,
                sources=[],
                reasoning_mode="Chain-of-Thought",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _parse_chain_of_thought_response(self, response: str) -> tuple[str, str]:
        """Parse the chain-of-thought response to separate thought process from final answer"""
        try:
            # Look for clear separators
            if "FINAL ANSWER:" in response:
                parts = response.split("FINAL ANSWER:", 1)
                thought_process = parts[0].replace("THOUGHT PROCESS:", "").strip()
                final_answer = parts[1].strip()
            elif "Final Answer:" in response:
                parts = response.split("Final Answer:", 1)
                thought_process = parts[0].replace("Thought Process:", "").strip()
                final_answer = parts[1].strip()
            elif "ANSWER:" in response:
                parts = response.split("ANSWER:", 1)
                thought_process = parts[0].replace("THOUGHT PROCESS:", "").strip()
                final_answer = parts[1].strip()
            else:
                # Fallback: try to split on common patterns
                lines = response.split('\n')
                thought_lines = []
                answer_lines = []
                in_answer = False
                
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['final answer', 'answer:', 'conclusion']):
                        in_answer = True
                        answer_lines.append(line)
                    elif in_answer:
                        answer_lines.append(line)
                    else:
                        thought_lines.append(line)
                
                thought_process = '\n'.join(thought_lines).strip()
                final_answer = '\n'.join(answer_lines).strip()
            
            # Clean up the extracted parts
            thought_process = thought_process.strip()
            final_answer = final_answer.strip()
            
            # If parsing failed, use fallback
            if not thought_process or not final_answer:
                # Split response roughly in half
                mid_point = len(response) // 2
                thought_process = response[:mid_point].strip()
                final_answer = response[mid_point:].strip()
            
            return thought_process, final_answer
            
        except Exception as e:
            logger.warning(f"Failed to parse chain-of-thought response: {e}")
            # Fallback: return the full response as thought process, empty final answer
            return response.strip(), ""

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
            
            # Extract content from AIMessage object
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            sub_queries = []
            for line in response_content.strip().split('\n'):
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
            answer = self.llm.invoke(prompt)
            
            # Extract content from AIMessage object
            answer_content = answer.content if hasattr(answer, 'content') else str(answer)
            
            logger.debug(f"Sub-query answered, length: {len(answer_content)} characters")
            return answer_content.strip()
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
            final_answer = self.llm.invoke(prompt)
            
            # Extract content from AIMessage object
            final_answer_content = final_answer.content if hasattr(final_answer, 'content') else str(final_answer)
            
            logger.info(f"Final answer synthesized, length: {len(final_answer_content)} characters")
            return final_answer_content.strip()
        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            return f"Error synthesizing final answer: {str(e)}"

    def step_by_step_reasoning(self, query: str, context: str, stream_callback=None) -> ReasoningResult:
        """Execute step-by-step reasoning by decomposing the query with clear separation of analysis and final answer."""
        logger.info(f"Executing multi-step reasoning for query: '{query}'")
        logger.debug(f"Context length: {len(context)} characters")
        
        try:
            start_time = time.time()
            intermediate_thoughts = []
            analysis_steps = []
            
            # 1. Generate sub-queries
            logger.info("Step 1: Generating sub-queries")
            if stream_callback:
                stream_callback("🔍 **Step 1: Analyzing Query**\nBreaking down the complex query into manageable sub-questions...\n")
            
            sub_queries = self._generate_sub_queries(query, context)
            intermediate_thoughts.append(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
            
            if stream_callback:
                stream_callback(f"✅ **Generated {len(sub_queries)} sub-queries:**\n")
                for i, sub_q in enumerate(sub_queries, 1):
                    stream_callback(f"{i}. {sub_q}\n")
                stream_callback("\n")
            
            analysis_steps.append(f"**Query Analysis:** Breaking down '{query}' into {len(sub_queries)} sub-questions")
            
            # 2. Answer each sub-query
            logger.info("Step 2: Answering sub-queries")
            if stream_callback:
                stream_callback("🔍 **Step 2: Answering Sub-Queries**\nProcessing each sub-question individually...\n")
            
            sub_answers = []
            for i, sub_q in enumerate(sub_queries, 1):
                logger.debug(f"Answering sub-query {i}/{len(sub_queries)}: '{sub_q}'")
                
                if stream_callback:
                    stream_callback(f"**Processing Sub-Question {i}:** {sub_q}\n")
                
                answer = self._answer_sub_query(sub_q, context)
                sub_answers.append({"question": sub_q, "answer": answer})
                intermediate_thoughts.append(f"Answered '{sub_q}': '{answer[:100]}...'")
                
                if stream_callback:
                    stream_callback(f"✅ **Answer {i}:** {answer[:200]}...\n\n")
                
                analysis_steps.append(f"**Step {i}:** {sub_q}")
                analysis_steps.append(f"  → {answer[:150]}...")

            # 3. Synthesize the final answer
            logger.info("Step 3: Synthesizing final answer")
            if stream_callback:
                stream_callback("🔍 **Step 3: Synthesizing Final Answer**\nCombining all sub-answers into a comprehensive response...\n")
            
            final_answer = self._synthesize_final_answer(query, sub_answers)
            intermediate_thoughts.append("Synthesized final answer.")
            
            if stream_callback:
                stream_callback("✅ **Synthesis Complete!**\n")
            
            analysis_steps.append("**Synthesis:** Combining sub-answers into final response")
            
            # Create reasoning steps for display
            reasoning_steps = [
                f"Original Query: {query}",
                f"Sub-queries Generated: {len(sub_queries)}",
                f"Analysis Completed: {len(sub_answers)} sub-answers collected",
                f"Final Synthesis: Complete"
            ]
            
            execution_time = time.time() - start_time
            
            logger.info("Multi-step reasoning completed successfully")
            return ReasoningResult(
                content=final_answer,
                reasoning_steps=reasoning_steps,
                thought_process="\n".join(analysis_steps),
                final_answer=final_answer,
                confidence=0.9,
                sources=["Document Analysis"],
                reasoning_mode="Multi-Step",
                intermediate_thoughts=intermediate_thoughts,
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Multi-step reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content=f"An error occurred during multi-step reasoning: {e}",
                reasoning_steps=[],
                thought_process="",
                final_answer="",
                confidence=0.0,
                sources=[],
                reasoning_mode="Multi-Step",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )

class StandardReasoning:
    """Standard reasoning for simple, direct responses."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing StandardReasoning with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_API_URL.replace("/api", "")
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def execute_reasoning(self, query: str, context: str, stream_callback=None) -> ReasoningResult:
        """Execute standard reasoning for simple, direct responses"""
        logger.info(f"Executing standard reasoning for query: '{query}'")
        
        try:
            start_time = time.time()
            
            if stream_callback:
                stream_callback("📝 **Standard Reasoning**\nProcessing query with direct response approach...\n")
            
            # Create a simple prompt for direct response
            prompt = f"""
            Context: {context}
            
            Question: {query}
            
            Please provide a clear, direct answer to the question above. 
            If context is provided, use it to inform your response.
            If no context is available, use your general knowledge.
            
            Answer:
            """
            
            if stream_callback:
                stream_callback("💭 **Generating Response**\nCreating direct answer...\n")
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from AIMessage object
            response_content = response.content if hasattr(response, 'content') else str(response)
            final_answer = response_content.strip()
            
            if stream_callback:
                stream_callback("✅ **Response Generated**\nFinalizing answer...\n")
            
            execution_time = time.time() - start_time
            
            # Create reasoning steps for display
            reasoning_steps = [
                f"Query: {query}",
                f"Context: {len(context)} characters provided" if context else "Context: None provided",
                f"Response: Direct answer generated",
                f"Final Answer: {final_answer[:100]}..." if len(final_answer) > 100 else f"Final Answer: {final_answer}"
            ]
            
            if stream_callback:
                stream_callback("🎯 **Standard Reasoning Complete!**\n")
            
            logger.info(f"Standard reasoning completed in {execution_time:.2f}s")
            return ReasoningResult(
                content=final_answer,
                reasoning_steps=reasoning_steps,
                thought_process=f"Direct response using standard reasoning approach",
                final_answer=final_answer,
                confidence=0.8,
                sources=["LLM"],
                reasoning_mode="Standard",
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Standard reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            if stream_callback:
                stream_callback(f"❌ **Standard Reasoning Error:** {str(e)}\n")
            
            return ReasoningResult(
                content=f"An error occurred during standard reasoning: {e}",
                reasoning_steps=[],
                thought_process="",
                final_answer="",
                confidence=0.0,
                sources=[],
                reasoning_mode="Standard",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )

class AutoReasoning:
    """Intelligently selects and applies the best reasoning mode for a given query"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing AutoReasoning with model: {model_name}")
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_API_URL.replace("/api", "")
            )
            self.model_name = model_name
            
            # Initialize all reasoning modes
            self.chain_of_thought = ReasoningChain(model_name)
            self.multi_step = MultiStepReasoning(model_name)
            self.agent = ReasoningAgent(model_name)
            self.standard = StandardReasoning(model_name)
            
            logger.info("AutoReasoning initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AutoReasoning: {e}")
            raise
    
    def _analyze_query_complexity(self, query: str, context: str) -> Dict[str, Any]:
        """Analyze query complexity and type to determine best reasoning approach"""
        logger.debug(f"Analyzing query complexity: '{query}'")
        
        analysis_prompt = f"""
        Analyze the following query and context to determine the best reasoning approach.
        
        QUERY: "{query}"
        CONTEXT: "{context[:500]}..." (truncated)
        
        Consider these factors:
        1. Query complexity (simple vs complex)
        2. Query type (factual, analytical, creative, mathematical, etc.)
        3. Whether tools might be needed
        4. Whether multi-step decomposition would help
        5. Whether chain-of-thought reasoning would be beneficial
        
        Respond with a JSON object:
        {{
            "complexity": "simple|moderate|complex",
            "query_type": "factual|analytical|creative|mathematical|problem_solving|other",
            "needs_tools": true|false,
            "needs_decomposition": true|false,
            "needs_chain_of_thought": true|false,
            "recommended_mode": "Standard|Chain-of-Thought|Multi-Step|Agent-Based",
            "reasoning": "brief explanation of why this mode is recommended"
        }}
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            
            # Extract content from AIMessage object
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            import json
            try:
                analysis = json.loads(response_content.strip())
                logger.info(f"Query analysis: {analysis['recommended_mode']} - {analysis['reasoning']}")
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                logger.warning("Failed to parse JSON analysis, using fallback")
                return self._fallback_analysis(query, context)
                
        except Exception as e:
            logger.error(f"Failed to analyze query complexity: {e}")
            return self._fallback_analysis(query, context)
    
    def _fallback_analysis(self, query: str, context: str) -> Dict[str, Any]:
        """Fallback analysis when the main analysis fails"""
        query_lower = query.lower()
        
        # Simple heuristics for mode selection
        needs_tools = any(word in query_lower for word in ['calculate', 'compute', 'time', 'date', 'search', 'web'])
        needs_decomposition = len(query.split()) > 15 or any(word in query_lower for word in ['analyze', 'compare', 'explain', 'how', 'why'])
        needs_chain_of_thought = any(word in query_lower for word in ['solve', 'problem', 'figure out', 'determine'])
        
        if needs_tools:
            return {
                "complexity": "moderate",
                "query_type": "problem_solving",
                "needs_tools": True,
                "needs_decomposition": needs_decomposition,
                "needs_chain_of_thought": needs_chain_of_thought,
                "recommended_mode": "Agent-Based",
                "reasoning": "Query requires tools for calculation, time, or web search"
            }
        elif needs_decomposition:
            return {
                "complexity": "complex",
                "query_type": "analytical",
                "needs_tools": False,
                "needs_decomposition": True,
                "needs_chain_of_thought": needs_chain_of_thought,
                "recommended_mode": "Multi-Step",
                "reasoning": "Complex query that benefits from step-by-step decomposition"
            }
        elif needs_chain_of_thought:
            return {
                "complexity": "moderate",
                "query_type": "problem_solving",
                "needs_tools": False,
                "needs_decomposition": False,
                "needs_chain_of_thought": True,
                "recommended_mode": "Chain-of-Thought",
                "reasoning": "Query requires step-by-step reasoning"
            }
        else:
            return {
                "complexity": "simple",
                "query_type": "factual",
                "needs_tools": False,
                "needs_decomposition": False,
                "needs_chain_of_thought": False,
                "recommended_mode": "Standard",
                "reasoning": "Simple factual query that doesn't require special reasoning"
            }
    
    def auto_reason(self, query: str, context: str, stream_callback=None) -> ReasoningResult:
        """Automatically select and apply the best reasoning mode"""
        logger.info(f"Executing auto reasoning for query: '{query}'")
        
        try:
            start_time = time.time()
            
            if stream_callback:
                stream_callback("🤖 **Auto Reasoning Mode**\nAnalyzing query complexity to select the best reasoning approach...\n")
            
            # Analyze query to determine best approach
            analysis = self._analyze_query_complexity(query, context)
            recommended_mode = analysis["recommended_mode"]
            
            if stream_callback:
                stream_callback(f"📊 **Query Analysis Complete**\n")
                stream_callback(f"• **Complexity:** {analysis.get('complexity', 'unknown')}\n")
                stream_callback(f"• **Query Type:** {analysis.get('query_type', 'unknown')}\n")
                stream_callback(f"• **Needs Tools:** {analysis.get('needs_tools', False)}\n")
                stream_callback(f"• **Needs Decomposition:** {analysis.get('needs_decomposition', False)}\n")
                stream_callback(f"• **Needs Chain-of-Thought:** {analysis.get('needs_chain_of_thought', False)}\n")
                stream_callback(f"\n🎯 **Selected Mode:** {recommended_mode}\n")
                stream_callback(f"💡 **Reasoning:** {analysis.get('reasoning', 'No explanation provided')}\n\n")
            
            logger.info(f"Auto-selected reasoning mode: {recommended_mode}")
            
            # Execute reasoning with the recommended mode
            if recommended_mode == "Chain-of-Thought":
                if stream_callback:
                    stream_callback("🧠 **Executing Chain-of-Thought Reasoning**\n")
                result = self.chain_of_thought.execute_reasoning(query, context, stream_callback)
            elif recommended_mode == "Multi-Step":
                if stream_callback:
                    stream_callback("🔍 **Executing Multi-Step Reasoning**\n")
                result = self.multi_step.step_by_step_reasoning(query, context, stream_callback)
            elif recommended_mode == "Agent-Based":
                if stream_callback:
                    stream_callback("🤖 **Executing Agent-Based Reasoning**\n")
                result = self.agent.run(query, context, stream_callback)
            elif recommended_mode == "Standard":
                if stream_callback:
                    stream_callback("📝 **Executing Standard Reasoning**\n")
                result = self.standard.execute_reasoning(query, context, stream_callback)
            
            # Update the result with auto reasoning metadata
            result.reasoning_mode = f"Auto ({recommended_mode})"
            result.execution_time = time.time() - start_time
            
            if stream_callback:
                stream_callback(f"✅ **Auto Reasoning Complete!**\nUsed {recommended_mode} mode in {result.execution_time:.2f}s\n")
            
            logger.info(f"Auto reasoning completed in {result.execution_time:.2f}s using {recommended_mode}")
            return result
            
        except Exception as e:
            logger.error(f"Auto reasoning failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ReasoningResult(
                content=f"An error occurred during auto reasoning: {e}",
                reasoning_steps=[],
                thought_process="",
                final_answer="",
                confidence=0.0,
                sources=[],
                reasoning_mode="Auto (Error)",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )

class ReasoningEngine:
    """Main reasoning engine"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Initializing ReasoningEngine with model: {model_name}")
        self.model_name = model_name
        self.agent_reasoner = None
        self.chain_reasoner = None
        self.multi_step_reasoner = None
        self.standard_reasoner = None
    
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
            
        elif mode == "Standard":
            if self.standard_reasoner is None:
                logger.info("Initializing standard reasoner")
                self.standard_reasoner = StandardReasoning(model_name=self.model_name)
            return self.standard_reasoner.execute_reasoning(query, context)
            
        else:
            error_msg = f"Unknown reasoning mode: {mode}"
            logger.error(error_msg)
            raise ValueError(error_msg)
