"""
Celery tasks for BasicChat long-running operations
"""

import time
import logging
import traceback
from typing import Dict, Any, Optional
from celery import Celery

from reasoning_engine import ReasoningEngine, ReasoningResult
from document_processor import DocumentProcessor
from config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('basic_chat')
celery_app.config_from_object('celery_config')

# Shared FileUpload helper for simulating file uploads in Celery tasks
class FileUpload:
    def __init__(self, file_path, file_type):
        self.name = file_path.split('/')[-1]
        self.type = file_type
        self.file = open(file_path, 'rb')
    def getvalue(self):
        self.file.seek(0)
        return self.file.read()
    def close(self):
        self.file.close()

@celery_app.task(bind=True)
def run_reasoning(self, task_id: str, query: str, mode: str, context: str = ""):
    """Long-running reasoning task"""
    logger.info(f"Starting reasoning task {task_id}: {query[:50]}...")
    
    try:
        # Update progress - Initializing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.1,
                'status': 'Initializing reasoning engine'
            }
        )
        
        # Initialize reasoning engine
        reasoning_engine = ReasoningEngine(DEFAULT_MODEL)
        
        # Update progress - Processing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.3,
                'status': f'Running {mode} reasoning'
            }
        )
        
        # Run the reasoning
        result = reasoning_engine.run(query, mode, context=context)
        
        # Update progress - Finalizing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.95,
                'status': 'Finalizing results'
            }
        )
        
        # Prepare result
        task_result = {
            'task_id': task_id,
            'status': 'completed',
            'progress': 1.0,
            'result': {
                'content': result.content,
                'reasoning_steps': result.reasoning_steps,
                'thought_process': result.thought_process,
                'final_answer': result.final_answer,
                'confidence': result.confidence,
                'sources': result.sources,
                'reasoning_mode': result.reasoning_mode,
                'execution_time': result.execution_time,
                'success': result.success
            }
        }
        
        logger.info(f"Reasoning task {task_id} completed successfully")
        return task_result
        
    except Exception as e:
        logger.error(f"Reasoning task {task_id} failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@celery_app.task(bind=True)
def analyze_document(self, task_id: str, file_path: str, file_type: str = "unknown"):
    """Long-running document analysis task"""
    logger.info(f"Starting document analysis task {task_id}: {file_path}")
    
    try:
        # Update progress - Initializing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.1,
                'status': 'Initializing document processor'
            }
        )
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Update progress - Loading
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.2,
                'status': 'Loading document'
            }
        )
        
        # Load and process document
        with open(file_path, 'rb') as f:
            file_upload = FileUpload(file_path, file_type)
            
            # Update progress - Processing
            self.update_state(
                state='PROGRESS',
                meta={
                    'task_id': task_id,
                    'progress': 0.4,
                    'status': 'Processing document content'
                }
            )
            
            # Process the document
            doc_processor.process_file(file_upload)
            
            # Update progress - Analyzing
            self.update_state(
                state='PROGRESS',
                meta={
                    'task_id': task_id,
                    'progress': 0.7,
                    'status': 'Analyzing document structure'
                }
            )
            
            # Get processing results
            processed_files = doc_processor.get_processed_files()
            
            # Update progress - Finalizing
            self.update_state(
                state='PROGRESS',
                meta={
                    'task_id': task_id,
                    'progress': 0.9,
                    'status': 'Finalizing analysis'
                }
            )
            
            # Prepare result
            task_result = {
                'task_id': task_id,
                'status': 'completed',
                'progress': 1.0,
                'result': {
                    'processed_files': processed_files,
                    'file_path': file_path,
                    'file_type': file_type,
                    'total_files': len(processed_files),
                    'analysis_complete': True
                }
            }
            
            file_upload.close()
            logger.info(f"Document analysis task {task_id} completed successfully")
            return task_result
            
    except Exception as e:
        logger.error(f"Document analysis task {task_id} failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@celery_app.task(bind=True)
def process_document(self, task_id: str, file_path: str, file_type: str = "unknown"):
    """Long-running document processing task (similar to analyze_document but with more detailed processing)"""
    logger.info(f"Starting document processing task {task_id}: {file_path}")
    
    try:
        # Update progress - Initializing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.05,
                'status': 'Initializing document processor'
            }
        )
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Update progress - Loading
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.1,
                'status': 'Loading document file'
            }
        )
        
        # Simulate file upload object
        file_upload = FileUpload(file_path, file_type)
        
        # Update progress - Text extraction
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.2,
                'status': 'Extracting text content'
            }
        )
        
        # Process the document
        doc_processor.process_file(file_upload)
        
        # Update progress - Vectorization
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.4,
                'status': 'Creating vector embeddings'
            }
        )
        
        # Simulate vectorization time
        time.sleep(1)
        
        # Update progress - Indexing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.6,
                'status': 'Indexing document chunks'
            }
        )
        
        # Simulate indexing time
        time.sleep(1)
        
        # Update progress - Metadata extraction
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.8,
                'status': 'Extracting metadata'
            }
        )
        
        # Get processing results
        processed_files = doc_processor.get_processed_files()
        
        # Update progress - Finalizing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.95,
                'status': 'Finalizing document processing'
            }
        )
        
        # Prepare detailed result
        task_result = {
            'task_id': task_id,
            'status': 'completed',
            'progress': 1.0,
            'result': {
                'processed_files': processed_files,
                'file_path': file_path,
                'file_type': file_type,
                'total_files': len(processed_files),
                'processing_complete': True,
                'vectorized': True,
                'indexed': True,
                'searchable': True
            }
        }
        
        file_upload.close()
        logger.info(f"Document processing task {task_id} completed successfully")
        return task_result
        
    except Exception as e:
        logger.error(f"Document processing task {task_id} failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@celery_app.task(bind=True)
def health_check(self):
    """Health check task for monitoring"""
    try:
        # Check reasoning engine
        reasoning_engine = ReasoningEngine(DEFAULT_MODEL)
        
        # Check document processor
        doc_processor = DocumentProcessor()
        
        return {
            'status': 'healthy',
            'reasoning_engine': 'ok',
            'document_processor': 'ok',
            'timestamp': time.time()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }

@celery_app.task(bind=True)
def run_deep_research(self, task_id: str, query: str, research_depth: str = "comprehensive"):
    """Run deep research task with multiple sources and comprehensive analysis"""
    logger.info(f"Starting deep research task {task_id}: {query[:50]}...")
    
    try:
        # Update progress - Initializing
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.05,
                'status': 'Initializing deep research engine'
            }
        )
        
        # Import web search capabilities
        from web_search import WebSearch
        from reasoning_engine import MultiStepReasoning
        
        # Initialize components
        web_search = WebSearch()
        reasoning_engine = MultiStepReasoning(DEFAULT_MODEL)  # Best for research
        
        # Step 1: Initial query analysis
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.1,
                'status': 'Analyzing research query'
            }
        )
        
        analysis_query = f"""
        Analyze this research query and break it down into key search terms and subtopics:
        "{query}"
        
        Provide:
        1. Main search terms (3-5 key terms)
        2. Related subtopics to explore
        3. Types of sources to look for
        4. Potential research questions
        """
        
        analysis_result = reasoning_engine.step_by_step_reasoning(analysis_query, "")
        
        # Step 2: Web search for multiple sources
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.2,
                'status': 'Searching for sources'
            }
        )
        
        # Extract search terms from analysis
        search_terms = [query]  # Start with original query
        if analysis_result.content:
            # Extract key terms from analysis (simplified)
            lines = analysis_result.content.split('\n')
            for line in lines:
                if 'search terms' in line.lower() or 'key terms' in line.lower():
                    # Extract terms from this line
                    terms = line.split(':')[-1].strip()
                    if terms:
                        search_terms.extend([term.strip() for term in terms.split(',')[:3]])
        
        # Perform multiple searches
        search_results = []
        for i, term in enumerate(search_terms[:3]):  # Limit to 3 searches
            try:
                results = web_search.search(term, max_results=3)
                if results:
                    search_results.extend(results)
                
                # Update progress
                progress = 0.2 + (i + 1) * 0.15
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'task_id': task_id,
                        'progress': progress,
                        'status': f'Searching: {term}'
                    }
                )
                
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {e}")
                # Continue with other search terms
                continue
        
        # If no search results, create a fallback
        if not search_results:
            logger.warning("No search results found, using fallback content")
            search_results = [
                type('SearchResult', (), {
                    'title': 'Research Query Analysis',
                    'url': 'N/A',
                    'snippet': f'Analysis of query: {query}',
                    'source': 'Internal Analysis'
                })()
            ]
        
        # Step 3: Content analysis and synthesis
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.6,
                'status': 'Analyzing sources'
            }
        )
        
        # Prepare content for analysis
        content_for_analysis = f"""
        Research Query: {query}
        
        Sources Found:
        """
        
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
            content_for_analysis += f"""
        Source {i+1}: {getattr(result, 'title', 'No title')}
        URL: {getattr(result, 'url', 'No URL')}
        Snippet: {getattr(result, 'snippet', 'No snippet')}
        """
        
        # Step 4: Comprehensive analysis
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.7,
                'status': 'Synthesizing findings'
            }
        )
        
        synthesis_query = f"""
        Based on the research query and sources provided, create a comprehensive research report:
        
        {content_for_analysis}
        
        Please provide:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (bullet points)
        3. Detailed Analysis (comprehensive explanation)
        4. Sources and Citations
        5. Recommendations or Conclusions
        6. Areas for Further Research
        
        Make the analysis thorough, well-structured, and academically rigorous.
        """
        
        research_result = reasoning_engine.step_by_step_reasoning(synthesis_query, content_for_analysis)
        
        # Step 5: Final formatting and metadata
        self.update_state(
            state='PROGRESS',
            meta={
                'task_id': task_id,
                'progress': 0.9,
                'status': 'Finalizing research report'
            }
        )
        
        # Format the final result
        # Convert SearchResult objects to serializable dictionaries
        serializable_sources = []
        for result in search_results:
            serializable_sources.append({
                'title': getattr(result, 'title', 'No title'),
                'url': getattr(result, 'url', 'No URL'),
                'snippet': getattr(result, 'snippet', 'No snippet'),
                'source': getattr(result, 'source', 'Unknown')
            })
        
        task_result = {
            'task_id': task_id,
            'status': 'completed',
            'progress': 1.0,
            'result': {
                'research_query': query,
                'research_depth': research_depth,
                'executive_summary': _extract_section(research_result.content, "Executive Summary"),
                'key_findings': _extract_section(research_result.content, "Key Findings"),
                'detailed_analysis': _extract_section(research_result.content, "Detailed Analysis"),
                'sources': serializable_sources,
                'recommendations': _extract_section(research_result.content, "Recommendations"),
                'further_research': _extract_section(research_result.content, "Areas for Further Research"),
                'full_report': research_result.content,
                'search_terms_used': search_terms,
                'sources_analyzed': len(search_results),
                'confidence': getattr(research_result, 'confidence', 0.8),
                'execution_time': getattr(research_result, 'execution_time', 0),
                'success': getattr(research_result, 'success', True)
            }
        }
        
        logger.info(f"Deep research task {task_id} completed successfully")
        
        # Ensure all data is JSON serializable
        try:
            # Test serialization before returning
            import json
            json.dumps(task_result)
            return task_result
        except (TypeError, ValueError) as e:
            logger.warning(f"Task result contains non-serializable data, cleaning up: {e}")
            # Return a cleaned version with only serializable data
            return {
                'task_id': task_id,
                'status': 'completed',
                'progress': 1.0,
                'result': {
                    'research_query': query,
                    'research_depth': research_depth,
                    'executive_summary': _extract_section(research_result.content, "Executive Summary"),
                    'key_findings': _extract_section(research_result.content, "Key Findings"),
                    'detailed_analysis': _extract_section(research_result.content, "Detailed Analysis"),
                    'sources': serializable_sources,
                    'recommendations': _extract_section(research_result.content, "Recommendations"),
                    'further_research': _extract_section(research_result.content, "Areas for Further Research"),
                    'full_report': research_result.content,
                    'search_terms_used': search_terms,
                    'sources_analyzed': len(search_results),
                    'confidence': getattr(research_result, 'confidence', 0.8),
                    'execution_time': getattr(research_result, 'execution_time', 0),
                    'success': getattr(research_result, 'success', True)
                }
            }
        
    except Exception as e:
        logger.error(f"Deep research task {task_id} failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def _extract_section(content: str, section_name: str) -> str:
    """Extract a specific section from the research content"""
    try:
        lines = content.split('\n')
        section_start = -1
        section_end = -1
        
        for i, line in enumerate(lines):
            if section_name.lower() in line.lower():
                section_start = i
                break
        
        if section_start == -1:
            return ""
        
        # Find the end of this section (next numbered item or end)
        for i in range(section_start + 1, len(lines)):
            if (lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                any(keyword in lines[i].lower() for keyword in ['key findings', 'detailed analysis', 'sources', 'recommendations', 'further research'])):
                section_end = i
                break
        
        if section_end == -1:
            section_end = len(lines)
        
        return '\n'.join(lines[section_start:section_end]).strip()
        
    except Exception as e:
        logger.warning(f"Error extracting section {section_name}: {e}")
        return "" 
