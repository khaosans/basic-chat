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
        
        # Run reasoning with progress callback
        def progress_callback(message: str):
            """Callback to update progress during reasoning"""
            # Estimate progress based on message content
            if "starting" in message.lower() or "initializing" in message.lower():
                progress = 0.4
            elif "processing" in message.lower() or "analyzing" in message.lower():
                progress = 0.6
            elif "generating" in message.lower() or "creating" in message.lower():
                progress = 0.8
            elif "complete" in message.lower() or "finalizing" in message.lower():
                progress = 0.9
            else:
                progress = 0.7
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'task_id': task_id,
                    'progress': progress,
                    'status': message
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
            # Simulate file upload object
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