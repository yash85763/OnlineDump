# tasks/pdf_processing.py - Celery tasks for PDF processing

from celery import Celery
from config.database import (
    store_pdf_data, store_analysis_data, store_clause_data,
    get_pdf_by_hash, update_batch_job_status
)
from utils.obfuscation import ContentObfuscator, ObfuscationConfig
from utils.pdf_handler import PDFHandler
from utils.contract_analyzer import ContractAnalyzer
from utils.hash_utils import calculate_file_hash
import json
import tempfile
import os
from datetime import datetime

# Get Celery app
from config.database import get_celery_app
celery_app = get_celery_app()

@celery_app.task(bind=True)
def process_pdf_async(self, pdf_bytes_b64, pdf_name, session_id, obfuscation_config=None):
    """
    Asynchronous PDF processing task with automatic obfuscation.
    
    Args:
        self: Celery task instance
        pdf_bytes_b64: Base64 encoded PDF bytes
        pdf_name: Name of the PDF file
        session_id: User session ID
        obfuscation_config: Optional custom obfuscation configuration
        
    Returns:
        dict: Processing results
    """
    
    try:
        # Update task status
        self.update_state(state='PROCESSING', meta={'step': 'Starting PDF processing'})
        
        # Decode PDF bytes
        import base64
        pdf_bytes = base64.b64decode(pdf_bytes_b64)
        
        # Calculate file hash for deduplication
        file_hash = calculate_file_hash(pdf_bytes)
        
        # Check if PDF already exists
        existing_pdf = get_pdf_by_hash(file_hash)
        if existing_pdf:
            return {
                'success': True,
                'pdf_id': existing_pdf['id'],
                'message': 'PDF already processed (deduplication)',
                'existing': True
            }
        
        # Stage 1: PDF Parsing and Obfuscation
        self.update_state(state='PROCESSING', meta={'step': 'Parsing PDF and applying obfuscation'})
        
        pdf_result = process_pdf_with_obfuscation(pdf_bytes, pdf_name, file_hash, session_id, obfuscation_config)
        
        if not pdf_result['success']:
            return pdf_result
        
        pdf_id = pdf_result['pdf_id']
        
        # Stage 2: Contract Analysis
        self.update_state(state='PROCESSING', meta={'step': 'Analyzing contract content'})
        
        analysis_result = analyze_contract_content(pdf_id, session_id)
        
        if not analysis_result['success']:
            return analysis_result
        
        analysis_id = analysis_result['analysis_id']
        
        # Stage 3: Clause Extraction
        self.update_state(state='PROCESSING', meta={'step': 'Extracting clauses'})
        
        clause_result = extract_and_store_clauses(analysis_id, analysis_result['analysis_data'])
        
        return {
            'success': True,
            'pdf_id': pdf_id,
            'analysis_id': analysis_id,
            'clause_ids': clause_result['clause_ids'],
            'analysis_data': analysis_result['analysis_data'],
            'obfuscation_stats': pdf_result.get('obfuscation_stats'),
            'message': 'PDF processing completed successfully'
        }
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        return {
            'success': False,
            'error': str(e),
            'message': 'PDF processing failed'
        }

def process_pdf_with_obfuscation(pdf_bytes, pdf_name, file_hash, session_id, obfuscation_config=None):
    """Process PDF with automatic obfuscation"""
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # Initialize PDF handler and obfuscator
            pdf_handler = PDFHandler()
            
            # Create default obfuscation config if none provided
            if not obfuscation_config:
                obfuscation_config = ObfuscationConfig(
                    enable_page_filtering=True,
                    word_count_threshold_multiplier=1.0,
                    enable_keyword_filtering=True,
                    obfuscation_keywords=[
                        "confidential", "proprietary", "ssn", "social security",
                        "credit card", "bank account", "password", "medical record"
                    ],
                    keyword_combinations=[
                        ["personal", "information"],
                        ["financial", "data"],
                        ["customer", "data"]
                    ]
                )
            
            obfuscator = ContentObfuscator(obfuscation_config)
            
            # Process PDF
            pdf_result = pdf_handler.process_pdf(temp_path)
            
            if not pdf_result.get("parsable", False):
                return {
                    'success': False,
                    'error': 'PDF is not parsable',
                    'message': 'Failed to extract text from PDF'
                }
            
            # Apply obfuscation
            obfuscated_result = obfuscator.obfuscate_content(pdf_result)
            
            # Calculate metrics
            original_pages = pdf_result.get("pages", [])
            obfuscated_pages = obfuscated_result.get("pages", [])
            
            original_word_count = sum(len(para.split()) for page in original_pages 
                                    for para in page.get("paragraphs", []))
            final_word_count = sum(len(para.split()) for page in obfuscated_pages 
                                 for para in page.get("paragraphs", []))
            
            original_page_count = len(original_pages)
            final_page_count = len(obfuscated_pages)
            
            # Extract content
            raw_content = "\n\n".join(para for page in original_pages 
                                    for para in page.get("paragraphs", []))
            final_content = "\n\n".join(para for page in obfuscated_pages 
                                       for para in page.get("paragraphs", []))
            
            # Get obfuscation statistics
            obfuscation_stats = obfuscator.get_obfuscation_report()
            
            # Store PDF data
            pdf_data = {
                'pdf_name': pdf_name,
                'file_hash': file_hash,
                'upload_date': datetime.now(),
                'processed_date': datetime.now(),
                'layout': pdf_result.get("layout", "unknown"),
                'original_word_count': original_word_count,
                'original_page_count': original_page_count,
                'parsability': True,
                'final_word_count': final_word_count,
                'final_page_count': final_page_count,
                'avg_words_per_page': final_word_count / final_page_count if final_page_count > 0 else 0,
                'raw_content': raw_content,
                'final_content': final_content,
                'obfuscation_applied': True,
                'pages_removed_count': obfuscation_stats['summary']['pages_removed'],
                'paragraphs_obfuscated_count': obfuscation_stats['summary']['paragraphs_obfuscated'],
                'obfuscation_summary': obfuscation_stats,
                'uploaded_by': session_id
            }
            
            pdf_id = store_pdf_data(pdf_data)
            
            return {
                'success': True,
                'pdf_id': pdf_id,
                'obfuscation_stats': obfuscation_stats,
                'message': 'PDF processed and obfuscated successfully'
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'PDF processing failed'
        }

def analyze_contract_content(pdf_id, session_id):
    """Analyze contract content using ContractAnalyzer"""
    
    try:
        # Get PDF data
        from config.database import get_pdf_by_hash
        # We need to get PDF by ID, let's add this function
        pdf_data = get_pdf_by_id(pdf_id)
        
        if not pdf_data:
            return {
                'success': False,
                'error': 'PDF not found',
                'message': 'PDF record not found in database'
            }
        
        # Initialize contract analyzer
        analyzer = ContractAnalyzer()
        
        # Analyze the obfuscated content
        analysis_result = analyzer.analyze_contract(pdf_data['final_content'])
        
        # Determine version number
        from config.database import get_next_analysis_version
        version = get_next_analysis_version(pdf_id)
        
        # Store analysis data
        analysis_data = {
            'pdf_id': pdf_id,
            'analysis_date': datetime.now(),
            'version': version,
            'form_number': analysis_result.get('form_number'),
            'pi_clause': analysis_result.get('pi_clause'),
            'ci_clause': analysis_result.get('ci_clause'),
            'data_usage_mentioned': analysis_result.get('data_usage_mentioned'),
            'data_limitations_exists': analysis_result.get('data_limitations_exists'),
            'summary': analysis_result.get('summary'),
            'raw_json': analysis_result,
            'processed_by': session_id,
            'processing_time': 0  # Could add timing here
        }
        
        analysis_id = store_analysis_data(analysis_data)
        
        return {
            'success': True,
            'analysis_id': analysis_id,
            'analysis_data': analysis_result,
            'message': 'Contract analysis completed successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Contract analysis failed'
        }

def extract_and_store_clauses(analysis_id, analysis_data):
    """Extract and store individual clauses"""
    
    try:
        relevant_clauses = analysis_data.get("relevant_clauses", [])
        
        if not relevant_clauses:
            return {
                'success': True,
                'clause_ids': [],
                'message': 'No clauses found to extract'
            }
        
        # Prepare clause data
        clause_list = []
        for idx, clause_data in enumerate(relevant_clauses):
            clause_list.append({
                'clause_type': clause_data.get('type'),
                'clause_text': clause_data.get('text'),
                'clause_order': idx + 1
            })
        
        # Store clauses
        clause_ids = store_clause_data(clause_list, analysis_id)
        
        return {
            'success': True,
            'clause_ids': clause_ids,
            'message': f'Successfully extracted and stored {len(clause_ids)} clauses'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Clause extraction failed'
        }

# Helper function we need to add to database.py
def get_pdf_by_id(pdf_id):
    """Get PDF record by ID"""
    
    from config.database import db
    import psycopg2.extras
    
    sql = "SELECT * FROM pdfs WHERE id = %s"
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            result = cur.fetchone()
            return dict(result) if result else None

@celery_app.task(bind=True)
def process_batch_async(self, batch_job_id, file_list, session_id):
    """
    Asynchronous batch processing task.
    
    Args:
        self: Celery task instance
        batch_job_id: Batch job ID
        file_list: List of files to process (with base64 encoded bytes)
        session_id: User session ID
        
    Returns:
        dict: Batch processing results
    """
    
    try:
        total_files = len(file_list)
        processed_files = 0
        failed_files = 0
        results = []
        
        # Update batch job status to running
        update_batch_job_status(
            batch_job_id, 
            'running',
            started_at=datetime.now(),
            total_files=total_files
        )
        
        self.update_state(
            state='PROCESSING', 
            meta={
                'current': 0,
                'total': total_files,
                'status': 'Starting batch processing'
            }
        )
        
        # Process each file
        for idx, file_info in enumerate(file_list):
            try:
                # Update progress
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'current': idx + 1,
                        'total': total_files,
                        'status': f'Processing {file_info["name"]}'
                    }
                )
                
                # Process individual PDF
                result = process_pdf_async.apply_async(
                    args=[file_info['bytes_b64'], file_info['name'], session_id]
                ).get()  # Wait for completion
                
                if result['success']:
                    results.append({
                        'file_name': file_info['name'],
                        'status': 'success',
                        'pdf_id': result['pdf_id'],
                        'analysis_data': result.get('analysis_data', {}),
                        'processed_at': datetime.now().isoformat()
                    })
                    processed_files += 1
                else:
                    results.append({
                        'file_name': file_info['name'],
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error'),
                        'processed_at': datetime.now().isoformat()
                    })
                    failed_files += 1
                
                # Update batch job progress
                update_batch_job_status(
                    batch_job_id,
                    'running',
                    processed_files=processed_files,
                    failed_files=failed_files
                )
                
            except Exception as e:
                results.append({
                    'file_name': file_info['name'],
                    'status': 'failed',
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                })
                failed_files += 1
        
        # Update final batch job status
        update_batch_job_status(
            batch_job_id,
            'completed',
            completed_at=datetime.now(),
            processed_files=processed_files,
            failed_files=failed_files,
            results_json=results
        )
        
        return {
            'success': True,
            'total_files': total_files,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'results': results,
            'message': f'Batch processing completed: {processed_files} successful, {failed_files} failed'
        }
        
    except Exception as e:
        # Update batch job status to failed
        update_batch_job_status(
            batch_job_id,
            'failed',
            completed_at=datetime.now(),
            error_log=str(e)
        )
        
        self.update_state(state='FAILURE', meta={'error': str(e)})
        
        return {
            'success': False,
            'error': str(e),
            'message': 'Batch processing failed'
        }

@celery_app.task
def cleanup_old_sessions_async(hours=24):
    """Asynchronous task to cleanup old user sessions"""
    
    try:
        from config.database import db
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        sql = "DELETE FROM users WHERE last_active < %s"
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (cutoff_time,))
                deleted_count = cur.rowcount
                conn.commit()
        
        return {
            'success': True,
            'deleted_sessions': deleted_count,
            'message': f'Cleaned up {deleted_count} old sessions'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Session cleanup failed'
        }

# Task monitoring and status functions

def get_task_status(task_id):
    """Get status of a Celery task"""
    
    task_result = celery_app.AsyncResult(task_id)
    
    return {
        'task_id': task_id,
        'state': task_result.state,
        'result': task_result.result,
        'info': task_result.info
    }

def cancel_task(task_id):
    """Cancel a running Celery task"""
    
    celery_app.control.revoke(task_id, terminate=True)
    
    return {
        'success': True,
        'message': f'Task {task_id} cancelled'
    }
