# database/helpers.py - Simplified database helper functions without users

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import hashlib
import uuid
import json

from database.models import (
    db_manager, PDF, Analysis, Clause, Feedback,
    generate_file_hash
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHelper:
    """Safe database operations with fallbacks for PDF handler and Streamlit"""
    
    @staticmethod
    def safe_convert_datetime(value: Any) -> Optional[str]:
        """Safely convert datetime to ISO string with fallback"""
        try:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, str):
                # Try to parse and reformat
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.isoformat()
            return None
        except Exception as e:
            logger.warning(f"Failed to convert datetime {value}: {e}")
            return None
    
    @staticmethod
    def safe_convert_number(value: Any, default: Any = None) -> Any:
        """Safely convert to number with fallback"""
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                # Try int first, then float
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            return default
        except Exception as e:
            logger.warning(f"Failed to convert number {value}: {e}")
            return default
    
    @staticmethod
    def safe_convert_json(value: Any) -> Optional[Dict]:
        """Safely convert to JSON with fallback"""
        try:
            if value is None:
                return None
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return json.loads(value)
            # Try to convert other types to dict
            return dict(value) if hasattr(value, '__iter__') else None
        except Exception as e:
            logger.warning(f"Failed to convert JSON {value}: {e}")
            return None

# SIMPLIFIED SESSION MANAGEMENT
def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

# PDF OPERATIONS
def store_pdf_from_upload(file_data: bytes, filename: str, session_id: str, 
                         pdf_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Store PDF from file upload with safe fallbacks"""
    try:
        # Generate file hash for deduplication
        file_hash = generate_file_hash(file_data)
        
        # Check if PDF already exists
        existing_pdf = get_pdf_by_hash(file_hash)
        if existing_pdf:
            logger.info(f"PDF already exists: {existing_pdf['pdf_name']}")
            return {**existing_pdf, 'is_duplicate': True}
        
        # Prepare PDF data with safe conversions
        pdf_data = {
            'pdf_name': filename,
            'file_hash': file_hash,
            'upload_date': datetime.utcnow(),
            'uploaded_by_session': session_id,  # Store session ID directly
            
            # Optional metadata with safe fallbacks
            'layout': pdf_metadata.get('layout') if pdf_metadata else None,
            'original_word_count': DatabaseHelper.safe_convert_number(
                pdf_metadata.get('word_count') if pdf_metadata else None
            ),
            'original_page_count': DatabaseHelper.safe_convert_number(
                pdf_metadata.get('page_count') if pdf_metadata else None
            ),
            'parsability': DatabaseHelper.safe_convert_number(
                pdf_metadata.get('parsability') if pdf_metadata else None, 0.0
            ),
            
            # Initialize processing fields as null
            'processed_date': None,
            'final_word_count': None,
            'final_page_count': None,
            'avg_words_per_page': None,
            'raw_content': None,
            'final_content': None,
            'obfuscation_applied': False,
            'pages_removed_count': 0,
            'paragraphs_obfuscated_count': 0,
            'obfuscation_summary': None
        }
        
        # Store in database using the simplified create_pdf function
        from database.models import create_pdf
        result = create_pdf(pdf_data)
        
        logger.info(f"PDF stored successfully: {filename}")
        return {**result, 'is_duplicate': False, 'status': 'uploaded'}
            
    except Exception as e:
        logger.error(f"Error storing PDF {filename}: {e}")
        return {
            'pdf_name': filename,
            'error': str(e),
            'status': 'failed',
            'upload_date': datetime.utcnow().isoformat()
        }def update_pdf_processing_data(pdf_id: int, processing_data: Dict[str, Any]) -> bool:
    """Update PDF with processing results (word count, obfuscation, etc.)"""
    try:
        with db_manager.get_session() as session:
            pdf = session.query(PDF).filter(PDF.id == pdf_id).first()
            if not pdf:
                logger.error(f"PDF not found: {pdf_id}")
                return False
            
            # Update fields with safe conversions
            if 'processed_date' in processing_data:
                pdf.processed_date = datetime.utcnow()
            
            if 'final_word_count' in processing_data:
                pdf.final_word_count = DatabaseHelper.safe_convert_number(
                    processing_data['final_word_count']
                )
            
            if 'final_page_count' in processing_data:
                pdf.final_page_count = DatabaseHelper.safe_convert_number(
                    processing_data['final_page_count']
                )
            
            # Calculate average words per page safely
            if pdf.final_word_count and pdf.final_page_count and pdf.final_page_count > 0:
                pdf.avg_words_per_page = pdf.final_word_count / pdf.final_page_count
            
            if 'raw_content' in processing_data:
                pdf.raw_content = DatabaseHelper.safe_convert_json(processing_data['raw_content'])
            
            if 'final_content' in processing_data:
                pdf.final_content = str(processing_data['final_content']) if processing_data['final_content'] else None
            
            if 'obfuscation_applied' in processing_data:
                pdf.obfuscation_applied = bool(processing_data['obfuscation_applied'])
            
            if 'pages_removed_count' in processing_data:
                pdf.pages_removed_count = DatabaseHelper.safe_convert_number(
                    processing_data['pages_removed_count'], 0
                )
            
            if 'paragraphs_obfuscated_count' in processing_data:
                pdf.paragraphs_obfuscated_count = DatabaseHelper.safe_convert_number(
                    processing_data['paragraphs_obfuscated_count'], 0
                )
            
            if 'obfuscation_summary' in processing_data:
                pdf.obfuscation_summary = DatabaseHelper.safe_convert_json(
                    processing_data['obfuscation_summary']
                )
            
            session.flush()
            logger.info(f"Updated processing data for PDF: {pdf.pdf_name}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating PDF processing data: {e}")
        return False

def get_pdf_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get PDF by file hash (for deduplication)"""
    try:
        from database.models import get_pdf_by_hash as db_get_pdf_by_hash
        return db_get_pdf_by_hash(file_hash)
    except Exception as e:
        logger.error(f"Error getting PDF by hash: {e}")
        return None

def get_pdfs_by_session(session_id: str, limit: int = 50) -> List[dict]:
    """Get all PDFs uploaded by a session"""
    with db_manager.get_session() as session:
        pdfs = session.query(PDF)\
                     .filter(PDF.uploaded_by_session == session_id)\
                     .order_by(PDF.upload_date.desc())\
                     .limit(limit)\
                     .all()
        
        return [
            {
                'id': pdf.id,
                'pdf_name': pdf.pdf_name,
                'upload_date': pdf.upload_date.isoformat() if pdf.upload_date else None,
                'final_word_count': pdf.final_word_count,
                'final_page_count': pdf.final_page_count,
                'status': 'processed' if pdf.processed_date else 'uploaded'
            }
            for pdf in pdfs
        ]

# ANALYSIS OPERATIONS
def store_analysis_results(pdf_id: int, analysis_results: Dict[str, Any], 
                          version: str = "v1.0") -> Dict[str, Any]:
    """Store contract analysis results with safe fallbacks"""
    try:
        analysis_data = {
            'pdf_id': pdf_id,
            'analysis_date': datetime.utcnow(),
            'version': version,
            'form_number': analysis_results.get('form_number'),
            'pi_clause': analysis_results.get('pi_clause'),
            'ci_clause': analysis_results.get('ci_clause'),
            'data_usage_mentioned': bool(analysis_results.get('data_usage_mentioned', False)),
            'data_limitations_exists': bool(analysis_results.get('data_limitations_exists', False)),
            'summary': analysis_results.get('summary'),
            'raw_json': DatabaseHelper.safe_convert_json(analysis_results.get('raw_analysis')),
            'processed_by': analysis_results.get('processed_by', 'AI_System'),
            'processing_time': DatabaseHelper.safe_convert_number(
                analysis_results.get('processing_time'), 0.0
            )
        }
        
        from database.models import create_analysis
        result = create_analysis(analysis_data)
        
        logger.info(f"Analysis stored for PDF {pdf_id}")
        return {**result, 'status': 'completed'}
            
    except Exception as e:
        logger.error(f"Error storing analysis for PDF {pdf_id}: {e}")
        return {
            'pdf_id': pdf_id,
            'error': str(e),
            'status': 'failed',
            'analysis_date': datetime.utcnow().isoformat()
        }

def store_extracted_clauses(analysis_id: int, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Store extracted clauses with safe fallbacks"""
    try:
        clause_data_list = []
        
        for i, clause_data in enumerate(clauses):
            try:
                clause_dict = {
                    'analysis_id': analysis_id,
                    'clause_type': clause_data.get('type', 'UNKNOWN'),
                    'clause_text': str(clause_data.get('text', '')) if clause_data.get('text') else None,
                    'page_number': DatabaseHelper.safe_convert_number(clause_data.get('page_number')),
                    'paragraph_index': DatabaseHelper.safe_convert_number(clause_data.get('paragraph_index')),
                    'clause_order': DatabaseHelper.safe_convert_number(clause_data.get('order', i + 1))
                }
                clause_data_list.append(clause_dict)
                
            except Exception as clause_error:
                logger.error(f"Error preparing clause {i}: {clause_error}")
        
        if clause_data_list:
            from database.models import create_clauses
            results = create_clauses(clause_data_list)
            
            for result in results:
                result['status'] = 'stored'
            
            logger.info(f"Stored {len(results)} clauses for analysis {analysis_id}")
            return results
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error storing clauses for analysis {analysis_id}: {e}")
        return [{'error': str(e), 'status': 'failed'}]

# FEEDBACK OPERATIONS
def store_user_feedback(pdf_id: int, session_id: str, 
                       feedback_text: str, rating: int = None) -> Dict[str, Any]:
    """Store user feedback with safe fallbacks"""
    try:
        feedback_data = {
            'pdf_id': pdf_id,
            'user_session_id': session_id,  # Store session ID directly
            'feedback_date': datetime.utcnow(),
            'general_feedback': feedback_text
        }
        
        from database.models import create_feedback
        result = create_feedback(feedback_data)
        
        logger.info(f"Feedback stored for PDF {pdf_id}")
        return {**result, 'status': 'stored'}
            
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        return {
            'pdf_id': pdf_id,
            'error': str(e),
            'status': 'failed',
            'feedback_date': datetime.utcnow().isoformat()
        }

# QUERY OPERATIONS
def get_session_dashboard_data(session_id: str) -> Dict[str, Any]:
    """Get comprehensive dashboard data for a session"""
    try:
        from database.models import get_pdfs_by_session
        
        pdfs = get_pdfs_by_session(session_id)
        
        # Calculate stats
        processed_pdfs = [p for p in pdfs if p['status'] == 'processed']
        
        # Get analysis and feedback counts
        total_analyses = 0
        total_feedbacks = 0
        
        with db_manager.get_session() as session:
            for pdf_data in pdfs:
                pdf_obj = session.query(PDF).filter(PDF.id == pdf_data['id']).first()
                if pdf_obj:
                    total_analyses += len(pdf_obj.analyses)
                    total_feedbacks += len(pdf_obj.feedbacks)
        
        dashboard_data = {
            'session_id': session_id,
            'stats': {
                'total_pdfs': len(pdfs),
                'processed_pdfs': len(processed_pdfs),
                'total_analyses': total_analyses,
                'total_feedbacks': total_feedbacks
            },
            'recent_pdfs': pdfs[:10]  # Show last 10 PDFs
        }
        
        return dashboard_data
            
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {
            'session_id': session_id,
            'stats': {'total_pdfs': 0, 'processed_pdfs': 0, 'total_analyses': 0, 'total_feedbacks': 0},
            'recent_pdfs': []
        }

def get_pdf_full_details(pdf_id: int) -> Dict[str, Any]:
    """Get complete PDF details with analyses and clauses"""
    try:
        with db_manager.get_session() as session:
            from sqlalchemy.orm import joinedload
            
            pdf = session.query(PDF)\
                        .options(
                            joinedload(PDF.analyses).joinedload(Analysis.clauses),
                            joinedload(PDF.feedbacks)
                        )\
                        .filter(PDF.id == pdf_id)\
                        .first()
            
            if not pdf:
                return {'error': 'PDF not found', 'pdf_id': pdf_id}
            
            return {
                'id': pdf.id,
                'pdf_name': pdf.pdf_name,
                'file_hash': pdf.file_hash,
                'upload_date': DatabaseHelper.safe_convert_datetime(pdf.upload_date),
                'processed_date': DatabaseHelper.safe_convert_datetime(pdf.processed_date),
                'status': 'processed' if pdf.processed_date else 'uploaded',
                'uploaded_by_session': pdf.uploaded_by_session,
                
                # Content metrics
                'original_word_count': pdf.original_word_count,
                'final_word_count': pdf.final_word_count,
                'original_page_count': pdf.original_page_count,
                'final_page_count': pdf.final_page_count,
                'avg_words_per_page': pdf.avg_words_per_page,
                'parsability': pdf.parsability,
                
                # Obfuscation info
                'obfuscation_applied': pdf.obfuscation_applied,
                'pages_removed_count': pdf.pages_removed_count,
                'paragraphs_obfuscated_count': pdf.paragraphs_obfuscated_count,
                'obfuscation_summary': pdf.obfuscation_summary,
                
                # Analyses with clauses
                'analyses': [
                    {
                        'id': analysis.id,
                        'version': analysis.version,
                        'analysis_date': DatabaseHelper.safe_convert_datetime(analysis.analysis_date),
                        'form_number': analysis.form_number,
                        'pi_clause': analysis.pi_clause,
                        'ci_clause': analysis.ci_clause,
                        'summary': analysis.summary,
                        'data_usage_mentioned': analysis.data_usage_mentioned,
                        'data_limitations_exists': analysis.data_limitations_exists,
                        'processing_time': analysis.processing_time,
                        
                        'clauses': [
                            {
                                'id': clause.id,
                                'clause_type': clause.clause_type,
                                'clause_text': clause.clause_text,
                                'page_number': clause.page_number,
                                'paragraph_index': clause.paragraph_index,
                                'clause_order': clause.clause_order
                            }
                            for clause in sorted(analysis.clauses, key=lambda x: x.clause_order or 0)
                        ]
                    }
                    for analysis in sorted(pdf.analyses, key=lambda x: x.analysis_date, reverse=True)
                ],
                
                # Feedbacks
                'feedbacks': [
                    {
                        'id': feedback.id,
                        'feedback_date': DatabaseHelper.safe_convert_datetime(feedback.feedback_date),
                        'general_feedback': feedback.general_feedback,
                        'user_session_id': feedback.user_session_id
                    }
                    for feedback in sorted(pdf.feedbacks, key=lambda x: x.feedback_date, reverse=True)
                ]
            }
            
    except Exception as e:
        logger.error(f"Error getting PDF details: {e}")
        return {'error': str(e), 'pdf_id': pdf_id}

# UTILITY FUNCTIONS
def check_database_health() -> Dict[str, Any]:
    """Check database connection and basic stats"""
    try:
        with db_manager.get_session() as session:
            stats = {
                'connection': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'total_pdfs': session.query(PDF).count(),
                'total_analyses': session.query(Analysis).count(),
                'total_clauses': session.query(Clause).count(),
                'total_feedbacks': session.query(Feedback).count()
            }
            return stats
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'connection': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing simplified database helper functions...")
    
    health = check_database_health()
    print(f"Database health: {health}")
    
    # Test session-based operations
    session_id = generate_session_id()
    print(f"Generated session ID: {session_id}")
    
    print("✅ Simplified helper functions ready for use!")

def update_pdf_processing_data(pdf_id: int, processing_data: Dict[str, Any]) -> bool:
    """Update PDF with processing results (word count, obfuscation, etc.)"""
    try:
        with db_manager.get_session() as session:
            pdf = session.query(PDF).filter(PDF.id == pdf_id).first()
            if not pdf:
                logger.error(f"PDF not found: {pdf_id}")
                return False
            
            # Update fields with safe conversions
            if 'processed_date' in processing_data:
                pdf.processed_date = datetime.utcnow()
            
            if 'final_word_count' in processing_data:
                pdf.final_word_count = DatabaseHelper.safe_convert_number(
                    processing_data['final_word_count']
                )
            
            if 'final_page_count' in processing_data:
                pdf.final_page_count = DatabaseHelper.safe_convert_number(
                    processing_data['final_page_count']
                )
            
            # Calculate average words per page safely
            if pdf.final_word_count and pdf.final_page_count and pdf.final_page_count > 0:
                pdf.avg_words_per_page = pdf.final_word_count / pdf.final_page_count
            
            if 'raw_content' in processing_data:
                pdf.raw_content = DatabaseHelper.safe_convert_json(processing_data['raw_content'])
            
            if 'final_content' in processing_data:
                pdf.final_content = str(processing_data['final_content']) if processing_data['final_content'] else None
            
            if 'obfuscation_applied' in processing_data:
                pdf.obfuscation_applied = bool(processing_data['obfuscation_applied'])
            
            if 'pages_removed_count' in processing_data:
                pdf.pages_removed_count = DatabaseHelper.safe_convert_number(
                    processing_data['pages_removed_count'], 0
                )
            
            if 'paragraphs_obfuscated_count' in processing_data:
                pdf.paragraphs_obfuscated_count = DatabaseHelper.safe_convert_number(
                    processing_data['paragraphs_obfuscated_count'], 0
                )
            
            if 'obfuscation_summary' in processing_data:
                pdf.obfuscation_summary = DatabaseHelper.safe_convert_json(
                    processing_data['obfuscation_summary']
                )
            
            session.flush()
            logger.info(f"Updated processing data for PDF: {pdf.pdf_name}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating PDF processing data: {e}")
        return False

def get_pdf_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get PDF by file hash (for deduplication)"""
    try:
        with db_manager.get_session() as session:
            pdf = session.query(PDF).filter(PDF.file_hash == file_hash).first()
            if not pdf:
                return None
            
            return {
                'id': pdf.id,
                'pdf_name': pdf.pdf_name,
                'file_hash': pdf.file_hash,
                'upload_date': DatabaseHelper.safe_convert_datetime(pdf.upload_date),
                'processed_date': DatabaseHelper.safe_convert_datetime(pdf.processed_date),
                'uploaded_by': str(pdf.uploaded_by),
                'original_word_count': pdf.original_word_count,
                'final_word_count': pdf.final_word_count,
                'original_page_count': pdf.original_page_count,
                'final_page_count': pdf.final_page_count,
                'status': 'processed' if pdf.processed_date else 'uploaded'
            }
    except Exception as e:
        logger.error(f"Error getting PDF by hash: {e}")
        return None

# ANALYSIS OPERATIONS
def store_analysis_results(pdf_id: int, analysis_results: Dict[str, Any], 
                          version: str = "v1.0") -> Dict[str, Any]:
    """Store contract analysis results with safe fallbacks"""
    try:
        analysis_data = {
            'pdf_id': pdf_id,
            'analysis_date': datetime.utcnow(),
            'version': version,
            'form_number': analysis_results.get('form_number'),
            'pi_clause': analysis_results.get('pi_clause'),
            'ci_clause': analysis_results.get('ci_clause'),
            'data_usage_mentioned': bool(analysis_results.get('data_usage_mentioned', False)),
            'data_limitations_exists': bool(analysis_results.get('data_limitations_exists', False)),
            'summary': analysis_results.get('summary'),
            'raw_json': DatabaseHelper.safe_convert_json(analysis_results.get('raw_analysis')),
            'processed_by': analysis_results.get('processed_by', 'AI_System'),
            'processing_time': DatabaseHelper.safe_convert_number(
                analysis_results.get('processing_time'), 0.0
            )
        }
        
        with db_manager.get_session() as session:
            analysis = Analysis(**analysis_data)
            session.add(analysis)
            session.flush()
            session.refresh(analysis)
            
            result = {
                'id': analysis.id,
                'pdf_id': analysis.pdf_id,
                'version': analysis.version,
                'analysis_date': DatabaseHelper.safe_convert_datetime(analysis.analysis_date),
                'form_number': analysis.form_number,
                'summary': analysis.summary,
                'data_usage_mentioned': analysis.data_usage_mentioned,
                'data_limitations_exists': analysis.data_limitations_exists,
                'processing_time': analysis.processing_time,
                'status': 'completed'
            }
            
            logger.info(f"Analysis stored for PDF {pdf_id}")
            return result
            
    except Exception as e:
        logger.error(f"Error storing analysis for PDF {pdf_id}: {e}")
        return {
            'pdf_id': pdf_id,
            'error': str(e),
            'status': 'failed',
            'analysis_date': datetime.utcnow().isoformat()
        }

def store_extracted_clauses(analysis_id: int, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Store extracted clauses with safe fallbacks"""
    try:
        results = []
        
        with db_manager.get_session() as session:
            for i, clause_data in enumerate(clauses):
                try:
                    clause = Clause(
                        analysis_id=analysis_id,
                        clause_type=clause_data.get('type', 'UNKNOWN'),
                        clause_text=str(clause_data.get('text', '')) if clause_data.get('text') else None,
                        page_number=DatabaseHelper.safe_convert_number(clause_data.get('page_number')),
                        paragraph_index=DatabaseHelper.safe_convert_number(clause_data.get('paragraph_index')),
                        clause_order=DatabaseHelper.safe_convert_number(clause_data.get('order', i + 1))
                    )
                    
                    session.add(clause)
                    session.flush()
                    session.refresh(clause)
                    
                    results.append({
                        'id': clause.id,
                        'analysis_id': clause.analysis_id,
                        'clause_type': clause.clause_type,
                        'clause_text': clause.clause_text,
                        'page_number': clause.page_number,
                        'paragraph_index': clause.paragraph_index,
                        'clause_order': clause.clause_order,
                        'status': 'stored'
                    })
                    
                except Exception as clause_error:
                    logger.error(f"Error storing clause {i}: {clause_error}")
                    results.append({
                        'clause_type': clause_data.get('type', 'UNKNOWN'),
                        'error': str(clause_error),
                        'status': 'failed'
                    })
            
            logger.info(f"Stored {len([r for r in results if r.get('status') == 'stored'])} clauses for analysis {analysis_id}")
            return results
            
    except Exception as e:
        logger.error(f"Error storing clauses for analysis {analysis_id}: {e}")
        return [{'error': str(e), 'status': 'failed'}]

# FEEDBACK OPERATIONS
def store_user_feedback(pdf_id: int, user_session_id: str, 
                       feedback_text: str, rating: int = None) -> Dict[str, Any]:
    """Store user feedback with safe fallbacks"""
    try:
        user = get_or_create_user_by_session(user_session_id)
        
        feedback_data = {
            'pdf_id': pdf_id,
            'user_id': user['id'],
            'feedback_date': datetime.utcnow(),
            'general_feedback': feedback_text,
            'rating': DatabaseHelper.safe_convert_number(rating) if rating else None
        }
        
        with db_manager.get_session() as session:
            feedback = Feedback(**feedback_data)
            session.add(feedback)
            session.flush()
            session.refresh(feedback)
            
            result = {
                'id': feedback.id,
                'pdf_id': feedback.pdf_id,
                'user_id': str(feedback.user_id),
                'feedback_date': DatabaseHelper.safe_convert_datetime(feedback.feedback_date),
                'general_feedback': feedback.general_feedback,
                'rating': feedback.rating,
                'status': 'stored'
            }
            
            logger.info(f"Feedback stored for PDF {pdf_id}")
            return result
            
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        return {
            'pdf_id': pdf_id,
            'error': str(e),
            'status': 'failed',
            'feedback_date': datetime.utcnow().isoformat()
        }

# QUERY OPERATIONS
def get_user_dashboard_data(user_session_id: str) -> Dict[str, Any]:
    """Get comprehensive user dashboard data"""
    try:
        user = get_or_create_user_by_session(user_session_id)
        
        with db_manager.get_session() as session:
            # Get user's PDFs with analysis counts
            pdfs = session.query(PDF).filter(PDF.uploaded_by == user['id']).all()
            
            dashboard_data = {
                'user': user,
                'stats': {
                    'total_pdfs': len(pdfs),
                    'processed_pdfs': len([p for p in pdfs if p.processed_date]),
                    'total_analyses': sum(len(p.analyses) for p in pdfs),
                    'total_feedbacks': sum(len(p.feedbacks) for p in pdfs)
                },
                'recent_pdfs': [
                    {
                        'id': pdf.id,
                        'pdf_name': pdf.pdf_name,
                        'upload_date': DatabaseHelper.safe_convert_datetime(pdf.upload_date),
                        'status': 'processed' if pdf.processed_date else 'uploaded',
                        'word_count': pdf.final_word_count or pdf.original_word_count,
                        'page_count': pdf.final_page_count or pdf.original_page_count,
                        'analyses_count': len(pdf.analyses)
                    }
                    for pdf in sorted(pdfs, key=lambda x: x.upload_date, reverse=True)[:10]
                ]
            }
            
            return dashboard_data
            
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {
            'user': {'username': 'Unknown User', 'error': str(e)},
            'stats': {'total_pdfs': 0, 'processed_pdfs': 0, 'total_analyses': 0, 'total_feedbacks': 0},
            'recent_pdfs': []
        }

def get_pdf_full_details(pdf_id: int) -> Dict[str, Any]:
    """Get complete PDF details with analyses and clauses"""
    try:
        with db_manager.get_session() as session:
            from sqlalchemy.orm import joinedload
            
            pdf = session.query(PDF)\
                        .options(
                            joinedload(PDF.analyses).joinedload(Analysis.clauses),
                            joinedload(PDF.feedbacks),
                            joinedload(PDF.uploader)
                        )\
                        .filter(PDF.id == pdf_id)\
                        .first()
            
            if not pdf:
                return {'error': 'PDF not found', 'pdf_id': pdf_id}
            
            return {
                'id': pdf.id,
                'pdf_name': pdf.pdf_name,
                'file_hash': pdf.file_hash,
                'upload_date': DatabaseHelper.safe_convert_datetime(pdf.upload_date),
                'processed_date': DatabaseHelper.safe_convert_datetime(pdf.processed_date),
                'status': 'processed' if pdf.processed_date else 'uploaded',
                
                # Content metrics
                'original_word_count': pdf.original_word_count,
                'final_word_count': pdf.final_word_count,
                'original_page_count': pdf.original_page_count,
                'final_page_count': pdf.final_page_count,
                'avg_words_per_page': pdf.avg_words_per_page,
                'parsability': pdf.parsability,
                
                # Obfuscation info
                'obfuscation_applied': pdf.obfuscation_applied,
                'pages_removed_count': pdf.pages_removed_count,
                'paragraphs_obfuscated_count': pdf.paragraphs_obfuscated_count,
                'obfuscation_summary': pdf.obfuscation_summary,
                
                # User info
                'uploader': {
                    'username': pdf.uploader.username if pdf.uploader else 'Unknown'
                },
                
                # Analyses with clauses
                'analyses': [
                    {
                        'id': analysis.id,
                        'version': analysis.version,
                        'analysis_date': DatabaseHelper.safe_convert_datetime(analysis.analysis_date),
                        'form_number': analysis.form_number,
                        'pi_clause': analysis.pi_clause,
                        'ci_clause': analysis.ci_clause,
                        'summary': analysis.summary,
                        'data_usage_mentioned': analysis.data_usage_mentioned,
                        'data_limitations_exists': analysis.data_limitations_exists,
                        'processing_time': analysis.processing_time,
                        
                        'clauses': [
                            {
                                'id': clause.id,
                                'clause_type': clause.clause_type,
                                'clause_text': clause.clause_text,
                                'page_number': clause.page_number,
                                'paragraph_index': clause.paragraph_index,
                                'clause_order': clause.clause_order
                            }
                            for clause in sorted(analysis.clauses, key=lambda x: x.clause_order or 0)
                        ]
                    }
                    for analysis in sorted(pdf.analyses, key=lambda x: x.analysis_date, reverse=True)
                ],
                
                # Feedbacks
                'feedbacks': [
                    {
                        'id': feedback.id,
                        'feedback_date': DatabaseHelper.safe_convert_datetime(feedback.feedback_date),
                        'general_feedback': feedback.general_feedback,
                        'rating': feedback.rating
                    }
                    for feedback in sorted(pdf.feedbacks, key=lambda x: x.feedback_date, reverse=True)
                ]
            }
            
    except Exception as e:
        logger.error(f"Error getting PDF details: {e}")
        return {'error': str(e), 'pdf_id': pdf_id}

# UTILITY FUNCTIONS
def check_database_health() -> Dict[str, Any]:
    """Check database connection and basic stats"""
    try:
        with db_manager.get_session() as session:
            stats = {
                'connection': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'total_users': session.query(User).count(),
                'total_pdfs': session.query(PDF).count(),
                'total_analyses': session.query(Analysis).count(),
                'total_clauses': session.query(Clause).count(),
                'total_feedbacks': session.query(Feedback).count()
            }
            return stats
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'connection': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing database helper functions...")
    
    health = check_database_health()
    print(f"Database health: {health}")
    
    # Test user creation
    session_id = generate_session_id()
    user = get_or_create_user_by_session(session_id, "test_user")
    print(f"User: {user}")
    
    print("✅ Helper functions ready for use!")
