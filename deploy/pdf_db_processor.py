"""
Enhanced PDF Processing Module with Database Integration

This module extends the PDFHandler functionality to include database storage
of PDF extraction results and provides methods for retrieving processed data.

Features:
- Database integration for PDF metadata and content storage
- Word count and content statistics
- Session-based tracking of PDF processing
- Integration with the contract analysis pipeline
"""

import os
import uuid
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from ecfr_api_wrapper import PDFHandler
import re

class PDFDatabaseProcessor:
    """
    Enhanced PDF handler that integrates with database storage.
    """
    
    def __init__(self, db_handler, pdf_handler=None):
        """
        Initialize the PDF database processor.
        
        Args:
            db_handler: Database handler instance
            pdf_handler: Optional PDFHandler instance, will create a new one if not provided
        """
        self.db = db_handler
        self.pdf_handler = pdf_handler or PDFHandler()
        
    def process_and_store_pdf(self, pdf_path: str, session_id: str, generate_embeddings: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a PDF file and store the results in the database.
        
        Args:
            pdf_path: Path to the PDF file
            session_id: User session identifier
            generate_embeddings: Whether to generate embeddings for paragraphs
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Process PDF with the handler
            result = self.pdf_handler.process_pdf(pdf_path, generate_embeddings)
            
            # Check if parsing was successful
            if not result.get('parsable', False):
                return False, result
            
            # Get PDF file metadata
            filename = os.path.basename(pdf_path)
            file_size = os.path.getsize(pdf_path)
            
            # Calculate word count and other statistics
            word_count = 0
            final_text = ""
            
            # Process pages to get word count and consolidated text
            for page in result.get('pages', []):
                page_text = " ".join([p for p in page.get('paragraphs', []) if p])
                final_text += page_text + "\n\n"
                word_count += len(re.findall(r'\b\w+\b', page_text))
            
            # Calculate average words per page
            page_count = len(result.get('pages', []))
            avg_word_count_per_page = word_count / page_count if page_count > 0 else 0
            
            # Store in database
            pdf_id = self.db.store_pdf_document(
                session_id=session_id,
                filename=filename,
                pdf_name=Path(filename).stem,
                file_size=file_size,
                page_count=page_count,
                word_count=word_count,
                avg_word_count_per_page=avg_word_count_per_page,
                pdf_layout=result.get('layout', 'unknown'),
                parsable=result.get('parsable', False),
                final_text=final_text,
                metadata={
                    'processing_time': result.get('processing_time', 0),
                    'quality_info': result.get('quality_info', ''),
                    'embeddings_generated': generate_embeddings
                }
            )
            
            # Add PDF ID to the result
            result['pdf_id'] = pdf_id
            result['word_count'] = word_count
            result['avg_word_count_per_page'] = avg_word_count_per_page
            
            return True, result
            
        except Exception as e:
            # Log the error
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            
            # Try to store failed processing attempt
            try:
                self.db.store_pdf_document(
                    session_id=session_id,
                    filename=os.path.basename(pdf_path),
                    pdf_name=Path(os.path.basename(pdf_path)).stem,
                    file_size=os.path.getsize(pdf_path),
                    page_count=0,
                    word_count=0,
                    avg_word_count_per_page=0,
                    pdf_layout='unknown',
                    parsable=False,
                    final_text=None,
                    metadata={'error': str(e)}
                )
            except:
                pass
            
            return False, {"parsable": False, "error": str(e)}
    
    def process_pdf_bytes(self, pdf_bytes: bytes, filename: str, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Process PDF from bytes and store in database.
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            session_id: User session identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_bytes)
        
        try:
            success, result = self.process_and_store_pdf(temp_path, session_id)
            return success, result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_pdf_by_id(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve PDF information by ID.
        
        Args:
            pdf_id: PDF UUID
            
        Returns:
            PDF document information or None if not found
        """
        return self.db.get_pdf_document(pdf_id=pdf_id)
    
    def get_session_pdfs(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all PDFs for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of PDF documents
        """
        return self.db.get_session_pdfs(session_id)
    
    def store_contract_analysis(self, pdf_id: str, session_id: str, analysis_data: Dict[str, Any]) -> Optional[str]:
        """
        Store contract analysis results and associated clauses.
        
        Args:
            pdf_id: PDF UUID
            session_id: Session identifier
            analysis_data: Analysis results
            
        Returns:
            Analysis ID or None if failed
        """
        try:
            # Extract key analysis fields
            form_number = analysis_data.get('form_number', '')
            summary = analysis_data.get('summary', '')
            
            # Extract boolean flags with default values
            data_usage_mentioned = analysis_data.get('data_usage_mentioned', False)
            data_limitations_exists = analysis_data.get('data_limitations_exists', False)
            pi_clause = analysis_data.get('pi_clause', False)
            ci_clause = analysis_data.get('ci_clause', False)
            
            # Additional metadata to store
            metadata = {k: v for k, v in analysis_data.items() 
                       if k not in ['form_number', 'summary', 'data_usage_mentioned', 
                                  'data_limitations_exists', 'pi_clause', 'ci_clause',
                                  'relevant_clauses']}
            
            # Store the analysis
            analysis_id = self.db.store_contract_analysis(
                pdf_id=pdf_id,
                session_id=session_id,
                form_number=form_number,
                summary=summary,
                data_usage_mentioned=data_usage_mentioned,
                data_limitations_exists=data_limitations_exists,
                pi_clause=pi_clause,
                ci_clause=ci_clause,
                metadata=metadata
            )
            
            # Store each clause if present
            if analysis_id and 'relevant_clauses' in analysis_data:
                for clause in analysis_data['relevant_clauses']:
                    self.db.store_contract_clause(
                        analysis_id=analysis_id,
                        clause_type=clause.get('type', 'unknown'),
                        clause_text=clause.get('text', ''),
                        confidence=clause.get('confidence', None),
                        page_number=clause.get('page_number', None),
                        metadata=clause.get('metadata', None)
                    )
            
            return analysis_id
            
        except Exception as e:
            print(f"Error storing contract analysis: {str(e)}")
            return None
    
    def get_complete_analysis(self, analysis_id: str = None, pdf_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis information including clauses.
        
        Args:
            analysis_id: Analysis UUID (required if pdf_id not provided)
            pdf_id: PDF UUID (used to look up the most recent analysis if analysis_id not provided)
            
        Returns:
            Complete analysis data or None if not found
        """
        try:
            # Get the analysis
            analysis = None
            if analysis_id:
                analysis = self.db.get_contract_analysis(analysis_id=analysis_id)
            elif pdf_id:
                # Get the most recent analysis for this PDF
                analysis = self.db.get_contract_analysis(pdf_id=pdf_id)
            
            if not analysis:
                return None
            
            # Get clauses
            clauses = self.db.get_analysis_clauses(analysis['analysis_id'])
            
            # Format the complete result
            result = dict(analysis)
            result['relevant_clauses'] = [
                {
                    'id': str(clause['clause_id']),
                    'type': clause['clause_type'],
                    'text': clause['clause_text'],
                    'confidence': clause['confidence'],
                    'page_number': clause['page_number']
                }
                for clause in clauses
            ]
            
            # Get feedback
            feedback = self.db.get_analysis_feedback(analysis['analysis_id'])
            if feedback:
                result['feedback'] = feedback
            
            return result
            
        except Exception as e:
            print(f"Error retrieving complete analysis: {str(e)}")
            return None
    
    def store_feedback(self, session_id: str, pdf_id: str, analysis_id: str, 
                     feedback_type: str, feedback_value: str, correct: bool = None,
                     suggested_correction: str = None, clause_id: str = None) -> Optional[str]:
        """
        Store user feedback on analysis results.
        
        Args:
            session_id: Session identifier
            pdf_id: PDF UUID
            analysis_id: Analysis UUID
            feedback_type: Type of feedback (e.g., 'summary', 'pi_clause')
            feedback_value: Feedback value
            correct: Whether the analysis was correct
            suggested_correction: Suggested correction
            clause_id: Clause UUID if feedback is for a specific clause
            
        Returns:
            Feedback ID or None if failed
        """
        try:
            return self.db.store_feedback(
                session_id=session_id,
                pdf_id=pdf_id,
                analysis_id=analysis_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                correct=correct,
                suggested_correction=suggested_correction,
                clause_id=clause_id
            )
        except Exception as e:
            print(f"Error storing feedback: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    from db_handler import DatabaseHandler
    from config import get_config
    
    # Get configuration
    config = get_config()
    
    # Create database handler
    db_handler = DatabaseHandler(config['database'])
    
    # Initialize schema if needed
    db_handler.initialize_schema()
    
    # Create PDF processor
    pdf_processor = PDFDatabaseProcessor(db_handler)
    
    # Process a PDF file
    pdf_path = "sample.pdf"
    session_id = str(uuid.uuid4())
    
    # Create session
    db_handler.create_session(session_id)
    
    # Process PDF
    success, result = pdf_processor.process_and_store_pdf(pdf_path, session_id)
    
    if success:
        print(f"Successfully processed PDF: {result['filename']}")
        print(f"PDF ID: {result['pdf_id']}")
        print(f"Word count: {result['word_count']}")
        print(f"Page count: {len(result['pages'])}")
    else:
        print(f"Failed to process PDF: {result.get('error', 'Unknown error')}")
    
    # Close database connections
    db_handler.close()
