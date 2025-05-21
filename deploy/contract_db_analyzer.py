"""
Enhanced Contract Analyzer with Database Integration

This module provides functionality for analyzing contracts and storing results
in a PostgreSQL database. It extends the existing ContractAnalyzer class with
database operations and session tracking.

Features:
- Contract clause extraction and classification
- Database storage of analysis results
- Session-based tracking of analysis operations
- PDF document ID linking
"""

import os
import json
import uuid
from typing import Dict, Any, List, Tuple, Optional
import logging
from contract_analyzer import ContractAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('contract_db_analyzer')

class ContractDatabaseAnalyzer:
    """
    Enhanced Contract Analyzer with database integration.
    """
    
    def __init__(self, db_handler, base_analyzer=None):
        """
        Initialize the contract database analyzer.
        
        Args:
            db_handler: Database handler instance
            base_analyzer: Optional ContractAnalyzer instance, will create a new one if not provided
        """
        self.db = db_handler
        self.base_analyzer = base_analyzer or ContractAnalyzer()
    
    def analyze_and_store(self, pdf_id: str, session_id: str, contract_text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a contract and store results in the database.
        
        Args:
            pdf_id: PDF UUID
            session_id: Session identifier
            contract_text: Extracted contract text to analyze
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Update session activity
            self.db.update_session_activity(session_id)
            
            # Run the base analysis
            analysis_result = self.base_analyzer.analyze_contract(contract_text)
            
            # Store analysis in database
            analysis_id = self.db.store_contract_analysis(
                pdf_id=pdf_id,
                session_id=session_id,
                form_number=analysis_result.get('form_number', ''),
                summary=analysis_result.get('summary', ''),
                data_usage_mentioned=analysis_result.get('data_usage_mentioned', False),
                data_limitations_exists=analysis_result.get('data_limitations_exists', False),
                pi_clause=analysis_result.get('pi_clause', False),
                ci_clause=analysis_result.get('ci_clause', False),
                metadata={k: v for k, v in analysis_result.items() 
                        if k not in ['form_number', 'summary', 'data_usage_mentioned', 
                                   'data_limitations_exists', 'pi_clause', 'ci_clause',
                                   'relevant_clauses']}
            )
            
            # Store clauses
            for clause in analysis_result.get('relevant_clauses', []):
                clause_id = self.db.store_contract_clause(
                    analysis_id=analysis_id,
                    clause_type=clause.get('type', 'unknown'),
                    clause_text=clause.get('text', ''),
                    confidence=clause.get('confidence', None),
                    page_number=clause.get('page_number', None),
                    metadata=clause.get('metadata', None)
                )
                # Add the database ID to the clause
                clause['id'] = clause_id
            
            # Add IDs to the result
            analysis_result['analysis_id'] = analysis_id
            analysis_result['pdf_id'] = pdf_id
            
            return True, analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {str(e)}")
            return False, {"error": str(e)}
    
    def analyze_from_pdf_id(self, pdf_id: str, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a contract from a previously stored PDF ID.
        
        Args:
            pdf_id: PDF UUID
            session_id: Session identifier
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Get PDF document
            pdf_doc = self.db.get_pdf_document(pdf_id=pdf_id)
            if not pdf_doc:
                return False, {"error": f"PDF with ID {pdf_id} not found"}
            
            # Check if PDF text is available
            if not pdf_doc.get('final_text'):
                return False, {"error": f"PDF with ID {pdf_id} has no extracted text"}
            
            # Analyze using the extracted text
            return self.analyze_and_store(pdf_id, session_id, pdf_doc['final_text'])
            
        except Exception as e:
            logger.error(f"Error analyzing from PDF ID {pdf_id}: {str(e)}")
            return False, {"error": str(e)}
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results by ID.
        
        Args:
            analysis_id: Analysis UUID
            
        Returns:
            Analysis data or None if not found
        """
        try:
            # Get basic analysis data
            analysis = self.db.get_contract_analysis(analysis_id=analysis_id)
            if not analysis:
                return None
            
            # Get clauses
            clauses = self.db.get_analysis_clauses(analysis_id)
            
            # Format the result
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting analysis {analysis_id}: {str(e)}")
            return None
    
    def get_pdf_analyses(self, pdf_id: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a PDF.
        
        Args:
            pdf_id: PDF UUID
            
        Returns:
            List of analysis data
        """
        try:
            # Get analyses for this PDF
            analyses = []
            pdf_analyses = self.db.execute_query(
                """
                SELECT * FROM contract_analysis 
                WHERE pdf_id = %s
                ORDER BY analysis_date DESC
                """,
                (pdf_id,),
                fetchall=True,
                as_dict=True
            )
            
            for analysis in pdf_analyses:
                # Get clauses for this analysis
                clauses = self.db.get_analysis_clauses(analysis['analysis_id'])
                
                # Format and add to results
                analysis_data = dict(analysis)
                analysis_data['relevant_clauses'] = [
                    {
                        'id': str(clause['clause_id']),
                        'type': clause['clause_type'],
                        'text': clause['clause_text'],
                        'confidence': clause['confidence'],
                        'page_number': clause['page_number']
                    }
                    for clause in clauses
                ]
                
                analyses.append(analysis_data)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting analyses for PDF {pdf_id}: {str(e)}")
            return []
    
    def get_session_analyses(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of analysis data
        """
        try:
            # Get analyses for this session
            session_analyses = self.db.execute_query(
                """
                SELECT ca.*, pd.filename, pd.pdf_name, pd.page_count, pd.word_count 
                FROM contract_analysis ca
                JOIN pdf_documents pd ON ca.pdf_id = pd.pdf_id
                WHERE ca.session_id = %s
                ORDER BY ca.analysis_date DESC
                """,
                (session_id,),
                fetchall=True,
                as_dict=True
            )
            
            analyses = []
            for analysis in session_analyses:
                analysis_id = analysis['analysis_id']
                
                # Get clauses for this analysis
                clauses = self.db.get_analysis_clauses(analysis_id)
                
                # Format and add to results
                analysis_data = dict(analysis)
                analysis_data['relevant_clauses'] = [
                    {
                        'id': str(clause['clause_id']),
                        'type': clause['clause_type'],
                        'text': clause['clause_text'],
                        'confidence': clause['confidence'],
                        'page_number': clause['page_number']
                    }
                    for clause in clauses
                ]
                
                analyses.append(analysis_data)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting analyses for session {session_id}: {str(e)}")
            return []
    
    def store_feedback(self, analysis_id: str, session_id: str, 
                     feedback_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Store feedback for an analysis.
        
        Args:
            analysis_id: Analysis UUID
            session_id: Session identifier
            feedback_data: Feedback data
                - pdf_id: PDF UUID
                - feedback_type: Type of feedback
                - feedback_value: Feedback value
                - correct: Whether the analysis was correct
                - suggested_correction: Suggested correction
                - clause_id: Clause UUID (optional)
            
        Returns:
            Tuple of (success, feedback_id)
        """
        try:
            # Check that analysis exists
            analysis = self.db.get_contract_analysis(analysis_id=analysis_id)
            if not analysis:
                return False, None
            
            # Extract feedback data
            pdf_id = feedback_data.get('pdf_id', analysis['pdf_id'])
            feedback_type = feedback_data.get('feedback_type', 'general')
            feedback_value = feedback_data.get('feedback_value', '')
            correct = feedback_data.get('correct')
            suggested_correction = feedback_data.get('suggested_correction')
            clause_id = feedback_data.get('clause_id')
            
            # Store feedback
            feedback_id = self.db.store_feedback(
                session_id=session_id,
                pdf_id=pdf_id,
                analysis_id=analysis_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                correct=correct,
                suggested_correction=suggested_correction,
                clause_id=clause_id
            )
            
            return True, feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback for analysis {analysis_id}: {str(e)}")
            return False, None


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
    
    # Create contract analyzer
    analyzer = ContractDatabaseAnalyzer(db_handler)
    
    # Create a test session
    session_id = str(uuid.uuid4())
    db_handler.create_session(session_id)
    
    # Example PDF ID (should be retrieved from database)
    pdf_id = "123e4567-e89b-12d3-a456-426614174000"
    
    # Example contract text
    contract_text = """
    This contract is made and entered into as of January 1, 2023, by and between
    Acme Corporation ("Acme") and XYZ Company ("XYZ").
    
    1. DATA USAGE: XYZ may use Acme's data for the purposes of providing services
    under this agreement. All data shall be treated as Confidential Information.
    
    2. LIMITATIONS: XYZ shall not use Acme's data for any purpose not expressly
    permitted in this agreement.
    
    3. PERSONAL INFORMATION: XYZ shall protect all Personal Information in accordance
    with applicable privacy laws.
    
    4. CONFIDENTIAL INFORMATION: All information shared under this agreement shall
    be considered Confidential and shall not be disclosed to third parties.
    """
    
    # Analyze and store
    success, result = analyzer.analyze_and_store(pdf_id, session_id, contract_text)
    
    if success:
        print(f"Successfully analyzed contract")
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"Form number: {result['form_number']}")
        print(f"Summary: {result['summary']}")
        print(f"Number of relevant clauses: {len(result['relevant_clauses'])}")
    else:
        print(f"Failed to analyze contract: {result.get('error', 'Unknown error')}")
    
    # Close database connections
    db_handler.close()
