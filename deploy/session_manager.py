"""
Session Manager for Streamlit Application

This module manages user sessions, database connections, and provides
a consistent interface for the Streamlit application to interact with
the database and processing modules.

Features:
- Session creation and management
- Database connection handling
- PDF processing and analysis operations
- User feedback collection
"""

import os
import uuid
import json
import streamlit as st
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import database and processing modules
from db_handler import DatabaseHandler
from pdf_db_processor import PDFDatabaseProcessor
from contract_db_analyzer import ContractDatabaseAnalyzer
from config import get_config, init_directories

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('session_manager')

class SessionManager:
    """
    Manages user sessions, database connections, and application state.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        # Load configuration
        self.config = get_config()
        
        # Set log level from config
        logging.getLogger().setLevel(self.config['application'].get('log_level', 'INFO'))
        
        # Initialize directories
        init_directories(self.config)
        
        # Create or get session ID
        if 'session_id' not in st.session_state:
            # Generate a new session ID
            st.session_state.session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {st.session_state.session_id}")
        else:
            logger.info(f"Existing session: {st.session_state.session_id}")
        
        # Initialize database connection if not already initialized
        if 'db_handler' not in st.session_state:
            st.session_state.db_handler = DatabaseHandler(self.config['database'])
            
            # Initialize schema if in debug mode
            if self.config['application'].get('debug', False):
                st.session_state.db_handler.initialize_schema()
                
            # Create session in database
            st.session_state.db_handler.create_session(st.session_state.session_id)
            logger.info(f"Database connection initialized for session {st.session_state.session_id}")
        
        # Initialize processors if not already initialized
        if 'pdf_processor' not in st.session_state:
            st.session_state.pdf_processor = PDFDatabaseProcessor(st.session_state.db_handler)
            logger.info("PDF processor initialized")
            
        if 'contract_analyzer' not in st.session_state:
            st.session_state.contract_analyzer = ContractDatabaseAnalyzer(st.session_state.db_handler)
            logger.info("Contract analyzer initialized")
    
    def get_session_id(self) -> str:
        """Get the current session ID.
        
        Returns:
            str: Session ID
        """
        return st.session_state.session_id
    
    def update_session_activity(self):
        """Update the session's last activity timestamp."""
        st.session_state.db_handler.update_session_activity(st.session_state.session_id)
    
    def process_pdf_upload(self, uploaded_file) -> Tuple[bool, Dict[str, Any]]:
        """Process an uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Get PDF bytes and filename
            pdf_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            
            # Process PDF
            return st.session_state.pdf_processor.process_pdf_bytes(
                pdf_bytes=pdf_bytes,
                filename=filename,
                session_id=st.session_state.session_id
            )
        except Exception as e:
            logger.error(f"Error processing PDF upload: {str(e)}")
            return False, {"error": str(e)}
    
    def process_pdf_file(self, pdf_path) -> Tuple[bool, Dict[str, Any]]:
        """Process a PDF file from disk.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Process PDF
            return st.session_state.pdf_processor.process_and_store_pdf(
                pdf_path=pdf_path,
                session_id=st.session_state.session_id
            )
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            return False, {"error": str(e)}
    
    def analyze_contract(self, pdf_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Analyze a contract from a previously processed PDF.
        
        Args:
            pdf_id: PDF UUID
            
        Returns:
            Tuple of (success, result_data)
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Analyze contract
            return st.session_state.contract_analyzer.analyze_from_pdf_id(
                pdf_id=pdf_id,
                session_id=st.session_state.session_id
            )
        except Exception as e:
            logger.error(f"Error analyzing contract: {str(e)}")
            return False, {"error": str(e)}
    
    def get_session_pdfs(self) -> List[Dict[str, Any]]:
        """Get all PDFs for the current session.
        
        Returns:
            List of PDF documents
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Get session PDFs
            return st.session_state.pdf_processor.get_session_pdfs(st.session_state.session_id)
        except Exception as e:
            logger.error(f"Error getting session PDFs: {str(e)}")
            return []
    
    def get_session_analyses(self) -> List[Dict[str, Any]]:
        """Get all analyses for the current session.
        
        Returns:
            List of analysis data
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Get session analyses
            return st.session_state.contract_analyzer.get_session_analyses(st.session_state.session_id)
        except Exception as e:
            logger.error(f"Error getting session analyses: {str(e)}")
            return []
    
    def get_pdf_analyses(self, pdf_id: str) -> List[Dict[str, Any]]:
        """Get all analyses for a PDF.
        
        Args:
            pdf_id: PDF UUID
            
        Returns:
            List of analysis data
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Get PDF analyses
            return st.session_state.contract_analyzer.get_pdf_analyses(pdf_id)
        except Exception as e:
            logger.error(f"Error getting PDF analyses: {str(e)}")
            return []
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis results by ID.
        
        Args:
            analysis_id: Analysis UUID
            
        Returns:
            Analysis data or None if not found
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Get analysis
            return st.session_state.contract_analyzer.get_analysis(analysis_id)
        except Exception as e:
            logger.error(f"Error getting analysis: {str(e)}")
            return None
    
    def store_feedback(self, analysis_id: str, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback for an analysis.
        
        Args:
            analysis_id: Analysis UUID
            feedback_data: Feedback data
                - pdf_id: PDF UUID
                - feedback_type: Type of feedback
                - feedback_value: Feedback value
                - correct: Whether the analysis was correct
                - suggested_correction: Suggested correction
                - clause_id: Clause UUID (optional)
            
        Returns:
            Success status
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Store feedback
            success, _ = st.session_state.contract_analyzer.store_feedback(
                analysis_id=analysis_id,
                session_id=st.session_state.session_id,
                feedback_data=feedback_data
            )
            
            return success
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            return False
    
    def load_preloaded_contracts(self, pdf_folder=None, json_folder=None):
        """Load pre-loaded contracts into the database.
        
        Args:
            pdf_folder: Path to folder containing PDFs
            json_folder: Path to folder containing JSONs
            
        Returns:
            List of loaded PDF IDs
        """
        try:
            # Update session activity
            self.update_session_activity()
            
            # Default folders from config
            pdf_folder = pdf_folder or f"{self.config['application']['preloaded_contracts_dir']}/pdfs"
            json_folder = json_folder or f"{self.config['application']['preloaded_contracts_dir']}/jsons"
            
            # Ensure directories exist
            os.makedirs(pdf_folder, exist_ok=True)
            os.makedirs(json_folder, exist_ok=True)
            
            # Find all PDFs
            pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
            
            loaded_pdfs = []
            for pdf_name in pdf_files:
                pdf_path = os.path.join(pdf_folder, pdf_name)
                
                # Process PDF
                success, result = self.process_pdf_file(pdf_path)
                
                if success:
                    pdf_id = result.get('pdf_id')
                    
                    # Check for corresponding JSON
                    json_path = os.path.join(json_folder, f"{os.path.splitext(pdf_name)[0]}.json")
                    if os.path.exists(json_path):
                        try:
                            # Load JSON data
                            with open(json_path, 'r') as f:
                                analysis_data = json.load(f)
                            
                            # Store analysis
                            analysis_success, _ = st.session_state.contract_analyzer.analyze_and_store(
                                pdf_id=pdf_id,
                                session_id=st.session_state.session_id,
                                contract_text=result.get('final_text', '')
                            )
                            
                            if analysis_success:
                                logger.info(f"Loaded and analyzed preloaded contract: {pdf_name}")
                            else:
                                logger.warning(f"Failed to analyze preloaded contract: {pdf_name}")
                        except Exception as e:
                            logger.error(f"Error loading JSON for {pdf_name}: {str(e)}")
                    
                    loaded_pdfs.append(pdf_id)
            
            return loaded_pdfs
        except Exception as e:
            logger.error(f"Error loading preloaded contracts: {str(e)}")
            return []
    
    def close(self):
        """Close database connections."""
        if 'db_handler' in st.session_state:
            st.session_state.db_handler.close()
            logger.info("Database connections closed")


# Singleton instance
def get_session_manager():
    """Get or create the session manager instance.
    
    Returns:
        SessionManager: Session manager instance
    """
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    return st.session_state.session_manager


# Example usage in Streamlit app:
if __name__ == "__main__":
    st.title("Session Manager Example")
    
    # Get session manager
    session_manager = get_session_manager()
    
    # Display session ID
    st.write(f"Session ID: {session_manager.get_session_id()}")
    
    # PDF upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            success, result = session_manager.process_pdf_upload(uploaded_file)
        
        if success:
            st.success(f"PDF processed: {result.get('filename')}")
            st.write(f"PDF ID: {result.get('pdf_id')}")
            st.write(f"Word count: {result.get('word_count')}")
            st.write(f"Page count: {result.get('page_count')}")
            
            # Analyze button
            if st.button("Analyze Contract"):
                with st.spinner("Analyzing contract..."):
                    success, analysis = session_manager.analyze_contract(result.get('pdf_id'))
                
                if success:
                    st.success("Contract analyzed successfully")
                    st.write(f"Analysis ID: {analysis.get('analysis_id')}")
                    st.write(f"Form number: {analysis.get('form_number')}")
                    st.write(f"Summary: {analysis.get('summary')}")
                    
                    # Feedback form
                    st.subheader("Provide Feedback")
                    feedback_type = st.selectbox("Feedback Type", ["summary", "form_number", "data_usage_mentioned", "data_limitations_exists", "pi_clause", "ci_clause"])
                    feedback_value = st.text_area("Feedback")
                    correct = st.radio("Is the analysis correct?", ["Yes", "No", "Partially"])
                    suggested_correction = st.text_area("Suggested Correction")
                    
                    if st.button("Submit Feedback"):
                        feedback_data = {
                            "pdf_id": result.get('pdf_id'),
                            "feedback_type": feedback_type,
                            "feedback_value": feedback_value,
                            "correct": correct == "Yes",
                            "suggested_correction": suggested_correction
                        }
                        
                        if session_manager.store_feedback(analysis.get('analysis_id'), feedback_data):
                            st.success("Feedback submitted successfully")
                        else:
                            st.error("Error submitting feedback")
                else:
                    st.error(f"Error analyzing contract: {analysis.get('error')}")
        else:
            st.error(f"Error processing PDF: {result.get('error')}")
    
    # Display session PDFs
    st.subheader("Session PDFs")
    pdfs = session_manager.get_session_pdfs()
    for pdf in pdfs:
        st.write(f"{pdf.get('filename')} - {pdf.get('pdf_id')}")
    
    # Display session analyses
    st.subheader("Session Analyses")
    analyses = session_manager.get_session_analyses()
    for analysis in analyses:
        st.write(f"{analysis.get('form_number')} - {analysis.get('analysis_id')}")
