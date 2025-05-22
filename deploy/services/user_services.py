# services/user_service.py - Multi-user session management

import streamlit as st
import uuid
import hashlib
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database_models import User, PDF, Analysis

class UserSessionService:
    """Manages user sessions for multi-user support"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def get_or_create_session(self):
        """Get or create a unique session for the current user"""
        # Check if session already exists in Streamlit session state
        if 'user_session_id' not in st.session_state:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            st.session_state.user_session_id = session_id
            
            # Store in database
            user = User(session_id=session_id)
            self.db_session.add(user)
            self.db_session.commit()
        
        # Update last active timestamp
        self.update_last_active(st.session_state.user_session_id)
        return st.session_state.user_session_id
    
    def update_last_active(self, session_id: str):
        """Update last active timestamp for session"""
        user = self.db_session.query(User).filter_by(session_id=session_id).first()
        if user:
            user.last_active = datetime.utcnow()
            self.db_session.commit()
    
    def cleanup_inactive_sessions(self, hours=24):
        """Clean up sessions inactive for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        inactive_users = self.db_session.query(User).filter(
            User.last_active < cutoff_time
        ).all()
        
        for user in inactive_users:
            self.db_session.delete(user)
        
        self.db_session.commit()
        return len(inactive_users)

class DeduplicationService:
    """Handles PDF deduplication across users"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_bytes).hexdigest()
    
    def check_existing_pdf(self, file_hash: str) -> PDF:
        """Check if PDF already exists in database"""
        return self.db_session.query(PDF).filter_by(file_hash=file_hash).first()
    
    def get_existing_analysis(self, pdf_id: int, latest_only=True) -> Analysis:
        """Get existing analysis for a PDF"""
        query = self.db_session.query(Analysis).filter_by(pdf_id=pdf_id)
        
        if latest_only:
            return query.order_by(Analysis.version.desc()).first()
        else:
            return query.all()

# Integration example in main processing function:
def process_pdf_with_deduplication(pdf_bytes, pdf_name, session_id):
    """
    Process PDF with deduplication and multi-user support
    
    Workflow:
    1. Calculate file hash
    2. Check if PDF already processed
    3. If exists, return existing analysis
    4. If not exists, process and store
    5. Handle versioning for re-runs
    """
    
    # Calculate hash for deduplication
    file_hash = DeduplicationService.calculate_file_hash(pdf_bytes)
    
    # Check if already processed
    existing_pdf = DeduplicationService.check_existing_pdf(file_hash)
    
    if existing_pdf:
        # PDF already processed, get latest analysis
        existing_analysis = DeduplicationService.get_existing_analysis(existing_pdf.id)
        
        if existing_analysis:
            # Return existing results to save processing time
            return True, existing_analysis.raw_json, existing_pdf.id
    
    # New PDF - process normally
    # ... processing logic here
    
    return process_new_pdf(pdf_bytes, pdf_name, file_hash, session_id)




# new implementation


# services/user_service.py - Complete user session management with PDF processing

import streamlit as st
import uuid
import hashlib
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database_models import User, PDF, Analysis
from utils.hash_utils import calculate_file_hash

class UserSessionService:
    """Manages user sessions for multi-user support"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def get_or_create_session(self):
        """Get or create a unique session for the current user"""
        # Check if session already exists in Streamlit session state
        if 'user_session_id' not in st.session_state:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            st.session_state.user_session_id = session_id
            
            # Store in database
            user = User(session_id=session_id)
            self.db_session.add(user)
            self.db_session.commit()
        
        # Update last active timestamp
        self.update_last_active(st.session_state.user_session_id)
        return st.session_state.user_session_id
    
    def update_last_active(self, session_id: str):
        """Update last active timestamp for session"""
        user = self.db_session.query(User).filter_by(session_id=session_id).first()
        if user:
            user.last_active = datetime.utcnow()
            self.db_session.commit()
    
    def cleanup_inactive_sessions(self, hours=24):
        """Clean up sessions inactive for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        inactive_users = self.db_session.query(User).filter(
            User.last_active < cutoff_time
        ).all()
        
        for user in inactive_users:
            self.db_session.delete(user)
        
        self.db_session.commit()
        return len(inactive_users)

class DeduplicationService:
    """Handles PDF deduplication across users"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def check_existing_pdf(self, file_hash: str) -> PDF:
        """Check if PDF already exists in database"""
        return self.db_session.query(PDF).filter_by(file_hash=file_hash).first()
    
    def get_existing_analysis(self, pdf_id: int, latest_only=True):
        """Get existing analysis for a PDF"""
        query = self.db_session.query(Analysis).filter_by(pdf_id=pdf_id)
        
        if latest_only:
            return query.order_by(Analysis.version.desc()).first()
        else:
            return query.all()

def process_pdf_with_deduplication(pdf_bytes: bytes, pdf_name: str, session_id: str, db_session):
    """
    Process PDF with deduplication and multi-user support.
    
    Complete workflow:
    1. Calculate file hash for deduplication
    2. Check if PDF already processed by any user
    3. If exists and analyzed, return existing analysis (saves processing time)
    4. If exists but not analyzed, run analysis only
    5. If not exists, do full processing (parsing + analysis)
    6. Handle versioning for re-runs
    
    Args:
        pdf_bytes: Raw PDF file bytes
        pdf_name: Name of the PDF file  
        session_id: Current user's session ID
        db_session: Database session
        
    Returns:
        Tuple of (success: bool, result_data: dict, pdf_id: int)
    """
    
    try:
        # Step 1: Calculate file hash for deduplication
        file_hash = calculate_file_hash(pdf_bytes)
        print(f"Processing PDF: {pdf_name}, Hash: {file_hash[:12]}...")
        
        # Step 2: Check if PDF already exists in database (from any user)
        dedup_service = DeduplicationService(db_session)
        existing_pdf = dedup_service.check_existing_pdf(file_hash)
        
        if existing_pdf:
            print(f"Found existing PDF in database: {existing_pdf.pdf_name}")
            
            # PDF already processed by someone - check if analysis exists
            existing_analysis = dedup_service.get_existing_analysis(existing_pdf.id, latest_only=True)
            
            if existing_analysis:
                # Analysis already done - return existing results (huge time saver!)
                print(f"Using existing analysis (version {existing_analysis.version})")
                import json
                return True, json.loads(existing_analysis.raw_json), existing_pdf.id
            
            else:
                # PDF parsed but not analyzed - just run contract analysis
                print("PDF exists but no analysis found - running analysis only")
                
                from services.analysis_service import AnalysisService
                analysis_service = AnalysisService(db_session)
                
                success, analysis_result = analysis_service.analyze_contract_with_storage(
                    existing_pdf, session_id, force_rerun=False
                )
                
                if success:
                    print("Analysis completed successfully")
                    return True, analysis_result, existing_pdf.id
                else:
                    print(f"Analysis failed: {analysis_result}")
                    return False, analysis_result, existing_pdf.id
        
        else:
            # Step 3: New PDF - do complete processing (parsing + analysis)
            print("New PDF - starting complete processing pipeline")
            
            from services.pdf_service import EnhancedPDFService
            pdf_service = EnhancedPDFService(db_session)
            
            success, result_data, pdf_id = pdf_service.process_pdf_pipeline(
                pdf_bytes=pdf_bytes,
                pdf_name=pdf_name,
                session_id=session_id,
                force_rerun=False
            )
            
            if success:
                print(f"Complete processing successful - PDF ID: {pdf_id}")
            else:
                print(f"Processing failed: {result_data}")
            
            return success, result_data, pdf_id
            
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error in PDF processing with deduplication: {str(e)}"
        print(error_msg)
        return False, {"error": error_msg}, None

def process_pdf_with_rerun(pdf_id: int, session_id: str, db_session):
    """
    Process PDF re-run (force new analysis on existing PDF).
    
    Args:
        pdf_id: ID of existing PDF to re-analyze
        session_id: Current user's session ID
        db_session: Database session
        
    Returns:
        Tuple of (success: bool, result_data: dict, pdf_id: int)
    """
    
    try:
        # Get existing PDF record
        pdf_record = db_session.query(PDF).filter_by(id=pdf_id).first()
        
        if not pdf_record:
            return False, {"error": "PDF not found"}, None
        
        print(f"Re-running analysis for PDF: {pdf_record.pdf_name}")
        
        # Force new analysis
        from services.analysis_service import AnalysisService
        analysis_service = AnalysisService(db_session)
        
        success, analysis_result = analysis_service.analyze_contract_with_storage(
            pdf_record, session_id, force_rerun=True  # This creates a new version
        )
        
        if success:
            print("Re-run analysis completed successfully")
            return True, analysis_result, pdf_id
        else:
            print(f"Re-run analysis failed: {analysis_result}")
            return False, analysis_result, pdf_id
            
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error in PDF re-run processing: {str(e)}"
        print(error_msg)
        return False, {"error": error_msg}, None


# Usage examples for integration with main application:

def integrate_with_streamlit_upload():
    """
    Example of how to integrate with Streamlit file upload.
    This replaces your existing process_pdf function.
    """
    
    # In your main Streamlit app:
    """
    uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            pdf_name = pdf.name
            pdf_bytes = pdf.getvalue()
            session_id = st.session_state.user_session_id
            db_session = st.session_state.db_session
            
            # Use the deduplication function
            with st.spinner(f"Processing {pdf_name}..."):
                success, result_data, pdf_id = process_pdf_with_deduplication(
                    pdf_bytes, pdf_name, session_id, db_session
                )
                
                if success:
                    st.success(f"✅ {pdf_name} processed successfully!")
                    
                    # Store results in session state for immediate use
                    st.session_state.pdf_files[pdf_name] = pdf_bytes
                    st.session_state.json_data[pdf_name] = result_data
                    st.session_state.analysis_status[pdf_name] = "Processed"
                    
                    # Set as current PDF if none selected
                    if not st.session_state.current_pdf:
                        st.session_state.current_pdf = pdf_name
                        
                else:
                    st.error(f"❌ Failed to process {pdf_name}: {result_data.get('error', 'Unknown error')}")
                    st.session_state.analysis_status[pdf_name] = f"Failed: {result_data.get('error', 'Unknown error')}"
    """

def integrate_with_rerun_button():
    """
    Example of how to integrate re-run functionality with Streamlit button.
    """
    
    # In your analysis display section:
    """
    if st.button("🔄 Re-run Analysis", key=f"rerun_{pdf_id}"):
        with st.spinner("Re-running analysis..."):
            success, result_data, _ = process_pdf_with_rerun(
                pdf_id, st.session_state.user_session_id, st.session_state.db_session
            )
            
            if success:
                st.success("✅ Re-run completed!")
                st.session_state.json_data[pdf_name] = result_data
                st.rerun()  # Refresh to show new results
            else:
                st.error(f"❌ Re-run failed: {result_data.get('error', 'Unknown error')}")
    """