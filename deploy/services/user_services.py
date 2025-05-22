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
