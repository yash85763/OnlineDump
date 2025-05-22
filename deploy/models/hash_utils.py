# utils/hash_utils.py - File hashing utilities

import hashlib

def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Calculate SHA256 hash of file content for deduplication.
    
    Args:
        file_bytes: Raw file bytes
        
    Returns:
        Hexadecimal string representation of SHA256 hash
    """
    return hashlib.sha256(file_bytes).hexdigest()

def calculate_content_hash(content: str) -> str:
    """
    Calculate hash of text content.
    
    Args:
        content: Text content to hash
        
    Returns:
        Hexadecimal string representation of SHA256 hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def verify_file_integrity(file_bytes: bytes, expected_hash: str) -> bool:
    """
    Verify file integrity by comparing hashes.
    
    Args:
        file_bytes: Raw file bytes
        expected_hash: Expected hash value
        
    Returns:
        True if hashes match, False otherwise
    """
    actual_hash = calculate_file_hash(file_bytes)
    return actual_hash == expected_hash


# services/feedback_service.py - Feedback management service

from sqlalchemy.orm import Session
from models.database_models import Feedback, PDF
from datetime import datetime
from typing import List, Dict, Any, Optional

class FeedbackService:
    """Service for managing user feedback on contract analyses"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def save_feedback(self, 
                     pdf_id: int, 
                     session_id: str, 
                     form_number_feedback: str = None, 
                     general_feedback: str = None,
                     rating: int = None) -> bool:
        """
        Save user feedback to database.
        
        Args:
            pdf_id: ID of the PDF being reviewed
            session_id: User session ID
            form_number_feedback: Specific feedback about form number extraction
            general_feedback: General feedback about the analysis
            rating: Optional rating (1-5)
            
        Returns:
            True if feedback saved successfully, False otherwise
        """
        try:
            feedback = Feedback(
                pdf_id=pdf_id,
                user_session_id=session_id,
                feedback_date=datetime.utcnow(),
                form_number_feedback=form_number_feedback.strip() if form_number_feedback else None,
                general_feedback=general_feedback.strip() if general_feedback else None,
                rating=rating
            )
            
            self.db_session.add(feedback)
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error saving feedback: {str(e)}")
            return False
    
    def get_feedback_history(self, pdf_id: int) -> List[Feedback]:
        """
        Get all feedback for a specific PDF.
        
        Args:
            pdf_id: ID of the PDF
            
        Returns:
            List of Feedback objects ordered by date (newest first)
        """
        return self.db_session.query(Feedback).filter_by(
            pdf_id=pdf_id
        ).order_by(Feedback.feedback_date.desc()).all()
    
    def get_user_feedback(self, session_id: str) -> List[Feedback]:
        """
        Get all feedback submitted by a specific user.
        
        Args:
            session_id: User session ID
            
        Returns:
            List of Feedback objects
        """
        return self.db_session.query(Feedback).filter_by(
            user_session_id=session_id
        ).order_by(Feedback.feedback_date.desc()).all()
    
    def get_feedback_stats(self, pdf_id: int = None) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            pdf_id: Optional PDF ID to get stats for specific PDF
            
        Returns:
            Dictionary with feedback statistics
        """
        query = self.db_session.query(Feedback)
        
        if pdf_id:
            query = query.filter_by(pdf_id=pdf_id)
        
        feedback_list = query.all()
        
        if not feedback_list:
            return {
                "total_feedback": 0,
                "avg_rating": 0,
                "rating_distribution": {i: 0 for i in range(1, 6)},
                "has_form_feedback": 0,
                "has_general_feedback": 0
            }
        
        # Calculate statistics
        total_feedback = len(feedback_list)
        ratings = [f.rating for f in feedback_list if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        rating_distribution = {i: 0 for i in range(1, 6)}
        for rating in ratings:
            if rating in rating_distribution:
                rating_distribution[rating] += 1
        
        has_form_feedback = sum(1 for f in feedback_list if f.form_number_feedback)
        has_general_feedback = sum(1 for f in feedback_list if f.general_feedback)
        
        return {
            "total_feedback": total_feedback,
            "avg_rating": round(avg_rating, 2),
            "rating_distribution": rating_distribution,
            "has_form_feedback": has_form_feedback,
            "has_general_feedback": has_general_feedback
        }
    
    def update_feedback(self, 
                       feedback_id: int, 
                       form_number_feedback: str = None, 
                       general_feedback: str = None,
                       rating: int = None) -> bool:
        """
        Update existing feedback.
        
        Args:
            feedback_id: ID of feedback to update
            form_number_feedback: Updated form number feedback
            general_feedback: Updated general feedback
            rating: Updated rating
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            feedback = self.db_session.query(Feedback).filter_by(id=feedback_id).first()
            
            if not feedback:
                return False
            
            if form_number_feedback is not None:
                feedback.form_number_feedback = form_number_feedback.strip()
            
            if general_feedback is not None:
                feedback.general_feedback = general_feedback.strip()
            
            if rating is not None:
                feedback.rating = rating
            
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error updating feedback: {str(e)}")
            return False
    
    def delete_feedback(self, feedback_id: int, session_id: str) -> bool:
        """
        Delete feedback (only by the user who created it).
        
        Args:
            feedback_id: ID of feedback to delete
            session_id: Session ID of user requesting deletion
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            feedback = self.db_session.query(Feedback).filter_by(
                id=feedback_id,
                user_session_id=session_id
            ).first()
            
            if not feedback:
                return False
            
            self.db_session.delete(feedback)
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error deleting feedback: {str(e)}")
            return False


# ui/feedback_form.py - Feedback form UI components

import streamlit as st
from datetime import datetime
from services.feedback_service import FeedbackService
from typing import Optional

def render_feedback_form(pdf_id: int, pdf_name: str, db_session):
    """
    Render feedback form for a specific PDF.
    
    Args:
        pdf_id: ID of the PDF to provide feedback for
        pdf_name: Name of the PDF file
        db_session: Database session
    """
    
    st.subheader(f"📝 Provide Feedback for: {pdf_name}")
    st.markdown("Your feedback helps improve the contract analysis accuracy!")
    
    with st.form(key=f"feedback_form_{pdf_id}", clear_on_submit=True):
        # Form Number Feedback Section
        st.markdown("### 📋 Form Number Feedback")
        st.markdown("*Provide specific feedback about the form number extraction:*")
        
        form_number_feedback = st.text_area(
            "Form Number Feedback",
            placeholder="Was the form number correctly identified? What should it be? Any suggestions for improvement?",
            height=100,
            key=f"form_feedback_{pdf_id}",
            help="Be specific about what was right or wrong with the form number detection"
        )
        
        # General Analysis Feedback Section
        st.markdown("### 💬 General Analysis Feedback")
        st.markdown("*Provide general feedback about the contract analysis:*")
        
        general_feedback = st.text_area(
            "General Feedback",
            placeholder="Comments about clause detection, PI/CI clause identification, summary accuracy, or overall analysis quality...",
            height=150,
            key=f"general_feedback_{pdf_id}",
            help="Share your thoughts on the overall analysis quality"
        )
        
        # Rating Section
        st.markdown("### ⭐ Overall Rating")
        rating = st.slider(
            "How would you rate the analysis quality?",
            min_value=1,
            max_value=5,
            value=3,
            key=f"rating_{pdf_id}",
            help="1 = Poor, 5 = Excellent"
        )
        
        # Rating labels
        rating_labels = {
            1: "😞 Poor - Major issues",
            2: "😐 Fair - Some issues", 
            3: "🙂 Good - Mostly accurate",
            4: "😊 Very Good - Minor issues",
            5: "😍 Excellent - Highly accurate"
        }
        
        st.markdown(f"**{rating_labels[rating]}**")
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            # Validate input
            if not form_number_feedback.strip() and not general_feedback.strip():
                st.warning("⚠️ Please provide at least some feedback before submitting.")
                return
            
            # Save feedback
            feedback_service = FeedbackService(db_session)
            success = feedback_service.save_feedback(
                pdf_id=pdf_id,
                session_id=st.session_state.user_session_id,
                form_number_feedback=form_number_feedback,
                general_feedback=general_feedback,
                rating=rating
            )
            
            if success:
                st.success("✅ Feedback submitted successfully! Thank you for helping us improve.")
                st.balloons()
                
                # Optional: Show what was submitted
                with st.expander("📋 Your Submitted Feedback"):
                    if form_number_feedback.strip():
                        st.markdown("**Form Number Feedback:**")
                        st.write(form_number_feedback)
                    
                    if general_feedback.strip():
                        st.markdown("**General Feedback:**")
                        st.write(general_feedback)
                    
                    st.markdown(f"**Rating:** {'⭐' * rating}")
                
                # Refresh the page to show updated feedback history
                st.rerun()
            else:
                st.error("❌ Failed to submit feedback. Please try again.")

def render_feedback_history(pdf_id: int, db_session, limit: Optional[int] = None):
    """
    Display feedback history for a PDF.
    
    Args:
        pdf_id: ID of the PDF
        db_session: Database session
        limit: Optional limit on number of feedback entries to show
    """
    
    feedback_service = FeedbackService(db_session)
    feedback_list = feedback_service.get_feedback_history(pdf_id)
    
    if limit:
        feedback_list = feedback_list[:limit]
    
    if feedback_list:
        st.subheader("📋 Previous Feedback")
        
        # Show feedback statistics
        stats = feedback_service.get_feedback_stats(pdf_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feedback", stats["total_feedback"])
        with col2:
            st.metric("Average Rating", f"{stats['avg_rating']}/5")
        with col3:
            st.metric("Response Rate", f"{(stats['total_feedback']/max(1, stats['total_feedback']))*100:.0f}%")
        
        st.divider()
        
        # Display individual feedback entries
        for i, feedback in enumerate(feedback_list):
            with st.expander(
                f"Feedback #{i+1} - {feedback.feedback_date.strftime('%Y-%m-%d %H:%M')} "
                f"{'⭐' * (feedback.rating or 0)}"
            ):
                # Form number feedback
                if feedback.form_number_feedback:
                    st.markdown("**📋 Form Number Feedback:**")
                    st.write(feedback.form_number_feedback)
                    st.markdown("---")
                
                # General feedback
                if feedback.general_feedback:
                    st.markdown("**💬 General Feedback:**")
                    st.write(feedback.general_feedback)
                    st.markdown("---")
                
                # Rating and metadata
                col1, col2 = st.columns(2)
                with col1:
                    if feedback.rating:
                        st.markdown(f"**Rating:** {'⭐' * feedback.rating} ({feedback.rating}/5)")
                
                with col2:
                    st.markdown(f"**Submitted:** {feedback.feedback_date.strftime('%Y-%m-%d %H:%M')}")
                
                # Edit/Delete options for own feedback
                if feedback.user_session_id == st.session_state.get('user_session_id'):
                    if st.button(f"🗑️ Delete", key=f"delete_feedback_{feedback.id}"):
                        if feedback_service.delete_feedback(feedback.id, st.session_state.user_session_id):
                            st.success("Feedback deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete feedback.")
    else:
        st.info("📝 No feedback available for this document yet. Be the first to provide feedback!")

def render_feedback_summary_widget(pdf_id: int, db_session):
    """
    Render a compact feedback summary widget.
    
    Args:
        pdf_id: ID of the PDF
        db_session: Database session
    """
    
    feedback_service = FeedbackService(db_session)
    stats = feedback_service.get_feedback_stats(pdf_id)
    
    if stats["total_feedback"] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("👥 Feedback Count", stats["total_feedback"])
        
        with col2:
            avg_rating = stats["avg_rating"]
            if avg_rating > 0:
                stars = "⭐" * int(avg_rating) + "☆" * (5 - int(avg_rating))
                st.metric("⭐ Avg Rating", f"{avg_rating}/5")
                st.markdown(f"<div style='font-size: 1.2em;'>{stars}</div>", unsafe_allow_html=True)
    else:
        st.info("No feedback yet - be the first!")


# Complete process_pdf_with_deduplication function for user_service.py

def process_pdf_with_deduplication(pdf_bytes: bytes, pdf_name: str, session_id: str, db_session):
    """
    Process PDF with deduplication and multi-user support.
    
    This function implements the complete workflow:
    1. Calculate file hash for deduplication
    2. Check if PDF already processed by any user
    3. If exists and analyzed, return existing analysis
    4. If exists but not analyzed, run analysis
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
    
    from utils.hash_utils import calculate_file_hash
    from services.pdf_service import EnhancedPDFService
    from services.analysis_service import AnalysisService
    from models.database_models import PDF, Analysis
    
    try:
        # Step 1: Calculate file hash for deduplication
        file_hash = calculate_file_hash(pdf_bytes)
        
        # Step 2: Check if PDF already exists in database
        existing_pdf = db_session.query(PDF).filter_by(file_hash=file_hash).first()
        
        if existing_pdf:
            # PDF already processed by someone - check if analysis exists
            analysis_service = AnalysisService(db_session)
            existing_analysis = analysis_service.get_latest_analysis(existing_pdf.id)
            
            if existing_analysis:
                # Analysis already done - return existing results to save time
                import json
                return True, json.loads(existing_analysis.raw_json), existing_pdf.id
            else:
                # PDF parsed but not analyzed - just run analysis
                success, analysis_result = analysis_service.analyze_contract_with_storage(
                    existing_pdf, session_id, force_rerun=False
                )
                
                if success:
                    return True, analysis_result, existing_pdf.id
                else:
                    return False, analysis_result, existing_pdf.id
        
        else:
            # Step 3: New PDF - do complete processing (parsing + analysis)
            pdf_service = EnhancedPDFService(db_session)
            
            success, result_data, pdf_id = pdf_service.process_pdf_pipeline(
                pdf_bytes=pdf_bytes,
                pdf_name=pdf_name,
                session_id=session_id,
                force_rerun=False
            )
            
            return success, result_data, pdf_id
            
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error in PDF processing with deduplication: {str(e)}"
        print(error_msg)
        return False, {"error": error_msg}, None