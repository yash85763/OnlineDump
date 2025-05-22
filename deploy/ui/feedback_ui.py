# ui/feedback_form.py - Feedback form implementation

import streamlit as st
from datetime import datetime
from services.feedback_service import FeedbackService

def render_feedback_form(pdf_id: int, pdf_name: str, db_session):
    """Render feedback form for a specific PDF"""
    
    st.subheader(f"📝 Provide Feedback for: {pdf_name}")
    
    with st.form(key=f"feedback_form_{pdf_id}"):
        st.markdown("### Form Number Feedback")
        st.markdown("*Provide specific feedback about the form number extraction:*")
        form_number_feedback = st.text_area(
            "Form Number Feedback",
            placeholder="Was the form number correctly identified? Any suggestions for improvement?",
            height=100,
            key=f"form_feedback_{pdf_id}"
        )
        
        st.markdown("### General Analysis Feedback")
        st.markdown("*Provide general feedback about the contract analysis:*")
        general_feedback = st.text_area(
            "General Feedback",
            placeholder="Comments about clause detection, PI/CI clause identification, or overall analysis quality...",
            height=150,
            key=f"general_feedback_{pdf_id}"
        )
        
        # Rating system (optional enhancement)
        st.markdown("### Overall Rating")
        rating = st.slider(
            "How would you rate the analysis quality?",
            min_value=1,
            max_value=5,
            value=3,
            key=f"rating_{pdf_id}"
        )
        
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if form_number_feedback.strip() or general_feedback.strip():
                feedback_service = FeedbackService(db_session)
                success = feedback_service.save_feedback(
                    pdf_id=pdf_id,
                    session_id=st.session_state.user_session_id,
                    form_number_feedback=form_number_feedback.strip(),
                    general_feedback=general_feedback.strip(),
                    rating=rating
                )
                
                if success:
                    st.success("✅ Feedback submitted successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to submit feedback. Please try again.")
            else:
                st.warning("⚠️ Please provide at least some feedback before submitting.")

def render_feedback_history(pdf_id: int, db_session):
    """Display feedback history for a PDF"""
    
    feedback_service = FeedbackService(db_session)
    feedback_list = feedback_service.get_feedback_history(pdf_id)
    
    if feedback_list:
        st.subheader("📋 Previous Feedback")
        
        for feedback in feedback_list:
            with st.expander(f"Feedback from {feedback.feedback_date.strftime('%Y-%m-%d %H:%M')}"):
                if feedback.form_number_feedback:
                    st.markdown("**Form Number Feedback:**")
                    st.write(feedback.form_number_feedback)
                
                if feedback.general_feedback:
                    st.markdown("**General Feedback:**")
                    st.write(feedback.general_feedback)
                
                if hasattr(feedback, 'rating') and feedback.rating:
                    st.markdown(f"**Rating:** {'⭐' * feedback.rating}")
                
                st.markdown("---")
    else:
        st.info("No feedback available for this document yet.")

# services/feedback_service.py - Feedback business logic

from sqlalchemy.orm import Session
from models.database_models import Feedback
from datetime import datetime

class FeedbackService:
    """Service for managing user feedback"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def save_feedback(self, pdf_id: int, session_id: str, 
                     form_number_feedback: str, general_feedback: str, 
                     rating: int = None) -> bool:
        """Save user feedback to database"""
        try:
            feedback = Feedback(
                pdf_id=pdf_id,
                user_session_id=session_id,
                feedback_date=datetime.utcnow(),
                form_number_feedback=form_number_feedback,
                general_feedback=general_feedback,
                rating=rating
            )
            
            self.db_session.add(feedback)
            self.db_session.commit()
            return True
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error saving feedback: {str(e)}")
            return False
    
    def get_feedback_history(self, pdf_id: int) -> list:
        """Get all feedback for a specific PDF"""
        return self.db_session.query(Feedback).filter_by(
            pdf_id=pdf_id
        ).order_by(Feedback.feedback_date.desc()).all()
    
    def get_feedback_stats(self, pdf_id: int = None) -> dict:
        """Get feedback statistics"""
        query = self.db_session.query(Feedback)
        
        if pdf_id:
            query = query.filter_by(pdf_id=pdf_id)
        
        feedback_list = query.all()
        
        if not feedback_list:
            return {"total": 0, "avg_rating": 0}
        
        total_feedback = len(feedback_list)
        ratings = [f.rating for f in feedback_list if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total": total_feedback,
            "avg_rating": round(avg_rating, 2),
            "rating_distribution": self._get_rating_distribution(ratings)
        }
    
    def _get_rating_distribution(self, ratings: list) -> dict:
        """Get distribution of ratings"""
        distribution = {i: 0 for i in range(1, 6)}
        for rating in ratings:
            if rating in distribution:
                distribution[rating] += 1
        return distribution
