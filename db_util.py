# Add this import at the top of your streamlit3.py file
from db_utils import store_feedback, get_all_feedback_for_pdf, init_db_pool, create_tables

# Initialize database connection when app starts
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = init_db_pool()
    if st.session_state.db_initialized:
        create_tables()

# Add this to your CSS in the streamlit3.py file
st.markdown("""
<style>
    /* ... your existing CSS ... */
    
    .feedback-form {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
        background-color: #f8f9fa;
    }
    .feedback-item {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #0068c9;
    }
    .feedback-date {
        color: #666;
        font-size: 12px;
        font-style: italic;
    }
    .feedback-field {
        font-weight: bold;
        color: #0068c9;
    }
    .feedback-button {
        background-color: #007bff;
        color: white;
        padding: 0.375rem 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #007bff;
        cursor: pointer;
        font-size: 1rem;
        line-height: 1.5;
        transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }
    .feedback-button:hover {
        background-color: #0069d9;
        border-color: #0062cc;
    }
</style>
""", unsafe_allow_html=True)

# Add this function to your streamlit3.py file
def render_feedback_ui(pdf_name):
    """Render feedback UI components for the current PDF"""
    # Initialize session state for feedback form visibility
    if 'show_feedback_form' not in st.session_state:
        st.session_state.show_feedback_form = False
    
    # Toggle feedback form visibility
    def toggle_feedback_form():
        st.session_state.show_feedback_form = not st.session_state.show_feedback_form
    
    # Feedback button
    st.button("📝 Provide Feedback", on_click=toggle_feedback_form, 
              key="feedback_toggle_button", help="Click to provide feedback on the analysis")
    
    # Feedback form
    if st.session_state.show_feedback_form:
        with st.container():
            st.markdown('<div class="feedback-form">', unsafe_allow_html=True)
            st.subheader("Feedback Form")
            
            # Field selection dropdown
            field_options = [
                "Form Number", 
                "Summary", 
                "Data Usage Mentioned",
                "Data Limitations",
                "PI Clause",
                "CI Clause",
                "Clause Extraction",
                "Other"
            ]
            selected_field = st.selectbox("Select field to provide feedback on:", field_options, key="feedback_field")
            
            # Feedback text area
            feedback_text = st.text_area("Your feedback:", height=100, key="feedback_text")
            
            # Submit button
            submit_col, cancel_col = st.columns([1, 3])
            
            with submit_col:
                if st.button("Submit Feedback", key="submit_feedback"):
                    if feedback_text.strip():
                        # Store feedback in database
                        success = store_feedback(pdf_name, selected_field, feedback_text)
                        if success:
                            st.session_state.show_feedback_form = False
                            st.success("Thank you for your feedback!")
                            st.rerun()
                        else:
                            st.error("Failed to save feedback. Please try again.")
                    else:
                        st.warning("Please enter your feedback before submitting.")
            
            with cancel_col:
                if st.button("Cancel", key="cancel_feedback"):
                    st.session_state.show_feedback_form = False
                    st.rerun()
                    
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display existing feedback
    display_existing_feedback(pdf_name)

def display_existing_feedback(pdf_name):
    """Display existing feedback for the current PDF"""
    feedback_list = get_all_feedback_for_pdf(pdf_name)
    
    if feedback_list:
        with st.expander("View Previous Feedback", expanded=False):
            for item in feedback_list:
                st.markdown(f"""
                <div class="feedback-item">
                    <div class="feedback-field">{item['field_name']}</div>
                    <div>{item['feedback_text']}</div>
                    <div class="feedback-date">{item['feedback_date']}</div>
                </div>
                """, unsafe_allow_html=True)
