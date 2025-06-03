"""
Contract Analysis Viewer with Database Integration

This Streamlit application provides a user interface for analyzing contracts,
with full database integration for storing and retrieving analysis results.

Features:
- PDF upload and processing
- Contract analysis and visualization
- User feedback collection
- Database persistence
- AWS EC2 deployment ready
"""

import os
import json
import base64
import streamlit as st
from pathlib import Path
import re
import glob
import tempfile
import urllib.parse
from PyPDF2 import PdfReader
from io import BytesIO
import uuid
import logging

# Import custom modules
from session_manager import get_session_manager
from config import get_config, init_directories

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('streamlit_app')

# Set page configuration
st.set_page_config(
    page_title="Contract Analysis Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and scrollable panes
st.markdown("""
<style>
    .left-pane {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        min-height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .pdf-viewer {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .json-details {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .extract-text {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #0068c9;
        margin: 10px 0;
    }
    .pdf-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .pdf-table th {
        background-color: #f2f2f2;
        padding: 8px;
        border: 1px solid #ddd;
        text-align: left;
    }
    .pdf-table td {
        border: 1px solid #ddd;
        padding: 8px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .pdf-table td:hover {
        background-color: #e6f3ff;
    }
    .pdf-table td.selected {
        background-color: #0068c9;
        color: white;
    }
    .status-button-true {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-false {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-missing {
        background-color: #ffc107;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .processing-message {
        color: #0068c9;
        font-size: 14px;
        margin: 5px 0;
    }
    .feedback-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        border: 1px solid #ddd;
    }
    .feedback-form {
        margin-top: 10px;
    }
    .feedback-submitted {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to validate PDF
def validate_pdf(pdf_bytes):
    """Validate PDF integrity and metadata"""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        metadata = pdf_reader.metadata
        if not pdf_reader.pages:
            return False, "Empty PDF or no pages detected"
        return True, metadata if metadata else "No metadata available"
    except Exception as e:
        return False, f"Invalid PDF: {str(e)}"

# Function to display PDF using iframe
def display_pdf_iframe(pdf_bytes, search_text=None):
    """Display PDF with optional search text"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}'
    if search_text:
        sanitized_text = sanitize_search_text(search_text)
        encoded_text = urllib.parse.quote(sanitized_text)
        pdf_display += f'#search={encoded_text}'
    pdf_display += '" width="100%" height="600px" type="application/pdf"></iframe>'
    
    if search_text:
        js_script = f"""
        <script>
            document.getElementById('pdfViewer').addEventListener('load', function() {{
                try {{
                    this.contentWindow.postMessage({{
                        type: 'search',
                        query: '{sanitize_search_text(search_text)}'
                    }}, '*');
                }} catch (e) {{
                    console.log('Error triggering PDF search:', e);
                }}
            }});
        </script>
        """
        pdf_display += js_script
    
    return pdf_display

# Fallback PDF display using object tag
def display_pdf_object(pdf_bytes):
    """Display PDF using object tag (fallback)"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    return f'<object data="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width="100%" height="600px"></object>'

def sanitize_search_text(text):
    """Clean up text for PDF search"""
    if not text:
        return ""
    text = text[:100]
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def set_current_pdf(pdf_id):
    """Set the current PDF to display"""
    st.session_state.current_pdf_id = pdf_id
    st.session_state.current_page = 1
    st.session_state.search_text = None

# Initialize session state variables
def init_session_state():
    """Initialize or reset session state variables"""
    if 'pdf_files' not in st.session_state:
        st.session_state.pdf_files = {}
    if 'current_pdf_id' not in st.session_state:
        st.session_state.current_pdf_id = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'search_text' not in st.session_state:
        st.session_state.search_text = None
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = {}
    if 'processing_messages' not in st.session_state:
        st.session_state.processing_messages = {}
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = {}
    if 'pdf_table_key' not in st.session_state:
        st.session_state.pdf_table_key = 0
    if 'loading_preloaded' not in st.session_state:
        st.session_state.loading_preloaded = False

def render_feedback_form(pdf_id, analysis_id=None, clause_id=None, feedback_type="general", label="Provide Feedback"):
    """Render a robust feedback form for an analysis or clause
    
    Args:
        pdf_id: PDF ID (required)
        analysis_id: Optional analysis ID
        clause_id: Optional clause ID
        feedback_type: Type of feedback
        label: Form label
    """
    import streamlit as st
    from datetime import datetime
    
    # Create unique form key
    form_key = f"feedback_{pdf_id}_{analysis_id}_{clause_id}_{feedback_type}"
    
    # Initialize session state for feedback tracking
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = {}
    
    # Check if user already provided feedback for this PDF
    def has_existing_feedback(pdf_id, user_session_id=None):
        """Check if feedback already exists for this PDF and user"""
        try:
            sql = """
                SELECT id FROM feedback 
                WHERE pdf_id = %s 
                AND user_session_id = %s 
                LIMIT 1
            """
            
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (pdf_id, user_session_id or 'anonymous'))
                    result = cur.fetchone()
                    return result is not None
        except Exception as e:
            st.error(f"Error checking existing feedback: {str(e)}")
            return False
    
    # Get or create session ID
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    user_session_id = st.session_state.session_id
    
    # Check if feedback was already submitted in current session
    if form_key in st.session_state.feedback_submitted and st.session_state.feedback_submitted[form_key]:
        st.markdown(f"<div class='feedback-submitted'>✅ Thank you for your feedback!</div>", unsafe_allow_html=True)
        return
    
    # Check if feedback already exists in database
    if has_existing_feedback(pdf_id, user_session_id):
        st.markdown(f"<div class='feedback-submitted'>✅ You have already provided feedback for this document.</div>", unsafe_allow_html=True)
        
        # Option to provide additional feedback
        if st.button("Provide Additional Feedback", key=f"additional_{form_key}"):
            st.session_state[f"allow_additional_{form_key}"] = True
            st.rerun()
        
        # Exit if not allowing additional feedback
        if not st.session_state.get(f"allow_additional_{form_key}", False):
            return
    
    # Render the feedback form
    with st.expander(label, expanded=True):
        st.markdown("<div class='feedback-form'>", unsafe_allow_html=True)
        
        # Form number selection
        form_number = st.selectbox(
            "Select Form Number",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            key=f"form_number_{form_key}",
            index=0,  # Default to form 1
            help="Select the form number that best matches this document"
        )
        
        # Feedback rating
        rating = st.radio(
            "Is this analysis correct?",
            ["Correct", "Partially Correct", "Incorrect"],
            key=f"rating_{form_key}",
            help="Rate the accuracy of the analysis"
        )
        
        # Convert rating to integer for database storage
        rating_map = {
            "Correct": 5,
            "Partially Correct": 3,
            "Incorrect": 1
        }
        rating_value = rating_map[rating]
        
        # Single feedback text box
        feedback_text = st.text_area(
            "Your feedback/comments",
            placeholder="Please provide your feedback or comments about this analysis...",
            key=f"feedback_{form_key}",
            help="Provide detailed feedback about the analysis accuracy, suggestions for improvement, etc."
        )
        
        # Validation
        if not feedback_text.strip():
            st.warning("Please provide some feedback before submitting.")
        
        # Submit button
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_clicked = st.button(
                "Submit Feedback", 
                key=f"submit_{form_key}",
                disabled=not feedback_text.strip(),
                type="primary"
            )
        
        if submit_clicked:
            # Show loading spinner
            with st.spinner("Submitting feedback..."):
                try:
                    # Prepare feedback data according to database schema
                    feedback_data = {
                        "pdf_id": pdf_id,
                        "feedback_date": datetime.now(),
                        "form_number_feedback": form_number,
                        "general_feedback": feedback_text.strip(),
                        "rating": rating_value,
                        "user_session_id": user_session_id
                    }
                    
                    # Store feedback using the database function
                    feedback_id = store_feedback_data(feedback_data)
                    
                    if feedback_id:
                        # Mark as submitted in session state
                        st.session_state.feedback_submitted[form_key] = True
                        
                        # Show success message
                        st.success("✅ Feedback submitted successfully!")
                        
                        # Optional: Store feedback ID for reference
                        if 'feedback_ids' not in st.session_state:
                            st.session_state.feedback_ids = []
                        st.session_state.feedback_ids.append(feedback_id)
                        
                        # Rerun to update UI
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. No feedback ID returned.")
                        
                except Exception as e:
                    st.error(f"❌ Error submitting feedback: {str(e)}")
                    
                    # Log error for debugging (if logging is set up)
                    print(f"Feedback submission error: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def get_user_feedback_history(user_session_id=None):
    """Get feedback history for a user session"""
    try:
        if not user_session_id:
            user_session_id = st.session_state.get('session_id', 'anonymous')
        
        sql = """
            SELECT f.*, p.pdf_name 
            FROM feedback f
            LEFT JOIN pdfs p ON f.pdf_id = p.id
            WHERE f.user_session_id = %s
            ORDER BY f.feedback_date DESC
        """
        
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (user_session_id,))
                results = cur.fetchall()
                return [dict(row) for row in results]
                
    except Exception as e:
        print(f"Error getting feedback history: {str(e)}")
        return []

def display_feedback_summary(pdf_id):
    """Display feedback summary for a PDF"""
    try:
        sql = """
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_feedback,
                COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_feedback
            FROM feedback 
            WHERE pdf_id = %s
        """
        
        with db.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (pdf_id,))
                result = cur.fetchone()
                
                if result and result['total_feedback'] > 0:
                    st.markdown("### Feedback Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Feedback", result['total_feedback'])
                    with col2:
                        st.metric("Average Rating", f"{result['avg_rating']:.1f}/5")
                    with col3:
                        st.metric("Positive", result['positive_feedback'])
                    with col4:
                        st.metric("Negative", result['negative_feedback'])
                
    except Exception as e:
        st.error(f"Error displaying feedback summary: {str(e)}")

# # Example usage function
# def example_usage():
#     """Example of how to use the robust feedback form"""
    
#     # Example: Render feedback form for a specific PDF
#     pdf_id = 123  # Replace with actual PDF ID
#     analysis_id = "abc-123"  # Optional analysis ID
    
#     render_feedback_form(
#         pdf_id=pdf_id,
#         analysis_id=analysis_id,
#         label="Rate This Analysis"
#     )
    
#     # Display feedback summary
#     display_feedback_summary(pdf_id)

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Get session manager
    session_manager = get_session_manager()
    
    # Get configuration
    config = get_config()
    
    # Update session activity
    session_manager.update_session_activity()
    
    # Display layout
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane: PDF upload and controls
    with col1:
        with st.container():
            st.markdown('<div class="left-pane">', unsafe_allow_html=True)
            st.header("Contracts")
            
            # Pre-loaded PDFs section
            st.subheader("Pre-loaded PDFs")
            
            if st.button("Load Pre-loaded Contracts"):
                with st.spinner("Loading pre-loaded contracts..."):
                    st.session_state.loading_preloaded = True
                    loaded_pdfs = session_manager.load_preloaded_contracts()
                    if loaded_pdfs:
                        st.success(f"Loaded {len(loaded_pdfs)} pre-loaded contracts")
                        
                        # Set current PDF if not already set
                        if not st.session_state.current_pdf_id and loaded_pdfs:
                            st.session_state.current_pdf_id = loaded_pdfs[0]
                            
                        st.session_state.loading_preloaded = False
                        st.rerun()
                    else:
                        st.warning("No pre-loaded contracts found or loaded")
                        st.session_state.loading_preloaded = False
            
            # PDF uploader
            st.subheader("Upload PDFs")
            uploaded_pdfs = st.file_uploader(
                "Upload Contract PDFs",
                type="pdf",
                key="pdf_uploader",
                accept_multiple_files=True
            )
            
            if uploaded_pdfs:
                for pdf in uploaded_pdfs:
                    pdf_key = f"{pdf.name}_{hash(pdf.name)}"
                    
                    if pdf_key not in st.session_state.pdf_files:
                        with st.spinner(f"Processing {pdf.name}..."):
                            # Process PDF
                            success, result = session_manager.process_pdf_upload(pdf)
                            
                            if success:
                                st.session_state.pdf_files[pdf_key] = {
                                    "name": pdf.name,
                                    "pdf_id": result.get("pdf_id"),
                                    "bytes": pdf.getvalue()
                                }
                                st.session_state.analysis_status[pdf_key] = "Not analyzed"
                                
                                # Set as current if no current PDF
                                if not st.session_state.current_pdf_id:
                                    st.session_state.current_pdf_id = result.get("pdf_id")
                                
                                st.success(f"Processed {pdf.name}")
                            else:
                                st.error(f"Failed to process {pdf.name}: {result.get('error', 'Unknown error')}")
            
            # Display PDF table - show PDFs from database
            st.subheader("Available PDFs")
            
            # Get PDFs from database
            pdfs = session_manager.get_session_pdfs()
            
            if pdfs:
                # Create a DataFrame for the table
                pdf_data = []
                for pdf in pdfs:
                    analyzed = False
                    pdf_id = pdf.get('pdf_id')
                    
                    # Check if this PDF has been analyzed
                    analyses = session_manager.get_pdf_analyses(pdf_id)
                    if analyses:
                        analyzed = True
                    
                    pdf_data.append({
                        "name": pdf.get('filename'),
                        "pdf_id": pdf_id,
                        "words": pdf.get('word_count', 0),
                        "pages": pdf.get('page_count', 0),
                        "analyzed": analyzed,
                        "selected": pdf_id == st.session_state.current_pdf_id
                    })
                
                # Create HTML table
                table_html = '''
                    <table class="pdf-table">
                        <tr>
                            <th>PDF Name</th>
                            <th>Pages</th>
                            <th>Status</th>
                        </tr>
                '''
                
                for i, pdf in enumerate(pdf_data):
                    selected_class = 'selected' if pdf["selected"] else ''
                    status = "Analyzed" if pdf["analyzed"] else "Not Analyzed"
                    status_class = "status-button-true" if pdf["analyzed"] else "status-button-false"
                    
                    table_html += f'''
                        <tr>
                            <td class="{selected_class}" onclick="document.getElementById('pdf_select_key').value='{pdf['pdf_id']}'; document.getElementById('pdf_select_form').submit();">
                                {pdf["name"]}
                            </td>
                            <td>{pdf["pages"]}</td>
                            <td><span class="{status_class}">{status}</span></td>
                        </tr>
                    '''
                
                table_html += '</table>'
                
                # Insert hidden form for click handling
                table_html += f'''
                    <form id="pdf_select_form" method="post" action="/?pdf_table_key={st.session_state.pdf_table_key}">
                        <input type="hidden" id="pdf_select_key" name="pdf_select_key" value="">
                    </form>
                '''
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Handle form submission
                query_params = st.experimental_get_query_params()
                if 'pdf_table_key' in query_params and st.session_state.pdf_table_key == int(query_params['pdf_table_key'][0]):
                    form_data = st.experimental_get_query_params()
                    if 'pdf_select_key' in form_data:
                        pdf_id = form_data['pdf_select_key'][0]
                        set_current_pdf(pdf_id)
                        
                        # Clear query params to prevent re-execution
                        st.experimental_set_query_params()
                        st.session_state.pdf_table_key += 1
                        st.rerun()
            else:
                st.info("No PDFs found. Please upload or load pre-loaded PDFs.")
            
            # Add an Analyze button if a PDF is selected but not analyzed
            if st.session_state.current_pdf_id:
                # Check if current PDF has been analyzed
                has_analysis = False
                analyses = session_manager.get_pdf_analyses(st.session_state.current_pdf_id)
                if analyses:
                    has_analysis = True
                
                if not has_analysis:
                    if st.button("Analyze Selected Contract"):
                        with st.spinner("Analyzing contract..."):
                            success, result = session_manager.analyze_contract(st.session_state.current_pdf_id)
                            if success:
                                st.success("Contract analyzed successfully!")
                                st.rerun()
                            else:
                                st.error(f"Error analyzing contract: {result.get('error', 'Unknown error')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle pane: PDF viewer
    with col2:
        with st.container():
            st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
            st.header("PDF Viewer")
            
            if st.session_state.current_pdf_id:
                # Get PDF metadata from database
                pdf_doc = None
                for pdf in pdfs:
                    if pdf.get('pdf_id') == st.session_state.current_pdf_id:
                        pdf_doc = pdf
                        break
                
                if pdf_doc:
                    st.subheader(f"Viewing: {pdf_doc.get('filename')}")
                    
                    # Get PDF bytes from session state or load from database
                    pdf_bytes = None
                    for pdf_key, pdf_data in st.session_state.pdf_files.items():
                        if pdf_data.get('pdf_id') == st.session_state.current_pdf_id:
                            pdf_bytes = pdf_data.get('bytes')
                            break
                    
                    # If not in session state, we need to load from a file
                    # In a production system, you would store the PDF content in the database
                    # or in S3, but for this example we'll look in the preloaded directory
                    if pdf_bytes is None:
                        pdf_filename = pdf_doc.get('filename')
                        pdf_path = os.path.join(config['application']['preloaded_contracts_dir'], 'pdfs', pdf_filename)
                        if os.path.exists(pdf_path):
                            with open(pdf_path, 'rb') as f:
                                pdf_bytes = f.read()
                                # Store in session state for future use
                                if pdf_filename not in st.session_state.pdf_files:
                                    st.session_state.pdf_files[pdf_filename] = {
                                        "name": pdf_filename,
                                        "pdf_id": st.session_state.current_pdf_id,
                                        "bytes": pdf_bytes
                                    }
                    
                    # Display PDF
                    if pdf_bytes:
                        try:
                            pdf_display = display_pdf_iframe(pdf_bytes, st.session_state.search_text)
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error with iframe: {e}")
                            try:
                                pdf_display = display_pdf_object(pdf_bytes)
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error with object tag: {e}")
                                is_valid, metadata_or_error = validate_pdf(pdf_bytes)
                                if not is_valid:
                                    st.error(f"Validation failed: {metadata_or_error}")
                                else:
                                    st.info(f"PDF metadata: {metadata_or_error}")
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_bytes,
                                    file_name=pdf_doc.get('filename'),
                                    mime="application/pdf",
                                    key="download_pdf"
                                )
                    else:
                        st.error("PDF content not available. This may be a database-only record.")
                else:
                    st.info("PDF metadata not found in database.")
            else:
                st.info("Select or upload a PDF to view.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right pane: Analysis details and feedback
    with col3:
        with st.container():
            st.markdown('<div class="json-details">', unsafe_allow_html=True)
            st.header("Contract Analysis")
            
            if st.session_state.current_pdf_id:
                # Get analyses for the current PDF
                analyses = session_manager.get_pdf_analyses(st.session_state.current_pdf_id)
                
                if analyses:
                    # Use the most recent analysis
                    analysis = analyses[0]
                    analysis_id = analysis.get('analysis_id')
                    
                    # Form number
                    st.subheader("Form Number")
                    st.markdown(f"<div class='extract-text'>{analysis.get('form_number', 'Not available')}</div>", 
                               unsafe_allow_html=True)
                    
                    # Add feedback form for form number
                    render_feedback_form(
                        analysis_id=analysis_id,
                        feedback_type="form_number",
                        label="Provide Feedback on Form Number"
                    )
                    
                    # Summary
                    st.subheader("Summary")
                    st.markdown(f"<div class='extract-text'>{analysis.get('summary', 'No summary available')}</div>", 
                               unsafe_allow_html=True)
                    
                    # Add feedback form for summary
                    render_feedback_form(
                        analysis_id=analysis_id,
                        feedback_type="summary",
                        label="Provide Feedback on Summary"
                    )
                    
                    # Contract status
                    st.subheader("Contract Status")
                    binary_keys = {
                        'data_usage_mentioned': 'Data Usage Mentioned',
                        'data_limitations_exists': 'Data Limitations Exists',
                        'pi_clause': 'Presence of PI Clause',
                        'ci_clause': 'Presence of CI Clause'
                    }
                    
                    for key, label in binary_keys.items():
                        value = analysis.get(key, None)
                        
                        # Handle different types of values, including "missing"
                        if value is True or (isinstance(value, str) and value.lower() in ["yes", "true", "1"]):
                            status = "True"
                            button_class = 'status-button-true'
                        elif value is False or (isinstance(value, str) and value.lower() in ["no", "false", "0"]):
                            status = "False" 
                            button_class = 'status-button-false'
                        elif value in ["missing", "Missing", "MISSING", "Absent"]:
                            status = "Missing"
                            button_class = 'status-button-missing'
                        else:
                            status = str(value) if value is not None else "None"
                            button_class = 'status-button-missing'
                        
                        # Display status
                        st.markdown(f"<div class='{button_class}'>{label}: {status}</div>", 
                                   unsafe_allow_html=True)
                        
                        # Add feedback form for this field
                        render_feedback_form(
                            analysis_id=analysis_id,
                            feedback_type=key,
                            label=f"Provide Feedback on {label}"
                        )
                    
                    # Relevant clauses
                    st.subheader("Relevant Clauses")
                    clauses = analysis.get("relevant_clauses", [])
                    if clauses:
                        for i, clause in enumerate(clauses):
                            with st.expander(f"Clause {i+1}: {clause['type'].capitalize()}"):
                                st.write(f"**Type:** {clause['type']}")
                                st.write(f"**Text:** {clause['text']}")
                                clause_id = clause.get('id')
                                
                                # Add search functionality
                                if st.button(f"Search clause {i+1} text", key=f"search_clause_{i}"):
                                    st.session_state.search_text = clause['text']
                                    st.success(f"Searching for clause {i+1}...")
                                    st.rerun()
                                
                                if len(clause['text']) > 100:
                                    st.warning("Text longer than 100 characters may not highlight fully.")
                                
                                # Add highlight button
                                if st.button("Highlight in PDF", key=f"highlight_{i}"):
                                    st.session_state.search_text = clause['text']
                                    st.success(f"Searching for clause {i+1}...")
                                    st.rerun()
                                
                                # Add feedback form for this clause
                                render_feedback_form(
                                    analysis_id=analysis_id,
                                    clause_id=clause_id,
                                    feedback_type=f"clause_{clause['type']}",
                                    label=f"Provide Feedback on {clause['type'].capitalize()} Clause"
                                )
                    else:
                        st.info("No relevant clauses found in the analysis.")
                else:
                    st.info("No analysis available for this PDF. Click 'Analyze Selected Contract' to analyze.")
            else:
                st.info("Select a PDF to view its analysis.")
            
            # Add general feedback section
            if st.session_state.current_pdf_id and analyses:
                st.subheader("Overall Analysis Feedback")
                render_feedback_form(
                    analysis_id=analyses[0].get('analysis_id'),
                    feedback_type="overall",
                    label="Provide Overall Feedback on Analysis"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
