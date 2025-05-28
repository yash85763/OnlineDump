import os
import json
import base64
import streamlit as st
import uuid
import tempfile
import urllib.parse
from pathlib import Path
from datetime import datetime
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Import custom modules
from utils.enhanced_pdf_handler import process_single_pdf_from_streamlit
from contract_analyzer import ContractAnalyzer
from ecfr_logger import ECFRLogger
from config.database import (
    initialize_database, 
    store_analysis_data, 
    store_clause_data, 
    store_feedback_data,
    get_next_analysis_version,
    check_database_connection
)
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Enhanced Contract Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📄"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .left-pane {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        height: 85vh;
        overflow-y: auto;
    }
    
    .pdf-viewer {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        height: 85vh;
        overflow-y: auto;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .analysis-panel {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        height: 85vh;
        overflow-y: auto;
    }
    
    .extract-text {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0068c9;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .status-button-true {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem 0;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(40,167,69,0.3);
    }
    
    .status-button-false {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem 0;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(220,53,69,0.3);
    }
    
    .status-button-missing {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: #212529;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem 0;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(255,193,7,0.3);
    }
    
    .processing-message {
        color: #0068c9;
        font-size: 0.9rem;
        padding: 0.3rem 0;
        border-left: 3px solid #0068c9;
        padding-left: 0.8rem;
        margin: 0.2rem 0;
        background-color: #f8f9ff;
        border-radius: 4px;
    }
    
    .database-status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .database-status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    
    .feedback-section {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .obfuscation-info {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Session Management Functions
def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def initialize_session_state():
    """Initialize all session state variables"""
    # Database initialization
    if 'database_initialized' not in st.session_state:
        try:
            if check_database_connection():
                initialize_database()
                st.session_state.database_initialized = True
                st.session_state.database_status = "Connected"
            else:
                st.session_state.database_initialized = False
                st.session_state.database_status = "Failed to connect"
        except Exception as e:
            st.session_state.database_initialized = False
            st.session_state.database_status = f"Error: {str(e)}"
    
    # Session state variables
    session_vars = {
        'pdf_files': {},
        'json_data': {},
        'current_pdf': None,
        'analysis_status': {},
        'processing_messages': {},
        'pdf_database_ids': {},  # Map PDF names to database IDs
        'search_text': None,
        'feedback_submitted': {},  # Track feedback submission per PDF
        'obfuscation_summaries': {},  # Store obfuscation info per PDF
        'session_id': get_session_id()
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# PDF Validation Functions
def validate_pdf(pdf_bytes):
    """Validate PDF integrity and metadata"""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        if not pdf_reader.pages:
            return False, "Empty PDF or no pages detected"
        return True, f"Valid PDF with {len(pdf_reader.pages)} pages"
    except Exception as e:
        return False, f"Invalid PDF: {str(e)}"

# PDF Display Functions
def display_pdf_iframe(pdf_bytes, search_text=None):
    """Display PDF with optional search text"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}'
    if search_text:
        sanitized_text = sanitize_search_text(search_text)
        encoded_text = urllib.parse.quote(sanitized_text)
        pdf_display += f'#search={encoded_text}'
    pdf_display += '" width="100%" height="600px" type="application/pdf"></iframe>'
    
    return pdf_display

def sanitize_search_text(text):
    """Clean up text for PDF search"""
    if not text:
        return ""
    text = text[:100]
    import re
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def set_current_pdf(pdf_name):
    """Set the current PDF to display"""
    st.session_state.current_pdf = pdf_name
    st.session_state.search_text = None

# Enhanced PDF Processing Function
def process_pdf_enhanced(pdf_bytes, pdf_name, message_placeholder):
    """Process a single PDF using the enhanced handler with database storage"""
    try:
        st.session_state.processing_messages[pdf_name] = []
        
        # Update processing message
        st.session_state.processing_messages[pdf_name].append("🔄 Starting PDF processing with obfuscation...")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" 
                      for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )
        
        # Process PDF with enhanced handler
        result = process_single_pdf_from_streamlit(
            pdf_name=pdf_name,
            pdf_bytes=pdf_bytes,
            enable_obfuscation=True,
            uploaded_by=get_session_id()
        )
        
        if result.get('success'):
            # Store processing information in session state
            st.session_state.pdf_database_ids[pdf_name] = result.get('pdf_id')
            st.session_state.obfuscation_summaries[pdf_name] = result.get('obfuscation_summary', {})
            
            # Update processing messages
            st.session_state.processing_messages[pdf_name].append("✅ PDF processed and stored in database")
            st.session_state.processing_messages[pdf_name].append(f"📊 Database ID: {result.get('pdf_id')}")
            
            # Add obfuscation info
            obf_summary = result.get('obfuscation_summary', {})
            pages_removed = obf_summary.get('pages_removed_count', 0)
            original_pages = obf_summary.get('total_original_pages', 0)
            final_pages = obf_summary.get('total_final_pages', 0)
            
            st.session_state.processing_messages[pdf_name].append(
                f"🔒 Privacy protection applied: {pages_removed} pages removed ({original_pages} → {final_pages} pages)"
            )
            
            message_placeholder.markdown(
                "\n".join([f"<div class='processing-message'>{msg}</div>" 
                          for msg in st.session_state.processing_messages[pdf_name]]),
                unsafe_allow_html=True
            )
            
            # Now run contract analysis on obfuscated content
            st.session_state.processing_messages[pdf_name].append("🔍 Starting contract analysis...")
            message_placeholder.markdown(
                "\n".join([f"<div class='processing-message'>{msg}</div>" 
                          for msg in st.session_state.processing_messages[pdf_name]]),
                unsafe_allow_html=True
            )
            
            # Get the processed content for analysis
            pages_content = result.get('pages', [])
            contract_text = '\n\n'.join([
                para for page in pages_content 
                for para in page.get('paragraphs', [])
            ])
            
            # Run contract analysis
            contract_analyzer = ContractAnalyzer()
            logger = ECFRLogger()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"{Path(pdf_name).stem}.json")
                
                try:
                    analysis_results = contract_analyzer.analyze_contract(contract_text, output_path)
                    
                    # Store analysis results in database
                    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
                    if pdf_id:
                        analysis_data = {
                            'pdf_id': pdf_id,
                            'analysis_date': datetime.now(),
                            'version': get_next_analysis_version(pdf_id),
                            'form_number': analysis_results.get('form_number'),
                            'pi_clause': analysis_results.get('pi_clause'),
                            'ci_clause': analysis_results.get('ci_clause'),
                            'data_usage_mentioned': analysis_results.get('data_usage_mentioned'),
                            'data_limitations_exists': analysis_results.get('data_limitations_exists'),
                            'summary': analysis_results.get('summary'),
                            'raw_json': analysis_results,
                            'processed_by': 'streamlit_analyzer',
                            'processing_time': 0.0
                        }
                        
                        try:
                            analysis_id = store_analysis_data(analysis_data)
                            
                            # Store clauses
                            clauses = analysis_results.get('relevant_clauses', [])
                            if clauses:
                                clause_data = []
                                for i, clause in enumerate(clauses):
                                    clause_data.append({
                                        'clause_type': clause.get('type', 'unknown'),
                                        'clause_text': clause.get('text', ''),
                                        'clause_order': i + 1
                                    })
                                store_clause_data(clause_data, analysis_id)
                            
                            st.session_state.processing_messages[pdf_name].append(f"💾 Analysis stored in database (ID: {analysis_id})")
                        except Exception as e:
                            st.session_state.processing_messages[pdf_name].append(f"⚠️ Warning: Could not store analysis in database: {str(e)}")
                    
                    # Store analysis results in session state for display
                    file_stem = Path(pdf_name).stem
                    st.session_state.json_data[file_stem] = analysis_results
                    
                    st.session_state.processing_messages[pdf_name].append("✅ Contract analysis completed successfully")
                    return True, "Analysis completed successfully"
                    
                except Exception as e:
                    logger.error(f"Contract analysis failed for {pdf_name}: {str(e)}")
                    return False, f"Contract analysis failed: {str(e)}"
        
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return False, error_msg
            
    except Exception as e:
        return False, f"Processing failed: {str(e)}"
    finally:
        # Keep processing messages for a short time
        if pdf_name in st.session_state.processing_messages:
            st.session_state.processing_messages[pdf_name].append("📝 Processing complete - Ready for review")

# Feedback System
def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF"""
    
    # Check if feedback already submitted
    feedback_key = f"feedback_{file_stem}"
    if st.session_state.feedback_submitted.get(feedback_key, False):
        st.success("✅ Thank you! Your feedback has been submitted for this document.")
        if st.button("Submit New Feedback", key=f"new_feedback_{file_stem}"):
            st.session_state.feedback_submitted[feedback_key] = False
            st.rerun()
        return
    
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    st.subheader("📝 Your Feedback Matters")
    st.write("Help us improve our analysis by providing feedback on the results:")
    
    with st.form(f"feedback_form_{file_stem}"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Form Number Feedback
            st.write("**Form Number Analysis**")
            form_number_correct = st.selectbox(
                "Is the form number correctly identified?",
                ["Select...", "Yes, correct", "No, incorrect", "Not applicable"],
                key=f"form_correct_{file_stem}"
            )
            form_number_feedback = st.text_area(
                "Form Number Comments",
                placeholder="Please provide details...",
                height=60,
                key=f"form_feedback_{file_stem}"
            )
            
            # PI Clause Feedback
            st.write("**PI Clause Detection**")
            pi_clause_correct = st.selectbox(
                "Is the PI clause detection accurate?",
                ["Select...", "Yes, accurate", "No, missed clauses", "No, false positives", "Not applicable"],
                key=f"pi_correct_{file_stem}"
            )
            pi_clause_feedback = st.text_area(
                "PI Clause Comments",
                placeholder="Please provide details...",
                height=60,
                key=f"pi_feedback_{file_stem}"
            )
        
        with col2:
            # CI Clause Feedback
            st.write("**CI Clause Detection**")
            ci_clause_correct = st.selectbox(
                "Is the CI clause detection accurate?",
                ["Select...", "Yes, accurate", "No, missed clauses", "No, false positives", "Not applicable"],
                key=f"ci_correct_{file_stem}"
            )
            ci_clause_feedback = st.text_area(
                "CI Clause Comments",
                placeholder="Please provide details...",
                height=60,
                key=f"ci_feedback_{file_stem}"
            )
            
            # Summary Feedback
            st.write("**Summary Quality**")
            summary_quality = st.selectbox(
                "How would you rate the summary quality?",
                ["Select...", "Excellent", "Good", "Fair", "Poor"],
                key=f"summary_quality_{file_stem}"
            )
            summary_feedback = st.text_area(
                "Summary Comments",
                placeholder="Please provide details...",
                height=60,
                key=f"summary_feedback_{file_stem}"
            )
        
        # General feedback and rating
        st.write("**Overall Assessment**")
        col3, col4 = st.columns([2, 1])
        with col3:
            general_feedback = st.text_area(
                "General Comments",
                placeholder="Any other comments, suggestions, or issues you noticed?",
                height=80,
                key=f"general_feedback_{file_stem}"
            )
        with col4:
            rating = st.slider(
                "Overall Rating", 
                1, 5, 3, 
                help="1=Poor, 5=Excellent",
                key=f"rating_{file_stem}"
            )
            st.write(f"Rating: {'⭐' * rating}")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            # Validate that at least some feedback is provided
            if (form_number_correct == "Select..." and pi_clause_correct == "Select..." and 
                ci_clause_correct == "Select..." and summary_quality == "Select..." and 
                not general_feedback.strip()):
                st.error("Please provide at least some feedback before submitting.")
                return
            
            # Get PDF ID from session state
            pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
            
            if pdf_id:
                # Prepare feedback data
                feedback_text_parts = []
                if form_number_correct != "Select...":
                    feedback_text_parts.append(f"Form Number: {form_number_correct}")
                    if form_number_feedback.strip():
                        feedback_text_parts.append(f"Form Details: {form_number_feedback}")
                
                if pi_clause_correct != "Select...":
                    feedback_text_parts.append(f"PI Clause: {pi_clause_correct}")
                    if pi_clause_feedback.strip():
                        feedback_text_parts.append(f"PI Details: {pi_clause_feedback}")
                
                if ci_clause_correct != "Select...":
                    feedback_text_parts.append(f"CI Clause: {ci_clause_correct}")
                    if ci_clause_feedback.strip():
                        feedback_text_parts.append(f"CI Details: {ci_clause_feedback}")
                
                if summary_quality != "Select...":
                    feedback_text_parts.append(f"Summary Quality: {summary_quality}")
                    if summary_feedback.strip():
                        feedback_text_parts.append(f"Summary Details: {summary_feedback}")
                
                structured_feedback = " | ".join(feedback_text_parts)
                
                feedback_data = {
                    'pdf_id': pdf_id,
                    'feedback_date': datetime.now(),
                    'form_number_feedback': f"{form_number_correct}: {form_number_feedback}" if form_number_feedback.strip() else form_number_correct,
                    'general_feedback': f"{structured_feedback} | General: {general_feedback}" if general_feedback.strip() else structured_feedback,
                    'rating': rating,
                    'user_session_id': get_session_id()
                }
                
                try:
                    feedback_id = store_feedback_data(feedback_data)
                    st.success("🎉 Thank you for your valuable feedback! It helps us improve our analysis.")
                    st.session_state.feedback_submitted[feedback_key] = True
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save feedback: {str(e)}")
            else:
                st.error("❌ Cannot submit feedback - PDF not found in database")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main Application
def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>📄 Enhanced Contract Analysis Platform</h1>
        <p>AI-powered contract analysis with privacy protection and intelligent feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Database status
        if st.session_state.database_initialized:
            st.markdown("<div class='database-status-success'>✅ Database Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='database-status-error'>❌ Database: {st.session_state.database_status}</div>", unsafe_allow_html=True)
        
        # Session info
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**PDFs Processed:** {len(st.session_state.pdf_files)}")
        st.write(f"**Analyses Complete:** {len(st.session_state.json_data)}")
        
        # Privacy notice
        st.markdown("""
        ---
        ### 🔒 Privacy Protection
        All uploaded documents are automatically processed with privacy protection:
        - Low-content pages are removed
        - Sensitive information is protected
        - Original documents are not stored permanently
        """)
    
    # Main layout
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane: PDF upload and management
    with col1:
        st.markdown('<div class="left-pane">', unsafe_allow_html=True)
        st.header("📁 Document Management")
        
        # PDF uploader
        st.subheader("📤 Upload Documents")
        uploaded_pdfs = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF contract files for analysis"
        )
        
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf.name not in st.session_state.pdf_files:
                    pdf_bytes = pdf.getvalue()
                    is_valid, validation_msg = validate_pdf(pdf_bytes)
                    if is_valid:
                        if len(pdf_bytes) > 10 * 1024 * 1024:  # 10MB limit
                            st.warning(f"⚠️ {pdf.name} is larger than 10MB. Processing may be slow.")
                        st.session_state.pdf_files[pdf.name] = pdf_bytes
                        st.session_state.analysis_status[pdf.name] = "Ready for processing"
                        st.success(f"✅ {pdf.name} uploaded successfully")
                    else:
                        st.error(f"❌ {pdf.name}: {validation_msg}")
        
        # Document list and selection
        if st.session_state.pdf_files:
            st.subheader("📋 Available Documents")
            
            # Create enhanced dataframe
            pdf_data = []
            for pdf_name in st.session_state.pdf_files.keys():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                file_size = len(st.session_state.pdf_files[pdf_name]) / 1024  # KB
                
                # Status emoji
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': f"{file_size:.1f}",
                    'DB ID': str(db_id)
                })
            
            pdf_df = pd.DataFrame(pdf_data)
            gb = GridOptionsBuilder.from_dataframe(pdf_df)
            gb.configure_selection(selection_mode='single', use_checkbox=False)
            gb.configure_grid_options(domLayout='normal')
            gb.configure_default_column(cellStyle={'fontSize': '14px'})
            gb.configure_column("PDF Name", cellStyle={'fontWeight': 'bold'})
            gridOptions = gb.build()

            grid_response = AgGrid(
                pdf_df,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                fit_columns_on_grid_load=True,
                theme='streamlit'
            )

            # Handle PDF selection
            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    
                    # Process PDF if not already processed
                    if st.session_state.analysis_status.get(selected_pdf) != "Processed":
                        st.session_state.processing_messages[selected_pdf] = []
                        with st.spinner(f"🔄 Processing {selected_pdf}..."):
                            message_placeholder = st.empty()
                            success, result = process_pdf_enhanced(
                                st.session_state.pdf_files[selected_pdf], 
                                selected_pdf, 
                                message_placeholder
                            )
                            
                            if success:
                                st.session_state.analysis_status[selected_pdf] = "Processed"
                                st.success(f"✅ Analysis complete for {selected_pdf}")
                            else:
                                st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                                st.error(f"❌ Failed to process {selected_pdf}: {result}")
                            
                            # Clear processing messages after showing
                            if selected_pdf in st.session_state.processing_messages:
                                final_messages = st.session_state.processing_messages[selected_pdf].copy()
                                with st.expander("📋 Processing Details", expanded=False):
                                    for msg in final_messages:
                                        st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle pane: PDF viewer
    with col2:
        st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
        st.header("📖 Document Viewer")
        
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            
            # PDF info header
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader(f"📄 {st.session_state.current_pdf}")
            with col_info2:
                file_size_mb = len(current_pdf_bytes) / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            # Show obfuscation info if available
            if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
                obf_summary = st.session_state.obfuscation_summaries[st.session_state.current_pdf]
                if obf_summary.get('obfuscation_applied', False):
                    col_obf1, col_obf2, col_obf3 = st.columns(3)
                    with col_obf1:
                        st.metric("Original Pages", obf_summary.get('total_original_pages', 0))
                    with col_obf2:
                        st.metric("Final Pages", obf_summary.get('total_final_pages', 0))
                    with col_obf3:
                        st.metric("Pages Removed", obf_summary.get('pages_removed_count', 0))
                    
                    st.markdown("""
                    <div class='obfuscation-info'>
                        🔒 <strong>Privacy Protection Applied:</strong> This document has been processed with our privacy protection system. 
                        Some pages with minimal content have been removed to protect confidentiality while preserving the core contract content for analysis.
                    </div>
                    """, unsafe_allow_html=True)
            
            # PDF display
            try:
                pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error displaying PDF: {e}")
                st.info("💡 Try downloading the PDF to view it externally.")
                st.download_button(
                    label="📥 Download PDF",
                    data=current_pdf_bytes,
                    file_name=st.session_state.current_pdf,
                    mime="application/pdf"
                )
        else:
            st.info("👆 Please select a PDF from the document list to view it here.")
            st.markdown("""
            ### 🚀 Getting Started
            1. **Upload** your PDF contract files using the uploader in the left panel
            2. **Select** a document from the list to view and analyze it
            3. **Review** the analysis results in the right panel
            4. **Provide feedback** to help improve our analysis
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right pane: Analysis results and feedback
    with col3:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.header("🔍 Analysis Results")
        
        file_stem = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else None
        if file_stem and file_stem in st.session_state.json_data:
            json_data = st.session_state.json_data[file_stem]
            
            # Analysis header with database info
            col_analysis1, col_analysis2 = st.columns(2)
            with col_analysis1:
                st.subheader(f"📊 Analysis: {file_stem}")
            with col_analysis2:
                pdf_db_id = st.session_state.pdf_database_ids.get(st.session_state.current_pdf)
                if pdf_db_id:
                    st.metric("Database ID", pdf_db_id)
            
            # Form Number
            st.markdown("### 📋 Form Number")
            form_number = json_data.get('form_number', 'Not available')
            st.markdown(f"<div class='extract-text'><strong>{form_number}</strong></div>", 
                       unsafe_allow_html=True)
            
            # Summary
            st.markdown("### 📝 Contract Summary")
            summary = json_data.get('summary', 'No summary available')
            st.markdown(f"<div class='extract-text'>{summary}</div>", 
                       unsafe_allow_html=True)
            
            # Contract Status - Enhanced UI
            st.markdown("### ✅ Contract Status")
            status_fields = {
                'data_usage_mentioned': 'Data Usage Mentioned',
                'data_limitations_exists': 'Data Limitations Exists',
                'pi_clause': 'Presence of PI Clause',
                'ci_clause': 'Presence of CI Clause'
            }
            
            # Create a nice grid for status
            col_status1, col_status2 = st.columns(2)
            status_items = list(status_fields.items())
            
            for i, (key, label) in enumerate(status_items):
                target_col = col_status1 if i % 2 == 0 else col_status2
                with target_col:
                    status = json_data.get(key, None)
                    status_str = str(status).lower() if status is not None else 'unknown'
                    
                    # Determine button style
                    if status_str in ['true', 'yes']:
                        button_class = 'status-button-true'
                        icon = "✅"
                    elif status_str in ['false', 'no']:
                        button_class = 'status-button-false' 
                        icon = "❌"
                    else:
                        button_class = 'status-button-missing'
                        icon = "❓"
                    
                    st.markdown(f"""
                    <div class='{button_class}'>
                        {icon} <strong>{label}</strong><br>
                        <small>{status}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Relevant Clauses
            st.markdown("### 📄 Relevant Clauses")
            clauses = json_data.get("relevant_clauses", [])
            
            if clauses:
                for i, clause in enumerate(clauses):
                    with st.expander(f"📑 Clause {i+1}: {clause['type'].capitalize()}", expanded=False):
                        st.markdown(f"**Type:** `{clause['type']}`")
                        st.markdown(f"**Content:**")
                        st.markdown(f"<div class='extract-text'>{clause['text']}</div>", unsafe_allow_html=True)
                        
                        # Search functionality
                        col_search1, col_search2 = st.columns(2)
                        with col_search1:
                            if st.button(f"🔍 Search in PDF", key=f"search_clause_{i}"):
                                st.session_state.search_text = clause['text'][:50]  # Limit search text
                                st.success(f"Searching for clause {i+1} in PDF...")
                                st.rerun()
                        with col_search2:
                            if len(clause['text']) > 100:
                                st.caption("⚠️ Long text may not highlight fully")
            else:
                st.info("No relevant clauses detected in this contract.")
            
            # Processing Statistics
            if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
                with st.expander("📊 Processing Statistics", expanded=False):
                    obf_summary = st.session_state.obfuscation_summaries[st.session_state.current_pdf]
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Original Words", f"{obf_summary.get('total_original_words', 0):,}")
                        st.metric("Original Paragraphs", obf_summary.get('total_original_paragraphs', 0))
                    with col_stat2:
                        st.metric("Final Words", f"{obf_summary.get('total_final_words', 0):,}")
                        retention_rate = obf_summary.get('word_retention_rate', 0)
                        st.metric("Word Retention", f"{retention_rate:.1%}")
                    
                    # Word count analysis if available
                    if 'word_count_analysis' in obf_summary:
                        wc_analysis = obf_summary['word_count_analysis']
                        st.write("**Word Count Analysis:**")
                        st.write(f"- Average words per page: {wc_analysis.get('average_word_count_per_page', 0):.1f}")
                        st.write(f"- Removal threshold: {wc_analysis.get('word_count_threshold', 0):.1f}")
                        st.write(f"- Pages removed: {wc_analysis.get('removed_pages_word_counts', [])}")
            
            # Feedback Section
            st.markdown("---")
            render_feedback_form(st.session_state.current_pdf, file_stem, json_data)
            
        else:
            st.info("👈 Select and process a PDF to see analysis results here.")
            
            # Show helpful information
            if st.session_state.pdf_files:
                unprocessed = [name for name, status in st.session_state.analysis_status.items() 
                             if status != "Processed"]
                if unprocessed:
                    st.markdown("### 📋 Pending Analysis")
                    for pdf_name in unprocessed[:3]:  # Show first 3
                        status = st.session_state.analysis_status[pdf_name]
                        if "processing" in status.lower():
                            st.write(f"🔄 {pdf_name} - {status}")
                        else:
                            st.write(f"⏳ {pdf_name} - Ready for processing")
                    
                    if len(unprocessed) > 3:
                        st.write(f"... and {len(unprocessed) - 3} more documents")
            
            # System capabilities info
            st.markdown("""
            ### 🎯 Analysis Capabilities
            Our AI system analyzes contracts for:
            - **Form identification** - Detect contract types and forms
            - **PI/CI clauses** - Identify privacy and confidentiality terms  
            - **Data usage terms** - Find data handling restrictions
            - **Contract summaries** - Generate concise overviews
            - **Clause extraction** - Extract relevant legal clauses
            
            ### 🔒 Privacy Features
            - Automatic content obfuscation
            - Low-value page removal
            - Secure database storage
            - Session-based processing
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()