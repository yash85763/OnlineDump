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
from difflib import SequenceMatcher

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
    
    .page-info {
        background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 100%);
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 4px solid #ffa500;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #8b4513;
        font-weight: 500;
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #f0fff0 0%, #e6ffe6 100%);
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #32cd32;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: #228b22;
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
    
    .batch-progress {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #ced4da;
    }
</style>
""", unsafe_allow_html=True)


def load_processed_pdfs_from_database():
    """Load all processed PDFs from database on app startup"""
    try:
        # Import at the function level to avoid import issues
        import sys
        import os
        
        # Add the current directory to Python path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        from config.database import get_all_processed_pdfs, initialize_database, check_database_connection
        
        # Check database connection first
        if not check_database_connection():
            st.warning("Database connection failed - cannot load previous PDFs")
            return
        
        # Initialize database connection
        initialize_database()
        
        # Get all processed PDFs
        processed_pdfs = get_all_processed_pdfs()
        
        if not processed_pdfs:
            st.info("No previously processed PDFs found in database")
            return
        
        loaded_count = 0
        
        for pdf_record in processed_pdfs:
            pdf_name = pdf_record['pdf_name']
            
            # Skip if already loaded in session
            if pdf_name in st.session_state.pdf_files:
                continue
            
            # Load PDF bytes from final_content
            if pdf_record.get('final_content'):
                try:
                    # Assuming final_content is stored as base64 string
                    pdf_bytes = base64.b64decode(pdf_record['final_content'])
                    st.session_state.pdf_files[pdf_name] = pdf_bytes
                    loaded_count += 1
                except Exception as e:
                    st.warning(f"Could not decode PDF bytes for {pdf_name}: {str(e)}")
                    continue
            else:
                st.warning(f"No PDF content found for {pdf_name}")
                continue
            
            # Load analysis data from raw_analysis_json
            if pdf_record.get('raw_analysis_json'):
                file_stem = Path(pdf_name).stem
                try:
                    if isinstance(pdf_record['raw_analysis_json'], str):
                        analysis_data = json.loads(pdf_record['raw_analysis_json'])
                    else:
                        analysis_data = pdf_record['raw_analysis_json']
                    
                    st.session_state.json_data[file_stem] = analysis_data
                    st.session_state.analysis_status[pdf_name] = "Processed"
                    
                except Exception as e:
                    st.warning(f"Could not parse analysis data for {pdf_name}: {str(e)}")
                    st.session_state.analysis_status[pdf_name] = "Error loading analysis"
            else:
                st.session_state.analysis_status[pdf_name] = "No analysis data"
        
        if loaded_count > 0:
            st.success(f"✅ Loaded {loaded_count} processed PDFs from database")
            
            # Set the first loaded PDF as current if none is selected
            if not st.session_state.current_pdf and st.session_state.pdf_files:
                st.session_state.current_pdf = list(st.session_state.pdf_files.keys())[0]
        
    except ImportError as e:
        st.error(f"Cannot import database module: {str(e)}")
        st.error("Make sure config/database.py exists and is properly configured")
    except Exception as e:
        st.warning(f"Could not load PDFs from database: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        
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
        'raw_pdf_data': {},  # Store raw PDF parsing data for page matching
        'current_pdf': None,
        'analysis_status': {},
        'processing_messages': {},
        'pdf_database_ids': {},
        'search_text': None,
        'feedback_submitted': {},
        'obfuscation_summaries': {},
        'session_id': get_session_id(),
        'batch_processing_status': None,
        'batch_processed_count': 0
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

# Page Number Detection Functions
def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize text by removing extra whitespace and converting to lowercase
    text1_norm = ' '.join(text1.lower().split())
    text2_norm = ' '.join(text2.lower().split())
    
    return SequenceMatcher(None, text1_norm, text2_norm).ratio()

def find_clause_page_number(clause_text, raw_pdf_data, similarity_threshold=0.9):
    """
    Find the page number where a clause appears in the raw PDF data
    
    Args:
        clause_text (str): The clause text to search for
        raw_pdf_data (dict): Raw PDF parsing data with pages and paragraphs
        similarity_threshold (float): Minimum similarity score (default 0.9 for 90%)
    
    Returns:
        dict: Contains page_number, similarity_score, and matched_text
    """
    if not clause_text or not raw_pdf_data:
        return {"page_number": None, "similarity_score": 0.0, "matched_text": ""}
    
    best_match = {
        "page_number": None,
        "similarity_score": 0.0,
        "matched_text": ""
    }
    
    # Get pages from raw PDF data
    pages = raw_pdf_data.get('pages', [])
    
    for page_idx, page_data in enumerate(pages):
        page_number = page_idx + 1  # Pages are 1-indexed for display
        paragraphs = page_data.get('paragraphs', [])
        
        # Check each paragraph in the page
        for paragraph in paragraphs:
            if not paragraph:
                continue
                
            # Calculate similarity between clause and paragraph
            similarity = calculate_text_similarity(clause_text, paragraph)
            
            # Update best match if this is better
            if similarity > best_match["similarity_score"]:
                best_match = {
                    "page_number": page_number,
                    "similarity_score": similarity,
                    "matched_text": paragraph
                }
        
        # Also check the entire page content as a single block
        page_content = ' '.join(paragraphs)
        if page_content:
            similarity = calculate_text_similarity(clause_text, page_content)
            if similarity > best_match["similarity_score"]:
                best_match = {
                    "page_number": page_number,
                    "similarity_score": similarity,
                    "matched_text": page_content[:200] + "..." if len(page_content) > 200 else page_content
                }
    
    # Only return match if it meets the threshold
    if best_match["similarity_score"] >= similarity_threshold:
        return best_match
    else:
        return {"page_number": None, "similarity_score": best_match["similarity_score"], "matched_text": ""}

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
def process_pdf_enhanced(pdf_bytes, pdf_name, message_placeholder, logger):
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
            
            # Store raw PDF data for page number matching
            st.session_state.raw_pdf_data[pdf_name] = {
                'pages': result.get('pages', [])
            }
            
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
            
            # Run contract analysis on obfuscated content
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
                            logger.error(f"Database storage failed for {pdf_name}: {str(e)}")
                    
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
        logger.error(f"Processing failed for {pdf_name}: {str(e)}")
        return False, f"Processing failed: {str(e)}"
    finally:
        st.session_state.processing_messages[pdf_name].append("📝 Processing complete - Ready for review")

# Batch Processing Function
def process_batch_pdfs(logger):
    """Process all uploaded PDFs one by one with progress updates"""
    pdf_files = list(st.session_state.pdf_files.keys())
    total_pdfs = len(pdf_files)
    
    if total_pdfs == 0:
        st.warning("⚠️ No PDFs uploaded for batch processing.")
        return
    
    st.session_state.batch_processing_status = "Running"
    st.session_state.batch_processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_name in enumerate(pdf_files):
        if st.session_state.analysis_status.get(pdf_name) == "Processed":
            continue  # Skip already processed PDFs
        
        status_text.markdown(
            f"<div class='batch-progress'>📄 Processing {pdf_name} ({i+1}/{total_pdfs})...</div>",
            unsafe_allow_html=True
        )
        
        message_placeholder = st.empty()
        success, result = process_pdf_enhanced(
            st.session_state.pdf_files[pdf_name], 
            pdf_name, 
            message_placeholder,
            logger
        )
        
        if success:
            st.session_state.analysis_status[pdf_name] = "Processed"
            st.session_state.batch_processed_count += 1
            status_text.markdown(
                f"<div class='batch-progress'>✅ {pdf_name} processed successfully ({st.session_state.batch_processed_count}/{total_pdfs})</div>",
                unsafe_allow_html=True
            )
        else:
            st.session_state.analysis_status[pdf_name] = f"❌ Failed: {result}"
            status_text.markdown(
                f"<div class='batch-progress'>❌ Failed to process {pdf_name}: {result} ({st.session_state.batch_processed_count}/{total_pdfs})</div>",
                unsafe_allow_html=True
            )
        
        # Update progress bar
        progress = (i + 1) / total_pdfs
        progress_bar.progress(progress)
        
        # Show processing messages
        if pdf_name in st.session_state.processing_messages:
            with st.expander(f"📋 Processing Details for {pdf_name}", expanded=False):
                for msg in st.session_state.processing_messages[pdf_name]:
                    st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
    
    st.session_state.batch_processing_status = "Completed"
    st.success(f"🎉 Batch processing completed: {st.session_state.batch_processed_count}/{total_pdfs} PDFs processed successfully")

# Feedback System
def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF"""
    feedback_key = f"feedback_{file_stem}"
    
    # Check if feedback was already submitted
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
        # Form number selection (1-10)
        form_number = st.selectbox(
            "Select Form Number",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=0,  # Default to form 1
            help="Select the form number that best matches this document",
            key=f"form_number_{file_stem}"
        )
        
        # Rating selection
        rating = st.radio(
            "Is this analysis correct?",
            ["Correct", "Partially Correct", "Incorrect"],
            help="Rate the accuracy of the analysis",
            key=f"rating_{file_stem}"
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
            height=120,
            help="Provide detailed feedback about the analysis accuracy, suggestions for improvement, etc.",
            key=f"feedback_text_{file_stem}"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            # Validation - require feedback text
            if not feedback_text.strip():
                st.error("Please provide some feedback before submitting.")
                return
            
            # Get PDF ID from session state
            pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
            
            if pdf_id:
                # Prepare feedback data according to database schema
                feedback_data = {
                    'pdf_id': pdf_id,
                    'feedback_date': datetime.now(),
                    'form_number_feedback': form_number,  # INTEGER form number
                    'general_feedback': feedback_text.strip(),  # Single feedback text
                    'rating': rating_value,  # INTEGER rating (1-5)
                    'user_session_id': get_session_id()
                }
                
                try:
                    # Store feedback using the database function
                    feedback_id = store_feedback_data(feedback_data)
                    
                    if feedback_id:
                        st.success("🎉 Thank you for your valuable feedback! It helps us improve our analysis.")
                        st.session_state.feedback_submitted[feedback_key] = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. No feedback ID returned.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to save feedback: {str(e)}")
                    print(f"Feedback submission error: {str(e)}")  # For debugging
            else:
                st.error("❌ Cannot submit feedback - PDF not found in database")
    
    st.markdown("</div>", unsafe_allow_html=True)

def get_session_id():
    """Get or create session ID for user tracking"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def display_feedback_summary_for_pdf(pdf_name):
    """Display feedback summary for a specific PDF"""
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    
    if not pdf_id:
        return
    
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
                    st.markdown(f"### Feedback Summary for '{pdf_name}'")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Feedback", result['total_feedback'])
                    with col2:
                        st.metric("Average Rating", f"{result['avg_rating']:.1f}/5")
                    with col3:
                        st.metric("Positive", result['positive_feedback'])
                    with col4:
                        st.metric("Negative", result['negative_feedback'])
                else:
                    st.info(f"No feedback yet for '{pdf_name}'")
                
    except Exception as e:
        st.error(f"Error displaying feedback summary: {str(e)}")

def check_existing_feedback_for_pdf(pdf_name, user_session_id=None):
    """Check if user already provided feedback for this PDF"""
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    
    if not pdf_id:
        return False
    
    if not user_session_id:
        user_session_id = get_session_id()
    
    try:
        sql = """
            SELECT id FROM feedback 
            WHERE pdf_id = %s 
            AND user_session_id = %s 
            LIMIT 1
        """
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (pdf_id, user_session_id))
                result = cur.fetchone()
                return result is not None
    except Exception as e:
        print(f"Error checking existing feedback: {str(e)}")
        return False
# Main Application
def main():
    # Initialize session state
    initialize_session_state()
    logger = ECFRLogger()
    
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
        
        if st.session_state.database_initialized:
            st.markdown("<div class='database-status-success'>✅ Database Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='database-status-error'>❌ Database: {st.session_state.database_status}</div>", unsafe_allow_html=True)
        
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**PDFs Processed:** {len([s for s in st.session_state.analysis_status.values() if s == 'Processed'])}/{len(st.session_state.pdf_files)}")
        st.write(f"**Batch Status:** {st.session_state.batch_processing_status or 'Not started'}")
        
        st.markdown("""
        ---
        ### 🔒 Privacy Protection
        All uploaded documents are automatically processed with privacy protection:
        - Low-content pages are removed
        - Sensitive information is protected
        - Original documents are not stored permanently
        
        ### 📍 Page Number Detection
        Enhanced clause analysis now includes:
        - Page number identification for each clause
        - 90% similarity matching with original content
        - Smart text comparison algorithms
        """)

    # Load processed PDFs from database on startup
    if 'database_loaded' not in st.session_state:
        with st.spinner("Loading previously processed PDFs from database..."):
            load_processed_pdfs_from_database()
            st.session_state.database_loaded = True
    
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
                        if len(pdf_bytes) > 10 * 1024 * 1024:
                            st.warning(f"⚠️ {pdf.name} is larger than 10MB. Processing may be slow.")
                        st.session_state.pdf_files[pdf.name] = pdf_bytes
                        st.session_state.analysis_status[pdf.name] = "Ready for processing"
                        st.success(f"✅ {pdf.name} uploaded successfully")
                    else:
                        st.error(f"❌ {pdf.name}: {validation_msg}")
        
        # Processing buttons
        st.subheader("⚙️ Process Documents")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Process Single PDF", disabled=not st.session_state.current_pdf):
                if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
                    if st.session_state.analysis_status.get(st.session_state.current_pdf) != "Processed":
                        st.session_state.processing_messages[st.session_state.current_pdf] = []
                        with st.spinner(f"🔄 Processing {st.session_state.current_pdf}..."):
                            message_placeholder = st.empty()
                            success, result = process_pdf_enhanced(
                                st.session_state.pdf_files[st.session_state.current_pdf], 
                                st.session_state.current_pdf, 
                                message_placeholder,
                                logger
                            )
                            
                            if success:
                                st.session_state.analysis_status[st.session_state.current_pdf] = "Processed"
                                st.success(f"✅ Analysis complete for {st.session_state.current_pdf}")
                            else:
                                st.session_state.analysis_status[st.session_state.current_pdf] = f"❌ Failed: {result}"
                                st.error(f"❌ Failed to process {st.session_state.current_pdf}: {result}")
                            
                            if st.session_state.current_pdf in st.session_state.processing_messages:
                                with st.expander("📋 Processing Details", expanded=False):
                                    for msg in st.session_state.processing_messages[st.session_state.current_pdf]:
                                        st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
        
        with col_btn2:
            if st.button("📚 Process All PDFs (Batch)", disabled=not st.session_state.pdf_files):
                process_batch_pdfs(logger)
        
        # Document list and selection
        if st.session_state.pdf_files:
            st.subheader("📋 Available Documents")
        
            st.write(f"DEBUG: Found {len(st.session_state.pdf_files)} PDFs in session state")
            for pdf_name, status in st.session_state.analysis_status.items():
                st.write(f" - {pdf_name}: {status}")
        
            pdf_data = []
            for pdf_name in st.session_state.pdf_files.keys():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
        
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
        
            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    # Fetch data from database for the selected PDF
                    pdf_id = st.session_state.pdf_database_ids.get(selected_pdf)
                    if pdf_id:
                        try:
                            # Fetch PDF data
                            pdf_record = get_pdf_by_id(pdf_id)
                            if pdf_record and pdf_record.get('final_content'):
                                try:
                                    pdf_bytes = base64.b64decode(pdf_record['final_content'])
                                    st.session_state.pdf_files[selected_pdf] = pdf_bytes
                                    st.session_state.analysis_status[selected_pdf] = "Processed"
                                except Exception as e:
                                    st.error(f"Could not decode PDF bytes for {selected_pdf}: {str(e)}")
                                    st.session_state.analysis_status[selected_pdf] = "Error loading PDF"
                                
                                # Store raw PDF data for page matching
                                if pdf_record.get('raw_analysis_json'):
                                    try:
                                        raw_analysis = json.loads(pdf_record['raw_analysis_json']) if isinstance(pdf_record['raw_analysis_json'], str) else pdf_record['raw_analysis_json']
                                        st.session_state.raw_pdf_data[selected_pdf] = {
                                            'pages': raw_analysis.get('pages', [])
                                        }
                                    except Exception as e:
                                        st.warning(f"Could not parse raw analysis data for {selected_pdf}: {str(e)}")
                                
                                # Fetch latest analysis data
                                analysis_record = get_latest_analysis(pdf_id)
                                if analysis_record and analysis_record.get('raw_json'):
                                    file_stem = Path(selected_pdf).stem
                                    try:
                                        analysis_data = json.loads(analysis_record['raw_json']) if isinstance(analysis_record['raw_json'], str) else analysis_record['raw_json']
                                        st.session_state.json_data[file_stem] = analysis_data
                                        st.session_state.analysis_status[selected_pdf] = "Processed"
                                    except Exception as e:
                                        st.warning(f"Could not parse analysis data for {selected_pdf}: {str(e)}")
                                        st.session_state.analysis_status[selected_pdf] = "Error loading analysis"
                                else:
                                    st.warning(f"No analysis data found for {selected_pdf}")
                                    st.session_state.analysis_status[selected_pdf] = "No analysis data"
                            else:
                                st.error(f"No PDF content found for {selected_pdf}")
                                st.session_state.analysis_status[selected_pdf] = "No PDF content"
                        except Exception as e:
                            st.error(f"Error fetching data for {selected_pdf}: {str(e)}")
                            st.session_state.analysis_status[selected_pdf] = f"Error: {str(e)}"
                    else:
                        st.error(f"No database ID found for {selected_pdf}")
                        st.session_state.analysis_status[selected_pdf] = "No database ID"
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle pane: PDF viewer
    with col2:
        st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
        st.header("📖 Document Viewer")
        
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader(f"📄 {st.session_state.current_pdf}")
            with col_info2:
                file_size_mb = len(current_pdf_bytes) / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
                obf_summary = st.session_state.obfuscation_summaries[st.session_state.current_pdf]
                if obf_summary.get('obfuscation_applied', False):
                    col_obf1, col_obf2, col_obf3 = st.columns(3)
                    with col_obf1:
                        st.metric("Original Pages", obf_summary.get('total_original_pages', 0))
                    with col_obf2:
        with col3:
            st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
            st.header("📊 Analysis Results")
            
            if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
                file_stem = Path(st.session_state.current_pdf).stem
                if file_stem in st.session_state.json_data:
                    analysis_data = st.session_state.json_data[file_stem]
                    
                    # Display key analysis metrics
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Form Number", analysis_data.get('form_number', 'N/A'))
                    with col_metric2:
                        st.metric("Data Usage", analysis_data.get('data_usage_mentioned', 'N/A'))
                    with col_metric3:
                        st.metric("Data Limitations", analysis_data.get('data_limitations_exists', 'N/A'))
                    
                    # Display summary
                    st.subheader("📝 Summary")
                    st.markdown(f"<div class='extract-text'>{analysis_data.get('summary', 'No summary available')}</div>", unsafe_allow_html=True)
                    
                    # Display clauses with page numbers
                    st.subheader("🔍 Relevant Clauses")
                    clauses = analysis_data.get('relevant_clauses', [])
                    raw_pdf_data = st.session_state.raw_pdf_data.get(st.session_state.current_pdf, {})
                    
                    for clause in clauses:
                        clause_type = clause.get('type', 'Unknown')
                        clause_text = clause.get('text', '')
                        
                        # Find page number for the clause
                        page_info = find_clause_page_number(clause_text, raw_pdf_data)
                        page_number = page_info.get('page_number', None)
                        similarity_score = page_info.get('similarity_score', 0.0)
                        
                        st.markdown(f"**{clause_type}**")
                        if page_number:
                            st.markdown(f"<div class='page-info'>📄 Page {page_number} (Similarity: {similarity_score:.2%})</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='page-info'>📄 Page not identified</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='extract-text'>{clause_text}</div>", unsafe_allow_html=True)
                    
                    # Display feedback form
                    st.subheader("📝 Feedback")
                    render_feedback_form(st.session_state.current_pdf, file_stem, analysis_data)
                else:
                    st.info("No analysis data available for the selected PDF.")
            else:
                st.info("Select a PDF to view analysis results.")
    
            st.markdown('</div>', unsafe_allow_html=True)
                        

if __name__ == "__main__":
    main()
    
#this is main
