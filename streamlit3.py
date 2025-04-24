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
from pdf_text_processor import PDFTextProcessor  # Assuming your custom class
from ecfr_logger import ECFRLogger  # Assuming your custom class
from contract_analyzer import ContractAnalyzer  # Assuming your custom class

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
        height: 85vh;
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
    .stButton>button {
        width: 100%;
    }
    .pdf-select-button {
        margin-bottom: 5px;
        padding: 8px;
        text-align: left;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pdf-select-button.selected {
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
    .processing-message {
        color: #0068c9;
        font-size: 14px;
        margin: 5px 0;
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

# Function to validate JSON
def validate_json(json_path):
    """Validate JSON file"""
    try:
        with open(json_path, 'r') as f:
            json.load(f)
        return True, None
    except Exception as e:
        return False, f"Invalid JSON: {str(e)}"

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

# Initialize session state
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = {}
if 'json_data' not in st.session_state:
    st.session_state.json_data = {}
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'search_text' not in st.session_state:
    st.session_state.search_text = None
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = {}
if 'processing_messages' not in st.session_state:
    st.session_state.processing_messages = {}

# Function to load pre-loaded PDFs and JSONs
def load_preloaded_data(preload_folder="./preloaded_contracts"):
    """Load pre-loaded PDFs and JSONs from specified folder"""
    pdf_files = glob.glob(os.path.join(preload_folder, "*.pdf"))
    preloaded_files = []
    logger = ECFRLogger()
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        file_stem = Path(pdf_name).stem
        json_path = os.path.join(preload_folder, f"{file_stem}.json")
        
        # Validate PDF
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
            is_valid_pdf, metadata_or_error = validate_pdf(pdf_bytes)
            if not is_valid_pdf:
                logger.error(f"Pre-loaded PDF {pdf_name} failed: {metadata_or_error}")
                continue
            
        # Check for corresponding JSON
        json_exists = os.path.exists(json_path)
        if json_exists:
            is_valid_json, json_error = validate_json(json_path)
            if not is_valid_json:
                logger.error(f"Pre-loaded JSON for {pdf_name} failed: {json_error}")
                continue
        
        preloaded_files.append((pdf_name, pdf_bytes, json_exists, json_path))
    
    return preloaded_files

def sanitize_search_text(text):
    """Clean up text for PDF search"""
    if not text:
        return ""
    text = text[:100]
    text = re.sub(r'[^\w\s.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def set_current_pdf(pdf_name):
    """Set the current PDF to display"""
    st.session_state.current_pdf = pdf_name
    st.session_state.current_page = 1
    st.session_state.search_text = None

def process_pdf(pdf_bytes, pdf_name, temp_dir, pdf_text_processor, contract_analyzer, logger, message_placeholder):
    """Process a single PDF and generate JSON with retries for ContractAnalyzer"""
    try:
        file_stem = Path(pdf_name).stem
        pdf_path = os.path.join(temp_dir, f"{file_stem}.pdf")
        preprocessed_path = os.path.join(temp_dir, f"{file_stem}.txt")
        output_path = os.path.join(temp_dir, f"{file_stem}.json")

        # Save PDF to temporary file
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

        # Process PDF text
        contract_text = pdf_text_processor.process_pdf(pdf_path, preprocessed_path)
        logger.info(f"Text extracted from {pdf_name}")
        st.session_state.processing_messages[pdf_name].append("Text extracted from PDF")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )

        # Analyze contract
        st.session_state.processing_messages[pdf_name].append("Analyzing the document")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )
        logger.info(f"Analyzing the document {pdf_name}")

        # Retry ContractAnalyzer up to 2 times
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                results = contract_analyzer.analyze_contract(contract_text, output_path)
                break  # Exit loop on success
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {pdf_name}: {str(e)}")
                    continue
                else:
                    logger.error(f"Failed after {max_retries} retries for {pdf_name}: {str(e)}")
                    return False, "An error occurred. Please submit another PDF or reach out to Support."

        # Load generated JSON
        with open(output_path, 'r') as f:
            json_data = json.load(f)

        return True, json_data
    except Exception as e:
        logger.error(f"Error processing {pdf_name}: {str(e)}")
        return False, "An error occurred. Please submit another PDF or reach out to Support."

def main():
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane: PDF upload and controls
    with col1:
        with st.container():
            st.markdown('<div class="left-pane">', unsafe_allow_html=True)
            st.header("Contracts")
            
            # Pre-loaded PDFs dropdown
            st.subheader("Pre-loaded PDFs")
            preloaded_files = load_preloaded_data()
            preloaded_pdf_names = [pdf_name for pdf_name, _, _, _ in preloaded_files]
            preloaded_pdf_names.insert(0, "Select a pre-loaded PDF")
            
            selected_preloaded_pdf = st.selectbox(
                "Choose a pre-loaded PDF",
                preloaded_pdf_names,
                key="preloaded_pdf_select"
            )
            
            if selected_preloaded_pdf and selected_preloaded_pdf != "Select a pre-loaded PDF":
                for pdf_name, pdf_bytes, json_exists, json_path in preloaded_files:
                    if pdf_name == selected_preloaded_pdf:
                        file_stem = Path(pdf_name).stem
                        if pdf_name not in st.session_state.pdf_files:
                            st.session_state.pdf_files[pdf_name] = pdf_bytes
                            st.session_state.analysis_status[pdf_name] = "Not processed"
                            if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
                                st.warning(f"{pdf_name} is larger than 1.5MB and may fail to display.")
                        
                        if json_exists and file_stem not in st.session_state.json_data:
                            with open(json_path, 'r') as f:
                                st.session_state.json_data[file_stem] = json.load(f)
                            st.session_state.analysis_status[pdf_name] = "Processed"
                        
                        if st.session_state.current_pdf is None:
                            st.session_state.current_pdf = pdf_name
                        
                        st.success(f"Loaded pre-loaded PDF: {pdf_name}")
                        break
            
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
                    if pdf.name not in st.session_state.pdf_files:
                        pdf_bytes = pdf.getvalue()
                        is_valid, metadata_or_error = validate_pdf(pdf_bytes)
                        if is_valid:
                            if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
                                st.warning(f"{pdf.name} is larger than 1.5MB and may fail to display.")
                            st.session_state.pdf_files[pdf.name] = pdf_bytes
                            st.session_state.analysis_status[pdf.name] = "Not processed"
                        else:
                            st.error(f"Failed to load {pdf.name}: {metadata_or_error}")
            
            # Analyze PDFs button
            if st.session_state.pdf_files:
                if st.button("Analyze PDFs", key="analyze_button"):
                    pdf_text_processor = PDFTextProcessor()
                    logger = ECFRLogger()
                    contract_analyzer = ContractAnalyzer()
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for pdf_name, pdf_bytes in st.session_state.pdf_files.items():
                            if st.session_state.analysis_status.get(pdf_name) != "Processed":
                                # Initialize processing messages for this PDF
                                st.session_state.processing_messages[pdf_name] = []
                                with st.spinner(f"Processing {pdf_name}..."):
                                    message_placeholder = st.empty()
                                    success, result = process_pdf(
                                        pdf_bytes, pdf_name, temp_dir, 
                                        pdf_text_processor, contract_analyzer, logger, message_placeholder
                                    )
                                    if success:
                                        st.session_state.json_data[Path(pdf_name).stem] = result
                                        st.session_state.analysis_status[pdf_name] = "Processed"
                                        st.success(f"Analysis complete for {pdf_name}")
                                    else:
                                        st.session_state.analysis_status[pdf_name] = result
                                        st.error(f"Failed to process {pdf_name}: {result}")
                                    # Clear processing messages after completion
                                    st.session_state.processing_messages[pdf_name] = []
                                    message_placeholder.empty()
            
            # Display analysis status
            if st.session_state.analysis_status:
                st.subheader("Analysis Status")
                for pdf_name, status in st.session_state.analysis_status.items():
                    st.write(f"{pdf_name}: {status}")
            
            # PDF selection buttons
            if st.session_state.pdf_files:
                st.subheader("Available PDFs")
                for pdf_name in st.session_state.pdf_files.keys():
                    if st.button(pdf_name, key=f"pdf_btn_{pdf_name}", 
                               type="primary" if pdf_name == st.session_state.current_pdf else "secondary",
                               use_container_width=True):
                        set_current_pdf(pdf_name)
                        st.rerun()
            
            # Page navigation
            if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
                st.subheader("Page Navigation")
                num_pages = 10
                page_num = st.number_input(
                    "Go to page:", 
                    min_value=1, 
                    max_value=num_pages, 
                    value=st.session_state.current_page,
                    step=1,
                    key="page_navigator"
                )
                if st.button("Navigate", key="nav_button"):
                    st.session_state.current_page = page_num
                    st.session_state.search_text = None
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle pane: PDF viewer
    with col2:
        with st.container():
            st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
            st.header("PDF Viewer")
            
            if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
                current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
                st.subheader(f"Viewing: {st.session_state.current_pdf}")
                
                try:
                    pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error with iframe: {e}")
                    try:
                        pdf_display = display_pdf_object(current_pdf_bytes)
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error with object tag: {e}")
                        is_valid, metadata_or_error = validate_pdf(current_pdf_bytes)
                        if not is_valid:
                            st.error(f"Validation failed: {metadata_or_error}")
                        else:
                            st.info(f"PDF metadata: {metadata_or_error}")
                        st.download_button(
                            label="Download PDF",
                            data=current_pdf_bytes,
                            file_name=st.session_state.current_pdf,
                            mime="application/pdf",
                            key="download_pdf"
                        )
            else:
                st.info("Select or upload a PDF.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right pane: JSON data display
    with col3:
        with st.container():
            st.markdown('<div class="json-details">', unsafe_allow_html=True)
            st.header("Contract Analysis")
            
            file_stem = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else None
            if file_stem and file_stem in st.session_state.json_data:
                json_data = st.session_state.json_data[file_stem]
                
                # Form number
                st.subheader("Form Number")
                st.markdown(f"<div class='extract-text'>{json_data.get('form_number', 'Not available')}</div>", 
                           unsafe_allow_html=True)
                
                # Summary
                st.subheader("Summary")
                st.markdown(f"<div class='extract-text'>{json_data.get('summary', 'No summary available')}</div>", 
                           unsafe_allow_html=True)
                
                # Contract status
                st.subheader("Contract Status")
                binary_keys = {
                    'data_usage_mentioned': 'Data Usage Mentioned',
                    'data_limitations_exists': 'Data Limitations Exists',
                    'pi_clause': 'Presence of PI Clause',
                    'ci_clause': 'Presence of CI Clause'
                }
                
                for key, label in binary_keys.items():
                    status = json_data.get(key, False)
                    button_class = 'status-button-true' if status else 'status-button-false'
                    st.markdown(f"<div class='{button_class}'>{label}: {status}</div>", 
                               unsafe_allow_html=True)
                
                # Relevant clauses
                st.subheader("Relevant Clauses")
                for i, clause in enumerate(json_data.get("relevant_clauses", [])):
                    with st.expander(f"Clause {i+1}: {clause['type'].capitalize()}"):
                        st.write(f"**Type:** {clause['type']}")
                        st.write(f"**Text:** {clause['text']}")
                        if st.button(f"Search clause {i+1} text", key=f"search_clause_{i}"):
                            st.session_state.search_text = clause['text']
                            st.success(f"Searching for clause {i+1}...")
                            st.rerun()
                        if len(clause['text']) > 100:
                            st.warning("Text longer than 100 characters may not highlight fully.")
                        if st.button("Highlight in PDF", key=f"highlight_{i}"):
                            st.session_state.search_text = clause['text']
                            st.success(f"Searching for clause {i+1}...")
                            st.rerun()
                
            else:
                st.info("Select a PDF and ensure analysis is complete.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
