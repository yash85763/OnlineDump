import os
import json
import base64
import streamlit as st
from pathlib import Path
import re
import urllib.parse
from PyPDF2 import PdfReader
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Contract Analysis Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .pdf-viewer {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
    }
    .json-details {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
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

# Check for pre-loaded data
def check_preloaded_data():
    pdf_exists = os.path.exists("shimi_paper.pdf")
    json_exists = os.path.exists("shimi_paper.json")
    return pdf_exists, json_exists

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

def main():
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane: PDF upload and controls
    with col1:
        st.header("Contracts")
        
        # Demo mode
        pdf_exists, json_exists = check_preloaded_data()
        use_demo_data = st.checkbox("Use pre-loaded SHIMI paper", 
                                    value=pdf_exists and json_exists,
                                    key="use_demo_data")
        
        if use_demo_data and pdf_exists and json_exists:
            if "shimi_paper.pdf" not in st.session_state.pdf_files:
                with open("shimi_paper.pdf", 'rb') as f:
                    pdf_bytes = f.read()
                    is_valid, metadata_or_error = validate_pdf(pdf_bytes)
                    if is_valid:
                        st.session_state.pdf_files["shimi_paper.pdf"] = pdf_bytes
                    else:
                        st.error(f"Pre-loaded SHIMI paper failed: {metadata_or_error}")
                    
            if "shimi_paper.json" not in st.session_state.json_data:
                with open("shimi_paper.json", 'r') as f:
                    st.session_state.json_data["shimi_paper"] = json.load(f)
            
            if st.session_state.current_pdf is None:
                st.session_state.current_pdf = "shimi_paper.pdf"
                
            st.success("Using pre-loaded SHIMI paper")
            
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
                        st.session_state.pdf_files[pdf.name] = pdf_bytes
                        if st.session_state.current_pdf is None:
                            st.session_state.current_pdf = pdf.name
                    else:
                        st.error(f"Failed to load {pdf.name}: {metadata_or_error}")
        
        # JSON uploader
        st.subheader("Upload JSON")
        uploaded_jsons = st.file_uploader(
            "Upload JSON Extracts",
            type="json",
            key="json_uploader",
            accept_multiple_files=True
        )
        
        if uploaded_jsons:
            for json_file in uploaded_jsons:
                file_stem = Path(json_file.name).stem
                st.session_state.json_data[file_stem] = json.load(json_file)
        
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
    
    # Middle pane: PDF viewer
    with col2:
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
    
    # Right pane: JSON data display
    with col3:
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
            st.info("Select a PDF and upload its JSON data.")

if __name__ == "__main__":
    main()
