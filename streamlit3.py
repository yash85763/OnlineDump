import os
import json
import base64
import streamlit as st
from pathlib import Path
import pandas as pd
import re
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="Research Paper Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-container {
        display: flex;
        flex-direction: row;
    }
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
    .button-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 15px;
    }
    .dict-button {
        padding: 5px 10px;
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        cursor: pointer;
    }
    .dict-button:hover {
        background-color: #ddd;
    }
    .active-button {
        background-color: #0068c9;
        color: white;
    }
    .extract-text {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #0068c9;
        margin: 10px 0;
    }
    .key-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
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
    .clause-button {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 8px;
        margin: 5px 0;
        width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Function to display PDF directly using iframe
def display_pdf_iframe(pdf_bytes, search_text=None):
    """Display PDF using a simple iframe with optional search text"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}'
    if search_text:
        sanitized_text = sanitize_search_text(search_text)
        encoded_text = urllib.parse.quote(sanitized_text)
        pdf_display += f'#search={encoded_text}'
    pdf_display += '" width="100%" height="600px" type="application/pdf"></iframe>'
    
    # Add JavaScript to trigger search (if supported)
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
    """Display PDF using an object tag (fallback method)"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<object data="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width="100%" height="600px"></object>'
    return pdf_display

# Initialize session state variables
if 'selected_extract_index' not in st.session_state:
    st.session_state.selected_extract_index = 0

if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = {}  # Dictionary to store multiple PDF files: {filename: bytes}

if 'json_data' not in st.session_state:
    st.session_state.json_data = {}

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

if 'search_text' not in st.session_state:
    st.session_state.search_text = None

# Check if pre-loaded data exists
def check_preloaded_data():
    pdf_exists = os.path.exists("shimi_paper.pdf")
    json_exists = os.path.exists("shimi_paper.json")
    return pdf_exists, json_exists

def sanitize_search_text(text):
    """Clean up text for searching in PDF"""
    if not text:
        return ""
    # Take first 50 characters to avoid long search strings
    text = text[:50]
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^\w\s.]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text

def set_current_pdf(pdf_name):
    """Set the current PDF to display"""
    st.session_state.current_pdf = pdf_name
    st.session_state.current_page = 1
    st.session_state.search_text = None

def main():
    # Check for pre-loaded data
    pdf_exists, json_exists = check_preloaded_data()
    
    # Main layout with three columns
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane for PDF upload and control options
    with col1:
        st.header("Research Papers")
        
        # Add a demo mode option
        use_demo_data = st.checkbox("Use pre-loaded SHIMI paper", 
                                    value=pdf_exists and json_exists,
                                    key="use_demo_data")
        
        if use_demo_data and pdf_exists and json_exists:
            # Load pre-existing data
            if "shimi_paper.pdf" not in st.session_state.pdf_files:
                with open("shimi_paper.pdf", 'rb') as f:
                    st.session_state.pdf_files["shimi_paper.pdf"] = f.read()
                    
            if "shimi_paper.json" not in st.session_state.json_data:
                with open("shimi_paper.json", 'r') as f:
                    st.session_state.json_data["shimi_paper"] = json.load(f)
            
            # Set as current PDF if none is selected
            if st.session_state.current_pdf is None:
                st.session_state.current_pdf = "shimi_paper.pdf"
                
            st.success("Pre-loaded SHIMI paper data is being used")
            
        # File uploader for multiple PDFs
        st.subheader("Upload PDFs")
        uploaded_pdfs = st.file_uploader(
            "Upload Research PDFs",
            type="pdf",
            key="pdf_uploader",
            accept_multiple_files=True
        )
        
        # Process uploaded PDFs
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf.name not in st.session_state.pdf_files:
                    st.session_state.pdf_files[pdf.name] = pdf.getvalue()
                    if st.session_state.current_pdf is None:
                        st.session_state.current_pdf = pdf.name
        
        # File uploader for JSON
        st.subheader("Upload JSON")
        uploaded_jsons = st.file_uploader(
            "Upload JSON Extracts",
            type="json",
            key="json_uploader",
            accept_multiple_files=True
        )
        
        # Process uploaded JSON
        if uploaded_jsons:
            for json_file in uploaded_jsons:
                file_stem = Path(json_file.name).stem
                st.session_state.json_data[file_stem] = json.load(json_file)
        
        # Display buttons for each uploaded PDF
        if st.session_state.pdf_files:
            st.subheader("Available PDFs")
            for pdf_name in st.session_state.pdf_files.keys():
                button_style = "selected" if pdf_name == st.session_state.current_pdf else ""
                if st.button(pdf_name, key=f"pdf_btn_{pdf_name}", 
                           help=f"Click to view {pdf_name}",
                           use_container_width=True, 
                           type="primary" if pdf_name == st.session_state.current_pdf else "secondary"):
                    set_current_pdf(pdf_name)
                    st.rerun()
        
        # Add page navigation
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
            if st.button("Navigate to Page", key="nav_button"):
                st.session_state.current_page = page_num
                st.session_state.search_text = None
                st.rerun()
    
    # Middle pane for PDF viewer
    with col2:
        st.header("PDF Viewer")
        
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            st.subheader(f"Viewing: {st.session_state.current_pdf}")
            
            try:
                pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying PDF with iframe: {e}")
                try:
                    pdf_display = display_pdf_object(current_pdf_bytes)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying PDF with object tag: {e}")
                    st.download_button(
                        label="Download PDF to view",
                        data=current_pdf_bytes,
                        file_name=st.session_state.current_pdf,
                        mime="application/pdf",
                        key="download_pdf"
                    )
        else:
            st.info("Select a PDF from the left panel or upload a new PDF.")
    
    # Right pane for JSON data display
    with col3:
        st.header("Contract Analysis")
        
        file_stem = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else None
        if file_stem and file_stem in st.session_state.json_data:
            json_data = st.session_state.json_data[file_stem]
            
            # Display summary
            st.subheader("Summary")
            st.markdown(f"<div class='extract-text'>{json_data.get('summary', 'No summary available')}</div>", 
                       unsafe_allow_html=True)
            
            # Display binary status indicators
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
            
            # Display relevant clauses
            st.subheader("Relevant Clauses")
            clauses = json_data.get("relevant_clauses", [])
            
            for i, clause in enumerate(clauses):
                with st.expander(f"Clause {i+1}: {clause['type'].capitalize()}", key=f"clause_{i}"):
                    st.write(f"**Type:** {clause['type']}")
                    st.write(f"**Text:** {clause['text']}")
                    if len(clause['text']) > 50:
                        st.warning("Long text may not highlight fully in PDF viewer.")
                    if st.button("Highlight in PDF", key=f"highlight_{i}"):
                        st.session_state.search_text = clause['text']
                        st.success(f"Searching for clause {i+1} in PDF...")
                        st.rerun()
            
        else:
            st.info("Select a PDF and ensure corresponding JSON data is uploaded.")

if __name__ == "__main__":
    main()
