import os
import json
import base64
import streamlit as st
from pathlib import Path
import pandas as pd
import re

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
</style>
""", unsafe_allow_html=True)

# Function to display PDF directly using iframe
def display_pdf_iframe(pdf_bytes):
    """Display PDF using a simple iframe"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
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
    st.session_state.json_data = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None  # Currently selected PDF filename

# Check if pre-loaded data exists
def check_preloaded_data():
    pdf_exists = os.path.exists("shimi_paper.pdf")
    json_exists = os.path.exists("shimi_paper.json")
    return pdf_exists, json_exists

def sanitize_search_text(text):
    """Clean up text for searching in PDF"""
    # Remove long numbers, multiple spaces, special characters
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
    text = re.sub(r'\s+', ' ', text)     # Normalize spaces
    text = text.strip()
    return text

def set_current_pdf(pdf_name):
    """Set the current PDF to display"""
    st.session_state.current_pdf = pdf_name
    # Reset page navigation for the new PDF
    st.session_state.current_page = 1

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
                    
            if st.session_state.json_data is None:
                with open("shimi_paper.json", 'r') as f:
                    st.session_state.json_data = json.load(f)
            
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
            accept_multiple_files=True  # Allow multiple file uploads
        )
        
        # Process uploaded PDFs
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                # Add to the dictionary if not already there
                if pdf.name not in st.session_state.pdf_files:
                    st.session_state.pdf_files[pdf.name] = pdf.getvalue()
                    
                    # Set as current PDF if none is selected
                    if st.session_state.current_pdf is None:
                        st.session_state.current_pdf = pdf.name
        
        # File uploader for JSON
        uploaded_json = st.file_uploader(
            "Upload JSON Extract",
            type="json",
            key="json_uploader"
        )
        
        # Process uploaded JSON
        if uploaded_json:
            st.session_state.json_data = json.load(uploaded_json)
        
        # Display buttons for each uploaded PDF
        if st.session_state.pdf_files:
            st.subheader("Available PDFs")
            
            # Create buttons for each PDF
            for pdf_name in st.session_state.pdf_files.keys():
                button_style = "selected" if pdf_name == st.session_state.current_pdf else ""
                if st.button(pdf_name, key=f"pdf_btn_{pdf_name}", 
                           help=f"Click to view {pdf_name}",
                           use_container_width=True, 
                           type="primary" if pdf_name == st.session_state.current_pdf else "secondary"):
                    set_current_pdf(pdf_name)
                    st.rerun()
        
        # Add page navigation for the current PDF
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            st.subheader("Page Navigation")
            
            # For simplicity, assume 10 pages if not specified in JSON
            num_pages = 10
            if st.session_state.json_data and 'num_pages' in st.session_state.json_data:
                num_pages = st.session_state.json_data.get('num_pages', 10)
            
            # Page navigation
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
                st.rerun()
    
    # Middle pane for PDF viewer
    with col2:
        st.header("PDF Viewer")
        
        # Check if a PDF is selected to display
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            
            # Display the PDF name
            st.subheader(f"Viewing: {st.session_state.current_pdf}")
            
            # First try the iframe method
            try:
                pdf_display = display_pdf_iframe(current_pdf_bytes)
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying PDF with iframe: {e}")
                # Fall back to object tag method
                try:
                    pdf_display = display_pdf_object(current_pdf_bytes)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying PDF with object tag: {e}")
                    # Last resort: provide download link
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
        st.header("Paper Analysis")
        
        if st.session_state.json_data is not None:
            json_data = st.session_state.json_data
            
            # Display metadata
            st.subheader("Document Information")
            st.markdown(f"<div class='key-info'><strong>Title:</strong> {json_data['metadata']['title']}<br>"
                       f"<strong>Author:</strong> {json_data['metadata']['authors']}<br>"
                       f"<strong>Affiliation:</strong> {json_data['metadata']['affiliation']}<br>"
                       f"<strong>Email:</strong> {json_data['metadata']['email']}</div>", 
                       unsafe_allow_html=True)
            
            # Display abstract
            st.subheader("Abstract")
            st.markdown(f"<div class='extract-text'>{json_data['abstract']}</div>", unsafe_allow_html=True)
            
            # Display key terms if available
            if json_data.get("key_terms"):
                st.subheader("Key Terms")
                st.write(", ".join(json_data["key_terms"]))
            
            # Display key findings
            st.subheader("Key Findings")
            for i, finding in enumerate(json_data.get("key_findings", [])):
                st.markdown(f"• {finding}")
            
            # Display binary questions
            st.subheader("Research Questions")
            for i, q in enumerate(json_data.get("binary_questions", [])):
                expander = st.expander(q["question"], key=f"question_{i}")
                with expander:
                    st.write(f"**Answer:** {q['answer']}")
                    st.write(f"**Explanation:** {q['explanation']}")
            
            # Create buttons for extracts
            st.subheader("Section Extracts")
            extracts = json_data.get("extracts", [])
            
            # Create button row for extracts in chunks of 3
            for i in range(0, len(extracts), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(extracts):
                        # Use only the section number/name as the button label
                        label = extracts[idx]["section"]
                        if cols[j].button(label, key=f"extract_btn_{idx}"):
                            st.session_state.selected_extract_index = idx
            
            # Display selected extract
            if extracts and 0 <= st.session_state.selected_extract_index < len(extracts):
                selected_extract = extracts[st.session_state.selected_extract_index]
                st.markdown(f"<div class='extract-text'>{selected_extract['text']}</div>", unsafe_allow_html=True)
                
                with st.expander("View full section content", key="full_content"):
                    st.text_area(
                        label="Section Content",
                        value=selected_extract["full_content"], 
                        height=300,
                        key=f"extract_content_{st.session_state.selected_extract_index}"
                    )
            
        else:
            st.info("Upload the JSON extract to view the paper analysis.")

if __name__ == "__main__":
    main()
