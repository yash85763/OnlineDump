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

if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None

if 'json_data' not in st.session_state:
    st.session_state.json_data = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

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

def main():
    # Check for pre-loaded data
    pdf_exists, json_exists = check_preloaded_data()
    
    # Main layout with three columns
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left pane for PDF upload and control options
    with col1:
        st.header("Research Paper")
        
        # Add a demo mode option
        use_demo_data = st.checkbox("Use pre-loaded SHIMI paper", 
                                    value=pdf_exists and json_exists,
                                    key="use_demo_data")
        
        if use_demo_data and pdf_exists and json_exists:
            # Load pre-existing data
            if st.session_state.pdf_bytes is None:
                with open("shimi_paper.pdf", 'rb') as f:
                    st.session_state.pdf_bytes = f.read()
            
            if st.session_state.json_data is None:
                with open("shimi_paper.json", 'r') as f:
                    st.session_state.json_data = json.load(f)
                    
            st.success("Pre-loaded SHIMI paper data is being used")
            
        else:
            # File uploader for PDF
            uploaded_pdf = st.file_uploader(
                "Upload Research PDF",
                type="pdf",
                key="pdf_uploader"
            )
            
            # File uploader for JSON
            uploaded_json = st.file_uploader(
                "Upload JSON Extract",
                type="json",
                key="json_uploader"
            )
            
            # Display status
            if uploaded_pdf and uploaded_json:
                st.success("Both PDF and JSON are loaded!")
                # Store the uploaded data
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()
                st.session_state.json_data = json.load(uploaded_json)
            elif uploaded_pdf:
                st.warning("PDF is loaded. Please upload the JSON extract.")
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()
            elif uploaded_json:
                st.warning("JSON extract is loaded. Please upload the PDF file.")
                st.session_state.json_data = json.load(uploaded_json)
            else:
                st.info("Please upload both the PDF and its JSON extract.")
                
        # Add page navigation
        if st.session_state.pdf_bytes is not None and st.session_state.json_data is not None:
            st.subheader("Page Navigation")
            
            # Get number of pages
            num_pages = st.session_state.json_data.get('num_pages', 10)  # Default to 10 if not specified
            
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
        
        if st.session_state.pdf_bytes is not None:
            # First try the iframe method
            try:
                pdf_display = display_pdf_iframe(st.session_state.pdf_bytes)
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying PDF with iframe: {e}")
                # Fall back to object tag method
                try:
                    pdf_display = display_pdf_object(st.session_state.pdf_bytes)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying PDF with object tag: {e}")
                    # Last resort: provide download link
                    st.download_button(
                        label="Download PDF to view",
                        data=st.session_state.pdf_bytes,
                        file_name="document.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
        else:
            st.info("Upload a PDF document to view it here.")
    
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
