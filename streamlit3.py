import os
import json
import time
import base64
import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, List, Any
import tempfile
import shutil
from pdfminer.high_level import extract_text
import concurrent.futures
import threading
import streamlit.components.v1 as components
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set page configuration
st.set_page_config(
    page_title="PDF Analyzer",
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
    .progress-container {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def process_pdf(pdf_file, output_folder: str) -> str:
    """Process a PDF file and save extracted text to a file"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text from PDF
        extracted_text = extract_text(tmp_path)
        
        # Create output filename
        pdf_name = Path(pdf_file.name).stem
        output_text_path = os.path.join(output_folder, f"{pdf_name}.txt")
        
        # Save extracted text
        with open(output_text_path, 'w', encoding="utf-8") as f:
            f.write(extracted_text)
            
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return output_text_path
    
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_file.name}: {e}")
        return None

# Function to process PDF and generate JSON
def generate_json(pdf_file, text_path: str, output_folder: str, progress_callback=None) -> str:
    """Generate JSON data from the extracted text"""
    try:
        # Simulate processing time with progress updates
        total_steps = 5
        pdf_name = Path(pdf_file.name).stem
        
        # Step 1: Initial processing
        if progress_callback:
            progress_callback(0.2)
        time.sleep(0.5)  # Simulate processing
        
        # Step 2: Text analysis
        if progress_callback:
            progress_callback(0.4)
        time.sleep(0.5)  # Simulate processing
        
        # Step 3: Information extraction
        if progress_callback:
            progress_callback(0.6)
        time.sleep(0.5)  # Simulate processing
        
        # Step 4: Finalize data
        if progress_callback:
            progress_callback(0.8)
        time.sleep(0.5)  # Simulate processing
        
        # Create sample JSON data (replace with actual extraction logic)
        json_data = {
            "filename": pdf_file.name,
            "pages": 5,  # Replace with actual page count
            "metadata": {
                "title": f"Document: {pdf_name}",
                "author": "Unknown",
                "date": "2023-01-01"
            },
            "extracted_info": [
                {"text": f"This is the first extracted section from {pdf_name}. It contains important information about the document."},
                {"text": f"Second section discusses key findings in the {pdf_name} document with relevant analysis."},
                {"text": f"Third section provides conclusions from the analysis of {pdf_name} with recommendations."}
            ],
            "key_metrics": {
                "total_words": 1500,
                "important_terms": ["term1", "term2", "term3"],
                "sentiment": "positive",
                "classification": "report"
            }
        }
        
        # Save JSON to file
        json_path = os.path.join(output_folder, f"{pdf_name}.json")
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
        
        # Step 5: Complete
        if progress_callback:
            progress_callback(1.0)
        
        return json_path
    
    except Exception as e:
        logging.error(f"Error generating JSON for {pdf_file.name}: {e}")
        if progress_callback:
            progress_callback(1.0, error=True)
        return None

# Function to display PDF
def display_pdf(pdf_path: str):
    """Display PDF file in Streamlit"""
    try:
        # Open and read the PDF file
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Embed PDF viewer
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="100%" 
            style="height: 80vh;" 
            type="application/pdf">
        </iframe>
        """
        return pdf_display
    except Exception as e:
        logging.error(f"Error displaying PDF: {e}")
        return f"<p>Error displaying PDF: {e}</p>"

# Initialize session state variables
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = {}  # {pdf_name: {"progress": 0.0, "json_path": None}}

if 'selected_pdf' not in st.session_state:
    st.session_state.selected_pdf = None

if 'selected_dict_index' not in st.session_state:
    st.session_state.selected_dict_index = 0

if 'temp_dir' not in st.session_state:
    # Create temporary directories for processing
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.text_dir = os.path.join(st.session_state.temp_dir, "text")
    st.session_state.json_dir = os.path.join(st.session_state.temp_dir, "json")
    st.session_state.pdf_dir = os.path.join(st.session_state.temp_dir, "pdfs")
    
    # Create directories
    os.makedirs(st.session_state.text_dir, exist_ok=True)
    os.makedirs(st.session_state.json_dir, exist_ok=True)
    os.makedirs(st.session_state.pdf_dir, exist_ok=True)

# Function to update PDF processing progress
def update_progress(pdf_name, progress, error=False):
    if pdf_name in st.session_state.processed_pdfs:
        st.session_state.processed_pdfs[pdf_name]["progress"] = progress
        if error:
            st.session_state.processed_pdfs[pdf_name]["error"] = True

# Function to process uploaded PDF files
def process_uploaded_pdf(uploaded_file):
    pdf_name = uploaded_file.name
    
    # Initialize progress tracking
    if pdf_name not in st.session_state.processed_pdfs:
        st.session_state.processed_pdfs[pdf_name] = {
            "progress": 0.0,
            "json_path": None,
            "pdf_path": None,
            "error": False
        }
    
    # Save the PDF to the temporary directory
    pdf_path = os.path.join(st.session_state.pdf_dir, pdf_name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.session_state.processed_pdfs[pdf_name]["pdf_path"] = pdf_path
    
    # Extract text from PDF
    text_path = process_pdf(uploaded_file, st.session_state.text_dir)
    if text_path:
        # Generate JSON
        progress_callback = lambda p, error=False: update_progress(pdf_name, p, error)
        json_path = generate_json(uploaded_file, text_path, st.session_state.json_dir, progress_callback)
        
        if json_path:
            st.session_state.processed_pdfs[pdf_name]["json_path"] = json_path
            return True
    
    st.session_state.processed_pdfs[pdf_name]["error"] = True
    return False

# Main layout with three columns
col1, col2, col3 = st.columns([25, 40, 35])

# Left pane for PDF upload and selection
with col1:
    st.header("PDF Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs (max 20)",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files:
        # Limit to 20 PDFs
        if len(uploaded_files) > 20:
            st.warning("Maximum 20 PDFs allowed. Only the first 20 will be processed.")
            uploaded_files = uploaded_files[:20]
        
        # Process new PDFs
        for pdf_file in uploaded_files:
            if pdf_file.name not in st.session_state.processed_pdfs:
                st.text(f"Processing: {pdf_file.name}")
                # Start processing in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_uploaded_pdf, pdf_file)
    
    # Display processing progress and PDF selection
    st.subheader("Document List")
    for pdf_name, pdf_data in st.session_state.processed_pdfs.items():
        # Progress bar for processing
        col_prog, col_btn = st.columns([7, 3])
        with col_prog:
            progress = pdf_data["progress"]
            progress_color = "#ff4b4b" if pdf_data.get("error", False) else "#00cc96"
            st.progress(progress, text=f"{int(progress*100)}%")
        
        # Button to select PDF
        with col_btn:
            if st.button(f"View", key=f"btn_{pdf_name}"):
                st.session_state.selected_pdf = pdf_name
                st.session_state.selected_dict_index = 0
                st.rerun()

# Middle pane for PDF viewer
with col2:
    st.header("PDF Viewer")
    
    # Display selected PDF
    if st.session_state.selected_pdf and st.session_state.selected_pdf in st.session_state.processed_pdfs:
        selected_data = st.session_state.processed_pdfs[st.session_state.selected_pdf]
        pdf_path = selected_data["pdf_path"]
        
        if pdf_path and os.path.exists(pdf_path):
            pdf_display = display_pdf(pdf_path)
            st.markdown(f"<div class='pdf-viewer'>{pdf_display}</div>", unsafe_allow_html=True)
        else:
            st.error("PDF file not available.")
    else:
        st.info("Select a PDF document from the left panel to view.")

# Right pane for JSON details
with col3:
    st.header("Document Analysis")
    
    if st.session_state.selected_pdf and st.session_state.selected_pdf in st.session_state.processed_pdfs:
        selected_data = st.session_state.processed_pdfs[st.session_state.selected_pdf]
        json_path = selected_data["json_path"]
        
        if json_path and os.path.exists(json_path):
            # Load JSON data
            with open(json_path, 'r', encoding="utf-8") as f:
                json_data = json.load(f)
            
            # Display key value at the top
            st.subheader("Document Information")
            st.write(f"**Title:** {json_data['metadata']['title']}")
            
            # Create buttons for dictionaries in extracted_info list
            st.write("**Extracted Information:**")
            
            # Create button row for extracted info sections
            extracted_info = json_data.get("extracted_info", [])
            
            st.markdown("<div class='button-row'>", unsafe_allow_html=True)
            for i, item in enumerate(extracted_info):
                # Get first 10 characters of text for button label
                label = item.get("text", "")[:10] + "..."
                
                # Create button HTML with active state if selected
                active_class = "active-button" if i == st.session_state.selected_dict_index else ""
                button_html = f"""
                <button class="dict-button {active_class}" 
                        onclick="document.getElementById('hidden_button_{i}').click()">
                    {label}
                </button>
                """
                st.markdown(button_html, unsafe_allow_html=True)
                
                # Hidden button to handle the click event
                if st.button(f"Select {i}", key=f"hidden_button_{i}", help="Hidden button"):
                    st.session_state.selected_dict_index = i
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display selected text
            if extracted_info and 0 <= st.session_state.selected_dict_index < len(extracted_info):
                selected_text = extracted_info[st.session_state.selected_dict_index].get("text", "")
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{selected_text}</div>", unsafe_allow_html=True)
            
            # Display table with key metrics
            st.subheader("Key Metrics")
            key_metrics = json_data.get("key_metrics", {})
            
            # Create a pandas DataFrame for display
            metrics_data = []
            for key, value in key_metrics.items():
                if isinstance(value, list):
                    value = ", ".join(value)
                metrics_data.append({"Metric": key.replace("_", " ").title(), "Value": value})
            
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
            
        else:
            if selected_data.get("error", False):
                st.error("Error processing the PDF. Please try again.")
            else:
                st.info("Document analysis in progress. Please wait.")
    else:
        st.info("Select a PDF document from the left panel to view analysis.")

# Cleanup on session end
def cleanup():
    """Clean up temporary files when app is closed"""
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

# Register cleanup function
# Note: This only works when app is properly closed
try:
    atexit.register(cleanup)
except:
    pass
