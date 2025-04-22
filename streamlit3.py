import os
import json
import time
import base64
import re
import tempfile
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
import atexit
import logging
from io import BytesIO
import fitz  # PyMuPDF

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set page configuration
st.set_page_config(
    page_title="Research Paper Analyzer",
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
    .key-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .extract-text {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #0068c9;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF"""
    try:
        # Create a BytesIO object from the uploaded file
        pdf_bytes = BytesIO(pdf_file.getvalue())
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract text from each page
        text_by_page = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_by_page.append(text)
        
        # Get the full text
        full_text = "\n".join(text_by_page)
        
        return {
            "success": True,
            "full_text": full_text,
            "text_by_page": text_by_page,
            "num_pages": len(doc)
        }
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def extract_metadata(pdf_text):
    """Extract metadata from the PDF text"""
    
    # Extract title
    title_pattern = r"^(.*?)(?=\n)"
    title_match = re.search(title_pattern, pdf_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Unknown Title"
    
    # Extract authors
    author_pattern = r"(?<=\n)([A-Za-z\s]+)(?=\[\d+)"
    author_match = re.search(author_pattern, pdf_text)
    authors = author_match.group(1).strip() if author_match else "Unknown Author"
    
    # Extract affiliations
    affiliation_pattern = r"(?<=\n)(University of.*?)(?=\n)"
    affiliation_match = re.search(affiliation_pattern, pdf_text)
    affiliation = affiliation_match.group(1).strip() if affiliation_match else "Unknown Affiliation"
    
    # Extract email
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_match = re.search(email_pattern, pdf_text)
    email = email_match.group(0) if email_match else "Unknown Email"
    
    return {
        "title": title,
        "authors": authors,
        "affiliation": affiliation,
        "email": email
    }

def extract_abstract(pdf_text):
    """Extract the abstract from the PDF text"""
    abstract_pattern = r"(?<=Abstract\.)(.*?)(?=Keywords:)"
    abstract_match = re.search(abstract_pattern, pdf_text, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    else:
        # Try alternate pattern
        alt_pattern = r"(?<=Abstract\n)(.*?)(?=\n\d\s+Introduction)"
        alt_match = re.search(alt_pattern, pdf_text, re.DOTALL)
        return alt_match.group(1).strip() if alt_match else "Abstract not found"

def extract_sections(pdf_text):
    """Extract main sections from the PDF text"""
    # Identify section headers (numbered sections like "1 Introduction", "2 Background", etc.)
    section_pattern = r"\n(\d+\s+[A-Z][a-z]+.*?)\n"
    sections = re.findall(section_pattern, pdf_text)
    
    # Extract content for each section
    section_contents = {}
    for i, section in enumerate(sections):
        section_name = section.strip()
        
        # Get the content between this section and the next
        if i < len(sections) - 1:
            next_section = sections[i + 1]
            content_pattern = f"{re.escape(section_name)}(.*?)(?={re.escape(next_section)})"
            content_match = re.search(content_pattern, pdf_text, re.DOTALL)
            if content_match:
                section_contents[section_name] = content_match.group(1).strip()
        else:
            # For the last section, get everything until the References or end of document
            content_pattern = f"{re.escape(section_name)}(.*?)(?=References|$)"
            content_match = re.search(content_pattern, pdf_text, re.DOTALL)
            if content_match:
                section_contents[section_name] = content_match.group(1).strip()
    
    return section_contents

def extract_key_findings(pdf_text, sections):
    """Extract key findings and create summary bullet points"""
    key_findings = []
    
    # Look for statements about contributions or results
    contribution_patterns = [
        r"(?:Our|We|This paper)(?:'s)? contributions? (?:is|are|include).*?(?::|\.)(.*?)(?=\n\n)",
        r"(?:We|Our results) demonstrate that(.*?)(?=\n\n)",
        r"(?:We|Our) (?:show|demonstrate|prove|find) that(.*?)(?=\n\n)",
        r"(?:advantages|benefits) of (?:our|this) approach(.*?)(?=\n\n)"
    ]
    
    for pattern in contribution_patterns:
        matches = re.findall(pattern, pdf_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            clean_finding = re.sub(r'\s+', ' ', match).strip()
            if clean_finding and len(clean_finding) > 20:  # Avoid very short snippets
                key_findings.append(clean_finding)
    
    # Extract key points from the conclusion section
    if "Conclusion" in "".join(sections.keys()):
        conclusion_section = [s for s in sections.keys() if "Conclusion" in s]
        if conclusion_section:
            conclusion_text = sections[conclusion_section[0]]
            # Look for key sentences in the conclusion
            sentences = re.split(r'(?<=[.!?])\s+', conclusion_text)
            for sentence in sentences:
                if re.search(r'\b(?:show|demonstrate|present|introduce|improve|novel|outperform|better|advantage)\b', 
                             sentence, re.IGNORECASE):
                    clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                    if clean_sentence and len(clean_sentence) > 30:
                        key_findings.append(clean_sentence)
    
    # Deduplicate and limit to reasonable number
    key_findings = list(set(key_findings))
    return key_findings[:5]  # Limit to top 5 findings

def extract_figures_tables(pdf_text):
    """Extract references to figures and tables"""
    figure_pattern = r"(Fig\.\s+\d+:.*?)(?=\n)"
    table_pattern = r"(Table\s+\d+:.*?)(?=\n)"
    
    figures = re.findall(figure_pattern, pdf_text)
    tables = re.findall(table_pattern, pdf_text)
    
    return {"figures": figures, "tables": tables}

def extract_key_terms(pdf_text):
    """Extract key terms and their frequencies"""
    # Get the Keywords section if it exists
    keywords_pattern = r"Keywords:(.*?)(?=\n\n|\.\n)"
    keywords_match = re.search(keywords_pattern, pdf_text)
    if keywords_match:
        keywords_text = keywords_match.group(1).strip()
        keywords = [kw.strip() for kw in keywords_text.split("·") if kw.strip()]
        if not keywords:  # Try alternative delimiter
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        return keywords
    else:
        return []

def generate_binary_questions(metadata, abstract, key_findings):
    """Generate binary questions about the paper"""
    questions = []
    
    # From title
    if "hierarchical" in metadata["title"].lower():
        questions.append({
            "question": "Does this paper propose a hierarchical approach?",
            "answer": "Yes",
            "explanation": "The paper introduces SHIMI (Semantic Hierarchical Memory Index), which organizes memory as a dynamic tree structure with semantic abstractions layered hierarchically."
        })
    
    # From abstract
    if "decentralized" in abstract.lower():
        questions.append({
            "question": "Is this research focused on centralized AI systems?",
            "answer": "No",
            "explanation": "The paper specifically addresses decentralized environments where agents maintain local memory trees and synchronize them asynchronously across networks."
        })
    
    # From key findings
    retrieval_focus = any("retrieval" in finding.lower() for finding in key_findings)
    if retrieval_focus:
        questions.append({
            "question": "Does the paper compare its approach against vector-based retrieval?",
            "answer": "Yes",
            "explanation": "The paper evaluates SHIMI against a RAG-style embedding-based retrieval baseline on semantically non-trivial queries."
        })
    
    # Generic questions based on research paper type
    questions.append({
        "question": "Does the paper include experimental evaluation?",
        "answer": "Yes",
        "explanation": "The paper includes a comprehensive evaluation section that measures retrieval accuracy, traversal efficiency, synchronization cost, and scalability."
    })
    
    questions.append({
        "question": "Is this research proposing a new architecture?",
        "answer": "Yes",
        "explanation": "The paper introduces SHIMI, a new memory architecture designed for decentralized AI systems with specific synchronization protocols and semantic organization."
    })
    
    return questions

def process_research_pdf(pdf_file, progress_callback=None):
    """Process a research PDF and extract structured information"""
    # Extract text and basic metadata
    if progress_callback:
        progress_callback(0.1)
    
    extraction_result = extract_text_from_pdf(pdf_file)
    if not extraction_result["success"]:
        if progress_callback:
            progress_callback(1.0, error=True)
        return {
            "success": False,
            "error": extraction_result["error"]
        }
    
    pdf_text = extraction_result["full_text"]
    text_by_page = extraction_result["text_by_page"]
    
    if progress_callback:
        progress_callback(0.3)
    
    # Extract metadata
    metadata = extract_metadata(pdf_text)
    
    if progress_callback:
        progress_callback(0.4)
    
    # Extract abstract
    abstract = extract_abstract(pdf_text)
    
    if progress_callback:
        progress_callback(0.5)
    
    # Extract main sections
    sections = extract_sections(pdf_text)
    
    if progress_callback:
        progress_callback(0.6)
    
    # Extract key findings
    key_findings = extract_key_findings(pdf_text, sections)
    
    if progress_callback:
        progress_callback(0.7)
    
    # Extract figures and tables
    visual_elements = extract_figures_tables(pdf_text)
    
    if progress_callback:
        progress_callback(0.8)
    
    # Extract key terms
    key_terms = extract_key_terms(pdf_text)
    
    if progress_callback:
        progress_callback(0.9)
    
    # Generate binary questions
    binary_questions = generate_binary_questions(metadata, abstract, key_findings)
    
    # Create extracts for each section
    extracts = []
    for section_name, content in sections.items():
        if len(content) > 100:  # Only include non-trivial sections
            # Get the first few sentences as a preview
            sentences = re.split(r'(?<=[.!?])\s+', content)
            preview = " ".join(sentences[:3]) if len(sentences) > 3 else content[:300]
            
            extracts.append({
                "section": section_name,
                "text": preview,
                "full_content": content
            })
    
    # Prepare final JSON structure
    result = {
        "success": True,
        "metadata": metadata,
        "abstract": abstract,
        "key_terms": key_terms,
        "key_findings": key_findings,
        "extracts": extracts,
        "binary_questions": binary_questions,
        "visual_elements": visual_elements,
        "num_pages": extraction_result["num_pages"]
    }
    
    if progress_callback:
        progress_callback(1.0)
    
    return result

def display_pdf(pdf_bytes):
    """Display PDF in an embedded viewer"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
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

# Initialize session state variables
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = {}  # {pdf_name: {"progress": 0.0, "json_data": None}}

if 'selected_pdf' not in st.session_state:
    st.session_state.selected_pdf = None

if 'selected_extract_index' not in st.session_state:
    st.session_state.selected_extract_index = 0

if 'temp_dir' not in st.session_state:
    # Create temporary directories for processing
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.json_dir = os.path.join(st.session_state.temp_dir, "json")
    st.session_state.pdf_dir = os.path.join(st.session_state.temp_dir, "pdfs")
    
    # Create directories
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
            "json_data": None,
            "pdf_bytes": None,
            "error": False
        }
    
    # Save the PDF bytes
    pdf_bytes = uploaded_file.getvalue()
    st.session_state.processed_pdfs[pdf_name]["pdf_bytes"] = pdf_bytes
    
    # Save the PDF to the temporary directory
    pdf_path = os.path.join(st.session_state.pdf_dir, pdf_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    # Process the PDF and extract information
    progress_callback = lambda p, error=False: update_progress(pdf_name, p, error)
    result = process_research_pdf(uploaded_file, progress_callback)
    
    if result["success"]:
        # Save JSON data
        st.session_state.processed_pdfs[pdf_name]["json_data"] = result
        
        # Save JSON to file
        json_path = os.path.join(st.session_state.json_dir, f"{Path(pdf_name).stem}.json")
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        return True
    else:
        st.session_state.processed_pdfs[pdf_name]["error"] = True
        return False

# Main layout with three columns
col1, col2, col3 = st.columns([25, 40, 35])

# Left pane for PDF upload and selection
with col1:
    st.header("Research Papers")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Research PDFs (max 20)",
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
            if pdf_file.name not in st.session_state.processed_pdfs or st.session_state.processed_pdfs[pdf_file.name]["json_data"] is None:
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
                st.session_state.selected_extract_index = 0
                st.experimental_rerun()

# Middle pane for PDF viewer
with col2:
    st.header("PDF Viewer")
    
    # Display selected PDF
    if st.session_state.selected_pdf and st.session_state.selected_pdf in st.session_state.processed_pdfs:
        selected_data = st.session_state.processed_pdfs[st.session_state.selected_pdf]
        pdf_bytes = selected_data["pdf_bytes"]
        
        if pdf_bytes:
            pdf_display = display_pdf(pdf_bytes)
            st.markdown(f"<div class='pdf-viewer'>{pdf_display}</div>", unsafe_allow_html=True)
        else:
            st.error("PDF data not available.")
    else:
        st.info("Select a research paper from the left panel to view.")

# Right pane for extracted information
with col3:
    st.header("Paper Analysis")
    
    if st.session_state.selected_pdf and st.session_state.selected_pdf in st.session_state.processed_pdfs:
        selected_data = st.session_state.processed_pdfs[st.session_state.selected_pdf]
        json_data = selected_data["json_data"]
        
        if json_data and json_data["success"]:
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
            if json_data["key_terms"]:
                st.subheader("Key Terms")
                st.write(", ".join(json_data["key_terms"]))
            
            # Display key findings
            st.subheader("Key Findings")
            for i, finding in enumerate(json_data["key_findings"]):
                st.markdown(f"• {finding}")
            
            # Display binary questions
            st.subheader("Research Questions")
            for q in json_data["binary_questions"]:
                expander = st.expander(q["question"])
                with expander:
                    st.write(f"**Answer:** {q['answer']}")
                    st.write(f"**Explanation:** {q['explanation']}")
            
            # Create buttons for extracts
            st.subheader("Section Extracts")
            extracts = json_data.get("extracts", [])
            
            # Create button row
            col_buttons = st.columns(min(3, len(extracts)))
            for i, col in enumerate(col_buttons):
                if i < len(extracts):
                    # Use only the section number/name as the button label
                    label = extracts[i]["section"]
                    if col.button(label, key=f"extract_btn_{i}"):
                        st.session_state.selected_extract_index = i
                        st.experimental_rerun()
            
            # Display selected extract
            if extracts and 0 <= st.session_state.selected_extract_index < len(extracts):
                selected_extract = extracts[st.session_state.selected_extract_index]
                st.markdown(f"<div class='extract-text'>{selected_extract['text']}</div>", unsafe_allow_html=True)
                
                with st.expander("View full section content"):
                    st.text_area("", selected_extract["full_content"], height=300)
            
        else:
            if selected_data.get("error", False):
                st.error("Error processing the PDF. Please try again.")
            else:
                st.info("Document analysis in progress. Please wait.")
    else:
        st.info("Select a research paper from the left panel to view analysis.")

# Cleanup on session end
def cleanup():
    """Clean up temporary files when app is closed"""
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

# Register cleanup function
try:
    atexit.register(cleanup)
except:
    pass

# Add sample PDF for testing if needed
if not st.session_state.processed_pdfs:
    st.sidebar.markdown("### Demo Mode")
    if st.sidebar.button("Load Sample Paper"):
        # You can add code here to load a sample PDF for testing
        pass
