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

# JavaScript for PDF navigation
# This will create PDF.js viewer with navigation capabilities
def get_pdf_display_with_navigation(pdf_base64, search_term=None, page_num=None):
    """Create PDF display with navigation capability"""
    
    js_code = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.min.js"></script>
    <script>
    // PDFjs viewer with search and navigation
    const viewerElement = document.getElementById('pdf-viewer');
    const pdfData = atob('PDF_BASE64_PLACEHOLDER');
    let currentPage = PAGE_NUM_PLACEHOLDER;
    let searchText = 'SEARCH_TEXT_PLACEHOLDER';
    
    // Initialize PDF.js
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.worker.min.js';
    
    const loadingTask = pdfjsLib.getDocument({data: pdfData});
    loadingTask.promise.then(pdf => {
        console.log('PDF loaded');
        
        // If search term is provided, search for it
        if (searchText !== 'null') {
            searchInPDF(pdf, searchText);
        } 
        // If page number is provided, go to that page
        else if (currentPage !== null) {
            renderPage(pdf, currentPage);
        }
        // Otherwise render first page
        else {
            renderPage(pdf, 1);
        }
    });
    
    function renderPage(pdf, pageNumber) {
        pdf.getPage(pageNumber).then(page => {
            const viewport = page.getViewport({scale: 1.5});
            
            // Prepare canvas
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            canvas.style.width = '100%';
            canvas.style.height = 'auto';
            
            // Clear viewer and add new canvas
            viewerElement.innerHTML = '';
            viewerElement.appendChild(canvas);
            
            // Render PDF page
            page.render({
                canvasContext: context,
                viewport: viewport
            });
            
            // Add page navigation controls
            addPageControls(pdf, pageNumber);
        });
    }
    
    function addPageControls(pdf, currentPage) {
        const controls = document.createElement('div');
        controls.style.padding = '10px';
        controls.style.backgroundColor = '#f8f9fa';
        controls.style.borderRadius = '5px';
        controls.style.margin = '10px 0';
        controls.style.display = 'flex';
        controls.style.justifyContent = 'space-between';
        
        // Previous page button
        const prevBtn = document.createElement('button');
        prevBtn.textContent = 'Previous Page';
        prevBtn.disabled = currentPage <= 1;
        prevBtn.onclick = () => renderPage(pdf, currentPage - 1);
        
        // Page indicator
        const pageIndicator = document.createElement('span');
        pageIndicator.textContent = `Page ${currentPage} of ${pdf.numPages}`;
        
        // Next page button
        const nextBtn = document.createElement('button');
        nextBtn.textContent = 'Next Page';
        nextBtn.disabled = currentPage >= pdf.numPages;
        nextBtn.onclick = () => renderPage(pdf, currentPage + 1);
        
        controls.appendChild(prevBtn);
        controls.appendChild(pageIndicator);
        controls.appendChild(nextBtn);
        
        viewerElement.appendChild(controls);
    }
    
    function searchInPDF(pdf, searchText) {
        console.log('Searching for:', searchText);
        
        // Regular expression to clean up search text (remove section numbers, etc.)
        const cleanSearchText = searchText.replace(/^\d+\s+/, '').substring(0, 30);
        let found = false;
        
        // Search through each page
        const promises = [];
        for (let i = 1; i <= pdf.numPages; i++) {
            promises.push(
                pdf.getPage(i).then(page => {
                    return page.getTextContent().then(textContent => {
                        const textItems = textContent.items.map(item => item.str).join(' ');
                        if (textItems.includes(cleanSearchText)) {
                            console.log(`Found "${cleanSearchText}" on page ${i}`);
                            if (!found) {
                                found = true;
                                renderPage(pdf, i);
                                // Highlight for user that content was found
                                const notification = document.createElement('div');
                                notification.textContent = `Found match on page ${i}`;
                                notification.style.backgroundColor = '#4CAF50';
                                notification.style.color = 'white';
                                notification.style.padding = '10px';
                                notification.style.borderRadius = '5px';
                                notification.style.margin = '10px 0';
                                notification.style.textAlign = 'center';
                                viewerElement.prepend(notification);
                                
                                // Remove notification after 3 seconds
                                setTimeout(() => {
                                    notification.style.display = 'none';
                                }, 3000);
                            }
                            return true;
                        }
                        return false;
                    });
                })
            );
        }
        
        Promise.all(promises).then(results => {
            if (!results.includes(true)) {
                console.log('Search text not found');
                renderPage(pdf, 1);
                // Show not found message
                const notification = document.createElement('div');
                notification.textContent = `Couldn't find exact match. Showing first page.`;
                notification.style.backgroundColor = '#f44336';
                notification.style.color = 'white';
                notification.style.padding = '10px';
                notification.style.borderRadius = '5px';
                notification.style.margin = '10px 0';
                notification.style.textAlign = 'center';
                viewerElement.prepend(notification);
                
                // Remove notification after 3 seconds
                setTimeout(() => {
                    notification.style.display = 'none';
                }, 3000);
            }
        });
    }
    </script>
    """
    
    # Replace placeholders with actual values
    js_code = js_code.replace('PDF_BASE64_PLACEHOLDER', pdf_base64)
    js_code = js_code.replace('SEARCH_TEXT_PLACEHOLDER', str(search_term))
    js_code = js_code.replace('PAGE_NUM_PLACEHOLDER', str(page_num if page_num else 'null'))
    
    # Create the HTML structure
    html = f"""
    <div class="pdf-viewer-container" style="height: 85vh; overflow: auto;">
        <div id="pdf-viewer" style="width: 100%;"></div>
        {js_code}
    </div>
    """
    
    return html

# Initialize session state variables
if 'selected_extract_index' not in st.session_state:
    st.session_state.selected_extract_index = 0

if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None

if 'json_data' not in st.session_state:
    st.session_state.json_data = None

if 'current_search_term' not in st.session_state:
    st.session_state.current_search_term = None

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
                                    value=pdf_exists and json_exists)
        
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
                
        # Add custom PDF navigation
        if st.session_state.pdf_bytes is not None:
            st.subheader("Manual Navigation")
            
            # Allow manual page navigation
            if 'num_pages' in st.session_state.json_data:
                num_pages = st.session_state.json_data['num_pages']
                page_num = st.number_input("Go to page:", min_value=1, max_value=num_pages, step=1)
                if st.button("Navigate to Page"):
                    st.session_state.current_search_term = None
                    st.session_state.current_page = page_num
                    st.rerun()  # Use st.rerun() instead of experimental_rerun()
            
            # Custom text search
            search_text = st.text_input("Search in PDF:")
            if st.button("Search") and search_text:
                st.session_state.current_search_term = search_text
                st.session_state.current_page = None
                st.rerun()  # Use st.rerun() instead of experimental_rerun()
    
    # Middle pane for PDF viewer
    with col2:
        st.header("PDF Viewer")
        
        if st.session_state.pdf_bytes is not None:
            # Convert PDF bytes to base64 for embedding
            pdf_base64 = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
            
            # Determine if we're showing a specific page or searching for text
            current_page = getattr(st.session_state, 'current_page', None)
            search_term = getattr(st.session_state, 'current_search_term', None)
            
            # Create the PDF display with navigation capabilities
            pdf_display = get_pdf_display_with_navigation(
                pdf_base64, 
                search_term=search_term,
                page_num=current_page
            )
            
            # Render the PDF viewer
            st.markdown(pdf_display, unsafe_allow_html=True)
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
            
            # Make abstract clickable to navigate to it in the PDF
            if st.button("View Abstract in PDF"):
                # Extract first few words of abstract to search
                abstract_search = sanitize_search_text(json_data['abstract'][:50])
                st.session_state.current_search_term = abstract_search
                st.session_state.current_page = None
                st.rerun()  # Use st.rerun() instead of experimental_rerun()
                
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
            for q in json_data.get("binary_questions", []):
                expander = st.expander(q["question"])
                with expander:
                    st.write(f"**Answer:** {q['answer']}")
                    st.write(f"**Explanation:** {q['explanation']}")
            
            # Create buttons for extracts that navigate to the content
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
                            
                            # Set the search term to navigate to this section
                            section_text = sanitize_search_text(extracts[idx]["section"])
                            st.session_state.current_search_term = section_text
                            st.session_state.current_page = None
                            st.rerun()  # Use st.rerun() instead of experimental_rerun()
            
            # Display selected extract
            if extracts and 0 <= st.session_state.selected_extract_index < len(extracts):
                selected_extract = extracts[st.session_state.selected_extract_index]
                
                # Add button to view this extract in PDF
                if st.button("View this extract in PDF", key="view_extract_in_pdf"):
                    extract_text = sanitize_search_text(selected_extract["text"][:50])
                    st.session_state.current_search_term = extract_text
                    st.session_state.current_page = None
                    st.rerun()  # Use st.rerun() instead of experimental_rerun()
                
                st.markdown(f"<div class='extract-text'>{selected_extract['text']}</div>", unsafe_allow_html=True)
                
                with st.expander("View full section content"):
                    st.text_area("", selected_extract["full_content"], height=300)
            
        else:
            st.info("Upload the JSON extract to view the paper analysis.")

if __name__ == "__main__":
    main()
