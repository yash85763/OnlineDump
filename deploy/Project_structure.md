The issue is likely in the document list section where the grid code has problems. Here are the specific fixes needed:

## **1. Fix the Grid Variables Issue:**

**In the document list section, replace the broken grid code with:**

```python
# Display PDF table - FIXED VERSION
if st.session_state.pdf_files:
    st.subheader("📋 Document Management")
    
    # Batch processing section
    st.markdown("### 🚀 Batch Processing")
    
    # Multi-select for batch processing
    available_pdfs = list(st.session_state.pdf_files.keys())
    unprocessed_pdfs = [pdf for pdf in available_pdfs 
                       if st.session_state.analysis_status.get(pdf, "Ready") != "Processed"]
    
    if unprocessed_pdfs:
        selected_for_batch = st.multiselect(
            "Select documents for batch processing (max 20):",
            options=unprocessed_pdfs,
            default=st.session_state.get('batch_selected_pdfs', []),
            max_selections=20,
            help="Select up to 20 documents to process simultaneously"
        )
        
        st.session_state.batch_selected_pdfs = selected_for_batch
        
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            batch_button_disabled = (len(selected_for_batch) == 0 or 
                                   st.session_state.get('batch_job_active', False) or
                                   not st.session_state.get('database_initialized', False))
            
            if st.button("🚀 Start Batch Processing", 
                        disabled=batch_button_disabled,
                        help="Process all selected documents"):
                if len(selected_for_batch) > 0:
                    st.session_state.batch_job_active = True
                    st.rerun()
        
        with col_batch2:
            if st.session_state.get('batch_job_active', False):
                if st.button("⏹️ Cancel Batch"):
                    st.session_state.batch_job_active = False
                    st.rerun()
        
        if len(selected_for_batch) > 0:
            st.info(f"📊 Selected: {len(selected_for_batch)} documents")
    else:
        st.info("All documents are already processed!")
    
    st.markdown("---")
    
    # Individual document selection - SIMPLIFIED VERSION
    st.subheader("📄 Individual Selection")
    
    # Create simple dataframe
    pdf_data = []
    for pdf_name in st.session_state.pdf_files.keys():
        status = st.session_state.analysis_status.get(pdf_name, "Ready")
        db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
        file_size = len(st.session_state.pdf_files[pdf_name]) / 1024  # KB
        
        # Status emoji
        if status == "Processed":
            status_emoji = "✅"
        elif "processing" in status.lower():
            status_emoji = "⏳"
        elif "failed" in status.lower() or "error" in status.lower():
            status_emoji = "❌"
        else:
            status_emoji = "📄"
        
        pdf_data.append({
            'Status': status_emoji,
            'PDF Name': pdf_name,
            'Size (KB)': f"{file_size:.1f}",
            'DB ID': str(db_id),
            'Analysis Status': status
        })
    
    pdf_df = pd.DataFrame(pdf_data)
    
    # Use simple selectbox instead of AgGrid if there are issues
    if len(pdf_df) > 0:
        try:
            # Try AgGrid first
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
                theme='streamlit',
                key='pdf_grid_main'
            )

            # Handle PDF selection
            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    st.rerun()
        
        except Exception as e:
            # Fallback to simple selectbox
            st.warning(f"Grid display error: {e}")
            st.write("**Available Documents:**")
            
            selected_pdf = st.selectbox(
                "Choose a document:",
                options=[""] + list(st.session_state.pdf_files.keys()),
                key="pdf_selector_fallback"
            )
            
            if selected_pdf and selected_pdf != st.session_state.get('current_pdf'):
                set_current_pdf(selected_pdf)
                st.rerun()
            
            # Show table for reference
            st.dataframe(pdf_df, use_container_width=True)
```

## **2. Add Missing Session State Variables:**

**Add these to your session_vars in `initialize_session_state()`:**

```python
session_vars = {
    'pdf_files': {},
    'json_data': {},
    'current_pdf': None,
    'analysis_status': {},
    'processing_messages': {},
    'pdf_database_ids': {},
    'search_text': None,
    'feedback_submitted': {},
    'obfuscation_summaries': {},
    'session_id': get_session_id(),
    'current_page_number': 1,
    'clause_page_mapping': {},
    'pages_content': {},
    'batch_job_active': False,  # Add this
    'batch_selected_pdfs': [], # Add this
    'batch_results': {}        # Add this
}
```

## **3. Add Error Handling for Missing Functions:**

**Add these missing functions if they don't exist:**

```python
# Add these functions if they're missing from your imports
def create_batch_job(selected_pdfs, session_id):
    """Placeholder - replace with actual implementation"""
    return "temp_job_id", 1

def run_batch_processing(selected_pdfs, job_id, progress_container):
    """Placeholder - replace with actual implementation"""
    return {
        'processed': [],
        'failed': [],
        'skipped': []
    }

def find_clause_in_pages(clause_text, pages_content):
    """Placeholder - replace with actual implementation"""
    return 1  # Return page 1 as default
```

## **4. Debug the Upload Section:**

**Add debugging after the upload section:**

```python
# Debug section - add after file upload
if uploaded_pdfs:
    st.write("DEBUG: Files uploaded successfully")
    st.write(f"Session state pdf_files: {list(st.session_state.pdf_files.keys())}")
    st.write(f"Analysis status: {st.session_state.analysis_status}")

# Debug the condition
if st.session_state.pdf_files:
    st.write(f"DEBUG: Found {len(st.session_state.pdf_files)} PDF files")
else:
    st.write("DEBUG: No PDF files in session state")
    st.write(f"PDF files content: {st.session_state.pdf_files}")
```

## **5. Simplify the Document List (Quick Fix):**

**Replace the entire document list section with this simpler version:**

```python
# Display PDF files - SIMPLE VERSION
if st.session_state.pdf_files:
    st.subheader("📋 Available Documents")
    
    # Show list of documents
    for pdf_name in st.session_state.pdf_files.keys():
        status = st.session_state.analysis_status.get(pdf_name, "Ready")
        
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"📄 **{pdf_name}**")
        with col2:
            st.write(f"Status: {status}")
        with col3:
            if st.button("Select", key=f"select_{pdf_name}"):
                set_current_pdf(pdf_name)
                st.rerun()
    
    # Batch processing
    st.markdown("---")
    st.subheader("🚀 Batch Processing")
    
    unprocessed = [name for name, status in st.session_state.analysis_status.items() 
                  if status != "Processed"]
    
    if unprocessed:
        selected_batch = st.multiselect(
            "Select for batch processing:",
            options=unprocessed,
            max_selections=20
        )
        
        if st.button("Start Batch", disabled=len(selected_batch) == 0):
            st.success(f"Would process {len(selected_batch)} files")
            # Add actual batch processing here
    else:
        st.info("All documents processed!")
else:
    st.info("Upload PDF documents to get started")
```

## **6. Check Import Issues:**

**Make sure all imports are working:**

```python
# Test imports at the top
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    st.write("DEBUG: AgGrid imported successfully")
except ImportError as e:
    st.error(f"AgGrid import error: {e}")
    st.write("Using fallback display method")

try:
    from utils.enhanced_pdf_handler import process_single_pdf_from_streamlit
    st.write("DEBUG: PDF handler imported successfully")  
except ImportError as e:
    st.error(f"PDF handler import error: {e}")
```

The most likely issue is that an error is occurring in the document list section that's preventing the rest of the code from executing. Try the simplified version first, then gradually add back the more complex features once you confirm the basic functionality works.​​​​​​​​​​​​​​​​