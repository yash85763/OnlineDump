## 1. Add new functions to `database.py`

```python
def get_all_pdfs_metadata() -> List[Dict[str, Any]]:
    """Get metadata for all PDFs in database (without heavy data like pdf_bytes)"""
    
    sql = """
        SELECT 
            p.id,
            p.pdf_name,
            p.file_hash,
            p.upload_date,
            p.processed_date,
            p.layout,
            p.original_page_count,
            p.final_page_count,
            p.final_word_count,
            p.avg_words_per_page,
            p.obfuscation_applied,
            p.pages_removed_count,
            p.uploaded_by,
            -- Analysis status
            CASE 
                WHEN COUNT(a.id) > 0 THEN 'Analyzed'
                ELSE 'Parsed Only'
            END as analysis_status,
            MAX(a.version) as latest_analysis_version,
            COUNT(a.id) as analysis_count,
            -- Size calculation (for display)
            CASE 
                WHEN p.pdf_bytes IS NOT NULL THEN LENGTH(p.pdf_bytes)
                ELSE 0
            END as file_size_bytes
        FROM pdfs p
        LEFT JOIN analyses a ON p.id = a.pdf_id
        GROUP BY p.id, p.pdf_name, p.file_hash, p.upload_date, p.processed_date,
                 p.layout, p.original_page_count, p.final_page_count, 
                 p.final_word_count, p.avg_words_per_page, p.obfuscation_applied,
                 p.pages_removed_count, p.uploaded_by, p.pdf_bytes
        ORDER BY p.processed_date DESC
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            results = cur.fetchall()
            return [dict(row) for row in results]

def check_pdf_exists_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if PDF with given hash exists in database"""
    
    sql = """
        SELECT id, pdf_name, file_hash, upload_date, uploaded_by
        FROM pdfs 
        WHERE file_hash = %s
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_hash,))
            result = cur.fetchone()
            return dict(result) if result else None
```

## 2. Update session state initialization in `main.py`

```python
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
        'raw_pdf_data': {},
        'current_pdf': None,
        'analysis_status': {},
        'processing_messages': {},
        'pdf_database_ids': {},
        'search_text': None,
        'feedback_submitted': {},
        'obfuscation_summaries': {},
        'session_id': get_session_id(),
        'batch_processing_status': None,
        'batch_processed_count': 0,
        
        # NEW: For database PDF management
        'available_pdfs_metadata': [],  # All PDFs in database (metadata only)
        'loaded_pdfs': set(),          # PDFs currently loaded in session
        'upload_rejections': {},       # Track rejected uploads with reasons
        'metadata_loaded': False       # Track if we've loaded metadata on startup
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
```

## 3. Add metadata loading function in `main.py`

```python
def load_pdfs_metadata_on_startup():
    """Load PDF metadata from database on app startup"""
    if st.session_state.metadata_loaded or not st.session_state.database_initialized:
        return
    
    try:
        from config.database import get_all_pdfs_metadata
        
        with st.spinner("Loading available PDFs from database..."):
            metadata = get_all_pdfs_metadata()
            st.session_state.available_pdfs_metadata = metadata
            st.session_state.metadata_loaded = True
            
            if metadata:
                st.success(f"✅ Found {len(metadata)} PDFs in database")
            else:
                st.info("No PDFs found in database")
                
    except Exception as e:
        st.error(f"Error loading PDF metadata: {str(e)}")
        st.session_state.available_pdfs_metadata = []
```

## 4. Add duplicate detection for uploads in `main.py`

```python
def check_upload_duplicates(uploaded_pdfs):
    """Check uploaded PDFs for duplicates against database"""
    if not uploaded_pdfs:
        return []
    
    from config.database import check_pdf_exists_by_hash
    import hashlib
    
    valid_uploads = []
    
    for pdf in uploaded_pdfs:
        pdf_bytes = pdf.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        
        # Check if already exists in database
        existing_pdf = check_pdf_exists_by_hash(file_hash)
        
        if existing_pdf:
            # Mark as rejected
            st.session_state.upload_rejections[pdf.name] = {
                'reason': 'duplicate',
                'existing_name': existing_pdf['pdf_name'],
                'existing_id': existing_pdf['id'],
                'upload_date': existing_pdf['upload_date']
            }
            st.warning(f"⚠️ {pdf.name} already exists in database as '{existing_pdf['pdf_name']}' (ID: {existing_pdf['id']})")
        else:
            # Check if already in current session
            if pdf.name in st.session_state.pdf_files:
                st.warning(f"⚠️ {pdf.name} already uploaded in current session")
            else:
                valid_uploads.append(pdf)
    
    return valid_uploads
```

## 5. Update the upload section in `main()` function

```python
# Replace the existing upload handling section with this:

# PDF uploader
st.subheader("📤 Upload Documents")
uploaded_pdfs = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF contract files for analysis"
)

if uploaded_pdfs:
    # Check for duplicates first
    valid_uploads = check_upload_duplicates(uploaded_pdfs)
    
    # Process valid uploads
    for pdf in valid_uploads:
        pdf_bytes = pdf.getvalue()
        is_valid, validation_msg = validate_pdf(pdf_bytes)
        if is_valid:
            if len(pdf_bytes) > 10 * 1024 * 1024:
                st.warning(f"⚠️ {pdf.name} is larger than 10MB. Processing may be slow.")
            st.session_state.pdf_files[pdf.name] = pdf_bytes
            st.session_state.analysis_status[pdf.name] = "Ready for processing"
            st.session_state.loaded_pdfs.add(pdf.name)
            st.success(f"✅ {pdf.name} uploaded successfully")
        else:
            st.error(f"❌ {pdf.name}: {validation_msg}")
    
    # Show rejection summary if any
    if st.session_state.upload_rejections:
        with st.expander("📋 Upload Rejections", expanded=False):
            for filename, rejection_info in st.session_state.upload_rejections.items():
                if rejection_info['reason'] == 'duplicate':
                    st.write(f"**{filename}**")
                    st.write(f"  - Reason: Already exists in database")
                    st.write(f"  - Existing file: {rejection_info['existing_name']}")
                    st.write(f"  - Database ID: {rejection_info['existing_id']}")
                    st.write(f"  - Original upload: {rejection_info['upload_date']}")
```

## 6. Add the AgGrid with all available PDFs

```python
def create_unified_pdf_grid():
    """Create AgGrid showing all available PDFs (database + session)"""
    pdf_data = []
    
    # Add PDFs from database metadata
    for pdf_record in st.session_state.available_pdfs_metadata:
        file_size_kb = pdf_record.get('file_size_bytes', 0) / 1024
        
        # Determine status emoji
        analysis_status = pdf_record.get('analysis_status', 'Unknown')
        if analysis_status == 'Analyzed':
            status_emoji = "✅"
        elif analysis_status == 'Parsed Only':
            status_emoji = "📄"
        else:
            status_emoji = "❓"
        
        # Check if loaded in session
        is_loaded = pdf_record['pdf_name'] in st.session_state.loaded_pdfs
        load_status = "🔵 Loaded" if is_loaded else "⚪ Available"
        
        pdf_data.append({
            'Status': status_emoji,
            'Load': load_status,
            'PDF Name': pdf_record['pdf_name'],
            'Size (KB)': f"{file_size_kb:.1f}",
            'Pages': f"{pdf_record.get('final_page_count', 0)}",
            'Words': f"{pdf_record.get('final_word_count', 0)}",
            'Analysis': analysis_status,
            'DB ID': str(pdf_record['id']),
            'Upload Date': pdf_record.get('upload_date', '').split('T')[0] if pdf_record.get('upload_date') else '',
            'Source': 'Database'
        })
    
    # Add session-only PDFs (newly uploaded, not yet processed)
    for pdf_name in st.session_state.pdf_files:
        # Skip if already in database
        if any(p['pdf_name'] == pdf_name for p in st.session_state.available_pdfs_metadata):
            continue
            
        file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
        status = st.session_state.analysis_status.get(pdf_name, "Ready")
        
        status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "🆕"
        
        pdf_data.append({
            'Status': status_emoji,
            'Load': "🔵 Loaded",
            'PDF Name': pdf_name,
            'Size (KB)': f"{file_size:.1f}",
            'Pages': "—",
            'Words': "—",
            'Analysis': status,
            'DB ID': "New",
            'Upload Date': "Today",
            'Source': 'Session'
        })
    
    return pdf_data

# Replace the document list section with this:
st.subheader("📋 All Available Documents")

# Load metadata on first run
load_pdfs_metadata_on_startup()

# Create unified grid
pdf_data = create_unified_pdf_grid()

if pdf_data:
    pdf_df = pd.DataFrame(pdf_data)
    gb = GridOptionsBuilder.from_dataframe(pdf_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_default_column(cellStyle={'fontSize': '14px'})
    gb.configure_column("PDF Name", cellStyle={'fontWeight': 'bold'})
    gb.configure_column("Status", width=70)
    gb.configure_column("Load", width=100)
    gb.configure_column("Size (KB)", width=90)
    gb.configure_column("Pages", width=70)
    gb.configure_column("Words", width=80)
    gb.configure_column("Analysis", width=100)
    gb.configure_column("DB ID", width=70)
    gb.configure_column("Source", width=90)
    gridOptions = gb.build()

    grid_response = AgGrid(
        pdf_df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=400,
        fit_columns_on_grid_load=True,
        theme='streamlit'
    )

    # Handle row selection (placeholder for now - we'll implement loading next)
    selected_rows = grid_response.get('selected_rows', pd.DataFrame())
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_pdf_name = selected_rows.iloc[0]['PDF Name']
        source = selected_rows.iloc[0]['Source']
        
        if source == 'Database':
            st.info(f"Selected database PDF: {selected_pdf_name} (Loading functionality coming next)")
        else:
            # Session PDF - just set as current
            set_current_pdf(selected_pdf_name)
else:
    st.info("No PDFs available. Upload some PDFs to get started!")

# Show summary
if st.session_state.available_pdfs_metadata or st.session_state.pdf_files:
    total_db_pdfs = len(st.session_state.available_pdfs_metadata)
    total_session_pdfs = len([p for p in st.session_state.pdf_files.keys() 
                             if not any(db_p['pdf_name'] == p for db_p in st.session_state.available_pdfs_metadata)])
    loaded_pdfs = len(st.session_state.loaded_pdfs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Database PDFs", total_db_pdfs)
    with col2:
        st.metric("New Session PDFs", total_session_pdfs)
    with col3:
        st.metric("Currently Loaded", loaded_pdfs)
```

## 7. Update the main function call

```python
# In your main() function, add this after initialize_session_state():
def main():
    # Initialize session state
    initialize_session_state()
    logger = ECFRLogger()
    
    # Load PDF metadata on startup
    load_pdfs_metadata_on_startup()
    
    # ... rest of your existing code ...
```

This implementation will:

1. **Load metadata on startup** - Shows all available PDFs immediately
2. **Detect duplicates** - Prevents uploading files already in database  
3. **Unified view** - Single AgGrid shows both database and session PDFs
4. **Clear indicators** - Visual feedback on what's loaded vs available
5. **Rejection tracking** - Shows users why uploads were rejected

The next step will be implementing the loading functionality when users click on database PDFs in the AgGrid.
