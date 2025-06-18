## Step 1: Basic PDF Loading from Database

Here's the code to implement Step 1:

### 1. Add the loading function to `database.py`

```python
def load_pdf_bytes_only(pdf_id: int) -> Dict[str, Any]:
    """Load only PDF bytes and basic metadata for display"""
    
    sql = """
        SELECT id, pdf_name, pdf_bytes, file_hash, layout, 
               final_page_count, final_word_count
        FROM pdfs 
        WHERE id = %s
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            result = cur.fetchone()
            return dict(result) if result else None
```

### 2. Add the loading function to `main.py`

```python
def load_pdf_from_database_basic(pdf_id: int, pdf_name: str):
    """Load PDF bytes from database and set as current PDF"""
    try:
        from config.database import load_pdf_bytes_only
        
        with st.spinner(f"Loading {pdf_name}..."):
            pdf_data = load_pdf_bytes_only(pdf_id)
            
            if not pdf_data:
                st.error(f"Could not load PDF data for {pdf_name}")
                return False
            
            # Check if PDF bytes exist
            if not pdf_data.get('pdf_bytes'):
                st.error(f"No PDF bytes found for {pdf_name}")
                return False
            
            # Load PDF bytes into session state
            pdf_bytes = bytes(pdf_data['pdf_bytes'])
            st.session_state.pdf_files[pdf_name] = pdf_bytes
            
            # Mark as loaded
            st.session_state.loaded_pdfs.add(pdf_name)
            
            # Set as current PDF for display
            st.session_state.current_pdf = pdf_name
            
            # Store database ID for later use
            st.session_state.pdf_database_ids[pdf_name] = pdf_data['id']
            
            # Set basic status
            st.session_state.analysis_status[pdf_name] = "Loaded from database"
            
            st.success(f"✅ {pdf_name} loaded successfully")
            return True
            
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return False
```

### 3. Update the AgGrid row selection handling in the left pane

Replace the row selection section in your left pane code with this:

```python
# Handle row selection
selected_rows = grid_response.get('selected_rows', pd.DataFrame())
if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_pdf_name = selected_rows.iloc[0]['PDF Name']
    source = selected_rows.iloc[0]['Source']
    db_id = selected_rows.iloc[0]['DB ID']
    
    if source == 'Database':
        # Check if already loaded
        if selected_pdf_name in st.session_state.loaded_pdfs:
            # Already loaded, just set as current
            if selected_pdf_name != st.session_state.get('current_pdf'):
                set_current_pdf(selected_pdf_name)
                st.rerun()
        else:
            # Load from database
            try:
                pdf_id = int(db_id)
                if load_pdf_from_database_basic(pdf_id, selected_pdf_name):
                    st.rerun()  # Refresh to show loaded PDF and update grid
            except ValueError:
                st.error(f"Invalid database ID: {db_id}")
    else:
        # Session PDF - just set as current
        if selected_pdf_name != st.session_state.get('current_pdf'):
            set_current_pdf(selected_pdf_name)
            st.rerun()
```

### 4. Update the `create_unified_pdf_grid()` function

Update the load status detection in this function:

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
        
        # Check if loaded in session - UPDATED LOGIC
        pdf_name = pdf_record['pdf_name']
        is_loaded = pdf_name in st.session_state.loaded_pdfs
        load_status = "🔵 Loaded" if is_loaded else "⚪ Available"
        
        pdf_data.append({
            'Status': status_emoji,
            'Load': load_status,
            'PDF Name': pdf_name,
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
```

## Testing Step 1:

1. **Start your app** and make sure you can see database PDFs in the AgGrid
2. **Click on a database PDF** (one with Source = 'Database' and Load = '⚪ Available')
3. **Verify these things happen**:
   - Loading spinner appears
   - Success message shows
   - PDF appears in the middle pane viewer
   - Grid updates to show "🔵 Loaded" for that PDF
   - PDF becomes the current selected PDF

4. **Test edge cases**:
   - Click on an already loaded PDF (should just switch to it)
   - Click on a session PDF (should work as before)

Let me know if Step 1 works correctly, and then we can move to Step 2!
