You're right! The code I provided was meant to replace specific sections, but it's unclear where exactly to place it in the left pane. Let me provide the complete left pane code:

```python
# Left pane: PDF upload and management
with col1:
    st.markdown('<div class="left-pane">', unsafe_allow_html=True)
    st.header("📁 Document Management")
    
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
    
    # Processing buttons
    st.subheader("⚙️ Process Documents")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Process Single PDF", disabled=not st.session_state.current_pdf):
            if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
                if st.session_state.analysis_status.get(st.session_state.current_pdf) != "Processed":
                    st.session_state.processing_messages[st.session_state.current_pdf] = []
                    with st.spinner(f"🔄 Processing {st.session_state.current_pdf}..."):
                        message_placeholder = st.empty()
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[st.session_state.current_pdf], 
                            st.session_state.current_pdf, 
                            message_placeholder,
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[st.session_state.current_pdf] = "Processed"
                            st.success(f"✅ Analysis complete for {st.session_state.current_pdf}")
                        else:
                            st.session_state.analysis_status[st.session_state.current_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {st.session_state.current_pdf}: {result}")
                        
                        if st.session_state.current_pdf in st.session_state.processing_messages:
                            with st.expander("📋 Processing Details", expanded=False):
                                for msg in st.session_state.processing_messages[st.session_state.current_pdf]:
                                    st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
    
    with col_btn2:
        if st.button("📚 Process All PDFs (Batch)", disabled=not st.session_state.pdf_files):
            process_batch_pdfs(logger)
    
    # Document list and selection
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

        # Handle row selection
        selected_rows = grid_response.get('selected_rows', pd.DataFrame())
        if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            selected_pdf_name = selected_rows.iloc[0]['PDF Name']
            source = selected_rows.iloc[0]['Source']
            
            if source == 'Database':
                st.info(f"Selected database PDF: {selected_pdf_name} (Loading functionality coming next)")
            else:
                # Session PDF - just set as current
                if selected_pdf_name != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf_name)
    else:
        st.info("No PDFs available. Upload some PDFs to get started!")
    
    # Show summary statistics
    if st.session_state.available_pdfs_metadata or st.session_state.pdf_files:
        total_db_pdfs = len(st.session_state.available_pdfs_metadata)
        total_session_pdfs = len([p for p in st.session_state.pdf_files.keys() 
                                 if not any(db_p['pdf_name'] == p for db_p in st.session_state.available_pdfs_metadata)])
        loaded_pdfs = len(st.session_state.loaded_pdfs)
        
        col1_stat, col2_stat, col3_stat = st.columns(3)
        with col1_stat:
            st.metric("Database PDFs", total_db_pdfs)
        with col2_stat:
            st.metric("New Session PDFs", total_session_pdfs)
        with col3_stat:
            st.metric("Currently Loaded", loaded_pdfs)
    
    st.markdown('</div>', unsafe_allow_html=True)
```

Make sure you also have these helper functions defined **before** the `main()` function (outside of any column or container):

```python
# Add these functions BEFORE main() function:

def load_pdfs_metadata_on_startup():
    """Load PDF metadata from database on app startup"""
    if st.session_state.metadata_loaded or not st.session_state.database_initialized:
        return
    
    try:
        from config.database import get_all_pdfs_metadata
        
        metadata = get_all_pdfs_metadata()
        st.session_state.available_pdfs_metadata = metadata
        st.session_state.metadata_loaded = True
        
        if metadata:
            # Don't show success message every time - just load silently
            pass
        
    except Exception as e:
        st.error(f"Error loading PDF metadata: {str(e)}")
        st.session_state.available_pdfs_metadata = []

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
```

The issue was likely that some of the helper functions were being defined inside the column context or not at all. Make sure all the helper functions are defined at the module level (outside of `main()` and outside of any `with col1:` blocks).
