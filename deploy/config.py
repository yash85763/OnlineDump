# Left pane: PDF upload and management
with col1:
    st.markdown('<div class="left-pane">', unsafe_allow_html=True)
    
    # Header with icon
    header_icon = get_icon_html('document', 24, '📁', 'Document')
    st.markdown(f'<h2 style="display: flex; align-items: center; gap: 10px;">{header_icon}Document Management</h2>', unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("### 📤 Upload Documents")
    
    # Upload widget
    uploaded_pdfs = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF contract files for analysis (Max 10MB per file)",
        key="main_pdf_uploader"
    )
    
    # Process uploaded files
    if uploaded_pdfs:
        upload_success_count = 0
        upload_error_count = 0
        
        for pdf in uploaded_pdfs:
            if pdf.name not in st.session_state.pdf_files:
                pdf_bytes = pdf.getvalue()
                
                # Validate file size (10MB limit)
                if len(pdf_bytes) > 10 * 1024 * 1024:
                    st.error(f"❌ {pdf.name}: File too large (>10MB)")
                    upload_error_count += 1
                    continue
                
                # Validate PDF
                is_valid, validation_msg = validate_pdf(pdf_bytes)
                if is_valid:
                    st.session_state.pdf_files[pdf.name] = pdf_bytes
                    st.session_state.analysis_status[pdf.name] = "Ready for processing"
                    upload_success_count += 1
                    
                    success_icon = get_icon_html('success', 16, '✅', 'Success')
                    st.markdown(f'<div style="color: green; display: flex; align-items: center; gap: 5px; margin: 5px 0;">{success_icon}<small>{pdf.name} uploaded successfully</small></div>', unsafe_allow_html=True)
                else:
                    st.error(f"❌ {pdf.name}: {validation_msg}")
                    upload_error_count += 1
        
        # Upload summary
        if upload_success_count > 0 or upload_error_count > 0:
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                if upload_success_count > 0:
                    st.metric("Uploaded", upload_success_count, delta=upload_success_count)
            with col_summary2:
                if upload_error_count > 0:
                    st.metric("Failed", upload_error_count, delta=upload_error_count, delta_color="inverse")
    
    # Document List and Management
    if st.session_state.pdf_files:
        st.markdown("---")
        
        # Statistics Section
        total_docs = len(st.session_state.pdf_files)
        processed_docs = len([status for status in st.session_state.analysis_status.values() 
                            if "Processed" in str(status)])
        ready_docs = len([status for status in st.session_state.analysis_status.values() 
                         if status == "Ready for processing"])
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("📄 Total", total_docs)
        with col_stat2:
            st.metric("✅ Processed", processed_docs)
        with col_stat3:
            st.metric("⏳ Ready", ready_docs)
        
        # Batch Processing Section
        st.markdown("### 🚀 Batch Processing")
        
        # Get unprocessed documents
        unprocessed_pdfs = []
        for pdf_name, status in st.session_state.analysis_status.items():
            if "Processed" not in str(status) and "processing" not in str(status).lower():
                unprocessed_pdfs.append(pdf_name)
        
        if unprocessed_pdfs:
            # Multi-select for batch processing
            selected_for_batch = st.multiselect(
                "Select documents for batch processing:",
                options=unprocessed_pdfs,
                default=st.session_state.get('batch_selected_pdfs', []),
                max_selections=20,
                help="Select up to 20 documents to process simultaneously",
                key="batch_multiselect"
            )
            
            # Update session state
            st.session_state.batch_selected_pdfs = selected_for_batch
            
            # Batch control buttons
            if selected_for_batch:
                col_batch1, col_batch2 = st.columns(2)
                
                with col_batch1:
                    batch_disabled = (
                        len(selected_for_batch) == 0 or 
                        st.session_state.get('batch_job_active', False) or
                        not st.session_state.get('database_initialized', False)
                    )
                    
                    batch_icon = get_icon_html('batch', 16, '🚀', 'Batch')
                    if st.button(
                        f"Start Batch ({len(selected_for_batch)})", 
                        disabled=batch_disabled,
                        help="Process all selected documents",
                        key="start_batch_btn",
                        use_container_width=True
                    ):
                        st.session_state.batch_job_active = True
                        st.rerun()
                
                with col_batch2:
                    if st.session_state.get('batch_job_active', False):
                        if st.button(
                            "⏹️ Cancel", 
                            help="Cancel batch processing",
                            key="cancel_batch_btn",
                            use_container_width=True
                        ):
                            st.session_state.batch_job_active = False
                            st.rerun()
                
                # Show batch selection info
                info_icon = get_icon_html('info', 16, 'ℹ️', 'Info')
                st.markdown(f'<div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; display: flex; align-items: center; gap: 8px;">{info_icon}<span><strong>Selected:</strong> {len(selected_for_batch)} documents ready for batch processing</span></div>', unsafe_allow_html=True)
                
                # Batch size warning
                if len(selected_for_batch) > 10:
                    warning_icon = get_icon_html('warning', 16, '⚠️', 'Warning')
                    st.markdown(f'<div style="background: #fff3cd; padding: 8px; border-radius: 5px; margin: 5px 0; display: flex; align-items: center; gap: 8px;">{warning_icon}<small>Large batch size may take several minutes to complete</small></div>', unsafe_allow_html=True)
            
            else:
                st.info("👆 Select documents above to enable batch processing")
        
        else:
            success_icon = get_icon_html('success', 16, '✅', 'Success')
            st.markdown(f'<div style="background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; display: flex; align-items: center; gap: 8px;">{success_icon}<span>All documents have been processed!</span></div>', unsafe_allow_html=True)
        
        # Batch Processing Execution
        if st.session_state.get('batch_job_active', False) and st.session_state.get('batch_selected_pdfs', []):
            st.markdown("### ⚡ Processing in Progress...")
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                # Create batch job
                try:
                    job_id, batch_db_id = create_batch_job(
                        st.session_state.batch_selected_pdfs, 
                        get_session_id()
                    )
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run batch processing
                    with st.spinner("Running batch analysis..."):
                        results = run_batch_processing(
                            st.session_state.batch_selected_pdfs, 
                            job_id, 
                            progress_container
                        )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.success("✅ Batch processing completed!")
                    
                    # Results metrics
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        success_icon = get_icon_html('success', 16, '✅', 'Success')
                        st.markdown(f'<div style="text-align: center; padding: 10px; background: #d4edda; border-radius: 5px;">{success_icon}<br><strong>{len(results.get("processed", []))}</strong><br><small>Processed</small></div>', unsafe_allow_html=True)
                    with col_result2:
                        skip_icon = get_icon_html('info', 16, '⏭️', 'Skipped')
                        st.markdown(f'<div style="text-align: center; padding: 10px; background: #cff4fc; border-radius: 5px;">{skip_icon}<br><strong>{len(results.get("skipped", []))}</strong><br><small>Skipped</small></div>', unsafe_allow_html=True)
                    with col_result3:
                        error_icon = get_icon_html('error', 16, '❌', 'Failed')
                        st.markdown(f'<div style="text-align: center; padding: 10px; background: #f8d7da; border-radius: 5px;">{error_icon}<br><strong>{len(results.get("failed", []))}</strong><br><small>Failed</small></div>', unsafe_allow_html=True)
                    
                    # Detailed results
                    if results.get('failed'):
                        with st.expander("❌ Failed Documents", expanded=False):
                            for failed in results['failed']:
                                error_icon = get_icon_html('error', 14, '❌', 'Error')
                                st.markdown(f'<div style="margin: 5px 0; display: flex; align-items: center; gap: 8px;">{error_icon}<span><strong>{failed["name"]}:</strong> {failed["error"]}</span></div>', unsafe_allow_html=True)
                    
                    if results.get('skipped'):
                        with st.expander("⏭️ Skipped Documents", expanded=False):
                            for skipped in results['skipped']:
                                skip_icon = get_icon_html('info', 14, '⏭️', 'Skipped')
                                st.markdown(f'<div style="margin: 5px 0; display: flex; align-items: center; gap: 8px;">{skip_icon}<span><strong>{skipped["name"]}:</strong> {skipped["reason"]}</span></div>', unsafe_allow_html=True)
                    
                    # Privacy stats
                    if results.get('total_pages_removed', 0) > 0:
                        privacy_icon = get_icon_html('privacy', 16, '🔒', 'Privacy')
                        st.markdown(f'<div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; display: flex; align-items: center; gap: 8px;">{privacy_icon}<span><strong>Privacy Protection:</strong> {results["total_pages_removed"]} pages removed from {results.get("total_original_pages", 0)} total pages</span></div>', unsafe_allow_html=True)
                    
                    # Store results and reset
                    if 'batch_results' not in st.session_state:
                        st.session_state.batch_results = {}
                    st.session_state.batch_results[job_id] = results
                    st.session_state.batch_job_active = False
                    st.session_state.batch_selected_pdfs = []
                    
                    # Refresh button
                    if st.button("🔄 Refresh Document List", key="refresh_after_batch"):
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")
                    st.session_state.batch_job_active = False
        
        # Individual Document Selection
        st.markdown("---")
        st.markdown("### 📄 Individual Document Selection")
        
        # Document grid/list
        try:
            # Create enhanced dataframe
            pdf_data = []
            for pdf_name in st.session_state.pdf_files.keys():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                file_size = len(st.session_state.pdf_files[pdf_name]) / 1024  # KB
                
                # Status display
                if "Processed" in str(status):
                    status_display = "✅ Processed"
                    status_color = "#d4edda"
                elif "processing" in str(status).lower():
                    status_display = "⏳ Processing"
                    status_color = "#cff4fc"
                elif "failed" in str(status).lower() or "error" in str(status).lower():
                    status_display = "❌ Failed"
                    status_color = "#f8d7da"
                else:
                    status_display = "📄 Ready"
                    status_color = "#f8f9fa"
                
                pdf_data.append({
                    'PDF Name': pdf_name,
                    'Size (KB)': f"{file_size:.1f}",
                    'Status': status_display,
                    'DB ID': str(db_id)
                })
            
            # Try to use AgGrid
            try:
                from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
                
                pdf_df = pd.DataFrame(pdf_data)
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
                    height=250,
                    fit_columns_on_grid_load=True,
                    theme='streamlit',
                    key='main_pdf_grid'
                )

                # Handle PDF selection
                selected_rows = grid_response.get('selected_rows', pd.DataFrame())
                if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                    selected_pdf = selected_rows.iloc[0]['PDF Name']
                    
                    if selected_pdf != st.session_state.get('current_pdf'):
                        set_current_pdf(selected_pdf)
                        
                        # Check if analysis exists in database first
                        pdf_id = st.session_state.pdf_database_ids.get(selected_pdf)
                        file_stem = Path(selected_pdf).stem
                        
                        if pdf_id and file_stem not in st.session_state.json_data:
                            # Try to load from database
                            if load_analysis_from_database(selected_pdf, pdf_id):
                                st.session_state.analysis_status[selected_pdf] = "Processed (from database)"
                                st.success(f"✅ Loaded existing analysis for {selected_pdf}")
                                st.rerun()
                        
                        # Process PDF if not already processed
                        current_status = st.session_state.analysis_status.get(selected_pdf, "")
                        if "Processed" not in current_status:
                            st.session_state.processing_messages[selected_pdf] = []
                            
                            with st.spinner(f"🔄 Processing {selected_pdf}..."):
                                message_placeholder = st.empty()
                                success, result = process_pdf_enhanced(
                                    st.session_state.pdf_files[selected_pdf], 
                                    selected_pdf, 
                                    message_placeholder
                                )
                                
                                if success:
                                    st.session_state.analysis_status[selected_pdf] = "Processed"
                                    st.success(f"✅ Analysis complete for {selected_pdf}")
                                else:
                                    st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                                    st.error(f"❌ Failed to process {selected_pdf}: {result}")
                                
                                # Show processing details
                                if selected_pdf in st.session_state.processing_messages:
                                    final_messages = st.session_state.processing_messages[selected_pdf].copy()
                                    if final_messages:
                                        with st.expander("📋 Processing Details", expanded=False):
                                            for msg in final_messages:
                                                processing_icon = get_icon_html('processing', 14, '🔄', 'Processing')
                                                st.markdown(f'<div style="margin: 3px 0; display: flex; align-items: center; gap: 8px; font-size: 0.9em;">{processing_icon}<span>{msg}</span></div>', unsafe_allow_html=True)
                                
                                message_placeholder.empty()
                                st.rerun()
            
            except ImportError:
                # Fallback to simple selectbox
                st.warning("AgGrid not available, using simple selector")
                
                pdf_options = [""] + list(st.session_state.pdf_files.keys())
                selected_pdf = st.selectbox(
                    "Choose a document:",
                    options=pdf_options,
                    key="pdf_selector_fallback"
                )
                
                if selected_pdf and selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    st.rerun()
                
                # Show document info table
                if pdf_data:
                    st.dataframe(pd.DataFrame(pdf_data), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in document display: {str(e)}")
            
            # Emergency fallback - simple list
            st.write("**Available Documents:**")
            for i, pdf_name in enumerate(st.session_state.pdf_files.keys()):
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                col_doc1, col_doc2, col_doc3 = st.columns([3, 2, 1])
                
                with col_doc1:
                    doc_icon = get_icon_html('document', 16, '📄', 'Document')
                    st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;">{doc_icon}<span><strong>{pdf_name}</strong></span></div>', unsafe_allow_html=True)
                with col_doc2:
                    st.write(f"Status: {status}")
                with col_doc3:
                    if st.button("Select", key=f"select_pdf_{i}"):
                        set_current_pdf(pdf_name)
                        st.rerun()
        
        # Clear all button
        st.markdown("---")
        col_clear1, col_clear2 = st.columns(2)
        
        with col_clear1:
            if st.button("🗑️ Clear All Documents", 
                        help="Remove all uploaded documents",
                        key="clear_all_docs"):
                # Confirm dialog would be nice here
                st.session_state.pdf_files = {}
                st.session_state.json_data = {}
                st.session_state.analysis_status = {}
                st.session_state.processing_messages = {}
                st.session_state.pdf_database_ids = {}
                st.session_state.current_pdf = None
                st.success("All documents cleared!")
                st.rerun()
        
        with col_clear2:
            if st.button("🔄 Refresh Status", 
                        help="Refresh document status",
                        key="refresh_status"):
                st.rerun()
    
    else:
        # No documents uploaded yet
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        
        info_icon = get_icon_html('info', 20, 'ℹ️', 'Info')
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                {info_icon}<h4 style="margin: 0;">Welcome to Contract Analysis</h4>
            </div>
            <ol style="margin: 0; padding-left: 20px;">
                <li><strong>Upload</strong> your PDF contract files using the uploader above</li>
                <li><strong>Select</strong> documents for individual or batch processing</li>
                <li><strong>Review</strong> AI-powered analysis results</li>
                <li><strong>Provide feedback</strong> to help improve accuracy</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)
        
        # Quick stats about the system
        st.markdown("### 📊 System Capabilities")
        
        capability_items = [
            ("🔍", "Smart contract analysis with AI"),
            ("🔒", "Privacy protection via content obfuscation"),
            ("📊", "Batch processing up to 20 documents"),
            ("💾", "Secure database storage"),
            ("📝", "Intelligent feedback system"),
            ("⚡", "Fast processing and results")
        ]
        
        for icon, text in capability_items:
            st.markdown(f"**{icon}** {text}")
    
    st.markdown('</div>', unsafe_allow_html=True)