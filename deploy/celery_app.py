def main():
    # Initialize session state
    initialize_session_state()
    logger = ECFRLogger()
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>📄 Enhanced Contract Analysis Platform</h1>
        <p>AI-powered contract analysis with privacy protection and intelligent feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with system status
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.session_state.database_initialized:
            st.markdown("<div class='database-status-success'>✅ Database Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='database-status-error'>❌ Database: {st.session_state.database_status}</div>", unsafe_allow_html=True)
        
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        # Show counts for uploaded vs database PDFs
        uploaded_count = len(st.session_state.pdf_files)
        database_count = len([name for name in st.session_state.pdf_database_ids.keys() if name not in st.session_state.pdf_files])
        processed_count = len([s for s in st.session_state.analysis_status.values() if s == "Processed"])
        
        st.write(f"**📤 Uploaded this session:** {uploaded_count}")
        st.write(f"**💾 Loaded from database:** {database_count}")
        st.write(f"**✅ Total processed:** {processed_count}")
        st.write(f"**🔄 Batch Status:** {st.session_state.batch_processing_status or 'Not started'}")
        
        st.markdown("""
        ---
        ### 🔒 Privacy Protection
        All uploaded documents are automatically processed with privacy protection:
        - Low-content pages are removed
        - Sensitive information is protected
        - Original documents are not stored permanently
        
        ### 📍 Page Number Detection
        Enhanced clause analysis now includes:
        - Page number identification for each clause
        - 90% similarity matching with original content
        - Smart text comparison algorithms
        
        ### ⚙️ Processing Modes
        **On-Demand**: Click any PDF to analyze instantly
        **Batch**: Process all PDFs automatically in sequence
        
        ### 💾 Database Integration
        **Auto-Load**: Previously processed PDFs load automatically
        **Cross-Session**: View documents processed by other users
        **Feedback History**: See all previous feedback for each document
        """)
    
    # Main layout
    col1, col2, col3 = st.columns([25, 40, 35])
    
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
            for pdf in uploaded_pdfs:
                if pdf.name not in st.session_state.pdf_files:
                    pdf_bytes = pdf.getvalue()
                    is_valid, validation_msg = validate_pdf(pdf_bytes)
                    if is_valid:
                        if len(pdf_bytes) > 10 * 1024 * 1024:
                            st.warning(f"⚠️ {pdf.name} is larger than 10MB. Processing may be slow.")
                        st.session_state.pdf_files[pdf.name] = pdf_bytes
                        st.session_state.analysis_status[pdf.name] = "Ready for processing"
                        st.success(f"✅ {pdf.name} uploaded successfully")
                    else:
                        st.error(f"❌ {pdf.name}: {validation_msg}")
        
        # Processing Strategy Selection
        st.subheader("⚙️ Processing Strategy")
        
        # Check if user has uploaded PDFs
        pdf_count = len(st.session_state.pdf_files)
        
        if pdf_count == 0:
            st.info("📤 Upload PDF files above to begin analysis, or select existing documents below")
        elif pdf_count == 1:
            # Single PDF - simple processing button
            pdf_name = list(st.session_state.pdf_files.keys())[0]
            current_status = st.session_state.analysis_status.get(pdf_name, "Ready")
            
            if current_status == "Processed":
                st.success("✅ PDF already processed")
                # Add option to reprocess
                if st.checkbox("🔄 Force reprocess", key="force_reprocess_single"):
                    if st.button("🔄 Reprocess PDF", use_container_width=True, type="secondary"):
                        st.session_state.analysis_status[pdf_name] = "Processing"
                        st.rerun()
            elif current_status == "Processing":
                st.info("🔄 Processing in progress...")
            else:
                # Show process button only if PDF is ready and selected
                if st.session_state.current_pdf == pdf_name:
                    if st.button("🔄 Process PDF", use_container_width=True, type="primary"):
                        st.session_state.analysis_status[pdf_name] = "Processing"
                        st.rerun()
                else:
                    st.info("👆 Please select the PDF from the list below to enable processing")
        else:
            # Multiple PDFs - show strategy selection
            st.markdown(f"📚 **{pdf_count} PDFs uploaded** - Choose your processing strategy:")
            
            # Initialize processing mode if not set
            if 'processing_mode' not in st.session_state:
                st.session_state.processing_mode = None
            
            col_strategy1, col_strategy2 = st.columns(2)
            
            with col_strategy1:
                st.markdown("""
                <div style='padding: 1rem; border: 2px solid #17a2b8; border-radius: 8px; background: linear-gradient(135deg, #e1f7fa 0%, #b3e5fc 100%); margin-bottom: 1rem;'>
                    <h4 style='color: #0c5460; margin: 0 0 0.5rem 0;'>🎯 On-Demand Analysis</h4>
                    <p style='color: #0c5460; margin: 0; font-size: 0.9rem;'>Process PDFs individually when you click them in the list. Immediate results for selected documents.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🎯 Choose On-Demand", use_container_width=True, type="primary" if st.session_state.processing_mode == "on_demand" else "secondary"):
                    st.session_state.processing_mode = "on_demand"
                    st.success("✅ On-Demand processing selected! Click any PDF in the list below to analyze it.")
                    st.rerun()
            
            with col_strategy2:
                st.markdown("""
                <div style='padding: 1rem; border: 2px solid #28a745; border-radius: 8px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); margin-bottom: 1rem;'>
                    <h4 style='color: #155724; margin: 0 0 0.5rem 0;'>📚 Batch Processing</h4>
                    <p style='color: #155724; margin: 0; font-size: 0.9rem;'>Process all PDFs automatically in sequence. Results available after all documents are analyzed.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📚 Choose Batch", use_container_width=True, type="primary" if st.session_state.processing_mode == "batch" else "secondary"):
                    st.session_state.processing_mode = "batch"
                    # Immediately start batch processing
                    st.success("✅ Batch processing started! Please wait while all PDFs are analyzed...")
                    process_batch_pdfs(logger)
                    st.rerun()
            
            # Show current mode status
            if st.session_state.processing_mode == "on_demand":
                st.info("🎯 **On-Demand Mode Active**: Click any PDF in the list below to analyze it instantly.")
            elif st.session_state.processing_mode == "batch":
                processed_count_mode = len([s for s in st.session_state.analysis_status.values() if s == "Processed"])
                if processed_count_mode == pdf_count:
                    st.success(f"✅ **Batch Processing Complete**: All {pdf_count} PDFs have been analyzed!")
                else:
                    st.info(f"📚 **Batch Mode**: {processed_count_mode}/{pdf_count} PDFs processed")
            
            # Reset button
            if st.session_state.processing_mode:
                if st.button("🔄 Change Processing Strategy", use_container_width=True):
                    st.session_state.processing_mode = None
                    st.rerun()
        
        # Document list and selection - ALWAYS SHOW (even without uploaded files)
        st.subheader("📋 Available Documents")
        
        # Combine uploaded PDFs and existing PDFs from database
        all_pdfs = {}
        
        # Add existing PDFs from database
        for pdf_name in st.session_state.pdf_database_ids.keys():
            if pdf_name not in st.session_state.pdf_files:  # Don't duplicate uploaded files
                all_pdfs[pdf_name] = "database"
        
        # Add currently uploaded PDFs
        for pdf_name in st.session_state.pdf_files.keys():
            all_pdfs[pdf_name] = "uploaded"
        
        if all_pdfs:
            pdf_data = []
            for pdf_name, source in all_pdfs.items():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                
                # Get file size
                if source == "uploaded":
                    file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
                    size_display = f"{file_size:.1f}"
                else:
                    size_display = "From DB"
                
                # Status and source emojis
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                source_emoji = "📤" if source == "uploaded" else "💾"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'Source': source_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': size_display,
                    'DB ID': str(db_id)
                })
            
            # Display the grid
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
                height=300,
                fit_columns_on_grid_load=True,
                theme='streamlit'
            )

            # Handle selection
            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                
                # Only process if it's an uploaded PDF and in on-demand mode
                if (st.session_state.processing_mode == "on_demand" and 
                    selected_pdf != st.session_state.get('current_pdf') and
                    st.session_state.analysis_status.get(selected_pdf) != "Processed" and
                    selected_pdf in st.session_state.pdf_files):  # Only process uploaded files
                    
                    # Process uploaded PDF
                    with st.spinner(f"🔄 Processing {selected_pdf} on-demand..."):
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[selected_pdf], 
                            selected_pdf, 
                            st.empty(),
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[selected_pdf] = "Processed"
                            st.success(f"✅ On-demand analysis complete for {selected_pdf}")
                        else:
                            st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {selected_pdf}: {result}")
                
                # Set current PDF for viewing (works for both uploaded and database PDFs)
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    
                    # Load clauses for database PDFs if not already loaded
                    file_stem = Path(selected_pdf).stem
                    if (selected_pdf in st.session_state.pdf_database_ids and 
                        file_stem in st.session_state.json_data and
                        not st.session_state.json_data[file_stem].get('relevant_clauses')):
                        
                        pdf_id = st.session_state.pdf_database_ids[selected_pdf]
                        load_pdf_clauses(pdf_id, file_stem)
        else:
            st.info("📄 No documents available. Upload PDFs above or check database connection.")
            
        # Show legend for source icons
        st.caption("📤 = Uploaded this session | 💾 = From database")
        
        # Processing Status Display (Single Column Below PDF List)
        st.subheader("📊 Processing Status")
        
        if st.session_state.pdf_files or database_count > 0:
            total_pdfs = len(all_pdfs) if 'all_pdfs' in locals() else 0
            
            # Handle single PDF processing execution
            if pdf_count == 1:
                pdf_name = list(st.session_state.pdf_files.keys())[0]
                current_status = st.session_state.analysis_status.get(pdf_name, "Ready")
                
                if current_status == "Processing":
                    with st.spinner(f"🔄 Processing {pdf_name}..."):
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[pdf_name], 
                            pdf_name, 
                            st.empty(),
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[pdf_name] = "Processed"
                            st.success(f"✅ Analysis complete for {pdf_name}")
                        else:
                            st.session_state.analysis_status[pdf_name] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {pdf_name}: {result}")
                        st.rerun()
            
            # Show current processing mode status
            if st.session_state.processing_mode == "on_demand":
                unprocessed = [name for name, status in st.session_state.analysis_status.items() 
                             if status != "Processed" and name in st.session_state.pdf_files]
                if unprocessed:
                    st.info(f"🎯 **On-Demand Mode**: {len(unprocessed)} uploaded PDFs ready. Click any PDF above to analyze instantly.")
                else:
                    st.success(f"✅ **All uploaded PDFs analyzed**: {uploaded_count} processed in on-demand mode.")
                    
            elif st.session_state.processing_mode == "batch":
                if st.session_state.batch_processing_status == "Running":
                    st.info(f"📚 **Batch Processing**: {processed_count}/{pdf_count} uploaded PDFs completed...")
                elif processed_count == pdf_count:
                    st.success(f"✅ **Batch Complete**: All {pdf_count} uploaded PDFs processed successfully!")
                else:
                    st.info(f"📚 **Batch Status**: {processed_count}/{pdf_count} uploaded PDFs processed")
                    
            elif pdf_count == 1:
                pdf_name = list(st.session_state.pdf_files.keys())[0]
                current_status = st.session_state.analysis_status.get(pdf_name, "Ready")
                if current_status == "Processed":
                    st.success("✅ **PDF Analysis Complete**")
                elif current_status == "Processing":
                    st.info("🔄 **Processing in progress**...")
                else:
                    if st.session_state.current_pdf == pdf_name:
                        st.info("📄 **Single PDF Selected**: Click 'Process PDF' button above to analyze")
                    else:
                        st.info("📄 **Single PDF Ready**: Select the PDF from the list above first")
            elif pdf_count > 1:
                st.info(f"📋 **{pdf_count} PDFs Uploaded**: Choose a processing strategy above to begin")
            
            # Show database status
            if database_count > 0:
                st.info(f"💾 **{database_count} documents loaded from database** - Ready for viewing and feedback")
            
            # Show detailed processing messages for current/recently processed PDFs
            current_pdf = st.session_state.current_pdf
            if current_pdf and current_pdf in st.session_state.processing_messages:
                if st.session_state.processing_messages[current_pdf]:
                    with st.expander(f"📋 Processing Details: {current_pdf}", expanded=False):
                        for msg in st.session_state.processing_messages[current_pdf]:
                            st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
            
            # Show any PDFs currently being processed (for batch mode)
            if st.session_state.processing_mode == "batch" and st.session_state.batch_processing_status == "Running":
                processing_pdfs = [name for name, status in st.session_state.analysis_status.items() 
                                 if "processing" in str(status).lower()]
                if processing_pdfs:
                    for pdf_name in processing_pdfs:
                        if pdf_name in st.session_state.processing_messages:
                            st.markdown(f"**🔄 Currently Processing: {pdf_name}**")
                            latest_msg = st.session_state.processing_messages[pdf_name][-1] if st.session_state.processing_messages[pdf_name] else "Processing..."
                            st.markdown(f"<div class='processing-message'>{latest_msg}</div>", unsafe_allow_html=True)
        else:
            st.info("📤 **No PDFs uploaded yet**. Use the uploader above to get started.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle pane: PDF viewer
    with col2:
        st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
        st.header("📖 Document Viewer")
        
        if st.session_state.current_pdf and (st.session_state.current_pdf in st.session_state.pdf_files or st.session_state.current_pdf in st.session_state.pdf_database_ids):
            # Get PDF bytes for display
            if st.session_state.current_pdf in st.session_state.pdf_files:
                current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
                pdf_source = "uploaded"
            else:
                # For database PDFs, we can't display them directly since we don't have the bytes
                current_pdf_bytes = None
                pdf_source = "database"
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader(f"📄 {st.session_state.current_pdf}")
                if pdf_source == "database":
                    st.caption("💾 Document loaded from database")
            with col_info2:
                if current_pdf_bytes:
                    file_size_mb = len(current_pdf_bytes) / (1024 * 1024)
                    st.metric("File Size", f"{file_size_mb:.2f} MB")
                else:
                    st.metric("Source", "Database")
            
            if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
                obf_summary = st.session_state.obfuscation_summaries[st.session_state.current_pdf]
                if obf_summary.get('obfuscation_applied', False):
                    col_obf1, col_obf2, col_obf3 = st.columns(3)
                    with col_obf1:
                        st.metric("Original Pages", obf_summary.get('total_original_pages', 0))
                    with col_obf2:
                        st.metric("Final Pages", obf_summary.get('total_final_pages', 0))
                    with col_obf3:
                        st.metric("Pages Removed", obf_summary.get('pages_removed_count', 0))
                    
                    st.markdown("""
                    <div class='obfuscation-info'>
                        🔒 <strong>Privacy Protection Applied:</strong> This document has been processed with our privacy protection system. 
                        Some pages with minimal content have been removed to protect confidentiality while preserving the core contract content for analysis.
                    </div>
                    """, unsafe_allow_html=True)
            
            if current_pdf_bytes:
                try:
                    pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error displaying PDF: {e}")
                    st.info("💡 Try downloading the PDF to view it externally.")
                    st.download_button(
                        label="📥 Download PDF",
                        data=current_pdf_bytes,
                        file_name=st.session_state.current_pdf,
                        mime="application/pdf"
                    )
            else:
                st.info("📄 **PDF Preview Not Available**")
                st.markdown("""
                This document was loaded from the database and the original PDF file is not available for viewing. 
                However, you can still:
                - View the complete analysis results in the right panel
                - See all extracted clauses with page numbers
                - Review and submit feedback
                - View previous feedback history
                """)
        else:
            st.info("👆 Please select a PDF from the document list to view it here.")
            st.markdown("""
            ### 🚀 Getting Started
            1. **Upload** your PDF contract files using the uploader in the left panel
            2. **Choose Strategy** for multiple PDFs:
               - **On-Demand**: Click any PDF to analyze instantly
               - **Batch**: Process all PDFs automatically
            3. **Select** a document from the list to view it
            4. **Review** the analysis results in the right panel
            5. **Provide feedback** to help improve our analysis
            
            ### ✨ New Features
            - **💾 Database Integration**: Previously processed PDFs load automatically
            - **🌐 Cross-Session Access**: View documents processed by other users
            - **📋 Feedback History**: See all previous feedback for each document
            - **📍 Page Number Detection**: Each clause shows its exact location
            - **🎯 Smart Matching**: 90% similarity algorithm for accurate page identification
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right pane: Analysis results and feedback
    with col3:
        st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)
        st.header("🔍 Analysis Results")
        
        file_stem = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else None
        if file_stem and file_stem in st.session_state.json_data:
            json_data = st.session_state.json_data[file_stem]
            
            col_analysis1, col_analysis2 = st.columns(2)
            with col_analysis1:
                st.subheader(f"📊 Analysis: {file_stem}")
                if json_data.get('loaded_from_database'):
                    st.caption("💾 Loaded from database")
            with col_analysis2:
                pdf_db_id = st.session_state.pdf_database_ids.get(st.session_state.current_pdf)
                if pdf_db_id:
                    st.metric("Database ID", pdf_db_id)
            
            st.markdown("### 📋 Form Number")
            form_number = json_data.get('form_number', 'Not available')
            st.markdown(f"<div class='extract-text'><strong>{form_number}</strong></div>", 
                       unsafe_allow_html=True)
            
            st.markdown("### 📝 Contract Summary")
            summary = json_data.get('summary', 'No summary available')
            st.markdown(f"<div class='extract-text'>{summary}</div>", 
                       unsafe_allow_html=True)
            
            st.markdown("### ✅ Contract Status")
            status_fields = {
                'data_usage_mentioned': 'Data Usage Mentioned',
                'data_limitations_exists': 'Data Limitations Exists',
                'pi_clause': 'Presence of PI Clause',
                'ci_clause': 'Presence of CI Clause'
            }
            
            col_status1, col_status2 = st.columns(2)
            status_items = list(status_fields.items())
            
            for i, (key, label) in enumerate(status_items):
                target_col = col_status1 if i % 2 == 0 else col_status2
                with target_col:
                    status = json_data.get(key, None)
                    status_str = str(status).lower() if status is not None else 'unknown'
                    
                    if status_str in ['true', 'yes']:
                        button_class = 'status-button-true'
                        icon = "✅"
                    elif status_str in ['false', 'no']:
                        button_class = 'status-button-false' 
                        icon = "❌"
                    else:
                        button_class = 'status-button-missing'
                        icon = "❓"
                    
                    st.markdown(f"""
                    <div class='{button_class}'>
                        {icon} <strong>{label}</strong><br>
                        <small>{status}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 📄 Relevant Clauses with Page Locations")
            clauses = json_data.get("relevant_clauses", [])
            
            if clauses:
                # Get raw PDF data for page matching
                raw_pdf_data = st.session_state.raw_pdf_data.get(st.session_state.current_pdf, {})
                
                for i, clause in enumerate(clauses):
                    with st.expander(f"📑 Clause {i+1}: {clause['type'].capitalize()}", expanded=False):
                        st.markdown(f"**Type:** `{clause['type']}`")
                        st.markdown(f"**Content:**")
                        st.markdown(f"<div class='extract-text'>{clause['text']}</div>", unsafe_allow_html=True)
                        
                        # Find page number for this clause
                        if raw_pdf_data:
                            page_match = find_clause_page_number(clause['text'], raw_pdf_data)
                            
                            if page_match['page_number']:
                                st.markdown(f"""
                                <div class='page-info'>
                                    📍 <strong>Found on Page {page_match['page_number']}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show similarity score
                                similarity_percent = page_match['similarity_score'] * 100
                                st.markdown(f"""
                                <div class='similarity-score'>
                                    🎯 Match Confidence: {similarity_percent:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show a preview of the matched text if available
                                if page_match['matched_text'] and len(page_match['matched_text']) > 50:
                                    with st.expander(f"👁️ Preview of matched content", expanded=False):
                                        preview_text = page_match['matched_text'][:300] + "..." if len(page_match['matched_text']) > 300 else page_match['matched_text']
                                        st.text(preview_text)
                            else:
                                st.markdown(f"""
                                <div class='page-info' style='border-left-color: #ff6b6b;'>
                                    ❓ <strong>Page location not found</strong><br>
                                    <small>Best match: {page_match['similarity_score']*100:.1f}% (below 90% threshold)</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='page-info' style='border-left-color: #ffa500;'>
                                ⚠️ <strong>Page matching not available</strong><br>
                                <small>Raw PDF data not found for this document</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Navigation button (now shows page number instead of search)
                        if raw_pdf_data:
                            page_match = find_clause_page_number(clause['text'], raw_pdf_data)
                            if page_match['page_number']:
                                col_nav1, col_nav2 = st.columns(2)
                                with col_nav1:
                                    st.info(f"📖 Navigate to Page {page_match['page_number']} in the PDF viewer")
                                with col_nav2:
                                    if len(clause['text']) > 100:
                                        st.caption("💡 Long clauses may span multiple sections")
            else:
                st.info("No relevant clauses detected in this contract.")
            
            # Processing statistics with page matching info
            if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
                with st.expander("📊 Processing Statistics", expanded=False):
                    obf_summary = st.session_state.obfuscation_summaries[st.session_state.current_pdf]
                    
                    col_stat1, col_stat2 = st.columns(2)
