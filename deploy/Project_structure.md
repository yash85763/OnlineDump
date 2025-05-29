Here are the code changes that need to be made:

## Code Changes for Single Column Layout Below PDF List

### 1. Replace the Document List and Selection Section

**Replace this existing code block:**
```python
        # Document list and selection
        if st.session_state.pdf_files:
            st.subheader("📋 Available Documents")
            
            pdf_data = []
            for pdf_name in st.session_state.pdf_files.keys():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
                
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': f"{file_size:.1f}",
                    'DB ID': str(db_id)
                })
            
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

            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                
                # Handle on-demand processing when PDF is clicked
                if (st.session_state.processing_mode == "on_demand" and 
                    selected_pdf != st.session_state.get('current_pdf') and
                    st.session_state.analysis_status.get(selected_pdf) != "Processed"):
                    
                    # Process the selected PDF immediately
                    st.session_state.processing_messages[selected_pdf] = []
                    with st.spinner(f"🔄 Processing {selected_pdf} on-demand..."):
                        message_placeholder = st.empty()
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[selected_pdf], 
                            selected_pdf, 
                            message_placeholder,
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[selected_pdf] = "Processed"
                            st.success(f"✅ On-demand analysis complete for {selected_pdf}")
                        else:
                            st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {selected_pdf}: {result}")
                
                # Set current PDF for viewing
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
        
        st.markdown('</div>', unsafe_allow_html=True)
```

**With this new code:**
```python
        # Document list and selection
        if st.session_state.pdf_files:
            st.subheader("📋 Available Documents")
            
            pdf_data = []
            for pdf_name in st.session_state.pdf_files.keys():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
                
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': f"{file_size:.1f}",
                    'DB ID': str(db_id)
                })
            
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

            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                
                # Handle on-demand processing when PDF is clicked
                if (st.session_state.processing_mode == "on_demand" and 
                    selected_pdf != st.session_state.get('current_pdf') and
                    st.session_state.analysis_status.get(selected_pdf) != "Processed"):
                    
                    # Process the selected PDF immediately
                    st.session_state.processing_messages[selected_pdf] = []
                    with st.spinner(f"🔄 Processing {selected_pdf} on-demand..."):
                        message_placeholder = st.empty()
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[selected_pdf], 
                            selected_pdf, 
                            message_placeholder,
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[selected_pdf] = "Processed"
                            st.success(f"✅ On-demand analysis complete for {selected_pdf}")
                        else:
                            st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {selected_pdf}: {result}")
                
                # Set current PDF for viewing
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
        
        # Processing Status Display (Single Column Below PDF List)
        st.subheader("📊 Processing Status")
        
        if st.session_state.pdf_files:
            pdf_count = len(st.session_state.pdf_files)
            processed_count = len([s for s in st.session_state.analysis_status.values() if s == "Processed"])
            
            # Show current processing mode status
            if st.session_state.processing_mode == "on_demand":
                unprocessed = [name for name, status in st.session_state.analysis_status.items() 
                             if status != "Processed"]
                if unprocessed:
                    st.info(f"🎯 **On-Demand Mode**: {len(unprocessed)} PDFs ready. Click any PDF above to analyze instantly.")
                else:
                    st.success(f"✅ **All PDFs Analyzed**: {processed_count}/{pdf_count} completed in on-demand mode.")
                    
            elif st.session_state.processing_mode == "batch":
                if st.session_state.batch_processing_status == "Running":
                    st.info(f"📚 **Batch Processing**: {processed_count}/{pdf_count} completed...")
                elif processed_count == pdf_count:
                    st.success(f"✅ **Batch Complete**: All {pdf_count} PDFs processed successfully!")
                else:
                    st.info(f"📚 **Batch Status**: {processed_count}/{pdf_count} PDFs processed")
                    
            elif pdf_count == 1:
                pdf_name = list(st.session_state.pdf_files.keys())[0]
                if st.session_state.analysis_status.get(pdf_name) == "Processed":
                    st.success("✅ **PDF Analysis Complete**")
                else:
                    st.info("📄 **Single PDF Ready**: Click 'Process PDF' button above to analyze")
            else:
                st.info(f"📋 **{pdf_count} PDFs Uploaded**: Choose a processing strategy above to begin")
            
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
```

## Summary of Changes:

### 📊 **New Single Column Processing Status Section**
- Added after the PDF grid, before closing the left pane div
- Shows different status messages based on processing mode
- Displays progress counters and completion status
- Includes expandable processing details for current PDF
- Shows live updates during batch processing

### 🔄 **How It Works:**
1. **Single PDF**: User selects from grid → clicks "Process PDF" button → processing happens
2. **Multiple PDFs**: User selects strategy → either click PDFs for on-demand or automatic batch
3. **Status Display**: Always appears in single column below PDF list regardless of mode
4. **Spinners**: Appear in the status section during processing operations

The key addition is the new "Processing Status Display" section that provides a unified, single-column view of all processing information below the document list.​​​​​​​​​​​​​​​​