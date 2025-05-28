Here are the specific changes needed to add batch processing functionality:

## **1. Add Batch Processing Functions:**

**Add these functions after the `check_all_services()` function:**

```python
def create_batch_job(selected_pdfs, session_id):
    """Create a new batch job in database"""
    from config.database import store_batch_job_data
    import uuid
    
    job_id = str(uuid.uuid4())
    batch_data = {
        'job_id': job_id,
        'created_at': datetime.now(),
        'total_files': len(selected_pdfs),
        'processed_files': 0,
        'failed_files': 0,
        'status': 'pending',
        'created_by': session_id,
        'total_pages_processed': 0,
        'total_pages_removed': 0,
        'total_paragraphs_obfuscated': 0,
        'results_json': {'files': selected_pdfs},
        'error_log': ''
    }
    
    batch_db_id = store_batch_job_data(batch_data)
    return job_id, batch_db_id

def run_batch_processing(selected_pdfs, job_id, progress_container):
    """Run batch processing for selected PDFs"""
    from utils.enhanced_pdf_handler import process_single_pdf_from_streamlit
    from config.database import get_pdf_by_hash
    import hashlib
    
    results = {
        'processed': [],
        'failed': [],
        'skipped': [],
        'total_pages_removed': 0,
        'total_original_pages': 0
    }
    
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for i, pdf_name in enumerate(selected_pdfs):
        try:
            status_text.text(f"Processing {i+1}/{len(selected_pdfs)}: {pdf_name}")
            
            # Check if already processed (by hash)
            pdf_bytes = st.session_state.pdf_files[pdf_name]
            file_hash = hashlib.sha256(pdf_bytes).hexdigest()
            existing_pdf = get_pdf_by_hash(file_hash)
            
            if existing_pdf:
                st.session_state.pdf_database_ids[pdf_name] = existing_pdf['id']
                results['skipped'].append({'name': pdf_name, 'reason': 'Already in database'})
                st.session_state.analysis_status[pdf_name] = "Processed (from database)"
            else:
                # Process the PDF
                result = process_single_pdf_from_streamlit(
                    pdf_name=pdf_name,
                    pdf_bytes=pdf_bytes,
                    enable_obfuscation=True,
                    uploaded_by=f"batch_{job_id}"
                )
                
                if result.get('success'):
                    st.session_state.pdf_database_ids[pdf_name] = result.get('pdf_id')
                    st.session_state.obfuscation_summaries[pdf_name] = result.get('obfuscation_summary', {})
                    
                    # Run contract analysis
                    pages_content = result.get('pages', [])
                    contract_text = '\n\n'.join([
                        para for page in pages_content 
                        for para in page.get('paragraphs', [])
                    ])
                    
                    # Store pages content for clause mapping
                    st.session_state.pages_content[pdf_name] = pages_content
                    
                    # Analyze contract
                    analysis_success = analyze_contract_for_batch(pdf_name, contract_text, result.get('pdf_id'))
                    
                    if analysis_success:
                        results['processed'].append(pdf_name)
                        st.session_state.analysis_status[pdf_name] = "Processed"
                        
                        # Track obfuscation stats
                        obf_summary = result.get('obfuscation_summary', {})
                        results['total_pages_removed'] += obf_summary.get('pages_removed_count', 0)
                        results['total_original_pages'] += obf_summary.get('total_original_pages', 0)
                    else:
                        results['failed'].append({'name': pdf_name, 'error': 'Analysis failed'})
                        st.session_state.analysis_status[pdf_name] = "Analysis failed"
                else:
                    results['failed'].append({'name': pdf_name, 'error': result.get('error', 'Unknown')})
                    st.session_state.analysis_status[pdf_name] = f"Failed: {result.get('error', 'Unknown')}"
            
            # Update progress
            progress_bar.progress((i + 1) / len(selected_pdfs))
            
        except Exception as e:
            results['failed'].append({'name': pdf_name, 'error': str(e)})
            st.session_state.analysis_status[pdf_name] = f"Error: {str(e)}"
    
    status_text.text("Batch processing completed!")
    return results

def analyze_contract_for_batch(pdf_name, contract_text, pdf_id):
    """Analyze contract and store results for batch processing"""
    try:
        from contract_analyzer import ContractAnalyzer
        from config.database import store_analysis_data, store_clause_data, get_next_analysis_version
        import tempfile
        
        contract_analyzer = ContractAnalyzer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, f"{Path(pdf_name).stem}.json")
            analysis_results = contract_analyzer.analyze_contract(contract_text, output_path)
            
            # Store analysis in database
            analysis_data = {
                'pdf_id': pdf_id,
                'analysis_date': datetime.now(),
                'version': get_next_analysis_version(pdf_id),
                'form_number': analysis_results.get('form_number'),
                'pi_clause': analysis_results.get('pi_clause'),
                'ci_clause': analysis_results.get('ci_clause'),
                'data_usage_mentioned': analysis_results.get('data_usage_mentioned'),
                'data_limitations_exists': analysis_results.get('data_limitations_exists'),
                'summary': analysis_results.get('summary'),
                'raw_json': analysis_results,
                'processed_by': 'batch_analyzer',
                'processing_time': 0.0
            }
            
            analysis_id = store_analysis_data(analysis_data)
            
            # Store clauses
            clauses = analysis_results.get('relevant_clauses', [])
            if clauses:
                clause_data = []
                for i, clause in enumerate(clauses):
                    clause_data.append({
                        'clause_type': clause.get('type', 'unknown'),
                        'clause_text': clause.get('text', ''),
                        'clause_order': i + 1
                    })
                store_clause_data(clause_data, analysis_id)
            
            # Store in session state for immediate display
            file_stem = Path(pdf_name).stem
            st.session_state.json_data[file_stem] = analysis_results
            
            # Create clause mapping
            if analysis_results.get('relevant_clauses') and pdf_name in st.session_state.pages_content:
                clause_mapping = {}
                for i, clause in enumerate(analysis_results['relevant_clauses']):
                    page_num = find_clause_in_pages(clause['text'], st.session_state.pages_content[pdf_name])
                    if page_num:
                        clause_mapping[i] = page_num
                st.session_state.clause_page_mapping[pdf_name] = clause_mapping
            
            return True
            
    except Exception as e:
        print(f"Error analyzing contract for {pdf_name}: {str(e)}")
        return False
```

## **2. Add Database Retrieval Function:**

**Add this function after the batch processing functions:**

```python
def load_analysis_from_database(pdf_name, pdf_id):
    """Load existing analysis from database"""
    try:
        from config.database import get_latest_analysis
        
        analysis_record = get_latest_analysis(pdf_id)
        if analysis_record:
            # Convert database record to display format
            analysis_data = analysis_record.get('raw_json', {})
            if isinstance(analysis_data, str):
                import json
                analysis_data = json.loads(analysis_data)
            
            file_stem = Path(pdf_name).stem
            st.session_state.json_data[file_stem] = analysis_data
            
            # Load clauses and create mapping if pages content exists
            if pdf_name in st.session_state.pages_content and analysis_data.get('relevant_clauses'):
                clause_mapping = {}
                for i, clause in enumerate(analysis_data['relevant_clauses']):
                    page_num = find_clause_in_pages(clause['text'], st.session_state.pages_content[pdf_name])
                    if page_num:
                        clause_mapping[i] = page_num
                st.session_state.clause_page_mapping[pdf_name] = clause_mapping
            
            return True
    except Exception as e:
        print(f"Error loading analysis from database: {str(e)}")
    
    return False
```

## **3. Update Session State:**

**Add to `session_vars` in `initialize_session_state()`:**

```python
session_vars = {
    # ... existing variables ...
    'batch_job_active': False,
    'batch_selected_pdfs': [],
    'batch_results': {},
}
```

## **4. Add Batch Selection UI to Left Pane:**

**Replace the document list section in col1 with:**

```python
# Document list and batch selection
if st.session_state.pdf_files:
    st.subheader("📋 Document Management")
    
    # Batch processing section
    st.markdown("### 🚀 Batch Processing")
    
    # Multi-select for batch processing
    available_pdfs = list(st.session_state.pdf_files.keys())
    unprocessed_pdfs = [pdf for pdf in available_pdfs 
                       if st.session_state.analysis_status.get(pdf) != "Processed"]
    
    if unprocessed_pdfs:
        selected_for_batch = st.multiselect(
            "Select documents for batch processing (max 20):",
            options=unprocessed_pdfs,
            default=st.session_state.batch_selected_pdfs,
            max_selections=20,
            help="Select up to 20 documents to process simultaneously"
        )
        
        st.session_state.batch_selected_pdfs = selected_for_batch
        
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            batch_button_disabled = (len(selected_for_batch) == 0 or 
                                   st.session_state.batch_job_active or
                                   not st.session_state.database_initialized)
            
            if st.button("🚀 Start Batch Processing", 
                        disabled=batch_button_disabled,
                        help="Process all selected documents"):
                if len(selected_for_batch) > 0:
                    st.session_state.batch_job_active = True
                    st.rerun()
        
        with col_batch2:
            if st.session_state.batch_job_active:
                if st.button("⏹️ Cancel Batch"):
                    st.session_state.batch_job_active = False
                    st.rerun()
        
        if len(selected_for_batch) > 0:
            st.info(f"📊 Selected: {len(selected_for_batch)} documents")
    else:
        st.info("All documents are already processed!")
    
    # Batch processing execution
    if st.session_state.batch_job_active and st.session_state.batch_selected_pdfs:
        st.markdown("### ⚡ Processing in Progress...")
        
        # Create batch job
        job_id, batch_db_id = create_batch_job(st.session_state.batch_selected_pdfs, get_session_id())
        
        # Progress container
        progress_container = st.container()
        
        # Run batch processing
        with st.spinner("Running batch analysis..."):
            results = run_batch_processing(
                st.session_state.batch_selected_pdfs, 
                job_id, 
                progress_container
            )
        
        # Display results
        st.success("✅ Batch processing completed!")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        with col_result1:
            st.metric("✅ Processed", len(results['processed']))
        with col_result2:
            st.metric("⏭️ Skipped", len(results['skipped']))
        with col_result3:
            st.metric("❌ Failed", len(results['failed']))
        
        if results['failed']:
            with st.expander("❌ Failed Documents", expanded=False):
                for failed in results['failed']:
                    st.write(f"• **{failed['name']}**: {failed['error']}")
        
        if results['skipped']:
            with st.expander("⏭️ Skipped Documents", expanded=False):
                for skipped in results['skipped']:
                    st.write(f"• **{skipped['name']}**: {skipped['reason']}")
        
        # Store results and reset
        st.session_state.batch_results[job_id] = results
        st.session_state.batch_job_active = False
        st.session_state.batch_selected_pdfs = []
        
        if st.button("🔄 Refresh Document List"):
            st.rerun()
    
    st.markdown("---")
    
    # Individual document selection (existing grid code)
    st.subheader("📄 Individual Selection")
    
    # Create enhanced dataframe (existing code remains the same)
    # ... existing grid code ...
```

## **5. Update PDF Selection Logic:**

**Replace the grid selection handling with:**

```python
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
        if st.session_state.analysis_status.get(selected_pdf) not in ["Processed", "Processed (from database)"]:
            # ... existing processing code ...
```

## **6. Add Batch Job History to Sidebar:**

**Add this to the sidebar after service status:**

```python
# Batch processing history
if st.session_state.batch_results:
    st.markdown("---")
    st.subheader("📊 Recent Batch Jobs")
    
    for job_id, results in list(st.session_state.batch_results.items())[-3:]:  # Show last 3
        with st.expander(f"Job {job_id[:8]}...", expanded=False):
            st.write(f"✅ Processed: {len(results['processed'])}")
            st.write(f"❌ Failed: {len(results['failed'])}")
            st.write(f"⏭️ Skipped: {len(results['skipped'])}")
            if results.get('total_pages_removed', 0) > 0:
                st.write(f"🔒 Pages removed: {results['total_pages_removed']}")
```

These changes will add:
- ✅ **Multi-select interface** for choosing up to 20 documents
- ✅ **Batch processing** with progress tracking
- ✅ **Database job tracking** with batch_jobs table
- ✅ **Automatic analysis retrieval** from database for processed documents
- ✅ **Comprehensive results display** with success/failure metrics
- ✅ **Prevention of duplicate processing** via file hash checks
- ✅ **Progress indicators** and cancellation capability
- ✅ **Batch job history** in sidebar

The system will now efficiently handle bulk document processing while avoiding redundant work by checking the database first!​​​​​​​​​​​​​​​​