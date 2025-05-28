Here are the specific fixes needed for both issues:

## **1. Fix Batch Processing to Include Analysis:**

**Replace the `run_batch_processing` function with:**

```python
def run_batch_processing(selected_pdfs, job_id, progress_container):
    """Run complete batch processing including analysis for selected PDFs"""
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
                # PDF already exists, load existing analysis
                st.session_state.pdf_database_ids[pdf_name] = existing_pdf['id']
                
                # Try to load analysis from database
                if load_analysis_from_database(pdf_name, existing_pdf['id']):
                    results['skipped'].append({'name': pdf_name, 'reason': 'Already processed (loaded from database)'})
                    st.session_state.analysis_status[pdf_name] = "Processed (from database)"
                else:
                    # PDF exists but no analysis - run analysis only
                    status_text.text(f"Analyzing {i+1}/{len(selected_pdfs)}: {pdf_name}")
                    
                    # Get pages content from existing PDF processing
                    result = process_single_pdf_from_streamlit(
                        pdf_name=pdf_name,
                        pdf_bytes=pdf_bytes,
                        enable_obfuscation=True,
                        uploaded_by=f"batch_{job_id}"
                    )
                    
                    if result.get('success'):
                        # Run contract analysis
                        pages_content = result.get('pages', [])
                        st.session_state.pages_content[pdf_name] = pages_content
                        
                        analysis_success = analyze_contract_for_batch(pdf_name, pages_content, existing_pdf['id'])
                        
                        if analysis_success:
                            results['processed'].append(pdf_name)
                            st.session_state.analysis_status[pdf_name] = "Processed"
                        else:
                            results['failed'].append({'name': pdf_name, 'error': 'Analysis failed'})
                            st.session_state.analysis_status[pdf_name] = "Analysis failed"
                    else:
                        results['failed'].append({'name': pdf_name, 'error': 'PDF processing failed'})
            else:
                # Complete processing: PDF + Analysis
                status_text.text(f"Processing & Analyzing {i+1}/{len(selected_pdfs)}: {pdf_name}")
                
                result = process_single_pdf_from_streamlit(
                    pdf_name=pdf_name,
                    pdf_bytes=pdf_bytes,
                    enable_obfuscation=True,
                    uploaded_by=f"batch_{job_id}"
                )
                
                if result.get('success'):
                    st.session_state.pdf_database_ids[pdf_name] = result.get('pdf_id')
                    st.session_state.obfuscation_summaries[pdf_name] = result.get('obfuscation_summary', {})
                    
                    # Store pages content for clause mapping
                    pages_content = result.get('pages', [])
                    st.session_state.pages_content[pdf_name] = pages_content
                    
                    # Run contract analysis
                    analysis_success = analyze_contract_for_batch(pdf_name, pages_content, result.get('pdf_id'))
                    
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
```

## **2. Update `analyze_contract_for_batch` Function:**

**Replace the function with:**

```python
def analyze_contract_for_batch(pdf_name, pages_content, pdf_id):
    """Analyze contract and store results for batch processing"""
    try:
        from contract_analyzer import ContractAnalyzer
        from config.database import store_analysis_data, store_clause_data, get_next_analysis_version
        import tempfile
        
        # Convert pages content to text for analysis
        contract_text = '\n\n'.join([
            para for page in pages_content 
            for para in page.get('paragraphs', [])
        ])
        
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
            
            # Create clause to page mapping using pages_content
            create_clause_page_mapping(pdf_name, analysis_results, pages_content)
            
            return True
            
    except Exception as e:
        print(f"Error analyzing contract for {pdf_name}: {str(e)}")
        return False
```

## **3. Fix Search Text Functionality - Update `find_clause_in_pages`:**

**Replace the `find_clause_in_pages` function with:**

```python
def find_clause_in_pages(clause_text, pages_content):
    """Find which page contains the clause text using exact paragraph matching"""
    if not clause_text or not pages_content:
        return None
    
    # Clean and prepare clause text for matching
    clause_clean = clause_text.lower().strip()
    clause_words = set(clause_clean.split())
    
    best_matches = []
    
    for page in pages_content:
        page_number = page.get('page_number', 0)
        paragraphs = page.get('paragraphs', [])
        
        for paragraph in paragraphs:
            if not paragraph:
                continue
                
            paragraph_clean = paragraph.lower().strip()
            
            # Method 1: Exact substring match (most reliable)
            if len(clause_clean) > 50:  # For longer clauses
                # Check if clause is contained in paragraph or vice versa
                if clause_clean in paragraph_clean or paragraph_clean in clause_clean:
                    return page_number
                
                # Check for significant overlap for long texts
                if len(clause_clean) > 100:
                    overlap_threshold = 0.6  # 60% overlap
                    clause_chunks = [clause_clean[i:i+50] for i in range(0, len(clause_clean), 50)]
                    matches = sum(1 for chunk in clause_chunks if chunk in paragraph_clean)
                    if matches / len(clause_chunks) >= overlap_threshold:
                        return page_number
            
            # Method 2: Word overlap for shorter clauses
            paragraph_words = set(paragraph_clean.split())
            common_words = clause_words.intersection(paragraph_words)
            
            if common_words:
                # Calculate match ratio
                match_ratio = len(common_words) / len(clause_words)
                coverage_ratio = len(common_words) / len(paragraph_words) if paragraph_words else 0
                
                # Higher threshold for better accuracy
                if match_ratio >= 0.7 or (match_ratio >= 0.5 and coverage_ratio >= 0.3):
                    best_matches.append((page_number, match_ratio, len(common_words)))
    
    # Return the page with the best match
    if best_matches:
        best_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by match ratio, then common words
        return best_matches[0][0]
    
    return None

def create_clause_page_mapping(pdf_name, analysis_results, pages_content):
    """Create and store clause to page mapping"""
    if not analysis_results.get('relevant_clauses') or not pages_content:
        return
    
    clause_mapping = {}
    for i, clause in enumerate(analysis_results['relevant_clauses']):
        page_num = find_clause_in_pages(clause['text'], pages_content)
        if page_num:
            clause_mapping[i] = page_num
    
    if clause_mapping:
        st.session_state.clause_page_mapping[pdf_name] = clause_mapping
```

## **4. Update Individual PDF Processing to Store Pages Content:**

**In the individual PDF selection section, after processing, add:**

```python
# In the individual PDF selection handling, after successful processing:
if success:
    st.session_state.analysis_status[selected_pdf] = "Processed"
    
    # Make sure pages content is stored for clause mapping
    if selected_pdf not in st.session_state.pages_content:
        # If not already stored, we need to get it from the result
        # This might require modifying process_pdf_enhanced to return pages_content
        pass
    
    st.success(f"✅ Analysis complete for {selected_pdf}")
```

## **5. Update `process_pdf_enhanced` to Store Pages Content:**

**Add this line in `process_pdf_enhanced` after getting the pages_content:**

```python
# In process_pdf_enhanced function, after getting pages_content from result:
if result.get('success'):
    # ... existing code ...
    
    # Get the processed content for analysis
    pages_content = result.get('pages', [])
    
    # Store pages content for clause mapping - ADD THIS LINE
    st.session_state.pages_content[pdf_name] = pages_content
    
    contract_text = '\n\n'.join([
        para for page in pages_content 
        for para in page.get('paragraphs', [])
    ])
    
    # ... rest of the function
    
    # After storing analysis results, create clause mapping - ADD THIS:
    if analysis_results.get('relevant_clauses'):
        create_clause_page_mapping(pdf_name, analysis_results, pages_content)
```

## **6. Update `load_analysis_from_database` Function:**

**Add pages content loading:**

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
            
            # If we don't have pages_content, we need to re-process the PDF to get it
            if pdf_name not in st.session_state.pages_content:
                # Re-process PDF to get pages content for clause mapping
                pdf_bytes = st.session_state.pdf_files.get(pdf_name)
                if pdf_bytes:
                    from utils.enhanced_pdf_handler import process_single_pdf_from_streamlit
                    result = process_single_pdf_from_streamlit(
                        pdf_name=pdf_name,
                        pdf_bytes=pdf_bytes,
                        enable_obfuscation=True,
                        uploaded_by="reload_for_mapping"
                    )
                    if result.get('success'):
                        st.session_state.pages_content[pdf_name] = result.get('pages', [])
            
            # Create clause mapping if we have pages content
            if pdf_name in st.session_state.pages_content and analysis_data.get('relevant_clauses'):
                create_clause_page_mapping(pdf_name, analysis_data, st.session_state.pages_content[pdf_name])
            
            return True
    except Exception as e:
        print(f"Error loading analysis from database: {str(e)}")
    
    return False
```

These changes will:

✅ **Complete batch processing** - Includes both PDF processing AND contract analysis
✅ **Database-first approach** - Checks existing data before processing
✅ **Accurate clause page detection** - Uses actual page-wise paragraphs from PDF handler
✅ **Better text matching** - Multiple matching algorithms for different clause types
✅ **Proper session state management** - Stores all required data for clause mapping
✅ **Fallback handling** - Re-processes if pages content is missing

The search functionality will now accurately detect which page contains each clause using the actual paragraph structure from your PDF handler!​​​​​​​​​​​​​​​​