The issue is that `st.session_state.clause_page_mapping` is empty because the clause mapping is not being created correctly. Here are the specific fixes:

## **1. Fix: Ensure Clause Mapping is Created During Processing**

**In the `process_pdf_enhanced()` function, after successful analysis, add:**

```python
# After storing analysis results in session state
file_stem = Path(pdf_name).stem
st.session_state.json_data[file_stem] = analysis_results

# FIX: Create clause mapping immediately after analysis
pages_content = result.get('pages', [])
if pages_content and analysis_results.get('relevant_clauses'):
    st.session_state.pages_content[pdf_name] = pages_content
    
    # Create clause to page mapping
    clause_mapping = {}
    for i, clause in enumerate(analysis_results['relevant_clauses']):
        page_num = find_clause_in_pages(clause['text'], pages_content)
        if page_num:
            clause_mapping[i] = page_num
        else:
            # Debug: Try with fewer words
            for word_count in [3, 2, 1]:
                page_num = find_clause_in_pages_with_word_count(clause['text'], pages_content, word_count)
                if page_num:
                    clause_mapping[i] = page_num
                    break
    
    st.session_state.clause_page_mapping[pdf_name] = clause_mapping
    st.session_state.processing_messages[pdf_name].append(f"📍 Mapped {len(clause_mapping)} clauses to pages")
```

## **2. Add Helper Function for Different Word Counts:**

**Add this function after `find_clause_in_pages`:**

```python
def find_clause_in_pages_with_word_count(clause_text, pages_content, max_words=4):
    """Find clause with specific word count"""
    if not clause_text or not pages_content:
        return None
    
    clause_words = clause_text.strip().split()[:max_words]
    if not clause_words:
        return None
    
    search_phrase = ' '.join(clause_words).lower()
    
    for page in pages_content:
        page_number = page.get('page_number', 0)
        page_text = ' '.join(page.get('paragraphs', [])).lower()
        
        if search_phrase in page_text:
            return page_number
    
    return None
```

## **3. Fix: Create Mapping When PDF is Selected (Fallback)**

**In the PDF selection logic, add clause mapping creation:**

```python
# Handle PDF selection
selected_rows = grid_response.get('selected_rows', pd.DataFrame())
if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_pdf = selected_rows.iloc[0]['PDF Name']
    if selected_pdf != st.session_state.get('current_pdf'):
        set_current_pdf(selected_pdf)
        
        # FIX: Create clause mapping if it doesn't exist
        file_stem = Path(selected_pdf).stem
        if (file_stem in st.session_state.json_data and 
            selected_pdf not in st.session_state.clause_page_mapping and
            selected_pdf in st.session_state.pages_content):
            
            # Create missing clause mapping
            json_data = st.session_state.json_data[file_stem]
            pages_content = st.session_state.pages_content[selected_pdf]
            
            if json_data.get('relevant_clauses'):
                clause_mapping = {}
                for i, clause in enumerate(json_data['relevant_clauses']):
                    page_num = find_clause_in_pages(clause['text'], pages_content)
                    if page_num:
                        clause_mapping[i] = page_num
                
                st.session_state.clause_page_mapping[selected_pdf] = clause_mapping
                st.success(f"✅ Created clause mapping: {len(clause_mapping)} clauses mapped")
        
        # ... rest of existing processing code ...
```

## **4. Fix: Ensure Pages Content is Always Stored**

**In the `analyze_contract_for_batch()` function, make sure pages content is stored:**

```python
def analyze_contract_for_batch(pdf_name, contract_text, pdf_id):
    # ... existing code ...
    
    # Store in session state for immediate display
    file_stem = Path(pdf_name).stem
    st.session_state.json_data[file_stem] = analysis_results
    
    # FIX: Ensure pages content exists before creating clause mapping
    if pdf_name not in st.session_state.pages_content:
        # Try to get pages content from the result that called this function
        # This should be passed as a parameter or retrieved from session state
        st.warning(f"Pages content missing for {pdf_name} - clause mapping skipped")
        return True
    
    # Create clause mapping
    pages_content = st.session_state.pages_content[pdf_name]
    if analysis_results.get('relevant_clauses'):
        clause_mapping = {}
        for i, clause in enumerate(analysis_results['relevant_clauses']):
            page_num = find_clause_in_pages(clause['text'], pages_content)
            if page_num:
                clause_mapping[i] = page_num
        
        st.session_state.clause_page_mapping[pdf_name] = clause_mapping
    
    return True
```

## **5. Add Debug Information to See What's Missing:**

**Add this debug section in your main function where clauses are displayed:**

```python
# DEBUG: Add this before the quick navigation section
if st.session_state.current_pdf:
    with st.expander("🔧 Debug Clause Mapping", expanded=False):
        pdf_name = st.session_state.current_pdf
        file_stem = Path(pdf_name).stem
        
        st.write(f"**Current PDF:** {pdf_name}")
        st.write(f"**Has JSON data:** {file_stem in st.session_state.json_data}")
        st.write(f"**Has pages content:** {pdf_name in st.session_state.pages_content}")
        st.write(f"**Has clause mapping:** {pdf_name in st.session_state.clause_page_mapping}")
        
        if file_stem in st.session_state.json_data:
            clauses = st.session_state.json_data[file_stem].get('relevant_clauses', [])
            st.write(f"**Number of clauses:** {len(clauses)}")
        
        if pdf_name in st.session_state.pages_content:
            pages = st.session_state.pages_content[pdf_name]
            st.write(f"**Number of pages:** {len(pages)}")
        
        clause_mapping = st.session_state.clause_page_mapping.get(pdf_name, {})
        st.write(f"**Clause mapping:** {clause_mapping}")
        
        # Manual trigger to create mapping
        if st.button("🔧 Create Clause Mapping Now"):
            if (file_stem in st.session_state.json_data and 
                pdf_name in st.session_state.pages_content):
                
                json_data = st.session_state.json_data[file_stem]
                pages_content = st.session_state.pages_content[pdf_name]
                
                clause_mapping = {}
                for i, clause in enumerate(json_data.get('relevant_clauses', [])):
                    page_num = find_clause_in_pages(clause['text'], pages_content)
                    st.write(f"Clause {i}: '{clause['text'][:50]}...' → Page {page_num}")
                    if page_num:
                        clause_mapping[i] = page_num
                
                st.session_state.clause_page_mapping[pdf_name] = clause_mapping
                st.success(f"Created mapping with {len(clause_mapping)} clauses")
                st.rerun()
```

## **6. Quick Fix: Force Create Mapping in Main Function**

**Add this right before the quick navigation section:**

```python
# FORCE CREATE clause mapping if missing
if (st.session_state.current_pdf and 
    file_stem in st.session_state.json_data and
    st.session_state.current_pdf not in st.session_state.clause_page_mapping):
    
    pdf_name = st.session_state.current_pdf
    json_data = st.session_state.json_data[file_stem]
    
    # Try to get or create pages content
    if pdf_name not in st.session_state.pages_content:
        st.warning("⚠️ Pages content missing - clause navigation unavailable")
    else:
        pages_content = st.session_state.pages_content[pdf_name]
        clause_mapping = {}
        
        for i, clause in enumerate(json_data.get('relevant_clauses', [])):
            page_num = find_clause_in_pages(clause['text'], pages_content)
            if page_num:
                clause_mapping[i] = page_num
        
        st.session_state.clause_page_mapping[pdf_name] = clause_mapping
```

The main issue is that the clause mapping creation is happening in `analyze_contract_for_batch()` but the pages content might not be available there. The fixes above ensure the mapping is created at multiple points where both the analysis results and pages content are available.

Run with the debug section first to see exactly what's missing!​​​​​​​​​​​​​​​​