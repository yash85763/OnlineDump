Here are the specific changes needed to add content highlighting and automatic page navigation:

## **1. Add New Functions for Page Detection:**

**Add these functions after the `sanitize_search_text` function:**

```python
def find_clause_in_pages(clause_text, pages_content):
    """Find which page contains the clause text"""
    clause_words = set(clause_text.lower().split())
    best_matches = []
    
    for page in pages_content:
        page_number = page.get('page_number', 0)
        page_text = ' '.join(page.get('paragraphs', [])).lower()
        page_words = set(page_text.split())
        
        # Calculate word overlap
        common_words = clause_words.intersection(page_words)
        if common_words:
            match_ratio = len(common_words) / len(clause_words)
            if match_ratio > 0.3:  # At least 30% word overlap
                best_matches.append((page_number, match_ratio, page_text))
    
    # Sort by match ratio and return best match
    if best_matches:
        best_matches.sort(key=lambda x: x[1], reverse=True)
        return best_matches[0][0]  # Return page number
    return None

def create_pdf_url_with_page(pdf_bytes, page_number=None, search_text=None):
    """Create PDF URL with specific page and search parameters"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_url = f'data:application/pdf;base64,{base64_pdf}'
    
    params = []
    if page_number:
        params.append(f'page={page_number}')
    if search_text:
        sanitized_text = sanitize_search_text(search_text)
        encoded_text = urllib.parse.quote(sanitized_text)
        params.append(f'search={encoded_text}')
    
    if params:
        pdf_url += '#' + '&'.join(params)
    
    return pdf_url
```

## **2. Update Session State Initialization:**

**Add to the `session_vars` dictionary in `initialize_session_state()`:**

```python
session_vars = {
    # ... existing variables ...
    'current_page_number': 1,  # Add this
    'clause_page_mapping': {},  # Add this - stores clause to page mapping
    'pages_content': {},  # Add this - stores pages content per PDF
}
```

## **3. Modify the PDF Processing Function:**

**In `process_pdf_enhanced()`, after storing the analysis results, add:**

```python
# Store pages content for clause mapping
st.session_state.pages_content[pdf_name] = pages_content

# Create clause to page mapping
if analysis_results.get('relevant_clauses'):
    clause_mapping = {}
    for i, clause in enumerate(analysis_results['relevant_clauses']):
        page_num = find_clause_in_pages(clause['text'], pages_content)
        if page_num:
            clause_mapping[i] = page_num
    st.session_state.clause_page_mapping[pdf_name] = clause_mapping
```

## **4. Update the PDF Viewer Display Function:**

**Replace the `display_pdf_iframe` function with:**

```python
def display_pdf_iframe_with_page(pdf_bytes, page_number=None, search_text=None):
    """Display PDF with specific page and optional search text"""
    pdf_url = create_pdf_url_with_page(pdf_bytes, page_number, search_text)
    
    iframe_html = f'''
    <iframe id="pdfViewer" 
            src="{pdf_url}" 
            width="100%" 
            height="600px" 
            type="application/pdf"
            style="border: 1px solid #ddd; border-radius: 5px;">
    </iframe>
    '''
    
    if search_text:
        iframe_html += f'''
        <script>
            setTimeout(function() {{
                const iframe = document.getElementById('pdfViewer');
                if (iframe && iframe.contentWindow) {{
                    try {{
                        iframe.contentWindow.postMessage({{
                            type: 'search',
                            query: '{sanitize_search_text(search_text)}'
                        }}, '*');
                    }} catch (e) {{
                        console.log('Search message failed:', e);
                    }}
                }}
            }}, 1000);
        </script>
        '''
    
    return iframe_html
```

## **5. Update the PDF Display Section:**

**In the middle pane (col2), replace the PDF display try block with:**

```python
# PDF display with page navigation
current_page = st.session_state.get('current_page_number', 1)

try:
    pdf_display = display_pdf_iframe_with_page(
        current_pdf_bytes, 
        current_page, 
        st.session_state.search_text
    )
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    # Page navigation controls
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
            total_pages = st.session_state.obfuscation_summaries[st.session_state.current_pdf].get('total_final_pages', 1)
        else:
            total_pages = 10  # Default fallback
            
        new_page = st.number_input(
            "Go to page:", 
            min_value=1, 
            max_value=total_pages,
            value=current_page,
            key=f"page_nav_{st.session_state.current_pdf}"
        )
        
        if new_page != current_page:
            st.session_state.current_page_number = new_page
            st.rerun()

except Exception as e:
    st.error(f"❌ Error displaying PDF: {e}")
    # ... rest of error handling stays the same
```

## **6. Enhance the Clause Display Section:**

**Replace the clause expander section in the right pane with:**

```python
# Relevant Clauses with enhanced navigation
st.markdown("### 📄 Relevant Clauses")
clauses = json_data.get("relevant_clauses", [])

if clauses:
    clause_mapping = st.session_state.clause_page_mapping.get(st.session_state.current_pdf, {})
    
    for i, clause in enumerate(clauses):
        # Get page number for this clause
        clause_page = clause_mapping.get(i)
        page_info = f" (Page {clause_page})" if clause_page else ""
        
        with st.expander(f"📑 Clause {i+1}: {clause['type'].capitalize()}{page_info}", expanded=False):
            st.markdown(f"**Type:** `{clause['type']}`")
            if clause_page:
                st.markdown(f"**Found on Page:** {clause_page}")
            
            st.markdown(f"**Content:**")
            st.markdown(f"<div class='extract-text'>{clause['text']}</div>", unsafe_allow_html=True)
            
            # Enhanced search functionality
            col_search1, col_search2 = st.columns(2)
            with col_search1:
                if st.button(f"🔍 Search Text", key=f"search_clause_{i}"):
                    st.session_state.search_text = clause['text'][:100]
                    st.success(f"Searching for clause {i+1}...")
                    st.rerun()
            
            with col_search2:
                if clause_page and st.button(f"📄 Go to Page {clause_page}", key=f"goto_page_{i}"):
                    st.session_state.current_page_number = clause_page
                    st.session_state.search_text = clause['text'][:50]  # Also search
                    st.success(f"Navigating to page {clause_page}...")
                    st.rerun()
            
            if not clause_page:
                st.info("💡 Page location not detected for this clause")
            
            if len(clause['text']) > 100:
                st.caption("⚠️ Long text may not highlight fully")
else:
    st.info("No relevant clauses detected in this contract.")
```

## **7. Add Page Reset on PDF Change:**

**In the `set_current_pdf` function, add:**

```python
def set_current_pdf(pdf_name):
    """Set the current PDF to display"""
    st.session_state.current_pdf = pdf_name
    st.session_state.search_text = None
    st.session_state.current_page_number = 1  # Add this line
```

## **8. Optional: Add Clause Summary with Page Numbers:**

**Add this section after the contract status in the right pane:**

```python
# Quick clause navigation
if clauses and st.session_state.current_pdf in st.session_state.clause_page_mapping:
    clause_mapping = st.session_state.clause_page_mapping[st.session_state.current_pdf]
    if clause_mapping:
        st.markdown("### 🚀 Quick Navigation")
        cols = st.columns(min(len(clause_mapping), 4))
        
        for idx, (clause_idx, page_num) in enumerate(clause_mapping.items()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                clause_type = clauses[clause_idx]['type'].capitalize()
                if st.button(f"📄 {clause_type}\nPage {page_num}", key=f"quick_nav_{clause_idx}"):
                    st.session_state.current_page_number = page_num
                    st.session_state.search_text = clauses[clause_idx]['text'][:50]
                    st.rerun()
```

These changes will:
- ✅ **Detect which page** each clause appears on
- ✅ **Automatically navigate** to the correct page when user clicks "Go to Page"
- ✅ **Highlight the clause text** in the PDF viewer
- ✅ **Show page numbers** in clause titles
- ✅ **Add quick navigation** buttons for rapid access
- ✅ **Maintain page state** when switching between PDFs
- ✅ **Provide visual feedback** when navigating

The system will now intelligently match clauses to pages and provide seamless navigation between the analysis results and the corresponding PDF content!​​​​​​​​​​​​​​​​