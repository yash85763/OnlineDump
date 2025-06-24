Excellent idea! For large PDFs, we can display the content in markdown format with automatic scrolling to clause locations. Here’s how to implement this:

## 1. Update the PDF viewer section in main.py

Replace your PDF viewer section in the middle pane:

```python
# Middle pane: PDF viewer
with col2:
    st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
    st.header("📖 Document Viewer")
    
    if st.session_state.current_pdf:
        if st.session_state.current_pdf in st.session_state.pdf_files:
            # PDF bytes available - show PDF viewer
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader(f"📄 {st.session_state.current_pdf}")
            with col_info2:
                file_size_mb = len(current_pdf_bytes) / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            # Show PDF viewer
            pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        else:
            # Large PDF - show markdown content
            st.subheader(f"📄 {st.session_state.current_pdf}")
            
            # Get file size info
            pdf_id = st.session_state.pdf_database_ids.get(st.session_state.current_pdf)
            if pdf_id:
                try:
                    from config.database import db
                    with db.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT file_size_mb FROM pdfs WHERE id = %s", (pdf_id,))
                            result = cur.fetchone()
                            file_size = f"{result[0]:.2f}" if result else "Unknown"
                except:
                    file_size = "Unknown"
            else:
                file_size = "Unknown"
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"📊 File Size: {file_size} MB")
            with col_info2:
                st.info("📖 Markdown View")
            
            # Display PDF content in markdown format
            display_pdf_as_markdown(st.session_state.current_pdf)
    else:
        st.info("Select a PDF to view")
    
    st.markdown('</div>', unsafe_allow_html=True)
```

## 2. Add markdown display function

Add this function to your `main.py`:

```python
def display_pdf_as_markdown(pdf_name):
    """Display PDF content as markdown with scrollable sections"""
    
    # Get content from pages_content
    if pdf_name in st.session_state.get('pages_content', {}):
        pages_content = st.session_state.pages_content[pdf_name]
        
        if not pages_content:
            st.warning("No content available for markdown display")
            return
        
        # Add search functionality
        search_term = st.text_input("🔍 Search in document", 
                                   placeholder="Enter text to search...",
                                   key=f"search_{pdf_name}")
        
        # Auto-scroll functionality
        scroll_to_clause = st.session_state.get('scroll_to_clause')
        
        if scroll_to_clause:
            st.info(f"🎯 Scrolling to: {scroll_to_clause[:50]}...")
        
        # Create markdown content with anchors
        markdown_content = generate_markdown_with_anchors(pages_content, search_term, scroll_to_clause)
        
        # Display in a scrollable container
        st.markdown("""
        <style>
        .markdown-content {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            background-color: white;
            font-family: 'Georgia', serif;
            line-height: 1.6;
        }
        
        .page-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            font-weight: bold;
        }
        
        .paragraph {
            margin-bottom: 1rem;
            text-align: justify;
        }
        
        .search-highlight {
            background-color: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .clause-highlight {
            background-color: #4caf50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 2px 0;
            display: inline-block;
        }
        
        .scroll-target {
            scroll-margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the markdown content
        st.markdown(f'<div class="markdown-content">{markdown_content}</div>', 
                   unsafe_allow_html=True)
        
        # Clear scroll target after displaying
        if scroll_to_clause:
            st.session_state.scroll_to_clause = None
            
    else:
        st.warning("PDF content not available. Please reload the PDF.")

def generate_markdown_with_anchors(pages_content, search_term=None, scroll_target=None):
    """Generate markdown content with search highlights and scroll anchors"""
    
    markdown_parts = []
    
    for page_idx, page in enumerate(pages_content):
        page_number = page.get('page_number', page_idx + 1)
        paragraphs = page.get('paragraphs', [])
        
        # Add page header
        markdown_parts.append(f'<div class="page-header">📄 Page {page_number}</div>')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Create unique anchor for this paragraph
            anchor_id = f"page_{page_number}_para_{para_idx}"
            
            # Apply search highlighting
            display_text = paragraph
            if search_term and search_term.strip():
                import re
                pattern = re.compile(re.escape(search_term.strip()), re.IGNORECASE)
                display_text = pattern.sub(
                    lambda m: f'<span class="search-highlight">{m.group()}</span>', 
                    display_text
                )
            
            # Apply clause highlighting if this is the scroll target
            css_class = "paragraph"
            if scroll_target and scroll_target.strip().lower() in paragraph.lower():
                display_text = f'<span class="clause-highlight">🎯 TARGET CLAUSE</span><br>{display_text}'
                css_class += " scroll-target"
                # Add JavaScript to scroll to this element
                markdown_parts.append(f'''
                <script>
                setTimeout(function() {{
                    document.getElementById('{anchor_id}').scrollIntoView({{
                        behavior: 'smooth',
                        block: 'center'
                    }});
                }}, 500);
                </script>
                ''')
            
            # Add paragraph with anchor
            markdown_parts.append(f'''
            <div id="{anchor_id}" class="{css_class}">
                {display_text}
            </div>
            ''')
    
    return '\n'.join(markdown_parts)
```

## 3. Add clause scroll functionality

Update your clause display in the right pane to enable scrolling:

```python
# In your right pane clause display section:
if clauses:
    clause_mapping = st.session_state.get('clause_page_mapping', {}).get(st.session_state.current_pdf, {})
    
    for i, clause in enumerate(clauses):
        clause_page = clause_mapping.get(i)
        page_info = f" (Page {clause_page})" if clause_page else ""
        
        with st.expander(f"Clause {i+1}: {clause.get('type', 'Unknown').capitalize()}{page_info}", expanded=False):
            if clause_page:
                st.markdown(f"**Found on page:** {clause_page}")
            else:
                st.markdown("Page location not found")
            
            # Display clause text
            clause_text = clause.get('text', 'No text available')
            st.markdown(f"**Text:** {clause_text}")
            
            # NEW: Add scroll to clause button for large PDFs
            if (st.session_state.current_pdf and 
                st.session_state.current_pdf not in st.session_state.pdf_files):
                
                if st.button(f"📍 Scroll to Clause {i+1}", key=f"scroll_clause_{i}"):
                    # Set scroll target and rerun to trigger scroll
                    st.session_state.scroll_to_clause = clause_text[:100]  # First 100 chars for matching
                    st.rerun()
```

## 4. Add content extraction for large PDFs

Update your loading function to ensure content is available for markdown display:

```python
def load_pdf_from_database_with_analysis(pdf_id, pdf_name):
    try:
        show_processing_overlay(
            message=f"Loading {pdf_name}",
            submessage="Retrieving PDF and analysis data from database..."
        )
        
        complete_data = load_complete_pdf_data(pdf_id)
        
        if not complete_data:
            st.error("Could not load PDF data")
            return False
        
        # Check if PDF bytes are available
        pdf_bytes = complete_data.get('pdf_bytes')
        file_size_mb = complete_data.get('file_size_mb', 0)
        
        if pdf_bytes:
            # Small PDF - load bytes for PDF viewer
            st.session_state.pdf_files[pdf_name] = bytes(pdf_bytes)
            st.session_state.loaded_pdfs.add(pdf_name)
            st.session_state.current_pdf = pdf_name
            st.success(f"✅ Loaded {pdf_name} ({file_size_mb:.2f} MB) - PDF viewer available")
        else:
            # Large PDF - no bytes, but ensure content is available for markdown
            st.session_state.loaded_pdfs.add(pdf_name)
            st.session_state.current_pdf = pdf_name
            st.success(f"✅ Loaded {pdf_name} ({file_size_mb:.2f} MB) - Markdown viewer available")
        
        st.session_state.pdf_database_ids[pdf_name] = complete_data["id"]
        
        # Load analysis data
        analyses = complete_data.get('analyses', [])
        if analyses and len(analyses) > 0:
            latest_analysis = analyses[0]
            analysis_data = latest_analysis.get('raw_json', {})
            
            if isinstance(analysis_data, str):
                try:
                    analysis_data = json.loads(analysis_data)
                except json.JSONDecodeError:
                    st.warning("Could not parse analysis data")
                    analysis_data = {}
            
            file_stem = Path(pdf_name).stem
            st.session_state.json_data[file_stem] = analysis_data
            st.session_state.analysis_status[pdf_name] = "Processed"
        else:
            st.session_state.analysis_status[pdf_name] = "Parsed Only"
        
        # Load PDF parsing data for both page navigation AND markdown display
        pdf_parsing_data = complete_data.get('pdf_parsing_data', {})
        if pdf_parsing_data:
            if isinstance(pdf_parsing_data, str):
                try:
                    pdf_parsing_data = json.loads(pdf_parsing_data)
                except json.JSONDecodeError:
                    st.warning("Could not parse PDF parsing data")
                    pdf_parsing_data = {}
            
            # Load original pages for page matching AND markdown display
            original_pages = pdf_parsing_data.get('original_pages', [])
            if original_pages:
                st.session_state.raw_pdf_data[pdf_name] = {'pages': original_pages}
                st.session_state.pages_content[pdf_name] = original_pages  # For markdown display
            else:
                st.warning("No pages data found")
                st.session_state.raw_pdf_data[pdf_name] = {'pages': []}
                st.session_state.pages_content[pdf_name] = []
        else:
            st.warning("No PDF parsing data found")
            st.session_state.raw_pdf_data[pdf_name] = {'pages': []}
            st.session_state.pages_content[pdf_name] = []
        
        return True
        
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return False
    finally:
        hide_processing_overlay()
```

## 5. Add scroll state to session state

Update your `initialize_session_state()` function:

```python
session_vars = {
    'pdf_files': {},
    'json_data': {},
    'raw_pdf_data': {},
    'pages_content': {},
    'clause_page_mapping': {},
    'scroll_to_clause': None,  # NEW: For clause scrolling
    # ... rest of your existing variables
}
```

## Features this implementation provides:

1. **Markdown Display**: Large PDFs show content in formatted text
1. **Search Functionality**: Search within the markdown content
1. **Auto-Scroll**: Click “Scroll to Clause” buttons to jump to specific clauses
1. **Page Headers**: Clear page separators in the markdown
1. **Highlighting**: Search terms and target clauses are highlighted
1. **Scrollable Container**: Fixed height with smooth scrolling

The markdown view provides a much better user experience for large PDFs while maintaining all the analysis and navigation functionality!​​​​​​​​​​​​​​​​