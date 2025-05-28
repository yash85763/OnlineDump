The console error is likely caused by the JavaScript code in the PDF viewer iframe. Here's how to fix the JavaScript issues:

## **1. Fix the PDF Display Function:**

**Replace the `display_pdf_iframe_with_page` function with:**

```python
def display_pdf_iframe_with_page(pdf_bytes, page_number=None, search_text=None):
    """Display PDF with specific page and optional search text"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    
    # Build PDF URL with parameters
    pdf_url = f'data:application/pdf;base64,{base64_pdf}'
    url_params = []
    
    if page_number and page_number > 1:
        url_params.append(f'page={page_number}')
    
    if search_text:
        # Clean and encode search text properly
        clean_search = sanitize_search_text(search_text)
        if clean_search:
            encoded_search = urllib.parse.quote(clean_search)
            url_params.append(f'search={encoded_search}')
    
    if url_params:
        pdf_url += '#' + '&'.join(url_params)
    
    # Create iframe without JavaScript initially
    iframe_html = f'''
    <iframe id="pdfViewer_{hash(search_text or '')}" 
            src="{pdf_url}" 
            width="100%" 
            height="600px" 
            type="application/pdf"
            style="border: 1px solid #ddd; border-radius: 5px;"
            onload="handlePdfLoad(this)">
    </iframe>
    '''
    
    # Add safer JavaScript
    if search_text:
        clean_search_js = sanitize_search_text(search_text).replace("'", "\\'").replace('"', '\\"')
        iframe_html += f'''
        <script>
        function handlePdfLoad(iframe) {{
            try {{
                // Wait a bit for PDF to load
                setTimeout(function() {{
                    if (iframe && iframe.contentWindow) {{
                        try {{
                            // Try to send search message
                            iframe.contentWindow.postMessage({{
                                type: 'search',
                                query: '{clean_search_js}',
                                find: '{clean_search_js}'
                            }}, '*');
                        }} catch (searchError) {{
                            console.log('PDF search not supported:', searchError.message);
                        }}
                    }}
                }}, 2000);
            }} catch (error) {{
                console.log('PDF viewer interaction not available:', error.message);
            }}
        }}
        
        // Alternative: Try URL fragment approach
        window.addEventListener('load', function() {{
            try {{
                var iframe = document.getElementById('pdfViewer_{hash(search_text or '')}');
                if (iframe) {{
                    var currentSrc = iframe.src;
                    if (currentSrc.indexOf('#search=') === -1 && '{clean_search_js}') {{
                        iframe.src = currentSrc + '#search={urllib.parse.quote(clean_search_js)}';
                    }}
                }}
            }} catch (error) {{
                console.log('URL update failed:', error.message);
            }}
        }});
        </script>
        '''
    
    return iframe_html
```

## **2. Improve the `sanitize_search_text` Function:**

**Replace with a more robust version:**

```python
def sanitize_search_text(text):
    """Clean up text for PDF search and JavaScript safety"""
    if not text:
        return ""
    
    # Limit length
    text = text[:50]
    
    # Remove potentially problematic characters
    import re
    # Keep only alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Escape for JavaScript
    text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    
    return text
```

## **3. Alternative: Simpler PDF Display Without JavaScript:**

**If JavaScript continues to cause issues, use this simpler version:**

```python
def display_pdf_simple(pdf_bytes, page_number=None, search_text=None):
    """Simple PDF display without complex JavaScript"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    
    # Build PDF URL
    pdf_url = f'data:application/pdf;base64,{base64_pdf}'
    
    # Add page parameter if specified
    if page_number and page_number > 1:
        pdf_url += f'#page={page_number}'
    elif search_text:
        # For search, try the search parameter
        clean_search = sanitize_search_text(search_text)
        if clean_search:
            encoded_search = urllib.parse.quote(clean_search)
            pdf_url += f'#search={encoded_search}'
    
    # Simple iframe without JavaScript
    iframe_html = f'''
    <div style="position: relative;">
        <iframe src="{pdf_url}" 
                width="100%" 
                height="600px" 
                type="application/pdf"
                style="border: 1px solid #ddd; border-radius: 5px;">
            <p>Your browser doesn't support PDF viewing. 
               <a href="{pdf_url}" target="_blank">Click here to download the PDF</a>
            </p>
        </iframe>
    </div>
    '''
    
    return iframe_html
```

## **4. Update the PDF Viewer Call:**

**In the middle pane, replace the PDF display section with:**

```python
# PDF display with error handling
current_page = st.session_state.get('current_page_number', 1)

try:
    # Use simple version first to avoid JavaScript errors
    pdf_display = display_pdf_simple(
        current_pdf_bytes, 
        current_page, 
        st.session_state.search_text
    )
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    # Show search info if active
    if st.session_state.search_text:
        st.info(f"🔍 Searching for: '{st.session_state.search_text}' on page {current_page}")
        
        # Clear search button
        if st.button("❌ Clear Search"):
            st.session_state.search_text = None
            st.rerun()
    
    # Page navigation controls
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        if st.session_state.current_pdf in st.session_state.obfuscation_summaries:
            total_pages = st.session_state.obfuscation_summaries[st.session_state.current_pdf].get('total_final_pages', 10)
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
    # Fallback to download option
    st.download_button(
        label="📥 Download PDF to view externally",
        data=current_pdf_bytes,
        file_name=st.session_state.current_pdf,
        mime="application/pdf"
    )
```

## **5. Add Browser Compatibility Check:**

**Add this function to check PDF support:**

```python
def check_pdf_support():
    """Add JavaScript to check PDF support"""
    return '''
    <script>
    function checkPdfSupport() {
        var supportsPdf = false;
        try {
            // Check if browser supports PDF viewing
            var navigator = window.navigator;
            var plugins = navigator.plugins;
            
            // Check for PDF plugin
            for (var i = 0; i < plugins.length; i++) {
                if (plugins[i].name.toLowerCase().indexOf('pdf') !== -1) {
                    supportsPdf = true;
                    break;
                }
            }
            
            // Chrome/Edge usually support PDF natively
            if (navigator.userAgent.indexOf('Chrome') !== -1 || 
                navigator.userAgent.indexOf('Edge') !== -1) {
                supportsPdf = true;
            }
            
        } catch (error) {
            console.log('PDF support check failed:', error);
        }
        
        if (!supportsPdf) {
            console.log('PDF viewing may not be supported in this browser');
        }
        
        return supportsPdf;
    }
    
    // Run check when page loads
    document.addEventListener('DOMContentLoaded', checkPdfSupport);
    </script>
    '''
```

## **6. Debug JavaScript Errors:**

**Add this to identify the specific error:**

```python
# Add to your PDF viewer section temporarily
st.markdown('''
<script>
window.addEventListener('error', function(e) {
    console.log('JavaScript Error Details:');
    console.log('Message:', e.message);
    console.log('Source:', e.filename);
    console.log('Line:', e.lineno);
    console.log('Column:', e.colno);
    console.log('Error object:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.log('Unhandled Promise Rejection:', e.reason);
});
</script>
''', unsafe_allow_html=True)
```

## **7. Quick Fix - Remove All JavaScript:**

**If errors persist, use this completely JavaScript-free version:**

```python
def display_pdf_no_js(pdf_bytes, page_number=None, search_text=None):
    """PDF display without any JavaScript"""
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    
    # Simple URL construction
    pdf_url = f'data:application/pdf;base64,{base64_pdf}'
    if page_number and page_number > 1:
        pdf_url += f'#page={page_number}'
    
    # Pure HTML iframe
    return f'''
    <iframe src="{pdf_url}" 
            width="100%" 
            height="600px" 
            style="border: 1px solid #ddd; border-radius: 5px;">
    </iframe>
    '''
```

The JavaScript errors are likely caused by:
- **Cross-origin restrictions** when trying to access iframe content
- **PDF viewer security** preventing script access
- **Malformed JavaScript** due to special characters in search text

Try the simpler versions first to eliminate the JavaScript errors, then gradually add back functionality if needed.​​​​​​​​​​​​​​​​