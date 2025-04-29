The **Contract Analysis Viewer** application does not use a dedicated PDF viewer library. Instead, it displays PDFs in the Streamlit interface using **browser-native PDF rendering** capabilities through HTML elements embedded via Streamlit's `st.markdown` function. Specifically, it employs two methods to display PDFs:

1. **Primary Method: iframe with Base64 Encoding**
   - The PDF is encoded as a Base64 string and embedded in an `<iframe>` element using a data URL.
   - This leverages the browser's built-in PDF viewer (e.g., Chrome, Firefox, or Edge's native PDF rendering).
   - Code in `streamlit.py`:
     ```python
     def display_pdf_iframe(pdf_bytes, search_text=None):
         base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
         pdf_display = f'<iframe id="pdfViewer" src="data:application/pdf;base64,{base64_pdf}'
         if search_text:
             sanitized_text = sanitize_search_text(search_text)
             encoded_text = urllib.parse.quote(sanitized_text)
             pdf_display += f'#search={encoded_text}'
         pdf_display += '" width="100%" height="600px" type="application/pdf"></iframe>'
         # JavaScript for search functionality
         if search_text:
             js_script = f"""
             <script>
                 document.getElementById('pdfViewer').addEventListener('load', function() {{
                     try {{
                         this.contentWindow.postMessage({{
                             type: 'search',
                             query: '{sanitize_search_text(search_text)}'
                         }}, '*');
                     }} catch (e) {{
                         console.log('Error triggering PDF search:', e);
                     }}
                 }});
             </script>
             """
             pdf_display += js_script
         return pdf_display
     ```
   - **How it works**:
     - `pdf_bytes` (raw PDF data) is encoded to Base64 using `base64.b64encode`.
     - The Base64 string is embedded in a data URL: `data:application/pdf;base64,{base64_pdf}`.
     - The `<iframe>` uses this URL as its `src`, instructing the browser to render the PDF.
     - If `search_text` is provided, it appends `#search={encoded_text}` to the URL and includes JavaScript to trigger text highlighting in the PDF.
   - **Usage**:
     ```python
     pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
     st.markdown(pdf_display, unsafe_allow_html=True)
     ```

2. **Fallback Method: object Tag with Base64 Encoding**
   - If the `<iframe>` method fails (e.g., due to browser compatibility or rendering issues), the application falls back to an `<object>` tag with the same Base64-encoded data URL.
   - Code in `streamlit.py`:
     ```python
     def display_pdf_object(pdf_bytes):
         base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
         return f'<object data="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width="100%" height="600px"></object>'
     ```
   - **How it works**:
     - Similar to the `<iframe>`, the PDF is Base64-encoded and embedded in a data URL.
     - The `<object>` tag is used to render the PDF, relying on the browser’s PDF viewer or plugins.
   - **Usage**:
     ```python
     try:
         pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
         st.markdown(pdf_display, unsafe_allow_html=True)
     except Exception as e:
         st.error(f"Error with iframe: {e}")
         try:
             pdf_display = display_pdf_object(current_pdf_bytes)
             st.markdown(pdf_display, unsafe_allow_html=True)
         except Exception as e:
             st.error(f"Error with object tag: {e}")
     ```

3. **Error Handling and Fallback**:
   - If both `<iframe>` and `<object>` fail, the application:
     - Validates the PDF using `validate_pdf` (via `PyPDF2`).
     - Displays validation errors or metadata.
     - Offers a download button to access the PDF directly:
       ```python
       st.download_button(
           label="Download PDF",
           data=current_pdf_bytes,
           file_name=st.session_state.current_pdf,
           mime="application/pdf",
           key="download_pdf"
       )
       ```
   - Large PDFs (>1.5MB) trigger a warning due to potential Base64 encoding limitations (noted April 24, 2025):
     ```python
     if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
         st.warning(f"{pdf_name} is larger than 1.5MB and may fail to display.")
     ```

### Dependencies Involved
- **Standard Python Libraries**:
  - `base64`: For encoding PDF bytes to Base64.
  - `urllib.parse`: For URL-encoding search text.
- **External Libraries**:
  - `streamlit`: For rendering HTML via `st.markdown(..., unsafe_allow_html=True)`.
  - `PyPDF2`: For validating PDFs (`validate_pdf` function), not for rendering.
- **No Dedicated PDF Viewer Library**:
  - The application relies on the browser’s native PDF rendering capabilities, not an external library like `pdf.js` or `PyMuPDF`.

### How PDFs Are Displayed
- **Process**:
  1. The PDF file is read as bytes (e.g., from `st.session_state.pdf_files` or uploaded files).
  2. Bytes are encoded to Base64 and embedded in a data URL.
  3. The data URL is set as the `src` of an `<iframe>` or `data` attribute of an `<object>` tag.
  4. Streamlit renders the HTML using `st.markdown`, and the browser displays the PDF.
  5. Search/highlight functionality is supported via URL parameters (`#search=`) and JavaScript postMessage.
- **Browser Dependency**:
  - Modern browsers (Chrome, Firefox, Edge, Safari) have built-in PDF viewers that handle the `data:application/pdf;base64,...` URL.
  - Compatibility may vary with older browsers or those with disabled PDF rendering.

### Limitations
- **Large PDFs**: Files >1.5MB may fail to render due to Base64 encoding or browser memory limits. Users are warned, and a download option is provided.
- **Search Functionality**: Limited to 100 characters and basic sanitization (`sanitize_search_text`), with potential inconsistencies across browsers.
- **OCR’d PDFs**: Rendering is unaffected, but text extraction depends on `PDFTextProcessor` (noted April 14, 2025).
- **Browser Variability**: Some browsers may require plugins or settings to render PDFs inline.

### Troubleshooting
- **PDF Not Displaying**:
  - Check logs for iframe/object errors:
    ```python
    st.error(f"Error with iframe: {e}")
    st.error(f"Error with object tag: {e}")
    ```
  - Verify PDF validity:
    ```bash
    file contract1.pdf  # Should show PDF document
    ```
  - Test with a smaller PDF (<1.5MB) or compress:
    ```bash
    qpdf --stream-data=compress input.pdf output.pdf
    ```
- **Search Not Working**:
  - Ensure `search_text` is short and sanitized:
    ```python
    text = text[:100]
    text = re.sub(r'[^\w\s.]', '', text)
    ```
  - Test in a different browser (e.g., Chrome vs. Firefox).
- **Browser Issues**:
  - Update the browser or enable PDF rendering in settings.
  - Use the download button as a fallback.

If you need to integrate a specific PDF viewer library (e.g., `pdf.js` for more control), let me know, and I can provide a modified version of the code!
