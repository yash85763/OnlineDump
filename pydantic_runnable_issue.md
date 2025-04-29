I'll help you implement PDF clause highlighting using PyMuPDF (fitz). This is an excellent choice since PyMuPDF offers better highlighting capabilities than the current solution with iframe search parameters.

Here's how to implement it:

```python
import fitz  # PyMuPDF

# Add this function to highlight clauses in the PDF
def highlight_text_in_pdf(pdf_bytes, search_text, output_filename=None):
    """
    Highlight specific text in a PDF using PyMuPDF and save to a new file
    Returns the bytes of the highlighted PDF
    """
    if not search_text or not pdf_bytes:
        return pdf_bytes
    
    # Load PDF from bytes
    try:
        mem_pdf = BytesIO(pdf_bytes)
        doc = fitz.open(stream=mem_pdf, filetype="pdf")
        
        # Search and highlight text
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_instances = page.search_for(search_text)
            
            # Add highlights for each found text instance
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=(1, 0.8, 0))  # Yellow highlight
                highlight.update()
        
        # Save the highlighted PDF
        output_stream = BytesIO()
        doc.save(output_stream)
        doc.close()
        
        # Return the highlighted PDF bytes
        return output_stream.getvalue()
    except Exception as e:
        st.error(f"Error highlighting PDF: {str(e)}")
        return pdf_bytes
```

Now modify your PDF display code to use the highlighted version when text is being searched. Update the relevant parts of your application:

1. First, add the import at the top:
```python
import fitz  # PyMuPDF
```

2. In the right pane where you handle the "Highlight in PDF" button, modify your code to use the new function:

```python
# Replace the existing button code with this:
if st.button("Highlight in PDF", key=f"highlight_{i}"):
    try:
        # Get current PDF bytes
        current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
        # Create highlighted version
        highlighted_pdf = highlight_text_in_pdf(current_pdf_bytes, clause['text'])
        # Temporarily store the highlighted version
        st.session_state.highlighted_pdf = highlighted_pdf
        st.session_state.highlighted_text = clause['text']
        st.success(f"Highlighted clause {i+1}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to highlight: {str(e)}")
```

3. Then update your PDF display code in the middle panel:

```python
# In the middle pane, modify the PDF display section
if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
    # Check if we have a highlighted version to display
    if hasattr(st.session_state, 'highlighted_pdf') and st.session_state.highlighted_pdf is not None:
        current_pdf_bytes = st.session_state.highlighted_pdf
        st.subheader(f"Viewing: {st.session_state.current_pdf} (Highlighted: {st.session_state.highlighted_text[:30]}...)")
        
        # Add a button to clear highlights
        if st.button("Clear Highlights", key="clear_highlights"):
            st.session_state.highlighted_pdf = None
            st.session_state.highlighted_text = None
            st.rerun()
    else:
        current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
        st.subheader(f"Viewing: {st.session_state.current_pdf}")
    
    try:
        pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        # Existing error handling code...
```

4. Finally, make sure to initialize the new session state variables:

```python
# Add these to your session state initialization
if 'highlighted_pdf' not in st.session_state:
    st.session_state.highlighted_pdf = None
if 'highlighted_text' not in st.session_state:
    st.session_state.highlighted_text = None
```

These changes will:
1. Create a function to highlight text in a PDF using PyMuPDF
2. Add a session state to store the highlighted PDF
3. Modify the display to show the highlighted PDF when available
4. Add a "Clear Highlights" button to revert to the original PDF

The PyMuPDF approach is more reliable than the iframe search parameter because it actually modifies the PDF to include highlight annotations, making the highlights persistent and visible regardless of the PDF viewer used.

Would you like me to explain any part of this implementation in more detail?​​​​​​​​​​​​​​​​