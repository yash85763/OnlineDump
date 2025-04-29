no I'll help you implement PDF clause highlighting using PyMuPDF (fitz). This is an excellent choice since PyMuPDF offers better highlighting capabilities than the current solution with iframe search parameters.

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

Would you like me to explain any part of this implementation in more detail?​​​​​​



I'll update the text search functionality to use fuzzy matching instead of exact matching, which will make it more flexible for finding similar text in the PDF. PyMuPDF doesn't have built-in fuzzy matching, but we can implement it using the `difflib` module, which is part of Python's standard library.

Here's how to modify the highlighting function to use fuzzy matching with a 90% similarity threshold:

```python
import difflib

def highlight_text_in_pdf(pdf_bytes, search_text, color="Yellow", similarity_threshold=0.9, output_filename=None):
    """
    Highlight text in a PDF using PyMuPDF with fuzzy matching.
    Highlights text that has at least 90% similarity to the search text.
    
    Args:
        pdf_bytes: The PDF file as bytes
        search_text: The text to search for
        color: Highlight color name
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0)
        output_filename: Optional filename to save the highlighted PDF
        
    Returns:
        The highlighted PDF as bytes
    """
    if not search_text or not pdf_bytes:
        return pdf_bytes
    
    # Color mapping
    color_map = {
        "Yellow": (1, 1, 0, 0.5),
        "Green": (0, 1, 0, 0.5),
        "Blue": (0, 0.7, 1, 0.5),
        "Pink": (1, 0.7, 0.7, 0.5),
        "Orange": (1, 0.6, 0, 0.5)
    }
    fill_color = color_map.get(color, (1, 1, 0, 0.5))
    
    try:
        # Load PDF from bytes
        mem_pdf = BytesIO(pdf_bytes)
        doc = fitz.open(stream=mem_pdf, filetype="pdf")
        
        # For each page in the PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get all text blocks on the page
            text_blocks = page.get_text("blocks")
            
            for block in text_blocks:
                block_text = block[4]  # Text content is in index 4
                
                # Skip very short blocks
                if len(block_text) < 3:
                    continue
                
                # Use difflib to compute similarity ratio
                similarity = difflib.SequenceMatcher(None, block_text.lower(), search_text.lower()).ratio()
                
                # If the block has high similarity, or contains the search text
                if similarity >= similarity_threshold or search_text.lower() in block_text.lower():
                    # Get the block rectangle coordinates
                    rect = fitz.Rect(block[:4])
                    
                    # Create highlight annotation
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=(0, 0, 0, 0), fill=fill_color)
                    highlight.update()
                    
                    # Also search for exact matches within this block to highlight precisely
                    # This helps when the block contains multiple sentences
                    words = page.search_for(search_text, clip=rect)
                    for word_rect in words:
                        word_highlight = page.add_highlight_annot(word_rect)
                        word_highlight.set_colors(stroke=(0, 0, 0, 0), fill=fill_color)
                        word_highlight.update()
        
        # Save the highlighted PDF
        output_stream = BytesIO()
        doc.save(output_stream)
        doc.close()
        
        return output_stream.getvalue()
    
    except Exception as e:
        st.error(f"Error highlighting PDF: {str(e)}")
        return pdf_bytes
```

This improved function:

1. Uses `difflib.SequenceMatcher` to calculate the similarity ratio between each text block and the search text
2. Highlights blocks that have at least 90% similarity (configurable via the `similarity_threshold` parameter)
3. Also highlights blocks that contain the search text as a substring, even if overall similarity is lower
4. Performs a second pass with exact matching to highlight specific words within matched blocks

When you call this function, you can specify the similarity threshold:

```python
# When highlighting a clause:
highlighted_pdf = highlight_text_in_pdf(
    current_pdf_bytes, 
    clause['text'], 
    color="Yellow", 
    similarity_threshold=0.9
)
```

You can also add a slider in the UI to let users adjust the similarity threshold:

```python
# Add this to the UI where you have the highlight button
similarity_threshold = st.slider(
    "Match precision", 
    min_value=0.7, 
    max_value=1.0, 
    value=0.9, 
    step=0.05,
    key=f"similarity_threshold_{i}",
    help="Lower values find more matches but may include false positives. Higher values require closer matches."
)

# Then pass this value to the highlight function
highlighted_pdf = highlight_text_in_pdf(
    current_pdf_bytes, 
    clause['text'], 
    color=highlight_color,
    similarity_threshold=similarity_threshold
)
```

This fuzzy matching approach will be much more effective at finding relevant text in PDFs, especially when the extracted text might have minor differences from the original PDF content due to extraction errors or formatting issues.​​​​​​​​​​​​​​​​
