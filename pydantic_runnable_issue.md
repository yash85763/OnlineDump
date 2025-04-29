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


Yes, I can definitely help you upgrade the PDF highlighting functionality to use a yellow color for better visibility. The good news is that PyMuPDF already supports colored highlighting, and we just need to modify the parameters in the highlighting function.

Here's how to improve the highlighting code:

```python
def highlight_text_in_pdf(pdf_bytes, search_text, output_filename=None):
    """
    Highlight specific text in a PDF using PyMuPDF with yellow highlighting
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
            
            # Add yellow highlights for each found text instance
            for inst in text_instances:
                # Create highlight annotation with yellow color
                highlight = page.add_highlight_annot(inst)
                # RGB values for yellow (1,1,0), but we can adjust opacity with the 4th value
                highlight.set_colors(stroke=(0, 0, 0, 0), fill=(1, 1, 0, 0.5))
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

The key changes here are:

1. Using `set_colors()` with the appropriate parameters:
   - `stroke=(0, 0, 0, 0)` - Sets the outline to transparent
   - `fill=(1, 1, 0, 0.5)` - Sets the fill color to yellow with 50% opacity

If you want a more vibrant yellow, you can adjust the opacity or use slightly different values. For example:
- Brighter yellow: `fill=(1, 0.9, 0, 0.7)`
- Softer yellow: `fill=(1, 1, 0.7, 0.4)`

You can also add these improvements to make the highlighting more user-friendly:

1. Add an option for the user to select highlight color:
```python
# Add to the right panel where you have the highlight button
highlight_color = st.select_slider(
    "Highlight color",
    options=["Yellow", "Green", "Blue", "Pink", "Orange"],
    value="Yellow",
    key=f"highlight_color_{i}"
)

# Then modify the highlight_text_in_pdf function to accept the color
def highlight_text_in_pdf(pdf_bytes, search_text, color="Yellow", output_filename=None):
    # Color mapping
    color_map = {
        "Yellow": (1, 1, 0, 0.5),      # Yellow
        "Green": (0, 1, 0, 0.5),       # Green
        "Blue": (0, 0.7, 1, 0.5),      # Blue
        "Pink": (1, 0.7, 0.7, 0.5),    # Pink
        "Orange": (1, 0.6, 0, 0.5)     # Orange
    }
    
    fill_color = color_map.get(color, (1, 1, 0, 0.5))  # Default to yellow
    
    # Rest of function as before, but use fill_color in set_colors:
    highlight.set_colors(stroke=(0, 0, 0, 0), fill=fill_color)
```

2. Add a brightness slider for the highlight:
```python
highlight_opacity = st.slider(
    "Highlight brightness", 
    min_value=0.3, 
    max_value=0.8, 
    value=0.5, 
    step=0.1,
    key=f"highlight_opacity_{i}"
)

# Then modify the color to use this opacity:
fill_color = list(color_map.get(color, (1, 1, 0)))
fill_color[3] = highlight_opacity  # Set the alpha/opacity
```



I'll update the contract status display to handle the three-value `pi_clause` field with appropriate color coding. Here's how to modify your code:

```python
# Update the contract status display in the right pane section
# Replace the binary_keys and status button code with this:

# Define color coding for different status values
def get_status_class(key, value):
    """Returns the appropriate CSS class based on the key and value"""
    if key == 'pi_clause':
        if value == 'yes':
            return 'status-button-yes'  # Green
        elif value == 'no':
            return 'status-button-no'   # Red
        else:  # 'missing' or any other value
            return 'status-button-missing'  # Yellow
    else:
        # For binary fields, use the original true/false classes
        return 'status-button-true' if value else 'status-button-false'

# Add these CSS classes to your styles section
st.markdown("""
<style>
    /* Add these new status button classes */
    .status-button-yes {
        background-color: #28a745;  /* Green */
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-no {
        background-color: #dc3545;  /* Red */
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-missing {
        background-color: #ffc107;  /* Yellow */
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    /* Keep your existing status button classes */
    /* ... */
</style>
""", unsafe_allow_html=True)

# Display status fields
st.subheader("Contract Status")
status_fields = {
    'data_usage_mentioned': 'Data Usage Mentioned',
    'data_limitations_exists': 'Data Limitations Exists',
    'pi_clause': 'Presence of PI Clause',
    'ci_clause': 'Presence of CI Clause'
}

for key, label in status_fields.items():
    value = json_data.get(key, False if key != 'pi_clause' else 'missing')
    button_class = get_status_class(key, value)
    
    # Display the status
    st.markdown(f"<div class='{button_class}'>{label}: {value}</div>", 
              unsafe_allow_html=True)
```

This code:

1. Creates a `get_status_class` function that returns different CSS classes based on the field and its value
2. Adds three new CSS classes for the three possible values of `pi_clause`
3. Updates the status fields display to handle both binary values and the three-state `pi_clause`

The key improvements:
- Green background for `pi_clause: yes`
- Red background for `pi_clause: no`
- Yellow background for `pi_clause: missing` (with black text for better readability on yellow)
- Original behavior for binary fields (green for true, red for false)

This approach maintains backward compatibility with your existing binary fields while properly supporting the new three-state field.​​​​​​​​​​​​​​​​
