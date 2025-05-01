I'll update the `streamlit.py` code to implement the requested changes to the Contract Analysis Viewer UI. The changes are:

1. **Replace Individual PDF Buttons with a Single-Column Table**:
   - Remove the individual buttons under "Available PDFs" in the left pane.
   - Replace them with a single-column table displaying the names of uploaded and pre-loaded PDFs.
   - Make each table cell clickable, triggering the selection of the corresponding PDF for viewing and analysis.

2. **Remove the "Analyze PDFs" Button**:
   - Remove the "Analyze PDFs" button.
   - Automatically analyze the PDF when its name is clicked in the table, if it hasn't been analyzed yet (i.e., `analysis_status` is not "Processed").

These changes will streamline the UI by consolidating PDF selection and analysis into a single action (clicking a table cell). All existing functionality will be preserved, including:
- Scrollable panes, processing messages ("Text extracted from PDF", "Analyzing the document"), pre-loaded PDFs/JSONs, folder creation, fallback directory search, and `pi_clause` styling (green for "True"/"yes"/"YES"/"Yes", red for "False"/"no"/"No"/"NO", yellow for "missing"/"Missing"/"MISSING"/"Absent").
- The table will be styled to fit within the scrollable left pane (`.left-pane`, `height: 85vh`, `overflow-y: auto`).

### Approach
1. **Remove PDF Buttons and Analyze Button**:
   - Delete the button loop under "Available PDFs" and the "Analyze PDFs" button in the left pane.
2. **Add Single-Column Table**:
   - Use Streamlit’s `st.dataframe` or HTML table (via `st.markdown`) to display PDF names.
   - Make cells clickable using Streamlit’s `st.button` within a table or JavaScript events in HTML.
   - On click, set the selected PDF as `st.session_state.current_pdf` and trigger analysis if needed.
3. **Automatic Analysis on Click**:
   - Check `st.session_state.analysis_status[pdf_name]` when a cell is clicked.
   - If not "Processed", call `process_pdf` to analyze the PDF, showing processing messages under a spinner.
4. **Styling**:
   - Style the table to match the UI (bordered, clickable cells, scrollable within `.left-pane`).
   - Ensure clickable cells are visually distinct (e.g., hover effects, selected state).

### Assumptions
- PDFs are stored in `st.session_state.pdf_files` (name: bytes) and `st.session_state.analysis_status` (name: status).
- Analysis results are stored in `st.session_state.json_data` (stem: JSON).
- The `PDFTextProcessor`, `ECFRLogger`, and `ContractAnalyzer` classes are available and functional.
- The JSON structure includes `form_number`, `summary`, `relevant_clauses`, and `pi_clause` (with values like "True", "missing", etc.).
- The table should display all PDFs (uploaded and pre-loaded) and fit within the left pane’s scrollable area.

### Changes to `streamlit.py`
I’ll provide the updated sections of `streamlit.py`, focusing on the left pane (`with col1:`) to replace the buttons with a table and remove the "Analyze PDFs" button. The rest of the code (CSS, middle/right panes, helper functions) remains unchanged unless specified.

#### 1. Update CSS for Table Styling
Add styles for the table and clickable cells to the `<style>` block. Replace the existing `<style>` block with:

```python
st.markdown("""
<style>
    .left-pane {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .pdf-viewer {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .json-details {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 85vh;
        overflow-y: auto;
        box-sizing: border-box;
    }
    .extract-text {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #0068c9;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
    .pdf-select-button {
        margin-bottom: 5px;
        padding: 8px;
        text-align: left;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pdf-select-button.selected {
        background-color: #0068c9;
        color: white;
    }
    .status-button-true {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-false {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .status-button-missing {
        background-color: #ffc107;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .processing-message {
        color: #0068c9;
        font-size: 14px;
        margin: 5px 0;
    }
    .pdf-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .pdf-table th, .pdf-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .pdf-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .pdf-table td {
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .pdf-table td:hover {
        background-color: #e6f3ff;
    }
    .pdf-table td.selected {
        background-color: #0068c9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
```

- **Changes**:
  - Added `.pdf-table`, `.pdf-table th`, `.pdf-table td` for table styling (bordered, full-width, padded).
  - Added `cursor: pointer` and `transition` for clickable cells with a hover effect (light blue, `#e6f3ff`).
  - Added `.pdf-table td.selected` to highlight the selected PDF (dark blue, `#0068c9`, white text).

#### 2. Update Left Pane in `main` Function
Replace the left pane code (within `with col1:`) in the `main` function with the following to implement the table and remove the buttons:

```python
# Left pane: PDF upload and controls
with col1:
    with st.container():
        st.markdown('<div class="left-pane">', unsafe_allow_html=True)
        st.header("Contracts")
        
        # Pre-loaded PDFs dropdown
        st.subheader("Pre-loaded PDFs")
        preloaded_files = load_preloaded_data(
            pdf_folder="./preloaded_contracts/pdfs",
            json_folder="./preloaded_contracts/jsons"
        )
        preloaded_pdf_names = [pdf_name for pdf_name, _, _, _ in preloaded_files] if preloaded_files else ["No pre-loaded PDFs available"]
        if preloaded_files:
            preloaded_pdf_names.insert(0, "Select a pre-loaded PDF")
            preloaded_pdf_names.append("Load all pre-loaded PDFs")
        
        selected_preloaded_pdf = st.selectbox(
            "Choose a pre-loaded PDF",
            preloaded_pdf_names,
            key="preloaded_pdf_select"
        )
        
        if selected_preloaded_pdf and selected_preloaded_pdf != "Select a pre-loaded PDF" and selected_preloaded_pdf != "No pre-loaded PDFs available":
            if selected_preloaded_pdf == "Load all pre-loaded PDFs":
                loaded_pdfs = []
                for pdf_name, pdf_bytes, json_exists, json_path in preloaded_files:
                    file_stem = Path(pdf_name).stem
                    if pdf_name not in st.session_state.pdf_files:
                        st.session_state.pdf_files[pdf_name] = pdf_bytes
                        st.session_state.analysis_status[pdf_name] = "Not processed"
                        if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
                            st.warning(f"{pdf_name} is larger than 1.5MB and may fail to display.")
                        loaded_pdfs.append(pdf_name)
                    
                    if json_exists and file_stem not in st.session_state.json_data:
                        with open(json_path, 'r') as f:
                            st.session_state.json_data[file_stem] = json.load(f)
                        st.session_state.analysis_status[pdf_name] = "Processed"
                    
                    if st.session_state.current_pdf is None and loaded_pdfs:
                        st.session_state.current_pdf = pdf_name
                
                if loaded_pdfs:
                    st.success(f"Loaded pre-loaded PDFs: {', '.join(loaded_pdfs)}")
                else:
                    st.warning("No valid pre-loaded PDFs found.")
            else:
                for pdf_name, pdf_bytes, json_exists, json_path in preloaded_files:
                    if pdf_name == selected_preloaded_pdf:
                        file_stem = Path(pdf_name).stem
                        if pdf_name not in st.session_state.pdf_files:
                            st.session_state.pdf_files[pdf_name] = pdf_bytes
                            st.session_state.analysis_status[pdf_name] = "Not processed"
                            if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
                                st.warning(f"{pdf_name} is larger than 1.5MB and may fail to display.")
                        
                        if json_exists and file_stem not in st.session_state.json_data:
                            with open(json_path, 'r') as f:
                                st.session_state.json_data[file_stem] = json.load(f)
                            st.session_state.analysis_status[pdf_name] = "Processed"
                        
                        if st.session_state.current_pdf is None:
                            st.session_state.current_pdf = pdf_name
                        
                        st.success(f"Loaded pre-loaded PDF: {pdf_name}")
                        break
        
        # PDF uploader
        st.subheader("Upload PDFs")
        uploaded_pdfs = st.file_uploader(
            "Upload Contract PDFs",
            type="pdf",
            key="pdf_uploader",
            accept_multiple_files=True
        )
        
        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                if pdf.name not in st.session_state.pdf_files:
                    pdf_bytes = pdf.getvalue()
                    is_valid, metadata_or_error = validate_pdf(pdf_bytes)
                    if is_valid:
                        if len(pdf_bytes) > 1500 * 1024:  # 1500 KB
                            st.warning(f"{pdf.name} is larger than 1.5MB and may fail to display.")
                        st.session_state.pdf_files[pdf.name] = pdf_bytes
                        st.session_state.analysis_status[pdf.name] = "Not processed"
                    else:
                        st.error(f"Failed to load {pdf.name}: {metadata_or_error}")
        
        # Display PDF table
        if st.session_state.pdf_files:
            st.subheader("Available PDFs")
            # Create HTML table
            table_html = '<table class="pdf-table"><tr><th>PDF Name</th></tr>'
            for pdf_name in st.session_state.pdf_files.keys():
                selected_class = 'selected' if pdf_name == st.session_state.current_pdf else ''
                table_html += f'<tr><td class="{selected_class}" onclick="document.getElementById(\'pdf_{pdf_name}\').click()">{pdf_name}</td></tr>'
            table_html += '</table>'
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Hidden buttons for click handling
            for pdf_name in st.session_state.pdf_files.keys():
                if st.button(pdf_name, key=f"pdf_{pdf_name}", type="secondary", use_container_width=True):
                    set_current_pdf(pdf_name)
                    if st.session_state.analysis_status.get(pdf_name) != "Processed":
                        pdf_text_processor = PDFTextProcessor()
                        logger = ECFRLogger()
                        contract_analyzer = ContractAnalyzer()
                        with tempfile.TemporaryDirectory() as temp_dir:
                            st.session_state.processing_messages[pdf_name] = []
                            with st.spinner(f"Processing {pdf_name}..."):
                                message_placeholder = st.empty()
                                success, result = process_pdf(
                                    st.session_state.pdf_files[pdf_name], pdf_name, temp_dir, 
                                    pdf_text_processor, contract_analyzer, logger, message_placeholder
                                )
                                if success:
                                    st.session_state.json_data[Path(pdf_name).stem] = result
                                    st.session_state.analysis_status[pdf_name] = "Processed"
                                    st.success(f"Analysis complete for {pdf_name}")
                                else:
                                    st.session_state.analysis_status[pdf_name] = result
                                    st.error(f"Failed to process {pdf_name}: {result}")
                                st.session_state.processing_messages[pdf_name] = []
                                message_placeholder.empty()
                    st.rerun()
        
        # Display analysis status
        if st.session_state.analysis_status:
            st.subheader("Analysis Status")
            for pdf_name, status in st.session_state.analysis_status.items():
                st.write(f"{pdf_name}: {status}")
        
        # Page navigation
        if st.session_state.current_pdf and st.session_state.current_pdf in st.session_state.pdf_files:
            st.subheader("Page Navigation")
            num_pages = 10
            page_num = st.number_input(
                "Go to page:", 
                min_value=1, 
                max_value=num_pages, 
                value=st.session_state.current_page,
                step=1,
                key="page_navigator"
            )
            if st.button("Navigate", key="nav_button"):
                st.session_state.current_page = page_num
                st.session_state.search_text = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
```

- **Changes**:
  - **Removed "Analyze PDFs" Button**: Deleted the button and its associated logic:
    ```python
    if st.button("Analyze PDFs", key="analyze_button"): ...
    ```
  - **Removed Individual PDF Buttons**: Deleted the button loop:
    ```python
    for pdf_name in st.session_state.pdf_files.keys():
        if st.button(pdf_name, key=f"pdf_btn_{pdf_name}", ...): ...
    ```
  - **Added Single-Column Table**:
    - Created an HTML table with one column ("PDF Name") using `<table class="pdf-table">`.
    - Each row’s `<td>` is clickable via `onclick` triggering a hidden `st.button`:
      ```python
      table_html += f'<tr><td class="{selected_class}" onclick="document.getElementById(\'pdf_{pdf_name}\').click()">{pdf_name}</td></tr>'
      ```
    - Highlighted the selected PDF with `class="selected"`:
      ```python
      selected_class = 'selected' if pdf_name == st.session_state.current_pdf else ''
      ```
    - Rendered the table with `st.markdown(table_html, unsafe_allow_html=True)`.
  - **Hidden Buttons for Click Handling**:
    - Added hidden `st.button` elements for each PDF, triggered by the table’s `onclick`:
      ```python
      for pdf_name in st.session_state.pdf_files.keys():
          if st.button(pdf_name, key=f"pdf_{pdf_name}", type="secondary", use_container_width=True): ...
      ```
    - On click:
      - Set the current PDF (`set_current_pdf(pdf_name)`).
      - Check if unprocessed (`analysis_status != "Processed"`) and run `process_pdf`:
        ```python
        if st.session_state.analysis_status.get(pdf_name) != "Processed":
            # Initialize processor, logger, analyzer
            with st.spinner(f"Processing {pdf_name}..."):
                message_placeholder = st.empty()
                success, result = process_pdf(...)
        ```
      - Display processing messages, update status, and show success/error messages.
      - Call `st.rerun()` to refresh the UI.
  - **Kept Other Sections**:
    - Pre-loaded PDFs dropdown, uploader, analysis status, and page navigation remain unchanged.

### Testing Instructions
1. **Verify Table Display**:
   - Run the app (`streamlit run streamlit.py`).
   - Upload PDFs or load pre-loaded PDFs (e.g., `contract1.pdf`, `contract2.pdf`).
   - Check the left pane under "Available PDFs":
     - A single-column table lists all PDFs (uploaded and pre-loaded).
     - Table has a header ("PDF Name") and bordered cells.
     - Current PDF is highlighted (blue background, white text, `.selected`).
     - Cells change to light blue on hover (`.pdf-table td:hover`).

2. **Test Clickable Cells**:
   - Click a PDF name in the table (e.g., `contract1.pdf`):
     - Verify it becomes the current PDF (highlighted in the table, displayed in the middle pane).
     - If unprocessed (`analysis_status` is "Not processed"):
       - Spinner shows `Processing contract1.pdf...`.
       - Messages appear: "Text extracted from PDF", "Analyzing the document" (blue, 14px, in `.left-pane`).
       - On success: `Analysis complete for contract1.pdf`, status updates to "Processed", JSON data appears in the right pane.
       - On failure: `Failed to process contract1.pdf: {error}`, status updates to the error message.
     - If already "Processed", no analysis occurs, and the PDF is displayed immediately.
   - Click a processed PDF and confirm no re-analysis occurs.

3. **Confirm Button Removal**:
   - Ensure the "Analyze PDFs" button is gone.
   - Ensure individual PDF buttons under "Available PDFs" are replaced by the table.

4. **Analysis Status**:
   - Check "Analysis Status" section:
     - Unprocessed PDFs show "Not processed".
     - Processed PDFs show "Processed".
     - Failed PDFs show the error message.

5. **Existing Functionality**:
   - Verify:
     - Pre-loaded PDFs dropdown (including "Load all pre-loaded PDFs").
     - PDF uploader.
     - Scrollable panes (`.left-pane`, `height: 85vh`).
     - Processing messages under spinner.
     - `pi_clause` styling (green for "True"/"yes"/"YES"/"Yes", red for "False"/"no"/"No"/"NO", yellow for "missing"/"Missing"/"MISSING"/"Absent").
     - Folder creation and fallback directory search (`./preloaded_contracts/pdfs`, `{current_dir}/preloaded_contracts/pdfs`).

6. **Table Styling**:
   - Use browser developer tools (F12):
     - Confirm `<table class="pdf-table">` with `<th>PDF Name</th>`.
     - Verify `<td>` cells have `cursor: pointer`, hover effect (`background-color: #e6f3ff`), and selected style (`background-color: #0068c9`).
     - Check `onclick` triggers the correct button (`id="pdf_{pdf_name}"`).

### Troubleshooting
- **Table Not Displaying**:
  - Verify the HTML table code:
    ```python
    table_html = '<table class="pdf-table"><tr><th>PDF Name</th></tr>'
    st.markdown(table_html, unsafe_allow_html=True)
    ```
  - Check `st.session_state.pdf_files` is populated:
    ```python
    st.write(st.session_state.pdf_files.keys())  # Debug
    ```
- **Cells Not Clickable**:
  - Confirm `onclick` events:
    ```python
    onclick="document.getElementById('pdf_{pdf_name}').click()"
    ```
  - Ensure hidden buttons exist:
    ```python
    for pdf_name in st.session_state.pdf_files.keys():
        if st.button(pdf_name, key=f"pdf_{pdf_name}", ...): ...
    ```
  - Check browser console for JavaScript errors.
- **Analysis Not Triggered**:
  - Verify the condition:
    ```python
    if st.session_state.analysis_status.get(pdf_name) != "Processed":
    ```
  - Ensure `process_pdf` is called and logs messages:
    ```python
    logger.info(f"Text extracted from {pdf_name}")
    ```
- **Processing Messages Missing**:
  - Check `st.session_state.processing_messages` and `message_placeholder`:
    ```python
    st.session_state.processing_messages[pdf_name].append("Text extracted from PDF")
    message_placeholder.markdown(...)
    ```
- **Table Not Scrollable**:
  - Confirm `.left-pane` CSS:
    ```css
    .left-pane {
        height: 85vh;
        overflow-y: auto;
    }
    ```
- **Styling Issues**:
  - Verify `.pdf-table` styles:
    ```css
    .pdf-table td { cursor: pointer; }
    .pdf-table td:hover { background-color: #e6f3ff; }
    .pdf-table td.selected { background-color: #0068c9; color: white; }
    ```
  - Clear browser cache or test in incognito mode.

### Assumptions
- The table replaces only the "Available PDFs" buttons; other left pane elements (dropdown, uploader, status, navigation) remain.
- Clicking a cell triggers analysis only if the PDF is unprocessed, avoiding redundant processing.
- The HTML table with JavaScript `onclick` is sufficient for clickable cells (Streamlit’s `st.dataframe` was considered but lacks direct click events).

If issues arise (e.g., table not clickable, analysis not triggering, or styling off), share a screenshot, JSON/PDF details, or browser console logs, and I’ll provide a targeted fix!
