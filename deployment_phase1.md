I'll help you implement these enhancements to your contract analysis application. Let's break this down into three parts as requested.

## 1. Update Streamlit Code to Use the Parsing Mechanism from the First File

First, let's create a new integration that uses the PDF handling capabilities from the `ecfr_api_wrapper.py` file instead of the current `PDFTextProcessor`. Here's the code you should add:

```python
# Import the PDFHandler from ecfr_api_wrapper
from ecfr_api_wrapper import PDFHandler

# Replace the process_pdf function with this updated version
def process_pdf(pdf_bytes, pdf_name, temp_dir, contract_analyzer, logger, message_placeholder):
    """Process a single PDF using the enhanced PDFHandler from ecfr_api_wrapper"""
    try:
        file_stem = Path(pdf_name).stem
        pdf_path = os.path.join(temp_dir, f"{file_stem}.pdf")
        output_json_path = os.path.join(temp_dir, f"{file_stem}.json")
        output_txt_path = os.path.join(temp_dir, f"{file_stem}.txt")

        # Save PDF to temporary file
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

        # Initialize PDFHandler with default settings
        pdf_handler = PDFHandler(
            min_quality_ratio=0.5,
            paragraph_spacing_threshold=10,
            page_continuity_threshold=0.1,
            min_words_threshold=5
        )
        
        # Process PDF and extract content
        logger.info(f"Processing PDF content from {pdf_name}")
        st.session_state.processing_messages[pdf_name].append("Extracting and analyzing PDF content")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )
        
        # Use the PDFHandler to extract content
        pdf_result = pdf_handler.process_pdf(pdf_path)
        
        if not pdf_result.get("parsable", False):
            error_msg = pdf_result.get("error", "Unknown error during PDF processing")
            logger.error(f"PDF processing failed for {pdf_name}: {error_msg}")
            return False, error_msg
        
        # Save text content to file for contract analyzer
        pdf_handler.save_to_txt(pdf_result, output_txt_path)
        
        # Get text content for analysis
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            contract_text = f.read()
        
        # Extract metadata for database
        layout = pdf_result.get("layout", "unknown")
        page_count = len(pdf_result.get("pages", []))
        
        # Count words in content
        word_count = sum(len(para.split()) for page in pdf_result.get("pages", []) 
                         for para in page.get("paragraphs", []))
        
        # Calculate average words per page
        avg_words_per_page = word_count / page_count if page_count > 0 else 0
        
        # Store metadata in session state for database insertion
        st.session_state.pdf_metadata = {
            "filename": pdf_name,
            "word_count": word_count,
            "page_count": page_count,
            "avg_words_per_page": avg_words_per_page,
            "layout": layout,
            "parsable": True
        }
        
        logger.info(f"Text extracted from {pdf_name}: {word_count} words across {page_count} pages")
        st.session_state.processing_messages[pdf_name].append(f"Text extracted: {word_count} words across {page_count} pages")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )

        # Analyze contract
        st.session_state.processing_messages[pdf_name].append("Analyzing the document")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )
        logger.info(f"Analyzing the document {pdf_name}")

        # Retry ContractAnalyzer up to 2 times
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                results = contract_analyzer.analyze_contract(contract_text, output_json_path)
                break  # Exit loop on success
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {pdf_name}: {str(e)}")
                    continue
                else:
                    logger.error(f"Failed after {max_retries} retries for {pdf_name}: {str(e)}")
                    return False, "An error occurred. Please submit another PDF or reach out to Support."

        # Load generated JSON
        with open(output_json_path, 'r') as f:
            json_data = json.load(f)

        return True, json_data
    except Exception as e:
        logger.error(f"Error processing {pdf_name}: {str(e)}")
        return False, f"An error occurred: {str(e)}"
```

Then, in your `main()` function, update the processing section to use this new version by removing the `pdf_text_processor` instantiation:

```python
# Replace this in the main() function where PDF processing occurs
if st.session_state.analysis_status.get(selected_pdf) != "Processed":
    logger = ECFRLogger()
    contract_analyzer = ContractAnalyzer()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        st.session_state.processing_messages[selected_pdf] = []
        with st.spinner(f"Processing {selected_pdf}..."):
            message_placeholder = st.empty()
            success, result = process_pdf(
                st.session_state.pdf_files[selected_pdf], selected_pdf, temp_dir, 
                contract_analyzer, logger, message_placeholder
            )
            if success:
                st.session_state.json_data[Path(selected_pdf).stem] = result
                st.session_state.analysis_status[selected_pdf] = "Processed"
                st.success(f"Analysis complete for {selected_pdf}")
                # Store PDF metadata and analysis results in database
                if hasattr(st.session_state, 'pdf_metadata'):
                    store_pdf_data(st.session_state.pdf_metadata, result)
            else:
                st.session_state.analysis_status[selected_pdf] = result
                st.error(f"Failed to process {selected_pdf}: {result}")
            st.session_state.processing_messages[selected_pdf] = []
            message_placeholder.empty()
```

## 2. Create Database Utility for PostgreSQL Integration

Create a new file called `db_utils.py`:

## 3. Update the UI to Add Feedback Mechanism

Here's the code you need to add to implement the feedback mechanism in your Streamlit app:

Now, you need to integrate the feedback UI into your main Streamlit application. Add this code to the right pane section in your `main()` function:

```python
# Add this at the end of the right pane section in main()
# Right before st.markdown('</div>', unsafe_allow_html=True)
if file_stem and st.session_state.current_pdf:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Feedback")
    render_feedback_ui(st.session_state.current_pdf)
```

## Implementation Instructions

Here are the steps to implement all these changes:

1. Create a new file called `db_utils.py` and copy the contents from the artifact above.

2. Update your Streamlit application by modifying the existing `process_pdf` function to use the `PDFHandler` from `ecfr_api_wrapper.py`.

3. Add the feedback UI code to your application.

4. Create a `.env` file in your project root with your PostgreSQL connection details:

```
DB_HOST=your-aurora-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

5. Install the required packages:

```bash
pip install psycopg2-binary python-dotenv
```

6. Update your imports at the top of the file:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from db_utils import store_pdf_data, store_feedback, get_all_feedback_for_pdf, init_db_pool, create_tables
from ecfr_api_wrapper import PDFHandler
```

## Summary of Changes

1. **PDF Processing Integration**: The code now uses the `PDFHandler` from `ecfr_api_wrapper.py` to extract text from PDFs more effectively, utilizing its advanced paragraph extraction and layout analysis capabilities.

2. **Database Integration**: A new `db_utils.py` module has been created to:
   - Connect to your AWS Aurora PostgreSQL database
   - Create the necessary database tables (input_data, analysis_data, clauses, feedback)
   - Store PDF metadata and analysis results
   - Store and retrieve user feedback

3. **Feedback Mechanism**: The UI now includes:
   - A "Provide Feedback" button
   - A feedback form where users can select a field and provide feedback
   - A display for viewing previous feedback
   - Database integration for storing feedback permanently

The implementation preserves your existing functionality while enhancing it with better PDF parsing, database storage, and user feedback capabilities.
