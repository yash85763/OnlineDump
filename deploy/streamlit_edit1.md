Looking at your code, I'll help you implement both functionalities. Here are the specific updates needed:

## 1. Database Schema Updates

### Update the `pdfs` table in `database.py`:

```sql
# In create_tables() function, update the 'pdfs' table schema:
'pdfs': """
    CREATE TABLE IF NOT EXISTS pdfs (
        id SERIAL PRIMARY KEY,
        pdf_name VARCHAR NOT NULL,
        file_hash VARCHAR UNIQUE NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed_date TIMESTAMP,
        
        -- PDF parsing data
        layout VARCHAR,
        original_word_count INTEGER,
        original_page_count INTEGER,
        parsability BOOLEAN,
        
        -- Final (obfuscated) content metrics
        final_word_count INTEGER,
        final_page_count INTEGER,
        avg_words_per_page FLOAT,
        
        -- Content storage
        raw_content TEXT,
        final_content TEXT,
        pdf_bytes BYTEA,  -- NEW: Store actual PDF file
        pdf_parsing_data JSONB,  -- NEW: Store complete PDF handler JSON data
        
        -- Obfuscation tracking
        obfuscation_applied BOOLEAN DEFAULT TRUE,
        pages_removed_count INTEGER DEFAULT 0,
        paragraphs_obfuscated_count INTEGER DEFAULT 0,
        obfuscation_summary JSONB,
        
        -- User tracking
        uploaded_by VARCHAR
    )
""",
```

## 2. Update PDF Storage Function

### Modify `store_pdf_data()` in `database.py`:

```python
def store_pdf_data(pdf_data: Dict[str, Any]) -> int:
    """Store PDF data after parsing and obfuscation process"""
    
    sql = """
        INSERT INTO pdfs (
            pdf_name, file_hash, upload_date, processed_date,
            layout, original_word_count, original_page_count, parsability,
            final_word_count, final_page_count, avg_words_per_page,
            raw_content, final_content, pdf_bytes, pdf_parsing_data,
            obfuscation_applied, pages_removed_count, paragraphs_obfuscated_count,
            obfuscation_summary, uploaded_by
        ) VALUES (
            %(pdf_name)s, %(file_hash)s, %(upload_date)s, %(processed_date)s,
            %(layout)s, %(original_word_count)s, %(original_page_count)s, %(parsability)s,
            %(final_word_count)s, %(final_page_count)s, %(avg_words_per_page)s,
            %(raw_content)s, %(final_content)s, %(pdf_bytes)s, %(pdf_parsing_data)s,
            %(obfuscation_applied)s, %(pages_removed_count)s, %(paragraphs_obfuscated_count)s,
            %(obfuscation_summary)s, %(uploaded_by)s
        ) RETURNING id
    """
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Convert datetime objects to strings if needed
            if 'upload_date' in pdf_data and isinstance(pdf_data['upload_date'], datetime):
                pdf_data['upload_date'] = pdf_data['upload_date'].isoformat()
            if 'processed_date' in pdf_data and isinstance(pdf_data['processed_date'], datetime):
                pdf_data['processed_date'] = pdf_data['processed_date'].isoformat()
            
            # Convert JSON objects to strings
            if 'obfuscation_summary' in pdf_data and isinstance(pdf_data['obfuscation_summary'], dict):
                pdf_data['obfuscation_summary'] = json.dumps(pdf_data['obfuscation_summary'])
            
            if 'pdf_parsing_data' in pdf_data and isinstance(pdf_data['pdf_parsing_data'], dict):
                pdf_data['pdf_parsing_data'] = json.dumps(pdf_data['pdf_parsing_data'])
            
            cur.execute(sql, pdf_data)
            pdf_id = cur.fetchone()[0]
            conn.commit()
            
            return pdf_id
```

## 3. Add Function to Get Analyses for PDF

### Add this function to `database.py`:

```python
def get_analyses_for_pdf(pdf_id: int) -> List[Dict[str, Any]]:
    """Get all analyses for a specific PDF"""
    
    sql = """
        SELECT a.*, 
               array_agg(
                   json_build_object(
                       'id', c.id,
                       'type', c.clause_type,
                       'text', c.clause_text,
                       'order', c.clause_order
                   ) ORDER BY c.clause_order
               ) as clauses
        FROM analyses a
        LEFT JOIN clauses c ON a.id = c.analysis_id
        WHERE a.pdf_id = %s
        GROUP BY a.id
        ORDER BY a.version DESC
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            results = cur.fetchall()
            return [dict(row) for row in results]
```

## 4. Update Enhanced PDF Handler

### Modify `process_pdf_with_database()` in `enhanced_Pdf_handler.py`:

```python
# At the beginning of process_pdf_with_database(), after calculating file_hash:

# Check if file already exists in database
if self.enable_database:
    try:
        existing_pdf = get_pdf_by_hash(file_hash)
        if existing_pdf:
            # Get analyses for this PDF
            from config.database import get_analyses_for_pdf
            analyses = get_analyses_for_pdf(existing_pdf["id"])
            
            return {
                "success": True,
                "message": "File already exists in database",
                "pdf_id": existing_pdf["id"],
                "duplicate": True,
                "existing_record": existing_pdf,
                "analyses": analyses,
                "pdf_bytes": existing_pdf.get("pdf_bytes"),  # Return stored PDF bytes
                "pdf_parsing_data": existing_pdf.get("pdf_parsing_data")  # Return stored parsing data
            }
    except Exception as e:
        print(f"Warning: Could not check for existing PDF: {str(e)}")

# In the data preparation section, add:
pdf_data = {
    # ... existing fields ...
    'pdf_bytes': pdf_bytes,  # Store the actual PDF bytes
    'pdf_parsing_data': {  # Store complete parsing data
        'pages': final_pages_content,
        'raw_pages': original_pages_content,
        'layout': layout_type,
        'quality_info': quality_info,
        'processing_timestamp': datetime.now().isoformat()
    },
    # ... rest of existing fields ...
}
```

## 5. Update Main Application Logic

### Modify `process_pdf_enhanced()` in `main.py`:

```python
def process_pdf_enhanced(pdf_bytes, pdf_name, message_placeholder, logger):
    """Process a single PDF using the enhanced handler with database storage"""
    try:
        st.session_state.processing_messages[pdf_name] = []
        
        # Update processing message
        st.session_state.processing_messages[pdf_name].append("🔍 Checking if PDF already exists in database...")
        message_placeholder.markdown(
            "\n".join([f"<div class='processing-message'>{msg}</div>" 
                      for msg in st.session_state.processing_messages[pdf_name]]),
            unsafe_allow_html=True
        )
        
        # Process PDF with enhanced handler
        result = process_single_pdf_from_streamlit(
            pdf_name=pdf_name,
            pdf_bytes=pdf_bytes,
            enable_obfuscation=True,
            uploaded_by=get_session_id()
        )
        
        if result.get('success'):
            if result.get('duplicate'):
                # Handle existing PDF
                st.session_state.processing_messages[pdf_name].append("✅ PDF found in database - loading existing data")
                
                # Load existing data into session state
                existing_record = result.get('existing_record', {})
                analyses = result.get('analyses', [])
                
                # Store PDF data in session state
                st.session_state.pdf_database_ids[pdf_name] = existing_record.get('id')
                st.session_state.obfuscation_summaries[pdf_name] = existing_record.get('obfuscation_summary', {})
                
                # Load PDF parsing data
                pdf_parsing_data = existing_record.get('pdf_parsing_data', {})
                if pdf_parsing_data:
                    st.session_state.raw_pdf_data[pdf_name] = {
                        'pages': pdf_parsing_data.get('pages', [])
                    }
                
                # Load most recent analysis if available
                if analyses:
                    latest_analysis = analyses[0]  # First one is most recent due to ORDER BY version DESC
                    file_stem = Path(pdf_name).stem
                    
                    # Convert raw_json back to dict if it's a string
                    analysis_data = latest_analysis.get('raw_json', {})
                    if isinstance(analysis_data, str):
                        analysis_data = json.loads(analysis_data)
                    
                    st.session_state.json_data[file_stem] = analysis_data
                    st.session_state.processing_messages[pdf_name].append(f"✅ Loaded existing analysis (Version {latest_analysis.get('version', 1)})")
                
                st.session_state.processing_messages[pdf_name].append(f"📊 Database ID: {existing_record.get('id')}")
                return True, "Existing PDF loaded successfully"
            
            else:
                # Handle new PDF processing (existing logic)
                # ... rest of your existing processing logic ...
```

## 6. Update File Loading Function

### Modify `load_processed_pdfs_from_database()` in `main.py`:

```python
def load_processed_pdfs_from_database():
    """Load all processed PDFs from database on app startup"""
    try:
        from config.database import get_all_processed_pdfs, get_analyses_for_pdf
        
        # Get all processed PDFs
        processed_pdfs = get_all_processed_pdfs()
        
        if not processed_pdfs:
            st.info("No previously processed PDFs found in database")
            return
        
        loaded_count = 0
        
        for pdf_record in processed_pdfs:
            pdf_name = pdf_record['pdf_name']
            
            # Skip if already loaded in session
            if pdf_name in st.session_state.pdf_files:
                continue
            
            # Load PDF bytes from stored pdf_bytes field
            if pdf_record.get('pdf_bytes'):
                try:
                    st.session_state.pdf_files[pdf_name] = bytes(pdf_record['pdf_bytes'])
                    loaded_count += 1
                except Exception as e:
                    st.warning(f"Could not load PDF bytes for {pdf_name}: {str(e)}")
                    continue
            else:
                st.warning(f"No PDF bytes found for {pdf_name}")
                continue
            
            # Load PDF parsing data
            if pdf_record.get('pdf_parsing_data'):
                try:
                    parsing_data = pdf_record['pdf_parsing_data']
                    if isinstance(parsing_data, str):
                        parsing_data = json.loads(parsing_data)
                    
                    st.session_state.raw_pdf_data[pdf_name] = {
                        'pages': parsing_data.get('pages', [])
                    }
                except Exception as e:
                    st.warning(f"Could not parse PDF data for {pdf_name}: {str(e)}")
            
            # Load analyses for this PDF
            try:
                analyses = get_analyses_for_pdf(pdf_record['id'])
                if analyses:
                    latest_analysis = analyses[0]  # Most recent
                    file_stem = Path(pdf_name).stem
                    
                    analysis_data = latest_analysis.get('raw_json', {})
                    if isinstance(analysis_data, str):
                        analysis_data = json.loads(analysis_data)
                    
                    st.session_state.json_data[file_stem] = analysis_data
                    st.session_state.analysis_status[pdf_name] = "Processed"
                else:
                    st.session_state.analysis_status[pdf_name] = "No analysis data"
            except Exception as e:
                st.warning(f"Could not load analyses for {pdf_name}: {str(e)}")
                st.session_state.analysis_status[pdf_name] = "Error loading analysis"
            
            # Store database metadata
            st.session_state.pdf_database_ids[pdf_name] = pdf_record['id']
            st.session_state.obfuscation_summaries[pdf_name] = pdf_record.get('obfuscation_summary', {})
        
        if loaded_count > 0:
            st.success(f"✅ Loaded {loaded_count} processed PDFs from database")
```

## Summary of Changes:

1. **Database Schema**: Added `pdf_bytes` (BYTEA) and `pdf_parsing_data` (JSONB) columns
2. **Duplicate Detection**: Enhanced to return full PDF data and analyses when duplicate found
3. **Data Loading**: Updated to use stored PDF bytes instead of base64 encoding
4. **Analysis Loading**: Added function to fetch all analyses for a PDF
5. **Session State**: Updated to properly handle existing vs new PDFs

These changes will ensure that when you upload duplicate files, they're detected early and all existing data (PDF bytes, parsing data, and analyses) are loaded from the database without reprocessing.
