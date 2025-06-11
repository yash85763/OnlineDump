You're absolutely right! Much simpler approach. Let's just add a new column to store the raw parsing JSON during the initial database table creation.

## Updated Database Schema

Just modify the `pdfs` table creation in your `create_tables()` function:

```python
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
        raw_parsing_json JSONB,  -- NEW COLUMN: Store complete parsing result
        
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

## Updated `store_pdf_data` Function

```python
def store_pdf_data(pdf_data: Dict[str, Any]) -> int:
    """Store PDF data after parsing and obfuscation process"""
    
    sql = """
        INSERT INTO pdfs (
            pdf_name, file_hash, upload_date, processed_date,
            layout, original_word_count, original_page_count, parsability,
            final_word_count, final_page_count, avg_words_per_page,
            raw_content, final_content, raw_parsing_json, obfuscation_applied,
            pages_removed_count, paragraphs_obfuscated_count,
            obfuscation_summary, uploaded_by
        ) VALUES (
            %(pdf_name)s, %(file_hash)s, %(upload_date)s, %(processed_date)s,
            %(layout)s, %(original_word_count)s, %(original_page_count)s, %(parsability)s,
            %(final_word_count)s, %(final_page_count)s, %(avg_words_per_page)s,
            %(raw_content)s, %(final_content)s, %(raw_parsing_json)s, %(obfuscation_applied)s,
            %(pages_removed_count)s, %(paragraphs_obfuscated_count)s,
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
            
            # Convert JSON objects to strings for JSONB columns
            if 'obfuscation_summary' in pdf_data and isinstance(pdf_data['obfuscation_summary'], dict):
                pdf_data['obfuscation_summary'] = json.dumps(pdf_data['obfuscation_summary'])
            
            if 'raw_parsing_json' in pdf_data and isinstance(pdf_data['raw_parsing_json'], dict):
                pdf_data['raw_parsing_json'] = json.dumps(pdf_data['raw_parsing_json'])
            
            cur.execute(sql, pdf_data)
            pdf_id = cur.fetchone()[0]
            conn.commit()
            
            return pdf_id
```

## Updated Loading Function

```python
def load_existing_pdfs_from_database():
    """Load all existing PDFs from database into session state with complete data integrity"""
    try:
        from config.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Query to get all processed PDFs with their latest analysis AND raw parsing JSON
                query = """
                SELECT DISTINCT p.id, p.pdf_name, p.final_word_count, p.upload_date, 
                       p.final_content, p.raw_parsing_json, p.obfuscation_summary,
                       a.form_number, a.summary, a.analysis_date,
                       a.pi_clause, a.ci_clause, a.data_usage_mentioned, 
                       a.data_limitations_exists, a.raw_json as analysis_raw_json
                FROM pdfs p
                LEFT JOIN analyses a ON p.id = a.pdf_id
                WHERE a.id = (
                    SELECT MAX(id) FROM analyses a2 WHERE a2.pdf_id = p.id
                )
                ORDER BY p.upload_date DESC
                LIMIT 50
                """
                
                cur.execute(query)
                existing_pdfs = cur.fetchall()
        
        # Load PDFs into session state with complete data
        loaded_count = 0
        for pdf_data in existing_pdfs:
            (pdf_id, filename, file_size, upload_date, final_content, raw_parsing_json, 
             obfuscation_summary, form_number, summary, analysis_date,
             pi_clause, ci_clause, data_usage_mentioned, data_limitations_exists, 
             analysis_raw_json) = pdf_data
            
            # Skip if already loaded
            if filename in st.session_state.pdf_database_ids:
                continue
            
            # Store PDF metadata
            st.session_state.pdf_database_ids[filename] = pdf_id
            st.session_state.analysis_status[filename] = "Processed"
            
            # Parse raw parsing JSON if available
            raw_parsing_data = None
            if raw_parsing_json:
                try:
                    if isinstance(raw_parsing_json, str):
                        raw_parsing_data = json.loads(raw_parsing_json)
                    else:
                        raw_parsing_data = raw_parsing_json
                except Exception as e:
                    print(f"Error parsing raw_parsing_json for {filename}: {e}")
            
            # Parse analysis JSON if available
            analysis_data = None
            if analysis_raw_json:
                try:
                    if isinstance(analysis_raw_json, str):
                        analysis_data = json.loads(analysis_raw_json)
                    else:
                        analysis_data = analysis_raw_json
                except Exception as e:
                    print(f"Error parsing analysis_raw_json for {filename}: {e}")
            
            # Create analysis data structure
            file_stem = Path(filename).stem
            if analysis_data:
                st.session_state.json_data[file_stem] = analysis_data
                st.session_state.json_data[file_stem]['loaded_from_database'] = True
                st.session_state.json_data[file_stem]['analysis_date'] = analysis_date
            else:
                # Fallback to individual fields
                st.session_state.json_data[file_stem] = {
                    'form_number': form_number or 'Not available',
                    'summary': summary or 'No summary available',
                    'pi_clause': pi_clause == 'true' if pi_clause else False,
                    'ci_clause': ci_clause == 'true' if ci_clause else False,
                    'data_usage_mentioned': data_usage_mentioned == 'true' if data_usage_mentioned else False,
                    'data_limitations_exists': data_limitations_exists == 'true' if data_limitations_exists else False,
                    'relevant_clauses': [],
                    'analysis_date': analysis_date,
                    'loaded_from_database': True
                }
            
            # Store raw PDF data for page matching
            if raw_parsing_data and 'pages' in raw_parsing_data:
                st.session_state.raw_pdf_data[filename] = {
                    'pages': raw_parsing_data['pages'],
                    'raw_parsing_json': raw_parsing_data
                }
            elif final_content:
                st.session_state.raw_pdf_data[filename] = {
                    'pages': convert_string_to_pages_structure(final_content, filename)
                }
            
            # Store obfuscation summary
            if obfuscation_summary:
                try:
                    if isinstance(obfuscation_summary, str):
                        st.session_state.obfuscation_summaries[filename] = json.loads(obfuscation_summary)
                    else:
                        st.session_state.obfuscation_summaries[filename] = obfuscation_summary
                except Exception as e:
                    print(f"Error parsing obfuscation_summary for {filename}: {e}")
            
            # Create processing messages
            st.session_state.processing_messages[filename] = [
                f"💾 Loaded from database (ID: {pdf_id})",
                f"📅 Originally processed: {analysis_date.strftime('%Y-%m-%d %H:%M') if analysis_date else 'Unknown'}",
                f"📊 File size: {file_size} words" if file_size else "📊 File size: Unknown",
                "📄 Raw parsing data: " + ("✅ Available" if raw_parsing_data else "❌ Missing"),
                "🔍 Raw analysis data: " + ("✅ Available" if analysis_data else "❌ Missing"),
                "✅ Analysis complete - Ready for viewing"
            ]
            
            loaded_count += 1
        
        return loaded_count
        
    except Exception as e:
        print(f"Error loading existing PDFs: {e}")
        return 0
```

That's it! Much cleaner approach:

1. **Add one column**: `raw_parsing_json JSONB` to the `pdfs` table
2. **Store during creation**: When storing PDF data, include the parsing JSON in the `raw_parsing_json` field
3. **Load directly**: When loading existing PDFs, read the `raw_parsing_json` column directly

No need for updates, no complex logic - just store the complete parsing JSON in the new column from the beginning.
