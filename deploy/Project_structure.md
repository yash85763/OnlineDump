Perfect! You want to store both the raw JSON from PDF parsing and the raw JSON from analysis so that when loading previous PDFs, you have complete data integrity. Let me update the code to handle this properly.

## 1. Update PDF Processing to Store Raw Parsing JSON

```python
def process_pdf_enhanced(pdf_bytes, pdf_name, message_placeholder, logger):
    """Process a single PDF using the enhanced handler with database storage"""
    try:
        st.session_state.processing_messages[pdf_name] = []
        
        # Update processing message
        st.session_state.processing_messages[pdf_name].append("🔄 Starting PDF processing with obfuscation...")
        st.session_state.processing_messages[pdf_name].append("🔍 Checking if PDF exists in database...")
        
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
            pdf_id = result.get('pdf_id')
            
            # Store the COMPLETE raw result JSON for future loading
            raw_parsing_json = {
                'success': result.get('success'),
                'pdf_id': pdf_id,
                'pages': result.get('pages', []),
                'final_content': result.get('final_content', ''),
                'obfuscation_summary': result.get('obfuscation_summary', {}),
                'processing_metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'processor': 'enhanced_pdf_handler',
                    'obfuscation_enabled': True
                }
            }
            
            # Update the PDF record with the raw parsing JSON
            try:
                from config.database import db
                with db.get_connection() as conn:
                    with conn.cursor() as cur:
                        update_sql = """
                        UPDATE pdfs 
                        SET raw_content = %s 
                        WHERE id = %s
                        """
                        cur.execute(update_sql, (json.dumps(raw_parsing_json), pdf_id))
                        conn.commit()
                        
                st.session_state.processing_messages[pdf_name].append("💾 Raw parsing data stored in database")
            except Exception as e:
                st.session_state.processing_messages[pdf_name].append(f"⚠️ Warning: Could not store raw parsing data: {str(e)}")
            
            # Check if this is an existing PDF (has final_content instead of pages)
            if 'final_content' in result and 'pages' not in result:
                st.session_state.processing_messages[pdf_name].append("📋 PDF found in database - using existing content...")
                
                # For existing PDFs, load the stored raw parsing JSON
                try:
                    from config.database import get_pdf_by_id
                    stored_pdf = get_pdf_by_id(pdf_id)
                    if stored_pdf and stored_pdf.get('raw_content'):
                        stored_parsing_data = json.loads(stored_pdf['raw_content'])
                        pages_content = stored_parsing_data.get('pages', [])
                        contract_text = result.get('final_content', '')
                        
                        st.session_state.processing_messages[pdf_name].append("📄 Loaded stored parsing data from database")
                    else:
                        # Fallback: convert final_content to pages structure
                        final_content = result.get('final_content', '')
                        pages_content = convert_string_to_pages_structure(final_content, pdf_name)
                        contract_text = final_content
                except Exception as e:
                    st.session_state.processing_messages[pdf_name].append(f"⚠️ Could not load stored parsing data: {str(e)}")
                    final_content = result.get('final_content', '')
                    pages_content = convert_string_to_pages_structure(final_content, pdf_name)
                    contract_text = final_content
                
            elif 'pages' in result:
                st.session_state.processing_messages[pdf_name].append("📄 New PDF - using fresh processing results...")
                
                # Extract text from pages structure (new PDF processing)
                pages_content = result.get('pages', [])
                contract_text = extract_contract_text_from_pages(pages_content, pdf_name)
                
            else:
                # Fallback: extract directly from PDF bytes
                st.session_state.processing_messages[pdf_name].append("⚠️ Unexpected result structure - extracting directly from PDF...")
                pages_content = extract_text_from_pdf_bytes(pdf_bytes, pdf_name)
                contract_text = extract_contract_text_from_pages(pages_content, pdf_name)
            
            # Store processing information in session state
            st.session_state.pdf_database_ids[pdf_name] = pdf_id
            st.session_state.obfuscation_summaries[pdf_name] = result.get('obfuscation_summary', {})
            
            # Store raw PDF data for page number matching
            st.session_state.raw_pdf_data[pdf_name] = {
                'pages': pages_content,
                'raw_parsing_json': raw_parsing_json  # Store complete parsing result
            }
            
            # Add debugging info
            st.session_state.processing_messages[pdf_name].append(f"📊 Extracted {len(contract_text)} characters for analysis")
            
            # Continue with contract analysis...
            st.session_state.processing_messages[pdf_name].append("🔍 Starting contract analysis...")
            message_placeholder.markdown(
                "\n".join([f"<div class='processing-message'>{msg}</div>" 
                          for msg in st.session_state.processing_messages[pdf_name]]),
                unsafe_allow_html=True
            )
            
            # Validate we have text for analysis
            if not contract_text or len(contract_text.strip()) < 50:
                st.session_state.processing_messages[pdf_name].append("❌ Insufficient text extracted for analysis")
                return False, "Insufficient text extracted from PDF"
            
            # Run contract analysis
            contract_analyzer = ContractAnalyzer()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"{Path(pdf_name).stem}.json")
                
                try:
                    st.session_state.processing_messages[pdf_name].append(f"🤖 Sending {len(contract_text)} characters to AI analyzer...")
                    
                    analysis_results = contract_analyzer.analyze_contract(contract_text, output_path)
                    
                    if not analysis_results:
                        raise ValueError("Contract analyzer returned empty results")
                    
                    # Store analysis results in database (with raw JSON)
                    if pdf_id:
                        analysis_data = {
                            'pdf_id': pdf_id,
                            'analysis_date': datetime.now(),
                            'version': get_next_analysis_version(pdf_id),
                            'form_number': analysis_results.get('form_number'),
                            'pi_clause': str(analysis_results.get('pi_clause', False)).lower(),  # Store as VARCHAR
                            'ci_clause': str(analysis_results.get('ci_clause', False)).lower(),
                            'data_usage_mentioned': str(analysis_results.get('data_usage_mentioned', False)).lower(),
                            'data_limitations_exists': str(analysis_results.get('data_limitations_exists', False)).lower(),
                            'summary': analysis_results.get('summary'),
                            'raw_json': analysis_results,  # Store complete analysis JSON
                            'processed_by': 'streamlit_analyzer',
                            'processing_time': 0.0
                        }
                        
                        try:
                            analysis_id = store_analysis_data(analysis_data)
                            
                            # Store clauses
                            clauses = analysis_results.get('relevant_clauses', [])
                            if clauses:
                                clause_data = []
                                for i, clause in enumerate(clauses):
                                    clause_data.append({
                                        'clause_type': clause.get('type', 'unknown'),
                                        'clause_text': clause.get('text', ''),
                                        'clause_order': i + 1
                                    })
                                store_clause_data(clause_data, analysis_id)
                            
                            st.session_state.processing_messages[pdf_name].append(f"💾 Analysis stored in database (ID: {analysis_id})")
                        except Exception as e:
                            st.session_state.processing_messages[pdf_name].append(f"⚠️ Warning: Could not store analysis in database: {str(e)}")
                            logger.error(f"Database storage failed for {pdf_name}: {str(e)}")
                    
                    # Store analysis results in session state for display
                    file_stem = Path(pdf_name).stem
                    st.session_state.json_data[file_stem] = analysis_results
                    
                    st.session_state.processing_messages[pdf_name].append("✅ Contract analysis completed successfully")
                    return True, "Analysis completed successfully"
                    
                except Exception as e:
                    logger.error(f"Contract analysis failed for {pdf_name}: {str(e)}")
                    return False, f"Contract analysis failed: {str(e)}"
        
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return False, error_msg
            
    except Exception as e:
        logger.error(f"Processing failed for {pdf_name}: {str(e)}")
        return False, f"Processing failed: {str(e)}"
    finally:
        st.session_state.processing_messages[pdf_name].append("📝 Processing complete - Ready for review")
```

## 2. Update Loading Function to Use Stored Raw JSONs

```python
def load_existing_pdfs_from_database():
    """Load all existing PDFs from database into session state with complete data integrity"""
    try:
        from config.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Query to get all processed PDFs with their latest analysis AND raw JSONs
                query = """
                SELECT DISTINCT p.id, p.pdf_name, p.final_word_count, p.upload_date, 
                       p.final_content, p.raw_content, p.obfuscation_summary,
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
            (pdf_id, filename, file_size, upload_date, final_content, raw_content, 
             obfuscation_summary, form_number, summary, analysis_date,
             pi_clause, ci_clause, data_usage_mentioned, data_limitations_exists, 
             analysis_raw_json) = pdf_data
            
            # Skip if already loaded (from current session uploads)
            if filename in st.session_state.pdf_database_ids:
                continue
            
            # Store PDF metadata
            st.session_state.pdf_database_ids[filename] = pdf_id
            st.session_state.analysis_status[filename] = "Processed"
            
            # Parse and store raw parsing JSON if available
            raw_parsing_data = None
            if raw_content:
                try:
                    raw_parsing_data = json.loads(raw_content)
                except Exception as e:
                    print(f"Error parsing raw_content for {filename}: {e}")
            
            # Parse and store raw analysis JSON if available
            analysis_data = None
            if analysis_raw_json:
                try:
                    if isinstance(analysis_raw_json, str):
                        analysis_data = json.loads(analysis_raw_json)
                    else:
                        analysis_data = analysis_raw_json
                except Exception as e:
                    print(f"Error parsing analysis_raw_json for {filename}: {e}")
            
            # Create analysis data structure from stored raw JSON
            file_stem = Path(filename).stem
            if analysis_data:
                # Use the complete stored analysis JSON
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
                    'relevant_clauses': [],  # Will be loaded when needed
                    'analysis_date': analysis_date,
                    'loaded_from_database': True
                }
            
            # Store raw PDF data for page matching
            if raw_parsing_data and 'pages' in raw_parsing_data:
                # Use stored pages structure
                st.session_state.raw_pdf_data[filename] = {
                    'pages': raw_parsing_data['pages'],
                    'raw_parsing_json': raw_parsing_data
                }
            elif final_content:
                # Fallback: convert final_content to pages structure
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
            
            # Create processing messages for UI consistency
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

def load_pdf_clauses(pdf_id, file_stem):
    """Load relevant clauses for a specific PDF - now checks raw JSON first"""
    try:
        # First check if clauses are already in the raw analysis JSON
        if (file_stem in st.session_state.json_data and 
            'relevant_clauses' in st.session_state.json_data[file_stem] and
            st.session_state.json_data[file_stem]['relevant_clauses']):
            return st.session_state.json_data[file_stem]['relevant_clauses']
        
        # Otherwise load from database
        from config.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT c.clause_type, c.clause_text, c.clause_order
                FROM clauses c
                JOIN analyses a ON c.analysis_id = a.id
                WHERE a.pdf_id = %s
                ORDER BY c.clause_order
                """
                
                cur.execute(query, (pdf_id,))
                clauses = cur.fetchall()
        
        # Convert to the expected format
        relevant_clauses = []
        for clause_type, clause_text, clause_order in clauses:
            relevant_clauses.append({
                'type': clause_type,
                'text': clause_text,
                'order': clause_order
            })
        
        # Update the session data
        if file_stem in st.session_state.json_data:
            st.session_state.json_data[file_stem]['relevant_clauses'] = relevant_clauses
        
        return relevant_clauses
        
    except Exception as e:
        print(f"Error loading clauses: {e}")
        return []
```

## 3. Helper Function for Text Extraction

```python
def extract_contract_text_from_pages(pages_content, pdf_name):
    """Extract contract text from pages structure (for new PDFs)"""
    try:
        contract_text_parts = []
        
        for page_idx, page_data in enumerate(pages_content):
            if isinstance(page_data, dict):
                paragraphs = page_data.get('paragraphs', [])
                if paragraphs:
                    page_text = '\n'.join(str(para) for para in paragraphs if para and str(para).strip())
                    if page_text.strip():
                        contract_text_parts.append(page_text)
            elif isinstance(page_data, list):
                page_text = '\n'.join(str(para) for para in page_data if para and str(para).strip())
                if page_text.strip():
                    contract_text_parts.append(page_text)
            elif isinstance(page_data, str):
                if page_data.strip():
                    contract_text_parts.append(page_data)
        
        return '\n\n'.join(contract_text_parts)
        
    except Exception as e:
        print(f"Error extracting text from pages for {pdf_name}: {e}")
        return ""
```

## Key Improvements:

1. **Complete Raw JSON Storage**: 
   - Stores complete PDF parsing result in `raw_content` column
   - Stores complete analysis result in `raw_json` column

2. **Data Integrity**: 
   - When loading existing PDFs, uses stored raw JSONs for complete data reconstruction
   - Maintains exact same data structure as fresh processing

3. **Fallback Mechanisms**: 
   - If raw JSON is missing, falls back to individual fields
   - If pages structure is missing, converts from final_content

4. **Enhanced Loading**: 
   - Loads obfuscation summaries from stored data
   - Loads complete analysis results including all fields
   - Preserves clauses from raw JSON when available

5. **Better Error Handling**: 
   - Graceful degradation if JSON parsing fails
   - Clear status messages about data availability

Now when you load previous PDFs, they'll have exactly the same data integrity as freshly processed ones, including complete pages structure for page number matching and all analysis results.
