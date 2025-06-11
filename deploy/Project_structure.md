Here are all the changes required to add the functionality to load previously processed PDFs from the database:

## 1. Update Database Schema (Add New Column)

**In `config/database.py`, update the `pdfs` table in `create_tables()` function:**

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

## 2. Update Database Storage Function

**In `config/database.py`, update `store_pdf_data()` function:**

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

## 3. Add Database Loading Functions

**Add these new functions to your main Streamlit file:**

```python
def get_previous_feedbacks(pdf_id):
    """Retrieve all previous feedbacks for a given PDF from database"""
    try:
        from config.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                SELECT feedback_date, form_number_feedback, general_feedback, rating, user_session_id
                FROM feedback 
                WHERE pdf_id = %s 
                ORDER BY feedback_date DESC
                """
                
                cur.execute(query, (pdf_id,))
                feedbacks = cur.fetchall()
                
                return feedbacks
        
    except Exception as e:
        print(f"Error retrieving feedbacks: {e}")
        return []

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
                LIMIT 100
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
            
            # Skip if already loaded (from current session uploads)
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
                # Use complete stored analysis JSON
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
    """Load relevant clauses for a specific PDF - checks raw JSON first"""
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

def convert_string_to_pages_structure(final_content, pdf_name):
    """Convert final_content string back to pages structure for page matching"""
    try:
        if not final_content:
            return []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in final_content.split('\n') if p.strip()]
        
        # Estimate pages (roughly 20-30 paragraphs per page)
        paragraphs_per_page = 25
        pages_content = []
        
        for i in range(0, len(paragraphs), paragraphs_per_page):
            page_paragraphs = paragraphs[i:i + paragraphs_per_page]
            page_data = {
                'page_number': (i // paragraphs_per_page) + 1,
                'paragraphs': page_paragraphs
            }
            pages_content.append(page_data)
        
        return pages_content
        
    except Exception as e:
        print(f"Error converting string to pages structure for {pdf_name}: {e}")
        return []
```

## 4. Update Session State Initialization

**In your main Streamlit file, update `initialize_session_state()` function:**

```python
def initialize_session_state():
    """Initialize all session state variables"""
    # Database initialization
    if 'database_initialized' not in st.session_state:
        try:
            from config.database import check_database_connection, initialize_database
            
            if check_database_connection():
                initialize_database()
                st.session_state.database_initialized = True
                st.session_state.database_status = "Connected"
            else:
                st.session_state.database_initialized = False
                st.session_state.database_status = "Failed to connect"
        except Exception as e:
            st.session_state.database_initialized = False
            st.session_state.database_status = f"Error: {str(e)}"
    
    # Session state variables
    session_vars = {
        'pdf_files': {},
        'json_data': {},
        'raw_pdf_data': {},
        'current_pdf': None,
        'analysis_status': {},
        'processing_messages': {},
        'pdf_database_ids': {},
        'search_text': None,
        'feedback_submitted': {},
        'obfuscation_summaries': {},
        'session_id': get_session_id(),
        'batch_processing_status': None,
        'batch_processed_count': 0,
        'processing_mode': None,
        'processing_in_progress': False,
        'existing_pdfs_loaded': False  # NEW: Track if existing PDFs are loaded
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    # Load existing PDFs from database on first run
    if (st.session_state.database_initialized and 
        not st.session_state.existing_pdfs_loaded):
        
        try:
            loaded_count = load_existing_pdfs_from_database()
            st.session_state.existing_pdfs_loaded = True
            
            if loaded_count > 0:
                print(f"📚 Loaded {loaded_count} existing PDFs from database")
        except Exception as e:
            print(f"❌ Failed to load existing PDFs: {e}")
```

## 5. Update Feedback Form with Previous Feedbacks

**Update the feedback form functions:**

```python
def render_previous_feedbacks(pdf_name):
    """Display previous feedbacks for the current PDF"""
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    
    if not pdf_id:
        return
    
    previous_feedbacks = get_previous_feedbacks(pdf_id)
    
    if not previous_feedbacks:
        st.info("📝 No previous feedback found for this document.")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Previous Feedback History")
    st.markdown(f"**{len(previous_feedbacks)} feedback(s) submitted for this document:**")
    
    for i, feedback in enumerate(previous_feedbacks):
        feedback_date, form_number, general_feedback, rating, session_id = feedback
        
        # Convert rating back to text
        rating_text = {5: "Correct", 3: "Partially Correct", 1: "Incorrect"}.get(rating, f"Rating: {rating}")
        
        with st.expander(f"📅 Feedback #{i+1} - {feedback_date.strftime('%Y-%m-%d %H:%M')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📋 Form Number:** {form_number}")
                st.markdown(f"**📝 Feedback:** {general_feedback}")
                
                # Show session info (truncated for privacy)
                session_display = f"{session_id[:8]}..." if session_id else "Unknown"
                st.caption(f"Session: {session_display}")
            
            with col2:
                # Show rating with appropriate emoji
                rating_emoji = {"Correct": "✅", "Partially Correct": "⚠️", "Incorrect": "❌"}.get(rating_text, "📊")
                st.markdown(f"**{rating_emoji} {rating_text}**")
                st.caption(f"📅 {feedback_date.strftime('%b %d, %Y')}")

def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF"""
    feedback_key = f"feedback_{file_stem}"
    
    # Check if feedback was already submitted
    if st.session_state.feedback_submitted.get(feedback_key, False):
        st.success("✅ Thank you! Your feedback has been submitted for this document.")
        if st.button("Submit New Feedback", key=f"new_feedback_{file_stem}"):
            st.session_state.feedback_submitted[feedback_key] = False
            st.rerun()
        
        # Show previous feedbacks after submission
        render_previous_feedbacks(pdf_name)
        return
    
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    st.subheader("📝 Your Feedback Matters")
    st.write("Help us improve our analysis by providing feedback on the results:")
    
    with st.form(f"feedback_form_{file_stem}"):
        # Form number selection (1-10)
        form_number = st.selectbox(
            "Select Form Number",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=0,  # Default to form 1
            help="Select the form number that best matches this document",
            key=f"form_number_{file_stem}"
        )
        
        # Rating selection
        rating = st.radio(
            "Is this analysis correct?",
            ["Correct", "Partially Correct", "Incorrect"],
            help="Rate the accuracy of the analysis",
            key=f"rating_{file_stem}"
        )
        
        # Convert rating to integer for database storage
        rating_map = {
            "Correct": 5,
            "Partially Correct": 3,
            "Incorrect": 1
        }
        rating_value = rating_map[rating]
        
        # Single feedback text box
        feedback_text = st.text_area(
            "Your feedback/comments",
            placeholder="Please provide your feedback or comments about this analysis...",
            height=120,
            help="Provide detailed feedback about the analysis accuracy, suggestions for improvement, etc.",
            key=f"feedback_text_{file_stem}"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            # Validation - require feedback text
            if not feedback_text.strip():
                st.error("Please provide some feedback before submitting.")
                return
            
            # Get PDF ID from session state
            pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
            
            if pdf_id:
                # Prepare feedback data according to database schema
                feedback_data = {
                    'pdf_id': pdf_id,
                    'feedback_date': datetime.now(),
                    'form_number_feedback': str(form_number),  # Store as TEXT
                    'general_feedback': feedback_text.strip(),
                    'rating': rating_value,  # INTEGER rating (1-5)
                    'user_session_id': get_session_id()
                }
                
                try:
                    from config.database import store_feedback_data
                    feedback_id = store_feedback_data(feedback_data)
                    
                    if feedback_id:
                        st.success("🎉 Thank you for your valuable feedback! It helps us improve our analysis.")
                        st.session_state.feedback_submitted[feedback_key] = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. No feedback ID returned.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to save feedback: {str(e)}")
                    print(f"Feedback submission error: {str(e)}")
            else:
                st.error("❌ Cannot submit feedback - PDF not found in database")
    
    # Show previous feedbacks below the form (always visible)
    render_previous_feedbacks(pdf_name)
    
    st.markdown("</div>", unsafe_allow_html=True)
```

## 6. Update Document List Section in Main Function

**In your main function, update the document list section:**

```python
        # Document list and selection - ALWAYS SHOW (even without uploaded files)
        st.subheader("📋 Available Documents")
        
        # Combine uploaded PDFs and existing PDFs from database
        all_pdfs = {}
        
        # Add existing PDFs from database
        for pdf_name in st.session_state.pdf_database_ids.keys():
            if pdf_name not in st.session_state.pdf_files:  # Don't duplicate uploaded files
                all_pdfs[pdf_name] = "database"
        
        # Add currently uploaded PDFs
        for pdf_name in st.session_state.pdf_files.keys():
            all_pdfs[pdf_name] = "uploaded"
        
        if all_pdfs:
            pdf_data = []
            for pdf_name, source in all_pdfs.items():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                
                # Get file size
                if source == "uploaded":
                    file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
                    size_display = f"{file_size:.1f}"
                else:
                    size_display = "From DB"
                
                # Status and source emojis
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                source_emoji = "📤" if source == "uploaded" else "💾"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'Source': source_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': size_display,
                    'DB ID': str(db_id)
                })
            
            # Display the grid
            pdf_df = pd.DataFrame(pdf_data)
            gb = GridOptionsBuilder.from_dataframe(pdf_df)
            gb.configure_selection(selection_mode='single', use_checkbox=False)
            gb.configure_grid_options(domLayout='normal')
            gb.configure_default_column(cellStyle={'fontSize': '14px'})
            gb.configure_column("PDF Name", cellStyle={'fontWeight': 'bold'})
            gridOptions = gb.build()

            grid_response = AgGrid(
                pdf_df,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                fit_columns_on_grid_load=True,
                theme='streamlit'
            )

            # Handle selection
            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                
                # Only process if it's an uploaded PDF and in on-demand mode
                if (st.session_state.processing_mode == "on_demand" and 
                    selected_pdf != st.session_state.get('current_pdf') and
                    st.session_state.analysis_status.get(selected_pdf) != "Processed" and
                    selected_pdf in st.session_state.pdf_files):  # Only process uploaded files
                    
                    # Process uploaded PDF
                    with st.spinner(f"🔄 Processing {selected_pdf} on-demand..."):
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[selected_pdf], 
                            selected_pdf, 
                            st.empty(),
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[selected_pdf] = "Processed"
                            st.success(f"✅ On-demand analysis complete for {selected_pdf}")
                        else:
                            st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {selected_pdf}: {result}")
                
                # Set current PDF for viewing (works for both uploaded and database PDFs)
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    
                    # Load clauses for database PDFs if not already loaded
                    file_stem = Path(selected_pdf).stem
                    if (selected_pdf in st.session_state.pdf_database_ids and 
                        file_stem in st.session_state.json_data and
                        not st.session_state.json_data[file_stem].get('relevant_clauses')):
                        
                        pdf_id = st.session_state.pdf_database_ids[selected_pdf]
                        load_pdf_clauses(pdf_id, file_stem)
        else:
            st.info("📄 No documents available. Upload PDFs above or check database connection.")
            
        # Show legend for source icons
        st.caption("📤 = Uploaded this session | 💾 = From database")
```

## 7. Update Sidebar Status

**In your main function, update the sidebar to show loaded PDFs:**

```python
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.session_state.database_initialized:
            st.markdown("<div class='database-status-success'>✅ Database Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='database-status-error'>❌ Database: {st.session_state.database_status}</div>", unsafe_allow_html=True)
        
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        # Show counts for uploaded vs database PDFs
        uploaded_count = len(st.session_state.pdf_files)
        database_count = len([name for name in st.session_state.pdf_database_ids.keys() if name not in st.session_state.pdf_files])
        processed_count = len([s for s in st.session_state.analysis_status.values() if s == "Processed"])
        
        st.write(f"**📤 Uploaded this session:** {uploaded_count}")
        st.write(f"**💾 Loaded from database:** {database_count}")
        st.write(f"**✅ Total processed:** {processed_count}")
        st.write(f"**🔄 Batch Status:** {st.session_state.batch_processing_status or 'Not started'}")
```

## Summary of Changes:

1. **Database Schema**: Added `raw_parsing_json JSONB` column to `pdfs` table
2. **Storage Function**: Updated to store raw parsing JSON during initial PDF creation
3. **Loading Functions**: New functions to load existing PDFs and their complete data
4. **Session Initialization**: Automatically loads existing PDFs on app start
5. **Feedback System**: Shows previous feedback history for each document
6. **UI Updates**: AgGrid shows both uploaded and database PDFs with source indicators
7. **Status Display**: Sidebar shows counts of uploaded vs database PDFs

The system now automatically loads all previously processed PDFs (from any user/session) when the app starts, providing complete data integrity and allowing users to view and provide feedback on any document in the database.
