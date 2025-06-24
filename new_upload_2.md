You want to manually create the database tables and only verify them programmatically. Here's what you need to update:

## 1. Remove table creation from `database.py`

**REMOVE** the entire `create_tables()` function and **REPLACE** it with a verification function:

```python
def verify_database_schema():
    """Verify that all required tables exist with correct schema"""
    
    required_tables = {
        'pdfs': [
            'id', 'pdf_name', 'file_hash', 'upload_date', 'processed_date',
            'layout', 'original_word_count', 'original_page_count', 'parsability',
            'final_word_count', 'final_page_count', 'avg_words_per_page',
            'raw_content', 'final_content', 'pdf_bytes', 'pdf_parsing_data',
            'obfuscation_applied', 'pages_removed_count', 'paragraphs_obfuscated_count',
            'obfuscation_summary', 'uploaded_by'
        ],
        'analyses': [
            'id', 'pdf_id', 'analysis_date', 'version', 'form_number',
            'pi_clause', 'ci_clause', 'data_usage_mentioned', 'data_limitations_exists',
            'summary', 'raw_json', 'processed_by', 'processing_time'
        ],
        'clauses': [
            'id', 'analysis_id', 'clause_type', 'clause_text', 'clause_order'
        ],
        'feedback': [
            'id', 'pdf_id', 'feedback_date', 'form_number_feedback',
            'general_feedback', 'rating', 'user_session_id'
        ],
        'batch_jobs': [
            'id', 'job_id', 'created_at', 'started_at', 'completed_at',
            'total_files', 'processed_files', 'failed_files', 'status',
            'total_pages_processed', 'total_pages_removed', 'total_paragraphs_obfuscated',
            'created_by', 'results_json', 'error_log'
        ],
        'users': [
            'id', 'session_id', 'created_at', 'last_active'
        ]
    }
    
    verification_results = {}
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for table_name, expected_columns in required_tables.items():
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table_name,))
                    
                    table_exists = cur.fetchone()[0]
                    
                    if table_exists:
                        # Check columns
                        cur.execute("""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_name = %s
                            ORDER BY ordinal_position
                        """, (table_name,))
                        
                        actual_columns = [row[0] for row in cur.fetchall()]
                        missing_columns = [col for col in expected_columns if col not in actual_columns]
                        extra_columns = [col for col in actual_columns if col not in expected_columns]
                        
                        verification_results[table_name] = {
                            'exists': True,
                            'missing_columns': missing_columns,
                            'extra_columns': extra_columns,
                            'status': 'OK' if not missing_columns else 'MISSING_COLUMNS'
                        }
                    else:
                        verification_results[table_name] = {
                            'exists': False,
                            'status': 'MISSING_TABLE'
                        }
        
        return verification_results
        
    except Exception as e:
        print(f"❌ Error verifying database schema: {str(e)}")
        return {}

def check_database_health():
    """Quick health check for database connection and basic tables"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if we can query the main tables
                cur.execute("SELECT COUNT(*) FROM pdfs")
                pdf_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM analyses")
                analysis_count = cur.fetchone()[0]
                
                return {
                    'connected': True,
                    'pdf_count': pdf_count,
                    'analysis_count': analysis_count
                }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e)
        }
```

## 2. Update `initialize_database()` function

**REPLACE** the existing `initialize_database()` function:

```python
def initialize_database():
    """Initialize database - verify tables exist (don't create them)"""
    
    try:
        print("🔌 Testing database connection...")
        
        # Test connection
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        print("✅ Database connection successful")
        
        # Verify schema instead of creating tables
        print("🔍 Verifying database schema...")
        verification_results = verify_database_schema()
        
        all_good = True
        for table_name, result in verification_results.items():
            if result['status'] == 'OK':
                print(f"✅ Table '{table_name}' - OK")
            elif result['status'] == 'MISSING_TABLE':
                print(f"❌ Table '{table_name}' - MISSING")
                all_good = False
            elif result['status'] == 'MISSING_COLUMNS':
                print(f"⚠️ Table '{table_name}' - Missing columns: {result['missing_columns']}")
                all_good = False
        
        if all_good:
            print("✅ Database schema verification complete - All tables OK")
        else:
            print("⚠️ Database schema verification complete - Some issues found")
            print("   Please create missing tables manually")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        raise
```

## 3. Update session state initialization in `main.py`

**REPLACE** the database initialization part in `initialize_session_state()`:

```python
def initialize_session_state():
    """Initialize all session state variables"""
    # Database initialization - ONLY VERIFY, DON'T CREATE
    if 'database_initialized' not in st.session_state:
        try:
            if check_database_connection():
                schema_ok = initialize_database()  # This now only verifies
                st.session_state.database_initialized = schema_ok
                st.session_state.database_status = "Connected" if schema_ok else "Schema issues"
            else:
                st.session_state.database_initialized = False
                st.session_state.database_status = "Failed to connect"
        except Exception as e:
            st.session_state.database_initialized = False
            st.session_state.database_status = f"Error: {str(e)}"
    
    # Rest of your session state variables...
```

## 4. Add a manual schema check function (optional)

Add this to `main.py` for debugging:

```python
def show_database_status():
    """Show detailed database status in sidebar"""
    with st.sidebar:
        st.subheader("🗄️ Database Status")
        
        if st.session_state.database_initialized:
            st.success("✅ Connected")
            
            # Show quick stats
            health = check_database_health()
            if health['connected']:
                st.write(f"PDFs: {health['pdf_count']}")
                st.write(f"Analyses: {health['analysis_count']}")
        else:
            st.error(f"❌ {st.session_state.database_status}")
        
        # Manual schema check button
        if st.button("🔍 Check Schema"):
            verification = verify_database_schema()
            
            for table, result in verification.items():
                if result['status'] == 'OK':
                    st.success(f"✅ {table}")
                elif result['status'] == 'MISSING_TABLE':
                    st.error(f"❌ {table} - Missing")
                else:
                    st.warning(f"⚠️ {table} - Issues")

# Call this in your main() function:
# show_database_status()
```

## 5. SQL scripts for manual table creation

Create these tables manually in your database:

```sql
-- Create tables manually in your database

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pdfs (
    id SERIAL PRIMARY KEY,
    pdf_name VARCHAR NOT NULL,
    file_hash VARCHAR UNIQUE NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP,
    layout VARCHAR,
    original_word_count INTEGER,
    original_page_count INTEGER,
    parsability BOOLEAN,
    final_word_count INTEGER,
    final_page_count INTEGER,
    avg_words_per_page FLOAT,
    raw_content TEXT,
    final_content TEXT,
    pdf_bytes BYTEA,
    pdf_parsing_data JSONB,
    obfuscation_applied BOOLEAN DEFAULT TRUE,
    pages_removed_count INTEGER DEFAULT 0,
    paragraphs_obfuscated_count INTEGER DEFAULT 0,
    obfuscation_summary JSONB,
    uploaded_by VARCHAR
);

CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version INTEGER DEFAULT 1,
    form_number VARCHAR,
    pi_clause VARCHAR,
    ci_clause VARCHAR,
    data_usage_mentioned VARCHAR,
    data_limitations_exists VARCHAR,
    summary TEXT,
    raw_json JSONB,
    processed_by VARCHAR,
    processing_time FLOAT
);

CREATE TABLE clauses (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    clause_type VARCHAR,
    clause_text TEXT,
    clause_order INTEGER
);

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    form_number_feedback TEXT,
    general_feedback TEXT,
    rating INTEGER,
    user_session_id VARCHAR
);

CREATE TABLE batch_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_files INTEGER,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'pending',
    total_pages_processed INTEGER DEFAULT 0,
    total_pages_removed INTEGER DEFAULT 0,
    total_paragraphs_obfuscated INTEGER DEFAULT 0,
    created_by VARCHAR,
    results_json JSONB,
    error_log TEXT
);

-- Create indexes
CREATE INDEX idx_pdfs_file_hash ON pdfs(file_hash);
CREATE INDEX idx_pdfs_upload_date ON pdfs(upload_date);
CREATE INDEX idx_analyses_pdf_id ON analyses(pdf_id);
CREATE INDEX idx_analyses_version ON analyses(pdf_id, version);
CREATE INDEX idx_clauses_analysis_id ON clauses(analysis_id);
CREATE INDEX idx_feedback_pdf_id ON feedback(pdf_id);
CREATE INDEX idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX idx_users_session_id ON users(session_id);
```

## Summary of changes:

**REMOVE:**
- `create_tables()` function
- Automatic table creation logic
- Index creation code

**ADD:**
- `verify_database_schema()` function
- `check_database_health()` function
- Schema verification on startup only
- Optional manual schema check button

**CHANGE:**
- `initialize_database()` now only verifies, doesn't create
- Database initialization happens only on app startup
- Better error handling for missing tables

The verification will only run when the app starts, not after each PDF upload.



---

Good point! If you can’t display PDFs larger than 1.5 MB, there’s no need to store the bytes. Here’s how to update the code:

## 1. Update the Enhanced PDF Handler

Modify the `process_pdf_with_database()` function in `enhanced_pdf_handler.py`:

```python
def process_pdf_with_database(self, pdf_path: str = None, pdf_bytes: bytes = None, 
                            pdf_name: str = None, uploaded_by: str = "system") -> Dict[str, Any]:
    """Process a PDF file through the complete pipeline and store in database."""
    
    try:
        # ... your existing validation code ...
        
        # Check PDF size and decide whether to store bytes
        pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
        store_pdf_bytes = pdf_size_mb <= 1.5
        
        print(f"🔍 DEBUG: PDF size: {pdf_size_mb:.2f} MB")
        print(f"🔍 DEBUG: Will store PDF bytes: {store_pdf_bytes}")
        
        # ... existing processing code (extract content, obfuscation, etc.) ...
        
        # Prepare data for database storage
        pdf_data = {
            'pdf_name': pdf_name,
            'file_hash': file_hash,
            'upload_date': datetime.now(),
            'processed_date': datetime.now(),
            'layout': layout_type,
            'original_word_count': original_word_count,
            'original_page_count': original_page_count,
            'parsability': True,
            'final_word_count': final_word_count,
            'final_page_count': final_page_count,
            'avg_words_per_page': avg_words_per_page,
            'raw_content': raw_content,
            'final_content': final_content,
            'pdf_bytes': pdf_bytes if store_pdf_bytes else None,  # Store NULL if too large
            'pdf_parsing_data': complete_processing_result,
            'obfuscation_applied': obfuscation_applied,
            'pages_removed_count': obfuscation_summary.get('pages_removed_count', 0),
            'paragraphs_obfuscated_count': obfuscation_summary.get('paragraphs_obfuscated_count', 0),
            'obfuscation_summary': obfuscation_summary,
            'uploaded_by': uploaded_by,
            'file_size_mb': pdf_size_mb  # Store size for reference
        }
        
        # Add file size info to the return result
        complete_processing_result["file_size_mb"] = pdf_size_mb
        complete_processing_result["pdf_bytes_stored"] = store_pdf_bytes
        
        # ... rest of your existing code ...
        
        return complete_processing_result
        
    except Exception as e:
        # ... existing error handling ...
```

## 2. Update Database Schema

Add a file size column to track the original size. Update your manual table creation:

```sql
-- Add this column to your pdfs table:
ALTER TABLE pdfs ADD COLUMN file_size_mb FLOAT;

-- Or if creating fresh table, include:
CREATE TABLE pdfs (
    -- ... your existing columns ...
    pdf_bytes BYTEA,                    -- Will be NULL for files > 1.5MB
    file_size_mb FLOAT,                 -- NEW: Store original file size
    pdf_parsing_data JSONB,
    -- ... rest of columns ...
);
```

## 3. Update the database storage function

Modify `store_pdf_data()` in `database.py`:

```python
def store_pdf_data(pdf_data: Dict[str, Any]) -> int:
    """Store PDF data after parsing and obfuscation process"""
    
    sql = """
        INSERT INTO pdfs (
            pdf_name, file_hash, upload_date, processed_date,
            layout, original_word_count, original_page_count, parsability,
            final_word_count, final_page_count, avg_words_per_page,
            raw_content, final_content, pdf_bytes, file_size_mb, pdf_parsing_data,
            obfuscation_applied, pages_removed_count, paragraphs_obfuscated_count,
            obfuscation_summary, uploaded_by
        ) VALUES (
            %(pdf_name)s, %(file_hash)s, %(upload_date)s, %(processed_date)s,
            %(layout)s, %(original_word_count)s, %(original_page_count)s, %(parsability)s,
            %(final_word_count)s, %(final_page_count)s, %(avg_words_per_page)s,
            %(raw_content)s, %(final_content)s, %(pdf_bytes)s, %(file_size_mb)s, %(pdf_parsing_data)s,
            %(obfuscation_applied)s, %(pages_removed_count)s, %(paragraphs_obfuscated_count)s,
            %(obfuscation_summary)s, %(uploaded_by)s
        ) RETURNING id
    """
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # ... your existing data preparation code ...
                
                # Handle NULL pdf_bytes explicitly
                if 'pdf_bytes' not in prepared_data or prepared_data['pdf_bytes'] is None:
                    prepared_data['pdf_bytes'] = None
                    print(f"🔍 DEBUG: Storing NULL pdf_bytes for large file")
                
                cur.execute(sql, prepared_data)
                pdf_id = cur.fetchone()[0]
                conn.commit()
                
                print(f"✅ PDF stored successfully with ID: {pdf_id}")
                return pdf_id
                
            except Exception as e:
                print(f"❌ Error storing PDF data: {str(e)}")
                conn.rollback()
                raise
```

## 4. Update the loading function

Modify `load_pdf_from_database_with_analysis()` in `main.py`:

```python
def load_pdf_from_database_with_analysis(pdf_id, pdf_name):
    try:
        show_processing_overlay(
            message=f"Loading {pdf_name}",
            submessage="Retrieving PDF and analysis data from database..."
        )
        
        complete_data = load_complete_pdf_data(pdf_id)
        
        if not complete_data:
            st.error("Could not load PDF data")
            return False
        
        # Check if PDF bytes are available
        pdf_bytes = complete_data.get('pdf_bytes')
        file_size_mb = complete_data.get('file_size_mb', 0)
        
        if pdf_bytes:
            # PDF bytes available - load for viewing
            st.session_state.pdf_files[pdf_name] = bytes(pdf_bytes)
            st.session_state.loaded_pdfs.add(pdf_name)
            st.session_state.current_pdf = pdf_name
            st.success(f"✅ Loaded {pdf_name} ({file_size_mb:.2f} MB)")
        else:
            # No PDF bytes - large file
            st.session_state.loaded_pdfs.add(pdf_name)
            st.session_state.current_pdf = pdf_name
            st.warning(f"⚠️ {pdf_name} ({file_size_mb:.2f} MB) - PDF viewer not available (file too large)")
            st.info("Analysis data and page navigation are still available")
        
        st.session_state.pdf_database_ids[pdf_name] = complete_data["id"]
        
        # ... rest of your analysis and parsing data loading code ...
        
        return True
        
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return False
    finally:
        hide_processing_overlay()
```

## 5. Update the PDF viewer to handle missing bytes

Update your PDF viewer section in the middle pane:

```python
# Middle pane: PDF viewer
with col2:
    st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
    st.header("📖 Document Viewer")
    
    if st.session_state.current_pdf:
        if st.session_state.current_pdf in st.session_state.pdf_files:
            # PDF bytes available - show viewer
            current_pdf_bytes = st.session_state.pdf_files[st.session_state.current_pdf]
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.subheader(f"📄 {st.session_state.current_pdf}")
            with col_info2:
                file_size_mb = len(current_pdf_bytes) / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            # Show PDF viewer
            pdf_display = display_pdf_iframe(current_pdf_bytes, st.session_state.search_text)
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        else:
            # PDF bytes not available - show placeholder
            st.subheader(f"📄 {st.session_state.current_pdf}")
            
            # Get file size from database if available
            pdf_id = st.session_state.pdf_database_ids.get(st.session_state.current_pdf)
            if pdf_id:
                try:
                    from config.database import db
                    with db.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT file_size_mb FROM pdfs WHERE id = %s", (pdf_id,))
                            result = cur.fetchone()
                            file_size = result[0] if result else "Unknown"
                except:
                    file_size = "Unknown"
            else:
                file_size = "Unknown"
            
            st.info(f"📊 File Size: {file_size} MB")
            st.warning("🔍 PDF viewer not available - file is too large (>1.5 MB)")
            st.markdown("""
            **Available options:**
            - ✅ View analysis results in the right pane
            - ✅ Use clause page navigation
            - ✅ Submit feedback
            - ❌ PDF visual preview (file too large)
            """)
    else:
        st.info("Select a PDF to view")
    
    st.markdown('</div>', unsafe_allow_html=True)
```

## 6. Update the AgGrid to show file size and PDF availability

Update your `create_unified_pdf_grid()` function:

```python
def create_unified_pdf_grid():
    """Create AgGrid showing all available PDFs (database + session)"""
    pdf_data = []
    
    # Add PDFs from database metadata
    for pdf_record in st.session_state.available_pdfs_metadata:
        file_size_kb = pdf_record.get('file_size_bytes', 0) / 1024
        file_size_mb = file_size_kb / 1024
        
        # Determine if PDF viewer is available
        has_pdf_bytes = pdf_record.get('file_size_bytes', 0) > 0 and file_size_mb <= 1.5
        viewer_status = "📖 Viewable" if has_pdf_bytes else "📊 Data Only"
        
        # ... existing status logic ...
        
        pdf_data.append({
            'Status': status_emoji,
            'Load': load_status,
            'PDF Name': pdf_record['pdf_name'],
            'Size (MB)': f"{file_size_mb:.2f}",  # Change to MB
            'Viewer': viewer_status,              # NEW column
            'Pages': f"{pdf_record.get('final_page_count', 0)}",
            'Words': f"{pdf_record.get('final_word_count', 0)}",
            'Analysis': analysis_status,
            'DB ID': str(pdf_record['id']),
            'Upload Date': pdf_record.get('upload_date', '').split('T')[0] if pdf_record.get('upload_date') else '',
            'Source': 'Database'
        })
    
    # ... rest of your existing code for session PDFs ...
    
    return pdf_data
```

## Summary of changes:

1. **Storage Logic**: Only store PDF bytes if file ≤ 1.5 MB
1. **Database Schema**: Added `file_size_mb` column
1. **Loading Logic**: Handle NULL pdf_bytes gracefully
1. **UI Updates**: Show appropriate messages for large files
1. **Grid Display**: Show viewer availability status

This saves significant database space while still providing full analysis functionality for large files!​​​​​​​​​​​​​​​​



---
---
---


The issue is that for PDFs larger than 1.5 MB, we’re not storing the PDF bytes but we’re still storing the PDF record in the database with all the metadata. The feedback system should still work because the PDF record exists with a valid ID.

Let me help you debug this:

## 1. Debug the feedback issue

Add debug information to your feedback submission to see what’s happening:

```python
def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF"""
    feedback_key = f"feedback_{file_stem}"
    
    # DEBUG: Check PDF database ID
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    st.write(f"🔍 DEBUG: PDF Name: {pdf_name}")
    st.write(f"🔍 DEBUG: PDF ID in session: {pdf_id}")
    st.write(f"🔍 DEBUG: PDF in database_ids: {pdf_name in st.session_state.pdf_database_ids}")
    
    # Check if feedback was already submitted
    if st.session_state.feedback_submitted.get(feedback_key, False):
        st.success("✅ Thank you! Your feedback has been submitted for this document.")
        if st.button("Submit New Feedback", key=f"new_feedback_{file_stem}"):
            st.session_state.feedback_submitted[feedback_key] = False
            st.rerun()
        return
    
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    st.subheader("📝 Your Feedback Matters")
    st.write("Help us improve our analysis by providing feedback on the results:")
    
    with st.form(f"feedback_form_{file_stem}"):
        # ... your existing form fields ...
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            # Validation
            if not feedback_text.strip():
                st.error("Please provide some feedback before submitting.")
                return
            
            # DEBUG: Show what we're trying to submit
            st.write(f"🔍 DEBUG: Attempting to submit feedback for PDF ID: {pdf_id}")
            
            if pdf_id:
                # Verify PDF exists in database
                try:
                    from config.database import get_pdf_by_id
                    pdf_record = get_pdf_by_id(pdf_id)
                    st.write(f"🔍 DEBUG: PDF record found: {pdf_record is not None}")
                    
                    if pdf_record:
                        st.write(f"🔍 DEBUG: PDF record name: {pdf_record.get('pdf_name')}")
                        st.write(f"🔍 DEBUG: PDF has bytes: {pdf_record.get('pdf_bytes') is not None}")
                    
                except Exception as e:
                    st.error(f"🔍 DEBUG: Error checking PDF record: {str(e)}")
                
                # Prepare feedback data
                feedback_data = {
                    'pdf_id': pdf_id,
                    'feedback_date': datetime.now(),
                    'form_number_feedback': form_number,
                    'general_feedback': feedback_text.strip(),
                    'rating': rating_value,
                    'user_session_id': get_session_id()
                }
                
                try:
                    feedback_id = store_feedback_data(feedback_data)
                    
                    if feedback_id:
                        st.success("🎉 Thank you for your valuable feedback!")
                        st.session_state.feedback_submitted[feedback_key] = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. No feedback ID returned.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to save feedback: {str(e)}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
            else:
                st.error("❌ Cannot submit feedback - PDF not found in database")
                st.write("🔍 DEBUG: Available PDF database IDs:")
                for name, id_val in st.session_state.pdf_database_ids.items():
                    st.write(f"  - {name}: {id_val}")
    
    st.markdown("</div>", unsafe_allow_html=True)
```

## 2. Check if the PDF ID is being stored correctly for large files

Update your loading function to ensure PDF ID is stored even for large files:

```python
def load_pdf_from_database_with_analysis(pdf_id, pdf_name):
    try:
        show_processing_overlay(
            message=f"Loading {pdf_name}",
            submessage="Retrieving PDF and analysis data from database..."
        )
        
        complete_data = load_complete_pdf_data(pdf_id)
        
        if not complete_data:
            st.error("Could not load PDF data")
            return False
        
        # IMPORTANT: Always store the PDF database ID, regardless of file size
        st.session_state.pdf_database_ids[pdf_name] = complete_data["id"]
        st.session_state.loaded_pdfs.add(pdf_name)
        st.session_state.current_pdf = pdf_name
        
        # Check if PDF bytes are available
        pdf_bytes = complete_data.get('pdf_bytes')
        file_size_mb = complete_data.get('file_size_mb', 0)
        
        if pdf_bytes:
            # PDF bytes available - load for viewing
            st.session_state.pdf_files[pdf_name] = bytes(pdf_bytes)
            st.success(f"✅ Loaded {pdf_name} ({file_size_mb:.2f} MB) - PDF viewer available")
        else:
            # No PDF bytes - large file, but still store ID for feedback
            st.warning(f"⚠️ {pdf_name} ({file_size_mb:.2f} MB) - PDF viewer not available (file too large)")
            st.info("✅ Analysis data and feedback are still available")
        
        # DEBUG: Confirm PDF ID is stored
        st.write(f"🔍 DEBUG: Stored PDF ID {complete_data['id']} for {pdf_name}")
        
        # ... rest of your analysis and parsing data loading code ...
        
        return True
        
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return False
    finally:
        hide_processing_overlay()
```

## 3. Verify the database query includes large files

Update your `get_all_pdfs_metadata()` function to ensure it includes files with NULL pdf_bytes:

```python
def get_all_pdfs_metadata() -> List[Dict[str, Any]]:
    """Get metadata for all PDFs in database (including large files with NULL bytes)"""
    
    sql = """
        SELECT 
            p.id,
            p.pdf_name,
            p.file_hash,
            p.upload_date,
            p.processed_date,
            p.layout,
            p.original_page_count,
            p.final_page_count,
            p.final_word_count,
            p.avg_words_per_page,
            p.obfuscation_applied,
            p.pages_removed_count,
            p.uploaded_by,
            p.file_size_mb,
            -- Analysis status
            CASE 
                WHEN COUNT(a.id) > 0 THEN 'Analyzed'
                ELSE 'Parsed Only'
            END as analysis_status,
            MAX(a.version) as latest_analysis_version,
            COUNT(a.id) as analysis_count,
            -- Size calculation - use file_size_mb if available, otherwise calculate from pdf_bytes
            CASE 
                WHEN p.file_size_mb IS NOT NULL THEN p.file_size_mb * 1024 * 1024
                WHEN p.pdf_bytes IS NOT NULL THEN LENGTH(p.pdf_bytes)
                ELSE 0
            END as file_size_bytes
        FROM pdfs p
        LEFT JOIN analyses a ON p.id = a.pdf_id
        GROUP BY p.id, p.pdf_name, p.file_hash, p.upload_date, p.processed_date,
                 p.layout, p.original_page_count, p.final_page_count, 
                 p.final_word_count, p.avg_words_per_page, p.obfuscation_applied,
                 p.pages_removed_count, p.uploaded_by, p.file_size_mb, p.pdf_bytes
        ORDER BY p.processed_date DESC
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            results = cur.fetchall()
            return [dict(row) for row in results]
```

## 4. Test the feedback system

Add a test button to verify the feedback system works:

```python
# Add this temporarily to test feedback for large files
if st.button("🧪 Test Feedback System"):
    current_pdf = st.session_state.get('current_pdf')
    if current_pdf:
        pdf_id = st.session_state.pdf_database_ids.get(current_pdf)
        st.write(f"Current PDF: {current_pdf}")
        st.write(f"PDF ID: {pdf_id}")
        
        if pdf_id:
            try:
                from config.database import get_pdf_by_id
                pdf_record = get_pdf_by_id(pdf_id)
                st.write(f"PDF found in database: {pdf_record is not None}")
                
                if pdf_record:
                    st.write(f"PDF name in DB: {pdf_record['pdf_name']}")
                    st.write(f"Has PDF bytes: {pdf_record.get('pdf_bytes') is not None}")
                    st.write(f"File size MB: {pdf_record.get('file_size_mb')}")
                    st.success("✅ PDF exists in database - feedback should work")
                else:
                    st.error("❌ PDF not found in database")
            except Exception as e:
                st.error(f"Error checking PDF: {str(e)}")
        else:
            st.error("No PDF ID found in session state")
    else:
        st.warning("No current PDF selected")
```

Run this debug code and let me know:

1. **What does the debug output show** when you try to submit feedback for a large PDF?
1. **Is the PDF ID being stored correctly** in `st.session_state.pdf_database_ids`?
1. **Does the PDF record exist** in the database when you run the test?
1. **What’s the exact error message** you’re getting?

This will help identify exactly where the feedback system is failing for large files.​​​​​​​​​​​​​​​​


