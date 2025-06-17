You're absolutely right! Let's take it step by step and focus on getting the database schema correct first.

## Step 1: Update Database Schema in `database.py`

### Update the `create_tables()` function:

```python
def create_tables():
    """Create all required tables if they don't exist"""
    
    table_schemas = {
        'users': """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
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
                pdf_bytes BYTEA,                    -- NEW: Store actual PDF file
                pdf_parsing_data JSONB,             -- NEW: Store complete PDF handler JSON data
                
                -- Obfuscation tracking
                obfuscation_applied BOOLEAN DEFAULT TRUE,
                pages_removed_count INTEGER DEFAULT 0,
                paragraphs_obfuscated_count INTEGER DEFAULT 0,
                obfuscation_summary JSONB,
                
                -- User tracking
                uploaded_by VARCHAR
            )
        """,
        
        'analyses': """
            CREATE TABLE IF NOT EXISTS analyses (
                id SERIAL PRIMARY KEY,
                pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                
                -- Analysis results
                form_number VARCHAR,
                pi_clause VARCHAR,
                ci_clause VARCHAR,
                data_usage_mentioned VARCHAR,
                data_limitations_exists VARCHAR,
                summary TEXT,
                raw_json JSONB,
                
                -- Processing metadata
                processed_by VARCHAR,
                processing_time FLOAT
            )
        """,
        
        'clauses': """
            CREATE TABLE IF NOT EXISTS clauses (
                id SERIAL PRIMARY KEY,
                analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
                clause_type VARCHAR,
                clause_text TEXT,
                clause_order INTEGER
            )
        """,
        
        'feedback': """
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
                feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                form_number_feedback TEXT,
                general_feedback TEXT,
                rating INTEGER,
                user_session_id VARCHAR
            )
        """,
        
        'batch_jobs': """
            CREATE TABLE IF NOT EXISTS batch_jobs (
                id SERIAL PRIMARY KEY,
                job_id VARCHAR UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                
                -- Job configuration
                total_files INTEGER,
                processed_files INTEGER DEFAULT 0,
                failed_files INTEGER DEFAULT 0,
                status VARCHAR DEFAULT 'pending',
                
                -- Batch processing summary
                total_pages_processed INTEGER DEFAULT 0,
                total_pages_removed INTEGER DEFAULT 0,
                total_paragraphs_obfuscated INTEGER DEFAULT 0,
                
                -- User tracking
                created_by VARCHAR,
                
                -- Job results
                results_json JSONB,
                error_log TEXT
            )
        """
    }
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                for table_name, schema in table_schemas.items():
                    cur.execute(schema)
                    print(f"✅ Table '{table_name}' created/verified")
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_pdfs_file_hash ON pdfs(file_hash)",
                    "CREATE INDEX IF NOT EXISTS idx_pdfs_upload_date ON pdfs(upload_date)",
                    "CREATE INDEX IF NOT EXISTS idx_analyses_pdf_id ON analyses(pdf_id)",
                    "CREATE INDEX IF NOT EXISTS idx_analyses_version ON analyses(pdf_id, version)",
                    "CREATE INDEX IF NOT EXISTS idx_clauses_analysis_id ON clauses(analysis_id)",
                    "CREATE INDEX IF NOT EXISTS idx_feedback_pdf_id ON feedback(pdf_id)",
                    "CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)",
                    "CREATE INDEX IF NOT EXISTS idx_users_session_id ON users(session_id)"
                ]
                
                for index in indexes:
                    cur.execute(index)
                
                conn.commit()
                print("✅ Database indexes created/verified")
                
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        raise
```

## Step 2: Update the `store_pdf_data()` function

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
            try:
                # Prepare data for storage
                prepared_data = pdf_data.copy()
                
                # Handle datetime objects
                if 'upload_date' in prepared_data and isinstance(prepared_data['upload_date'], datetime):
                    prepared_data['upload_date'] = prepared_data['upload_date'].isoformat()
                if 'processed_date' in prepared_data and isinstance(prepared_data['processed_date'], datetime):
                    prepared_data['processed_date'] = prepared_data['processed_date'].isoformat()
                
                # Handle JSON fields - convert dicts to JSON strings for JSONB storage
                if 'obfuscation_summary' in prepared_data and isinstance(prepared_data['obfuscation_summary'], dict):
                    # For JSONB columns, psycopg2 can handle dicts directly, but let's be explicit
                    pass  # Keep as dict for JSONB
                
                if 'pdf_parsing_data' in prepared_data and isinstance(prepared_data['pdf_parsing_data'], dict):
                    # Keep as dict for JSONB - psycopg2 will handle the conversion
                    pass
                
                # Ensure pdf_bytes is bytes
                if 'pdf_bytes' in prepared_data and not isinstance(prepared_data['pdf_bytes'], bytes):
                    raise ValueError(f"pdf_bytes must be bytes, got {type(prepared_data['pdf_bytes'])}")
                
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

## Step 3: Add a simple test function in `main.py`

```python
def test_database_storage():
    """Test function to verify database storage works correctly"""
    st.subheader("🧪 Database Storage Test")
    
    if st.button("Test Simple Storage"):
        try:
            from config.database import store_pdf_data
            from datetime import datetime
            import hashlib
            
            # Create test data
            test_pdf_bytes = b"This is test PDF content for database storage test"
            test_hash = hashlib.sha256(test_pdf_bytes).hexdigest()
            
            test_data = {
                'pdf_name': f'test_{datetime.now().strftime("%H%M%S")}.pdf',
                'file_hash': test_hash,
                'upload_date': datetime.now(),
                'processed_date': datetime.now(),
                'layout': 'single_column',
                'original_word_count': 100,
                'original_page_count': 1,
                'parsability': True,
                'final_word_count': 95,
                'final_page_count': 1,
                'avg_words_per_page': 95.0,
                'raw_content': 'This is test raw content',
                'final_content': 'This is test final content',
                'pdf_bytes': test_pdf_bytes,
                'pdf_parsing_data': {
                    'test_key': 'test_value',
                    'pages': [
                        {'page_number': 1, 'paragraphs': ['Test paragraph 1', 'Test paragraph 2']}
                    ],
                    'metadata': {'processing_time': '2023-01-01T10:00:00'}
                },
                'obfuscation_applied': True,
                'pages_removed_count': 0,
                'paragraphs_obfuscated_count': 2,
                'obfuscation_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'methods_applied': {'test': True}
                },
                'uploaded_by': 'test_user'
            }
            
            # Try to store
            pdf_id = store_pdf_data(test_data)
            st.success(f"✅ Test data stored successfully with ID: {pdf_id}")
            
            # Try to retrieve
            from config.database import get_pdf_by_id
            retrieved = get_pdf_by_id(pdf_id)
            
            if retrieved:
                st.success("✅ Test data retrieved successfully")
                st.json({
                    'id': retrieved['id'],
                    'pdf_name': retrieved['pdf_name'],
                    'file_hash': retrieved['file_hash'][:10] + '...',
                    'pdf_bytes_length': len(retrieved['pdf_bytes']) if retrieved['pdf_bytes'] else 0,
                    'pdf_parsing_data_keys': list(retrieved['pdf_parsing_data'].keys()) if retrieved['pdf_parsing_data'] else []
                })
            else:
                st.error("❌ Could not retrieve test data")
                
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    if st.button("Check Database Schema"):
        try:
            from config.database import db
            
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check pdfs table schema
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = 'pdfs'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()
                    
                    st.write("**PDFs table schema:**")
                    for col_name, data_type, nullable in columns:
                        st.write(f"  - `{col_name}`: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                    
                    # Check if new columns exist
                    column_names = [col[0] for col in columns]
                    required_new_columns = ['pdf_bytes', 'pdf_parsing_data']
                    
                    st.write("**New columns check:**")
                    for col in required_new_columns:
                        if col in column_names:
                            st.write(f"  ✅ `{col}` exists")
                        else:
                            st.write(f"  ❌ `{col}` missing")
                    
                    # Count existing records
                    cur.execute("SELECT COUNT(*) FROM pdfs")
                    count = cur.fetchone()[0]
                    st.write(f"**Current PDF records:** {count}")
                    
        except Exception as e:
            st.error(f"Schema check failed: {str(e)}")
```

## Step 4: Add the test to your main function

```python
# In your main() function, add this temporarily at the top:
def main():
    # Initialize session state
    initialize_session_state()
    logger = ECFRLogger()
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>📄 Enhanced Contract Analysis Platform</h1>
        <p>AI-powered contract analysis with privacy protection and intelligent feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ADD THIS TEST SECTION TEMPORARILY
    with st.expander("🧪 Database Testing", expanded=True):
        test_database_storage()
    
    # ... rest of your existing main() function ...
```

## Step 5: Test the setup

1. **Run your application**
2. **Check the database schema** using the "Check Database Schema" button
3. **Test storage** using the "Test Simple Storage" button
4. **Verify** that both operations work correctly

Once we confirm the database schema and storage work correctly, we can move to the next step of updating the PDF handler to use this new schema properly.

Let me know the results of these tests, and we'll proceed step by step!
