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
