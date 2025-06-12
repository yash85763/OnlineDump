# config/database.py - Simple database configuration using psycopg2

import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from urllib.parse import quote_plus
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class DatabaseConfig:
    """Database configuration and connection management using psycopg2"""
    
    def __init__(self):
        self.connection_pool = None
        self._initialize_connection_pool()
    
    def _get_database_config(self):
        """Get database configuration from environment variables"""
        
        # Option 1: Use full DATABASE_URL if provided
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
        
        # Option 2: Construct from individual parameters
        db_username = os.getenv('DB_USERNAME')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME')
        
        # Validate required parameters
        if not all([db_username, db_password, db_host, db_name]):
            missing_vars = []
            if not db_username: missing_vars.append('DB_USERNAME')
            if not db_password: missing_vars.append('DB_PASSWORD') 
            if not db_host: missing_vars.append('DB_HOST')
            if not db_name: missing_vars.append('DB_NAME')
            
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_username,
            'password': db_password,
            'options': '-c search_path=public'  # Use public schema
        }
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for efficient database connections"""
        
        try:
            db_config = self._get_database_config()
            
            if isinstance(db_config, str):
                # Using DATABASE_URL
                self.connection_pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dsn=db_config
                )
            else:
                # Using individual parameters
                self.connection_pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    **db_config
                )
            
            print("✅ Database connection pool initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize database connection pool: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool with automatic cleanup"""
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            # Set public schema
            with conn.cursor() as cur:
                cur.execute("SET search_path TO public")
            conn.commit()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()

# Global database instance
db = DatabaseConfig()

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

def initialize_database():
    """Initialize database - create tables if they don't exist"""
    
    try:
        print("🔌 Testing database connection...")
        
        # Test connection
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        print("✅ Database connection successful")
        
        # Create tables
        print("🏗️ Creating/verifying database tables...")
        create_tables()
        
        print("✅ Database initialization complete")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        raise

# Data storage functions for different processing stages

def store_pdf_data(pdf_data: Dict[str, Any]) -> int:
    """Store PDF data after parsing and obfuscation process"""
    
    sql = """
        INSERT INTO pdfs (
            pdf_name, file_hash, upload_date, processed_date,
            layout, original_word_count, original_page_count, parsability,
            final_word_count, final_page_count, avg_words_per_page,
            raw_content, final_content, obfuscation_applied,
            pages_removed_count, paragraphs_obfuscated_count,
            obfuscation_summary, uploaded_by
        ) VALUES (
            %(pdf_name)s, %(file_hash)s, %(upload_date)s, %(processed_date)s,
            %(layout)s, %(original_word_count)s, %(original_page_count)s, %(parsability)s,
            %(final_word_count)s, %(final_page_count)s, %(avg_words_per_page)s,
            %(raw_content)s, %(final_content)s, %(obfuscation_applied)s,
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
            
            # Convert obfuscation_summary to JSON string if it's a dict
            if 'obfuscation_summary' in pdf_data and isinstance(pdf_data['obfuscation_summary'], dict):
                pdf_data['obfuscation_summary'] = json.dumps(pdf_data['obfuscation_summary'])
            
            cur.execute(sql, pdf_data)
            pdf_id = cur.fetchone()[0]
            conn.commit()
            
            return pdf_id

def store_analysis_data(analysis_data: Dict[str, Any]) -> int:
    """Store analysis data after contract analysis is completed"""
    
    sql = """
        INSERT INTO analyses (
            pdf_id, analysis_date, version, form_number,
            pi_clause, ci_clause, data_usage_mentioned,
            data_limitations_exists, summary, raw_json,
            processed_by, processing_time
        ) VALUES (
            %(pdf_id)s, %(analysis_date)s, %(version)s, %(form_number)s,
            %(pi_clause)s, %(ci_clause)s, %(data_usage_mentioned)s,
            %(data_limitations_exists)s, %(summary)s, %(raw_json)s,
            %(processed_by)s, %(processing_time)s
        ) RETURNING id
    """
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Convert datetime objects and JSON data
            if 'analysis_date' in analysis_data and isinstance(analysis_data['analysis_date'], datetime):
                analysis_data['analysis_date'] = analysis_data['analysis_date'].isoformat()
            
            if 'raw_json' in analysis_data and isinstance(analysis_data['raw_json'], dict):
                analysis_data['raw_json'] = json.dumps(analysis_data['raw_json'])
            
            cur.execute(sql, analysis_data)
            analysis_id = cur.fetchone()[0]
            conn.commit()
            
            return analysis_id

def store_clause_data(clause_list: List[Dict[str, Any]], analysis_id: int) -> List[int]:
    """Store clause data after clause extraction"""
    
    sql = """
        INSERT INTO clauses (analysis_id, clause_type, clause_text, clause_order)
        VALUES (%(analysis_id)s, %(clause_type)s, %(clause_text)s, %(clause_order)s)
        RETURNING id
    """
    
    clause_ids = []
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for clause_data in clause_list:
                clause_data['analysis_id'] = analysis_id
                cur.execute(sql, clause_data)
                clause_id = cur.fetchone()[0]
                clause_ids.append(clause_id)
            
            conn.commit()
    
    return clause_ids

def store_feedback_data(feedback_data: Dict[str, Any]) -> int:
    """Store user feedback data"""
    
    sql = """
        INSERT INTO feedback (
            pdf_id, feedback_date, form_number_feedback,
            general_feedback, rating, user_session_id
        ) VALUES (
            %(pdf_id)s, %(feedback_date)s, %(form_number_feedback)s,
            %(general_feedback)s, %(rating)s, %(user_session_id)s
        ) RETURNING id
    """
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            if 'feedback_date' in feedback_data and isinstance(feedback_data['feedback_date'], datetime):
                feedback_data['feedback_date'] = feedback_data['feedback_date'].isoformat()
            
            cur.execute(sql, feedback_data)
            feedback_id = cur.fetchone()[0]
            conn.commit()
            
            return feedback_id

def store_batch_job_data(batch_data: Dict[str, Any]) -> int:
    """Store batch processing job data"""
    
    sql = """
        INSERT INTO batch_jobs (
            job_id, created_at, started_at, completed_at,
            total_files, processed_files, failed_files, status,
            total_pages_processed, total_pages_removed,
            total_paragraphs_obfuscated, created_by,
            results_json, error_log
        ) VALUES (
            %(job_id)s, %(created_at)s, %(started_at)s, %(completed_at)s,
            %(total_files)s, %(processed_files)s, %(failed_files)s, %(status)s,
            %(total_pages_processed)s, %(total_pages_removed)s,
            %(total_paragraphs_obfuscated)s, %(created_by)s,
            %(results_json)s, %(error_log)s
        ) RETURNING id
    """
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Convert datetime objects and JSON data
            datetime_fields = ['created_at', 'started_at', 'completed_at']
            for field in datetime_fields:
                if field in batch_data and isinstance(batch_data[field], datetime):
                    batch_data[field] = batch_data[field].isoformat()
            
            if 'results_json' in batch_data and isinstance(batch_data['results_json'], (dict, list)):
                batch_data['results_json'] = json.dumps(batch_data['results_json'])
            
            cur.execute(sql, batch_data)
            batch_id = cur.fetchone()[0]
            conn.commit()
            
            return batch_id

# Query helper functions

def get_pdf_by_id(pdf_id: int) -> Optional[Dict[str, Any]]:
    """Get PDF record by ID"""
    
    sql = "SELECT * FROM pdfs WHERE id = %s"
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            result = cur.fetchone()
            return dict(result) if result else None

def get_pdf_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get PDF record by file hash for deduplication"""
    
    sql = "SELECT * FROM pdfs WHERE file_hash = %s"
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (file_hash,))
            result = cur.fetchone()
            return dict(result) if result else None

def get_latest_analysis(pdf_id: int) -> Optional[Dict[str, Any]]:
    """Get latest analysis for a PDF"""
    
    sql = """
        SELECT * FROM analyses 
        WHERE pdf_id = %s 
        ORDER BY version DESC 
        LIMIT 1
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            result = cur.fetchone()
            return dict(result) if result else None

def get_analysis_history(pdf_id: int) -> List[Dict[str, Any]]:
    """Get all analyses for a PDF ordered by version"""
    
    sql = """
        SELECT * FROM analyses 
        WHERE pdf_id = %s 
        ORDER BY version DESC
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            results = cur.fetchall()
            return [dict(row) for row in results]

def get_next_analysis_version(pdf_id: int) -> int:
    """Get next version number for PDF analysis"""
    
    sql = "SELECT MAX(version) FROM analyses WHERE pdf_id = %s"
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (pdf_id,))
            max_version = cur.fetchone()[0]
            return (max_version or 0) + 1

def check_database_connection() -> bool:
    """Simple database connection test"""
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False

def get_all_processed_pdfs() -> List[Dict[str, Any]]:
    """Get all PDFs that have been processed and have analysis data"""
    
    sql = """
        SELECT p.*, a.raw_json 
        FROM pdfs p
        JOIN analyses a ON p.id = a.pdf_id
        WHERE p.raw_analysis_json IS NOT NULL
        ORDER BY p.upload_date DESC
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            results = cur.fetchall()
            return [dict(row) for row in results]

if __name__ == "__main__":
    print("🧪 Testing database setup with psycopg2...")
    
    if check_database_connection():
        print("✅ Database connection successful")
        initialize_database()
        print("✅ Database initialization complete")
    else:
        print("❌ Database connection failed")
