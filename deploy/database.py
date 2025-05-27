# config/database.py - Database configuration using psycopg2 and Celery integration

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

# Database schema creation
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
    """
    Store PDF data after parsing and obfuscation process.
    
    Args:
        pdf_data: Dictionary containing PDF information
        
    Returns:
        int: PDF ID
    """
    
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
    """
    Store analysis data after contract analysis is completed.
    
    Args:
        analysis_data: Dictionary containing analysis results
        
    Returns:
        int: Analysis ID
    """
    
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
    """
    Store clause data after clause extraction.
    
    Args:
        clause_list: List of clause dictionaries
        analysis_id: ID of the analysis record
        
    Returns:
        List[int]: List of clause IDs
    """
    
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

def update_batch_job_status(job_id: str, status: str, **kwargs):
    """Update batch job status and other fields"""
    
    # Build dynamic update query
    update_fields = ['status = %s']
    values = [status]
    
    for field, value in kwargs.items():
        if field in ['processed_files', 'failed_files', 'total_pages_processed', 
                    'total_pages_removed', 'total_paragraphs_obfuscated']:
            update_fields.append(f"{field} = %s")
            values.append(value)
        elif field == 'results_json' and isinstance(value, (dict, list)):
            update_fields.append("results_json = %s")
            values.append(json.dumps(value))
        elif field in ['started_at', 'completed_at']:
            update_fields.append(f"{field} = %s")
            values.append(value.isoformat() if isinstance(value, datetime) else value)
    
    values.append(job_id)  # For WHERE clause
    
    sql = f"UPDATE batch_jobs SET {', '.join(update_fields)} WHERE job_id = %s"
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)
            conn.commit()

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

# Celery integration for asynchronous processing

def get_celery_app():
    """Get Celery app configuration for asynchronous processing"""
    
    from celery import Celery
    
    # Configure Celery with Redis or RabbitMQ
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    celery_app = Celery(
        'contract_analyzer',
        broker=broker_url,
        backend=result_backend,
        include=['tasks.pdf_processing', 'tasks.batch_processing']
    )
    
    # Celery configuration
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_disable_rate_limits=False,
        task_compression='gzip',
        result_compression='gzip'
    )
    
    return celery_app

# Example usage functions

def example_complete_processing_pipeline():
    """Example of storing data through complete processing pipeline"""
    
    # Stage 1: Store PDF data after parsing and obfuscation
    pdf_data = {
        'pdf_name': 'contract.pdf',
        'file_hash': 'abc123hash',
        'upload_date': datetime.now(),
        'processed_date': datetime.now(),
        'layout': 'single_column',
        'original_word_count': 1000,
        'original_page_count': 10,
        'parsability': True,
        'final_word_count': 800,
        'final_page_count': 8,
        'avg_words_per_page': 100.0,
        'raw_content': 'Original PDF content...',
        'final_content': 'Obfuscated content...',
        'obfuscation_applied': True,
        'pages_removed_count': 2,
        'paragraphs_obfuscated_count': 5,
        'obfuscation_summary': {'pages_removed': 2, 'paragraphs_obfuscated': 5},
        'uploaded_by': 'session_123'
    }
    
    pdf_id = store_pdf_data(pdf_data)
    print(f"✅ PDF stored with ID: {pdf_id}")
    
    # Stage 2: Store analysis data
    analysis_data = {
        'pdf_id': pdf_id,
        'analysis_date': datetime.now(),
        'version': 1,
        'form_number': 'FORM-ABC-123',
        'pi_clause': 'yes',
        'ci_clause': 'no',
        'data_usage_mentioned': 'yes',
        'data_limitations_exists': 'no',
        'summary': 'Contract analysis summary...',
        'raw_json': {'form_number': 'FORM-ABC-123', 'clauses': []},
        'processed_by': 'session_123',
        'processing_time': 15.5
    }
    
    analysis_id = store_analysis_data(analysis_data)
    print(f"✅ Analysis stored with ID: {analysis_id}")
    
    # Stage 3: Store clauses
    clauses = [
        {
            'clause_type': 'pi_clause',
            'clause_text': 'Personal information clause text...',
            'clause_order': 1
        },
        {
            'clause_type': 'data_usage',
            'clause_text': 'Data usage clause text...',
            'clause_order': 2
        }
    ]
    
    clause_ids = store_clause_data(clauses, analysis_id)
    print(f"✅ Clauses stored with IDs: {clause_ids}")

if __name__ == "__main__":
    print("🧪 Testing database setup with psycopg2...")
    
    if check_database_connection():
        print("✅ Database connection successful")
        initialize_database()
        print("✅ Database initialization complete")
        
        # Run example
        example_complete_processing_pipeline()
        print("✅ Example processing pipeline completed")
    else:
        print("❌ Database connection failed")from models.database_models import Analysis
    
    analysis_record = Analysis(**analysis_data)
    session.add(analysis_record)
    session.flush()  # Get the ID without committing
    return analysis_record.id

def store_clause_data(session, clause_list, analysis_id):
    """
    Store clause data after clause extraction.
    
    Args:
        session: Database session
        clause_list: List of clause dictionaries
        analysis_id: ID of the analysis record
        
    Example:
        clause_list = [
            {'clause_type': 'pi_clause', 'clause_text': 'Personal info clause...', 'clause_order': 1},
            {'clause_type': 'ci_clause', 'clause_text': 'Confidential info clause...', 'clause_order': 2}
        ]
    """
    from models.database_models import Clause
    
    clause_ids = []
    for clause_data in clause_list:
        clause_data['analysis_id'] = analysis_id
        clause_record = Clause(**clause_data)
        session.add(clause_record)
        session.flush()
        clause_ids.append(clause_record.id)
    
    return clause_ids

def store_feedback_data(session, feedback_data):
    """
    Store user feedback data.
    
    Args:
        session: Database session
        feedback_data: Dictionary containing feedback information
    """
    from models.database_models import Feedback
    
    feedback_record = Feedback(**feedback_data)
    session.add(feedback_record)
    session.flush()
    return feedback_record.id

def store_batch_job_data(session, batch_data):
    """
    Store batch processing job data.
    
    Args:
        session: Database session
        batch_data: Dictionary containing batch job information
    """
    from models.database_models import BatchJob
    
    batch_record = BatchJob(**batch_data)
    session.add(batch_record)
    session.flush()
    return batch_record.id

# Query helper functions

def get_pdf_by_hash(session, file_hash):
    """Get PDF record by file hash for deduplication."""
    from models.database_models import PDF
    return session.query(PDF).filter_by(file_hash=file_hash).first()

def get_latest_analysis(session, pdf_id):
    """Get latest analysis for a PDF."""
    from models.database_models import Analysis
    return session.query(Analysis).filter_by(pdf_id=pdf_id).order_by(Analysis.version.desc()).first()

def get_analysis_history(session, pdf_id):
    """Get all analyses for a PDF ordered by version."""
    from models.database_models import Analysis
    return session.query(Analysis).filter_by(pdf_id=pdf_id).order_by(Analysis.version.desc()).all()

def get_pdf_clauses(session, pdf_id):
    """Get all clauses for a PDF (from latest analysis)."""
    from models.database_models import Analysis, Clause
    
    latest_analysis = get_latest_analysis(session, pdf_id)
    if not latest_analysis:
        return []
    
    return session.query(Clause).filter_by(analysis_id=latest_analysis.id).order_by(Clause.clause_order).all()

def get_next_analysis_version(session, pdf_id):
    """Get next version number for PDF analysis."""
    from models.database_models import Analysis
    
    max_version = session.query(
        session.func.max(Analysis.version)
    ).filter_by(pdf_id=pdf_id).scalar()
    
    return (max_version or 0) + 1

# Simple usage examples:

def example_store_complete_processing_pipeline():
    """
    Example of how to store data through the complete processing pipeline.
    """
    
    with DatabaseSession() as session:
        # Stage 1: Store PDF data after parsing and obfuscation
        pdf_data = {
            'pdf_name': 'contract.pdf',
            'file_hash': 'abc123hash',
            'raw_content': 'Original PDF content...',
            'final_content': 'Obfuscated content...',
            'original_word_count': 1000,
            'final_word_count': 800,
            'original_page_count': 10,
            'final_page_count': 8,
            'layout': 'single_column',
            'parsability': True,
            'obfuscation_applied': True,
            'pages_removed_count': 2,
            'paragraphs_obfuscated_count': 5,
            'uploaded_by': 'session_123'
        }
        
        pdf_id = store_pdf_data(session, pdf_data)
        print(f"✅ PDF stored with ID: {pdf_id}")
        
        # Stage 2: Store analysis data after contract analysis
        analysis_data = {
            'pdf_id': pdf_id,
            'form_number': 'FORM-ABC-123',
            'pi_clause': 'yes',
            'ci_clause': 'no',
            'data_usage_mentioned': 'yes',
            'data_limitations_exists': 'no',
            'summary': 'Contract analysis summary...',
            'raw_json': {'form_number': 'FORM-ABC-123', 'clauses': []},
            'version': 1,
            'processed_by': 'session_123'
        }
        
        analysis_id = store_analysis_data(session, analysis_data)
        print(f"✅ Analysis stored with ID: {analysis_id}")
        
        # Stage 3: Store clauses after clause extraction
        clauses = [
            {
                'clause_type': 'pi_clause',
                'clause_text': 'This contract contains personal information clauses...',
                'clause_order': 1
            },
            {
                'clause_type': 'data_usage',
                'clause_text': 'Data usage is restricted to...',
                'clause_order': 2
            }
        ]
        
        clause_ids = store_clause_data(session, clauses, analysis_id)
        print(f"✅ Clauses stored with IDs: {clause_ids}")
        
        # Session automatically commits all changes here

def example_batch_processing():
    """
    Example of storing batch processing data.
    """
    
    with DatabaseSession() as session:
        # Create batch job record
        batch_data = {
            'job_id': 'batch_001',
            'total_files': 5,
            'status': 'running',
            'created_by': 'session_123'
        }
        
        batch_id = store_batch_job_data(session, batch_data)
        print(f"✅ Batch job created with ID: {batch_id}")
        
        # Update batch job as files are processed
        batch_record = session.query(BatchJob).filter_by(id=batch_id).first()
        batch_record.processed_files = 3
        batch_record.status = 'completed'
        # Session will commit automatically

if __name__ == "__main__":
    print("🧪 Testing database setup...")
    
    if check_database_connection():
        print("✅ Database connection successful")
        initialize_database()
        print("✅ Database initialization complete")
    else:
        print("❌ Database connection failed")
