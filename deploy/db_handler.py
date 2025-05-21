"""
PostgreSQL Database Handler for Contract Analysis Application

This module provides functionality for connecting to AWS Aurora PostgreSQL
and manages all database operations for the contract analysis application.

Features:
- Connection pooling with psycopg2
- Database schema creation and management
- CRUD operations for PDF storage, analysis results, and user feedback
- Session tracking for users
"""

import os
import json
import uuid
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json, RealDictCursor
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_handler')

class DatabaseHandler:
    """Handles all database operations for the Contract Analysis application."""
    
    def __init__(self, config=None):
        """Initialize the database handler with connection parameters.
        
        Args:
            config (dict): Database configuration parameters
                - host: Database hostname or endpoint (required)
                - port: Database port (default: 5432)
                - dbname: Database name (required)
                - user: Database username (required)
                - password: Database password (required)
                - min_connections: Minimum connections in pool (default: 1)
                - max_connections: Maximum connections in pool (default: 10)
        """
        self.config = config or {}
        
        # Set defaults if not provided
        if 'port' not in self.config:
            self.config['port'] = 5432
        if 'min_connections' not in self.config:
            self.config['min_connections'] = 1
        if 'max_connections' not in self.config:
            self.config['max_connections'] = 10
            
        # Validate required config parameters
        required_params = ['host', 'dbname', 'user', 'password']
        missing_params = [param for param in required_params if param not in self.config]
        
        if missing_params:
            raise ValueError(f"Missing required database configuration parameters: {', '.join(missing_params)}")
            
        # Initialize connection pool
        self.connection_pool = None
        self.create_connection_pool()
        
    def create_connection_pool(self):
        """Create a connection pool for database operations."""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config['min_connections'],
                maxconn=self.config['max_connections'],
                host=self.config['host'],
                port=self.config['port'],
                dbname=self.config['dbname'],
                user=self.config['user'],
                password=self.config['password']
            )
            logger.info(f"Connection pool created successfully with {self.config['min_connections']} to {self.config['max_connections']} connections")
        except Exception as e:
            logger.error(f"Error creating connection pool: {str(e)}")
            raise
            
    def get_connection(self):
        """Get a connection from the pool.
        
        Returns:
            connection: Database connection object
        """
        if not self.connection_pool:
            self.create_connection_pool()
        return self.connection_pool.getconn()
        
    def return_connection(self, conn):
        """Return a connection to the pool.
        
        Args:
            conn: Database connection to return to the pool
        """
        if self.connection_pool:
            self.connection_pool.putconn(conn)
            
    def execute_query(self, query, params=None, fetchone=False, fetchall=False, as_dict=False):
        """Execute a database query with proper connection handling.
        
        Args:
            query (str): SQL query to execute
            params (tuple or dict): Parameters for the query
            fetchone (bool): Whether to fetch a single result
            fetchall (bool): Whether to fetch all results
            as_dict (bool): Whether to return results as dictionaries
            
        Returns:
            Result of the query, if any
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if as_dict:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
                
            cursor.execute(query, params)
            
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()
            else:
                result = None
                
            conn.commit()
            return result
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)
                
    def initialize_schema(self):
        """Create the database schema if it doesn't exist."""
        schema_sql = """
        -- Create sessions table to track user sessions
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id VARCHAR(255) NULL
        );
        
        -- Create table for PDF documents
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id SERIAL PRIMARY KEY,
            pdf_id UUID DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) REFERENCES sessions(session_id),
            filename VARCHAR(255) NOT NULL,
            pdf_name VARCHAR(255) NOT NULL,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            avg_word_count_per_page FLOAT NOT NULL,
            pdf_layout VARCHAR(50) NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parsable BOOLEAN NOT NULL,
            final_text TEXT,
            obfuscation_applied BOOLEAN DEFAULT FALSE,
            metadata JSONB
        );
        
        -- Create table for contract analysis results
        CREATE TABLE IF NOT EXISTS contract_analysis (
            id SERIAL PRIMARY KEY,
            analysis_id UUID DEFAULT gen_random_uuid(),
            pdf_id UUID REFERENCES pdf_documents(pdf_id),
            session_id VARCHAR(255) REFERENCES sessions(session_id),
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            form_number VARCHAR(255),
            summary TEXT,
            data_usage_mentioned BOOLEAN,
            data_limitations_exists BOOLEAN,
            pi_clause BOOLEAN,
            ci_clause BOOLEAN,
            metadata JSONB
        );
        
        -- Create table for clauses extracted from contracts
        CREATE TABLE IF NOT EXISTS contract_clauses (
            id SERIAL PRIMARY KEY,
            clause_id UUID DEFAULT gen_random_uuid(),
            analysis_id UUID REFERENCES contract_analysis(analysis_id),
            clause_type VARCHAR(100) NOT NULL,
            clause_text TEXT NOT NULL,
            confidence FLOAT,
            page_number INTEGER,
            metadata JSONB
        );
        
        -- Create table for user feedback
        CREATE TABLE IF NOT EXISTS user_feedback (
            id SERIAL PRIMARY KEY,
            feedback_id UUID DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) REFERENCES sessions(session_id),
            pdf_id UUID REFERENCES pdf_documents(pdf_id),
            analysis_id UUID REFERENCES contract_analysis(analysis_id),
            clause_id UUID REFERENCES contract_clauses(clause_id),
            feedback_type VARCHAR(50) NOT NULL,
            feedback_value TEXT NOT NULL,
            correct BOOLEAN,
            suggested_correction TEXT,
            feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_pdf_session ON pdf_documents(session_id);
        CREATE INDEX IF NOT EXISTS idx_pdf_uuid ON pdf_documents(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_pdf ON contract_analysis(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_session ON contract_analysis(session_id);
        CREATE INDEX IF NOT EXISTS idx_clauses_analysis ON contract_clauses(analysis_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_session ON user_feedback(session_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_pdf ON user_feedback(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON user_feedback(analysis_id);
        """
        
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Execute schema creation
            cursor.execute(schema_sql)
            conn.commit()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error initializing database schema: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)
    
    # Session management methods
    def create_session(self, session_id, user_id=None):
        """Create a new session record.
        
        Args:
            session_id (str): Unique session identifier
            user_id (str, optional): User identifier if authenticated
            
        Returns:
            bool: Success status
        """
        query = """
        INSERT INTO sessions (session_id, user_id)
        VALUES (%s, %s)
        ON CONFLICT (session_id) 
        DO UPDATE SET last_active = CURRENT_TIMESTAMP, user_id = EXCLUDED.user_id
        RETURNING session_id
        """
        result = self.execute_query(query, (session_id, user_id), fetchone=True)
        return result is not None
        
    def update_session_activity(self, session_id):
        """Update the last_active timestamp for a session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: Success status
        """
        query = """
        UPDATE sessions 
        SET last_active = CURRENT_TIMESTAMP
        WHERE session_id = %s
        RETURNING session_id
        """
        result = self.execute_query(query, (session_id,), fetchone=True)
        return result is not None
    
    # PDF document methods
    def store_pdf_document(self, session_id, filename, pdf_name, file_size, page_count, 
                          word_count, avg_word_count_per_page, pdf_layout, parsable, 
                          final_text=None, metadata=None):
        """Store PDF document information in the database.
        
        Args:
            session_id (str): Session identifier
            filename (str): Original filename
            pdf_name (str): Normalized PDF name
            file_size (int): Size in bytes
            page_count (int): Number of pages
            word_count (int): Total word count
            avg_word_count_per_page (float): Average words per page
            pdf_layout (str): Layout type (e.g., 'single_column', 'double_column')
            parsable (bool): Whether PDF is parsable
            final_text (str, optional): Processed text content
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: PDF ID (UUID)
        """
        query = """
        INSERT INTO pdf_documents (
            session_id, filename, pdf_name, file_size, page_count, 
            word_count, avg_word_count_per_page, pdf_layout, parsable, 
            final_text, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING pdf_id
        """
        params = (
            session_id, filename, pdf_name, file_size, page_count, 
            word_count, avg_word_count_per_page, pdf_layout, parsable, 
            final_text, Json(metadata) if metadata else None
        )
        result = self.execute_query(query, params, fetchone=True)
        return result[0] if result else None
    
    def get_pdf_document(self, pdf_id=None, session_id=None, filename=None):
        """Retrieve PDF document information.
        
        Args:
            pdf_id (str, optional): PDF UUID
            session_id (str, optional): Session identifier
            filename (str, optional): Original filename
            
        Returns:
            dict: PDF document information
        """
        query = """
        SELECT * FROM pdf_documents 
        WHERE 1=1
        """
        params = []
        
        if pdf_id:
            query += " AND pdf_id = %s"
            params.append(pdf_id)
        
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
            
        if filename:
            query += " AND filename = %s"
            params.append(filename)
            
        if not (pdf_id or session_id or filename):
            raise ValueError("At least one search parameter required")
            
        return self.execute_query(query, tuple(params), fetchone=True, as_dict=True)
    
    def get_session_pdfs(self, session_id):
        """Get all PDFs for a session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            list: List of PDF documents
        """
        query = """
        SELECT * FROM pdf_documents 
        WHERE session_id = %s
        ORDER BY upload_date DESC
        """
        return self.execute_query(query, (session_id,), fetchall=True, as_dict=True)
    
    # Contract analysis methods
    def store_contract_analysis(self, pdf_id, session_id, form_number, summary, 
                              data_usage_mentioned, data_limitations_exists, pi_clause, 
                              ci_clause, metadata=None):
        """Store contract analysis results.
        
        Args:
            pdf_id (str): PDF UUID
            session_id (str): Session identifier
            form_number (str): Contract form number
            summary (str): Contract summary
            data_usage_mentioned (bool): Data usage mentioned flag
            data_limitations_exists (bool): Data limitations exist flag
            pi_clause (bool): PI clause present flag
            ci_clause (bool): CI clause present flag
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Analysis ID (UUID)
        """
        query = """
        INSERT INTO contract_analysis (
            pdf_id, session_id, form_number, summary, data_usage_mentioned,
            data_limitations_exists, pi_clause, ci_clause, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING analysis_id
        """
        params = (
            pdf_id, session_id, form_number, summary, data_usage_mentioned,
            data_limitations_exists, pi_clause, ci_clause, 
            Json(metadata) if metadata else None
        )
        result = self.execute_query(query, params, fetchone=True)
        return result[0] if result else None
    
    def get_contract_analysis(self, analysis_id=None, pdf_id=None, session_id=None):
        """Retrieve contract analysis results.
        
        Args:
            analysis_id (str, optional): Analysis UUID
            pdf_id (str, optional): PDF UUID
            session_id (str, optional): Session identifier
            
        Returns:
            dict: Contract analysis information
        """
        query = """
        SELECT * FROM contract_analysis 
        WHERE 1=1
        """
        params = []
        
        if analysis_id:
            query += " AND analysis_id = %s"
            params.append(analysis_id)
        
        if pdf_id:
            query += " AND pdf_id = %s"
            params.append(pdf_id)
            
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
            
        if not (analysis_id or pdf_id or session_id):
            raise ValueError("At least one search parameter required")
            
        return self.execute_query(query, tuple(params), fetchone=True, as_dict=True)
    
    # Contract clause methods
    def store_contract_clause(self, analysis_id, clause_type, clause_text, 
                            confidence=None, page_number=None, metadata=None):
        """Store a contract clause.
        
        Args:
            analysis_id (str): Analysis UUID
            clause_type (str): Type of clause
            clause_text (str): Clause text
            confidence (float, optional): Confidence score
            page_number (int, optional): Page number where clause appears
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Clause ID (UUID)
        """
        query = """
        INSERT INTO contract_clauses (
            analysis_id, clause_type, clause_text, confidence, 
            page_number, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING clause_id
        """
        params = (
            analysis_id, clause_type, clause_text, confidence, 
            page_number, Json(metadata) if metadata else None
        )
        result = self.execute_query(query, params, fetchone=True)
        return result[0] if result else None
    
    def get_analysis_clauses(self, analysis_id):
        """Get all clauses for an analysis.
        
        Args:
            analysis_id (str): Analysis UUID
            
        Returns:
            list: List of clauses
        """
        query = """
        SELECT * FROM contract_clauses 
        WHERE analysis_id = %s
        ORDER BY id
        """
        return self.execute_query(query, (analysis_id,), fetchall=True, as_dict=True)
    
    # User feedback methods
    def store_feedback(self, session_id, pdf_id, analysis_id, feedback_type, feedback_value, 
                      correct=None, suggested_correction=None, clause_id=None):
        """Store user feedback.
        
        Args:
            session_id (str): Session identifier
            pdf_id (str): PDF UUID
            analysis_id (str): Analysis UUID
            feedback_type (str): Type of feedback (e.g., 'summary', 'pi_clause')
            feedback_value (str): Feedback value
            correct (bool, optional): Whether the analysis was correct
            suggested_correction (str, optional): Suggested correction
            clause_id (str, optional): Clause UUID if feedback is for a specific clause
            
        Returns:
            str: Feedback ID (UUID)
        """
        query = """
        INSERT INTO user_feedback (
            session_id, pdf_id, analysis_id, clause_id, feedback_type,
            feedback_value, correct, suggested_correction
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING feedback_id
        """
        params = (
            session_id, pdf_id, analysis_id, clause_id, feedback_type,
            feedback_value, correct, suggested_correction
        )
        result = self.execute_query(query, params, fetchone=True)
        return result[0] if result else None
    
    def get_analysis_feedback(self, analysis_id):
        """Get all feedback for an analysis.
        
        Args:
            analysis_id (str): Analysis UUID
            
        Returns:
            list: List of feedback items
        """
        query = """
        SELECT * FROM user_feedback 
        WHERE analysis_id = %s
        ORDER BY feedback_date DESC
        """
        return self.execute_query(query, (analysis_id,), fetchall=True, as_dict=True)
        
    def close(self):
        """Close the connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")


# Example usage
if __name__ == "__main__":
    # Example configuration (replace with actual values)
    db_config = {
        'host': 'your-aurora-endpoint.rds.amazonaws.com',
        'port': 5432,
        'dbname': 'contracts_db',
        'user': 'dbadmin',
        'password': 'your-password',
        'min_connections': 1,
        'max_connections': 5
    }
    
    # Create database handler
    db = DatabaseHandler(db_config)
    
    # Initialize schema
    db.initialize_schema()
    
    # Test operations
    session_id = str(uuid.uuid4())
    db.create_session(session_id)
    
    # Close connections
    db.close()
