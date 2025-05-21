import os
import psycopg2
from psycopg2 import pool
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_utils")

# Database connection pool
connection_pool = None

def init_db_pool():
    """Initialize database connection pool from environment variables"""
    global connection_pool
    
    try:
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        
        # Validate connection parameters
        if not all([db_host, db_name, db_user, db_password]):
            logger.error("Missing database credentials. Check environment variables.")
            return False
        
        # Create a connection pool with min 1, max 10 connections
        connection_pool = pool.SimpleConnectionPool(
            1, 10,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        logger.info("Database connection pool initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing database connection pool: {str(e)}")
        return False

def get_db_connection():
    """Get a connection from the pool"""
    global connection_pool
    if connection_pool is None:
        init_db_pool()
    
    if connection_pool:
        return connection_pool.getconn()
    return None

def release_db_connection(conn):
    """Return a connection to the pool"""
    global connection_pool
    if connection_pool and conn:
        connection_pool.putconn(conn)

def close_all_connections():
    """Close all connections in the pool"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        logger.info("All database connections closed")

def create_tables():
    """Create all the required tables if they don't exist"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create input_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id SERIAL PRIMARY KEY,
            pdf_name VARCHAR(255) NOT NULL,
            file_extension VARCHAR(10),
            word_count INTEGER,
            num_pages INTEGER,
            avg_word_count_per_page FLOAT,
            page_layout VARCHAR(50),
            parsable BOOLEAN DEFAULT TRUE,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create analysis_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_data (
            id SERIAL PRIMARY KEY,
            pdf_name VARCHAR(255) NOT NULL,
            form_number VARCHAR(100),
            summary TEXT,
            data_usage_mentioned BOOLEAN,
            data_limitations_exists BOOLEAN,
            pi_clause BOOLEAN,
            ci_clause BOOLEAN,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_data_id INTEGER REFERENCES input_data(id)
        );
        """)
        
        # Create clauses table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clauses (
            id SERIAL PRIMARY KEY,
            pdf_name VARCHAR(255) NOT NULL,
            clause_type VARCHAR(50),
            clause_text TEXT,
            extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_data_id INTEGER REFERENCES analysis_data(id)
        );
        """)
        
        # Create feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            pdf_name VARCHAR(255) NOT NULL,
            field_name VARCHAR(100) NOT NULL,
            feedback_text TEXT NOT NULL,
            feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_data_id INTEGER REFERENCES analysis_data(id)
        );
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
        return True
    
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error creating database tables: {str(e)}")
        return False
    
    finally:
        if conn:
            release_db_connection(conn)

def store_pdf_data(pdf_metadata, analysis_result):
    """Store PDF metadata and analysis results in the database"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Insert into input_data table
        cursor.execute("""
        INSERT INTO input_data (
            pdf_name, file_extension, word_count, num_pages, 
            avg_word_count_per_page, page_layout, parsable
        ) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """, (
            pdf_metadata['filename'],
            os.path.splitext(pdf_metadata['filename'])[1],
            pdf_metadata['word_count'],
            pdf_metadata['page_count'],
            pdf_metadata['avg_words_per_page'],
            pdf_metadata['layout'],
            pdf_metadata['parsable']
        ))
        
        input_data_id = cursor.fetchone()[0]
        
        # 2. Insert into analysis_data table
        cursor.execute("""
        INSERT INTO analysis_data (
            pdf_name, form_number, summary, data_usage_mentioned,
            data_limitations_exists, pi_clause, ci_clause, input_data_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """, (
            pdf_metadata['filename'],
            analysis_result.get('form_number', ''),
            analysis_result.get('summary', ''),
            analysis_result.get('data_usage_mentioned', False),
            analysis_result.get('data_limitations_exists', False),
            analysis_result.get('pi_clause', False),
            analysis_result.get('ci_clause', False),
            input_data_id
        ))
        
        analysis_data_id = cursor.fetchone()[0]
        
        # 3. Insert clauses
        for clause in analysis_result.get('relevant_clauses', []):
            cursor.execute("""
            INSERT INTO clauses (
                pdf_name, clause_type, clause_text, analysis_data_id
            ) VALUES (%s, %s, %s, %s);
            """, (
                pdf_metadata['filename'],
                clause.get('type', ''),
                clause.get('text', ''),
                analysis_data_id
            ))
        
        conn.commit()
        logger.info(f"Stored PDF data for {pdf_metadata['filename']} successfully")
        return True
    
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error storing PDF data: {str(e)}")
        return False
    
    finally:
        if conn:
            release_db_connection(conn)

def store_feedback(pdf_name, field_name, feedback_text):
    """Store user feedback in the database"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, find the analysis_data_id for this PDF
        cursor.execute("""
        SELECT id FROM analysis_data WHERE pdf_name = %s ORDER BY analysis_date DESC LIMIT 1;
        """, (pdf_name,))
        
        result = cursor.fetchone()
        if result:
            analysis_data_id = result[0]
            
            # Insert feedback
            cursor.execute("""
            INSERT INTO feedback (
                pdf_name, field_name, feedback_text, analysis_data_id
            ) VALUES (%s, %s, %s, %s);
            """, (
                pdf_name,
                field_name,
                feedback_text,
                analysis_data_id
            ))
            
            conn.commit()
            logger.info(f"Stored feedback for {pdf_name}, field {field_name}")
            return True
        else:
            logger.warning(f"No analysis data found for {pdf_name}, feedback not stored")
            return False
    
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error storing feedback: {str(e)}")
        return False
    
    finally:
        if conn:
            release_db_connection(conn)

def get_all_feedback_for_pdf(pdf_name):
    """Retrieve all feedback for a specific PDF"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT field_name, feedback_text, feedback_date 
        FROM feedback 
        WHERE pdf_name = %s 
        ORDER BY feedback_date DESC;
        """, (pdf_name,))
        
        feedback_data = cursor.fetchall()
        
        # Convert to list of dictionaries
        feedback_list = []
        for item in feedback_data:
            feedback_list.append({
                'field_name': item[0],
                'feedback_text': item[1],
                'feedback_date': item[2].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return feedback_list
    
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        return []
    
    finally:
        if conn:
            release_db_connection(conn)

# Initialize database on import
if __name__ == "__main__":
    if init_db_pool():
        create_tables()
        logger.info("Database initialized successfully")
    else:
        logger.error("Failed to initialize database")