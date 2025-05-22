import sys
import os

# Add the project root directory to sys.path to fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
from psycopg2 import sql
from urllib.parse import quote_plus
from models.database_models import get_default_system_configs, SystemConfig

def get_database_url():
    """
    Construct database connection parameters from environment variables for AWS Aurora PostgreSQL.
    Returns a dictionary for psycopg2 connection.
    """
    # Option 1: Parse DATABASE_URL if provided
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        # Assuming DATABASE_URL format: postgresql://username:password@host:port/dbname
        try:
            # Simple parsing for psycopg2
            user_part, host_part = database_url.replace('postgresql://', '').split('@')
            username, password = user_part.split(':')
            host_port, dbname = host_part.split('/')
            host, port = host_port.split(':')
            return {
                'user': username,
                'password': password,
                'host': host,
                'port': port,
                'dbname': dbname
            }
        except Exception as e:
            raise ValueError(f"Could not parse DATABASE_URL: {str(e)}")

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
        'user': db_username,
        'password': db_password,
        'host': db_host,
        'port': db_port,
        'dbname': db_name
    }

def get_connection():
    """
    Create a psycopg2 connection to the database with public schema set.
    Returns:
        psycopg2 connection object
    """
    conn_params = get_database_url()
    try:
        conn = psycopg2.connect(
            **conn_params,
            connect_timeout=30,
            application_name="contract_analyzer_app",
            options="-c search_path=public"
        )
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {str(e)}")
        raise

def initialize_database():
    """
    Initialize database tables in public schema and default system configuration.
    Call this once when the application starts.
    """
    try:
        print("🔌 Testing database connection...")
        if not check_database_connection():
            raise Exception("Failed to connect to database")

        print("✅ Database connection successful")

        # Create tables in public schema
        print("🏗️ Creating database tables...")
        create_tables()
        print("✅ Database tables created successfully")

        # Initialize default system configuration
        print("⚙️ Initializing system configuration...")
        _initialize_default_system_config()
        print("✅ System configuration initialized")

    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        print("🔍 Please check your database connection settings:")
        print(f"   - DB_HOST: {os.getenv('DB_HOST', 'Not set')}")
        print(f"   - DB_USERNAME: {os.getenv('DB_USERNAME', 'Not set')}")
        print(f"   - DB_NAME: {os.getenv('DB_NAME', 'Not set')}")
        print(f"   - DB_PORT: {os.getenv('DB_PORT', '5432')}")
        raise

def create_tables():
    """
    Create necessary tables in the public schema.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Set search path to public
            cur.execute("SET search_path TO public")

            # Create system_config table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.system_config (
                    id SERIAL PRIMARY KEY,
                    config_key VARCHAR(100) NOT NULL,
                    config_value TEXT NOT NULL,
                    description TEXT,
                    updated_by VARCHAR(50) NOT NULL,
                    CONSTRAINT unique_config_key UNIQUE (config_key)
                )
            """)

            # Add other tables as needed (e.g., users, pdfs, analyses, clauses, feedback)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.pdfs (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.analyses (
                    id SERIAL PRIMARY KEY,
                    pdf_id INTEGER REFERENCES public.pdfs(id),
                    analysis_result TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.clauses (
                    id SERIAL PRIMARY KEY,
                    analysis_id INTEGER REFERENCES public.analyses(id),
                    clause_text TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.feedback (
                    id SERIAL PRIMARY KEY,
                    analysis_id INTEGER REFERENCES public.analyses(id),
                    feedback_text TEXT
                )
            """)

            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating tables: {str(e)}")
        raise
    finally:
        conn.close()

def _initialize_default_system_config():
    """
    Initialize default system configuration for obfuscation if not exists.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Set search path to public
            cur.execute("SET search_path TO public")

            # Check if system config already exists
            cur.execute("""
                SELECT COUNT(*) 
                FROM public.system_config 
                WHERE config_key LIKE 'obfuscation_%'
            """)
            existing_configs = cur.fetchone()[0]

            if existing_configs == 0:
                # No obfuscation config exists, create defaults
                default_configs = get_default_system_configs()
                print(f"📝 Creating {len(default_configs)} default configuration entries...")

                for config_data in default_configs:
                    cur.execute(
                        sql.SQL("""
                            INSERT INTO public.system_config (config_key, config_value, description, updated_by)
                            VALUES (%s, %s, %s, %s)
                        """),
                        (
                            config_data["config_key"],
                            config_data["config_value"],
                            config_data["description"],
                            "system_init"
                        )
                    )
                conn.commit()
                print("✅ Default obfuscation configuration initialized")
            else:
                print(f"✅ System configuration already exists ({existing_configs} entries found)")
    except Exception as e:
        conn.rollback()
        print(f"⚠️ Warning: Could not initialize system config: {str(e)}")
        raise
    finally:
        conn.close()

def check_database_connection():
    """
    Check if database connection is working and accessible.
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Test basic connectivity
            cur.execute("SELECT 1")
            # Test schema access
            cur.execute("SET search_path TO public")
            # Test if we can query system tables
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 1")
        conn.close()
        print("✅ Database connection and schema access verified")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

def verify_database_setup():
    """
    Comprehensive database setup verification.
    Returns:
        dict: Status of various database components
    """
    status = {
        "connection": False,
        "schema_access": False,
        "tables_exist": False,
        "system_config_exists": False,
        "error_messages": []
    }

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Test connection
            cur.execute("SELECT 1")
            status["connection"] = True

            # Test schema access
            cur.execute("SET search_path TO public")
            status["schema_access"] = True

            # Check if our tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('pdfs', 'analyses', 'clauses', 'feedback', 'users', 'system_config')
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            expected_tables = ['pdfs', 'analyses', 'clauses', 'feedback', 'users', 'system_config']

            if len(existing_tables) == len(expected_tables):
                status["tables_exist"] = True
            else:
                missing_tables = set(expected_tables) - set(existing_tables)
                status["error_messages"].append(f"Missing tables: {missing_tables}")

            # Test system config
            cur.execute("SELECT COUNT(*) FROM public.system_config")
            config_count = cur.fetchone()[0]
            if config_count > 0:
                status["system_config_exists"] = True
            else:
                status["error_messages"].append("No system configuration found")
        conn.close()
    except Exception as e:
        status["error_messages"].append(f"Database error: {str(e)}")

    return status

def create_database_if_not_exists():
    """
    Create database if it doesn't exist (for initial setup).
    Connects to the default 'postgres' database to create the target database.
    """
    db_name = os.getenv('DB_NAME')
    if not db_name:
        raise ValueError("DB_NAME environment variable is required")

    # Connect to default postgres database
    conn_params = get_database_url()
    conn_params['dbname'] = 'postgres'

    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_session(autocommit=True)  # Database creation can't be in a transaction
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name])
            if not cur.fetchone():
                # Create database
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"✅ Database '{db_name}' created successfully")
            else:
                print(f"✅ Database '{db_name}' already exists")
        conn.close()
    except Exception as e:
        print(f"⚠️ Could not create database: {str(e)}")
        raise

def database_health_check():
    """
    Perform comprehensive database health check for monitoring/alerting.
    Returns:
        dict: Health status and metrics
    """
    import time
    from datetime import datetime

    health_status = {
        "status": "unhealthy",
        "timestamp": None,
        "connection_time_ms": None,
        "active_connections": None,
        "table_counts": {},
        "errors": []
    }

    start_time = time.time()

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Record connection time
            health_status["connection_time_ms"] = round((time.time() - start_time) * 1000, 2)
            health_status["timestamp"] = datetime.utcnow().isoformat()

            # Check active connections
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
            health_status["active_connections"] = cur.fetchone()[0]

            # Check table record counts
            tables = ['users', 'pdfs', 'analyses', 'clauses', 'feedback', 'system_config']
            for table in tables:
                try:
                    cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
                    health_status["table_counts"][table] = cur.fetchone()[0]
                except Exception as e:
                    health_status["errors"].append(f"Could not count {table}: {str(e)}")
            health_status["status"] = "healthy"
        conn.close()
    except Exception as e:
        health_status["errors"].append(f"Database connection failed: {str(e)}")

    return health_status

# Context manager for database connections
class DatabaseConnection:
    """Context manager for psycopg2 connections with automatic cleanup and schema setting"""
    def __init__(self):
        self.conn = None

    def __enter__(self):
        self.conn = get_connection()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()

# Example usage and testing functions
if __name__ == "__main__":
    print("🧪 Testing database configuration...")

    # Test connection
    if check_database_connection():
        print("✅ Database connection successful")

        # Test initialization
        try:
            initialize_database()
            print("✅ Database initialization successful")

            # Verify setup
            status = verify_database_setup()
            print(f"📊 Database status: {status}")

            # Health check
            health = database_health_check()
            print(f"💗 Health check: {health}")

        except Exception as e:
            print(f"❌ Database initialization failed: {str(e)}")
    else:
        print("❌ Database connection failed")