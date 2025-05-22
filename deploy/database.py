# config/database.py - Database configuration for AWS Aurora PostgreSQL with public schema

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models.database_models import Base, get_default_system_configs, SystemConfig
from urllib.parse import quote_plus

def get_database_url():
    """
    Construct database URL from environment variables for AWS Aurora PostgreSQL.
    Supports both individual connection parameters and full DATABASE_URL.
    """
    
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
    
    # URL encode password to handle special characters
    encoded_password = quote_plus(db_password)
    
    # Construct PostgreSQL connection URL
    database_url = f"postgresql://{db_username}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    return database_url

# Database configuration
DATABASE_URL = get_database_url()

# Create engine with AWS Aurora PostgreSQL optimized settings
engine = create_engine(
    DATABASE_URL,
    
    # Connection pool settings for AWS Aurora
    pool_size=10,                    # Number of connections to maintain in pool
    max_overflow=20,                 # Maximum additional connections beyond pool_size
    pool_pre_ping=True,              # Verify connections before use (important for AWS)
    pool_recycle=3600,               # Recycle connections every hour (AWS best practice)
    
    # Connection timeout settings
    connect_args={
        "connect_timeout": 30,       # Connection timeout in seconds
        "application_name": "contract_analyzer_app",  # App name for monitoring
        "options": "-c search_path=public"  # Ensure we use public schema
    },
    
    # Debugging (set to True for development)
    echo=os.getenv('SQL_DEBUG', 'false').lower() == 'true',
    
    # AWS specific settings
    isolation_level="READ_COMMITTED"  # Good for AWS Aurora
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

def initialize_database():
    """
    Initialize database tables in public schema and default system configuration.
    Call this once when the application starts.
    """
    try:
        # Test database connection first
        print("🔌 Testing database connection...")
        if not check_database_connection():
            raise Exception("Failed to connect to database")
        
        print("✅ Database connection successful")
        
        # Ensure we're using public schema
        with engine.connect() as conn:
            # Set search path to public schema
            conn.execute(text("SET search_path TO public"))
            conn.commit()
        
        print("📁 Using public schema for tables")
        
        # Create all tables in public schema
        print("🏗️ Creating database tables...")
        Base.metadata.create_all(bind=engine)
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

def _initialize_default_system_config():
    """Initialize default system configuration for obfuscation if not exists"""
    
    session = SessionLocal()
    try:
        # Ensure we're in public schema
        session.execute(text("SET search_path TO public"))
        
        # Check if system config already exists
        existing_configs = session.query(SystemConfig).filter(
            SystemConfig.config_key.like('obfuscation_%')
        ).count()
        
        if existing_configs == 0:
            # No obfuscation config exists, create defaults
            default_configs = get_default_system_configs()
            
            print(f"📝 Creating {len(default_configs)} default configuration entries...")
            
            for config_data in default_configs:
                system_config = SystemConfig(
                    config_key=config_data["config_key"],
                    config_value=config_data["config_value"],
                    description=config_data["description"],
                    updated_by="system_init"
                )
                session.add(system_config)
            
            session.commit()
            print("✅ Default obfuscation configuration initialized")
        else:
            print(f"✅ System configuration already exists ({existing_configs} entries found)")
            
    except Exception as e:
        session.rollback()
        print(f"⚠️ Warning: Could not initialize system config: {str(e)}")
        raise
    finally:
        session.close()

def get_session():
    """
    Get a new database session with public schema set.
    
    Returns:
        SQLAlchemy session configured for public schema
    """
    session = SessionLocal()
    # Ensure we're using public schema
    session.execute(text("SET search_path TO public"))
    return session

def get_engine():
    """
    Get the database engine.
    
    Returns:
        SQLAlchemy engine
    """
    return engine

def check_database_connection():
    """
    Check if database connection is working and accessible.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            
            # Test schema access
            conn.execute(text("SET search_path TO public"))
            
            # Test if we can query system tables
            conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 1"))
            
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
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["connection"] = True
            
            # Test schema access
            conn.execute(text("SET search_path TO public"))
            status["schema_access"] = True
            
            # Check if our tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('pdfs', 'analyses', 'clauses', 'feedback', 'users', 'system_config')
            """))
            
            existing_tables = [row[0] for row in result]
            expected_tables = ['pdfs', 'analyses', 'clauses', 'feedback', 'users', 'system_config']
            
            if len(existing_tables) == len(expected_tables):
                status["tables_exist"] = True
            else:
                missing_tables = set(expected_tables) - set(existing_tables)
                status["error_messages"].append(f"Missing tables: {missing_tables}")
        
        # Test system config
        session = get_session()
        try:
            config_count = session.query(SystemConfig).count()
            if config_count > 0:
                status["system_config_exists"] = True
            else:
                status["error_messages"].append("No system configuration found")
        finally:
            session.close()
            
    except Exception as e:
        status["error_messages"].append(f"Database error: {str(e)}")
    
    return status

def create_database_if_not_exists():
    """
    Create database if it doesn't exist (for initial setup).
    This connects to the default 'postgres' database to create the target database.
    """
    
    db_name = os.getenv('DB_NAME')
    if not db_name:
        raise ValueError("DB_NAME environment variable is required")
    
    # Connect to default postgres database to create our database
    default_url = get_database_url().replace(f"/{db_name}", "/postgres")
    
    try:
        default_engine = create_engine(default_url)
        
        with default_engine.connect() as conn:
            # Don't use transactions for database creation
            conn.execute(text("COMMIT"))
            
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            
            if not result.fetchone():
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"✅ Database '{db_name}' created successfully")
            else:
                print(f"✅ Database '{db_name}' already exists")
                
    except Exception as e:
        print(f"⚠️ Could not create database: {str(e)}")
        print("Please ensure the database exists or you have permission to create it")

# Context manager for database sessions with proper schema handling
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup and schema setting"""
    
    def __init__(self):
        self.session = None
    
    def __enter__(self):
        self.session = get_session()  # This already sets search_path to public
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()

# Database health check function for monitoring
def database_health_check():
    """
    Perform comprehensive database health check for monitoring/alerting.
    
    Returns:
        dict: Health status and metrics
    """
    health_status = {
        "status": "unhealthy",
        "timestamp": None,
        "connection_time_ms": None,
        "active_connections": None,
        "table_counts": {},
        "errors": []
    }
    
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    try:
        with engine.connect() as conn:
            # Record connection time
            health_status["connection_time_ms"] = round((time.time() - start_time) * 1000, 2)
            health_status["timestamp"] = datetime.utcnow().isoformat()
            
            # Check active connections
            result = conn.execute(text("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"))
            health_status["active_connections"] = result.scalar()
            
            # Check table record counts
            tables = ['users', 'pdfs', 'analyses', 'clauses', 'feedback', 'system_config']
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT count(*) FROM {table}"))
                    health_status["table_counts"][table] = result.scalar()
                except Exception as e:
                    health_status["errors"].append(f"Could not count {table}: {str(e)}")
            
            health_status["status"] = "healthy"
            
    except Exception as e:
        health_status["errors"].append(f"Database connection failed: {str(e)}")
    
    return health_status

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

# Environment variables documentation
"""
Required Environment Variables for AWS Aurora PostgreSQL:

Option 1 - Full DATABASE_URL:
DATABASE_URL=postgresql://username:password@aurora-cluster.region.rds.amazonaws.com:5432/database_name

Option 2 - Individual Parameters:
DB_USERNAME=your_username
DB_PASSWORD=your_password  
DB_HOST=aurora-cluster.region.rds.amazonaws.com
DB_PORT=5432
DB_NAME=contract_analysis

Optional:
SQL_DEBUG=true  # Enable SQL query logging for development

Example .env file:
DB_USERNAME=contract_user
DB_PASSWORD=secure_password_123
DB_HOST=contract-aurora-cluster.cluster-xyz123.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=contract_analysis
SQL_DEBUG=false
"""