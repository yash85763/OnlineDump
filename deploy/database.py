# config/database.py - Database configuration and connection management

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database_models import Base, get_default_system_configs, SystemConfig

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    # Default to PostgreSQL - update with your AWS Aurora PostgreSQL connection string
    'postgresql://username:password@aurora-cluster.region.rds.amazonaws.com:5432/contract_analysis'
)

# Alternative: For local development, you can use SQLite
# DATABASE_URL = 'sqlite:///contract_analysis.db'

# Create engine with connection pooling for production
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # Number of connections to maintain in pool
    max_overflow=20,           # Maximum additional connections
    pool_pre_ping=True,        # Verify connections before use
    echo=False,                # Set to True for SQL debugging
    # For SQLite (local development only):
    # connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database():
    """
    Initialize database tables and default system configuration.
    Call this once when the application starts.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Initialize default system configuration
        _initialize_default_system_config()
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        raise

def _initialize_default_system_config():
    """Initialize default system configuration for obfuscation if not exists"""
    
    session = SessionLocal()
    try:
        # Check if system config already exists
        existing_configs = session.query(SystemConfig).filter(
            SystemConfig.config_key.like('obfuscation_%')
        ).count()
        
        if existing_configs == 0:
            # No obfuscation config exists, create defaults
            default_configs = get_default_system_configs()
            
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
            print("✅ System configuration already exists")
            
    except Exception as e:
        session.rollback()
        print(f"⚠️ Warning: Could not initialize system config: {str(e)}")
    finally:
        session.close()

def get_session():
    """
    Get a new database session.
    
    Returns:
        SQLAlchemy session
    """
    return SessionLocal()

def get_engine():
    """
    Get the database engine.
    
    Returns:
        SQLAlchemy engine
    """
    return engine

# Health check function
def check_database_connection():
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        session = get_session()
        # Try a simple query
        session.execute("SELECT 1")
        session.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False

# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup"""
    
    def __init__(self):
        self.session = None
    
    def __enter__(self):
        self.session = get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()

# Example usage:
# with DatabaseSession() as session:
#     pdf = session.query(PDF).first()
#     # Session automatically committed and closed