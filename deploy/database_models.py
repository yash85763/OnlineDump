# database/models.py - SQLAlchemy models for contract analysis application

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Boolean, Float, JSON, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import hashlib
import uuid

# Base class for all models
Base = declarative_base()

class User(Base):
    """User model for authentication and session management"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=True)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    uploaded_pdfs = relationship("PDF", back_populates="uploader", cascade="all, delete-orphan")
    feedback_entries = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', session_id='{self.session_id}')>"

class PDF(Base):
    """PDF model for storing contract documents and processing metadata"""
    __tablename__ = 'pdfs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_name = Column(String(500), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime, nullable=True)
    
    # PDF parsing data
    layout = Column(String(100), nullable=True)
    original_word_count = Column(Integer, nullable=True)
    original_page_count = Column(Integer, nullable=True)
    parsability = Column(Float, nullable=True)
    
    # Final (processed) content metrics
    final_word_count = Column(Integer, nullable=True)
    final_page_count = Column(Integer, nullable=True)
    avg_words_per_page = Column(Float, nullable=True)
    
    # Content storage
    raw_content = Column(JSON, nullable=True)
    final_content = Column(Text, nullable=True)
    
    # Obfuscation tracking
    obfuscation_applied = Column(Boolean, default=True)
    pages_removed_count = Column(Integer, default=0)
    paragraphs_obfuscated_count = Column(Integer, default=0)
    obfuscation_summary = Column(JSON, nullable=True)
    
    # Foreign key to users
    uploaded_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Relationships
    uploader = relationship("User", back_populates="uploaded_pdfs")
    analyses = relationship("Analysis", back_populates="pdf", cascade="all, delete-orphan")
    feedback_entries = relationship("Feedback", back_populates="pdf", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_pdfs_file_hash', 'file_hash'),
        Index('idx_pdfs_upload_date', 'upload_date'),
        Index('idx_pdfs_uploaded_by', 'uploaded_by'),
    )
    
    def __repr__(self):
        return f"<PDF(id={self.id}, name='{self.pdf_name}', hash='{self.file_hash[:8]}...')>"

class Analysis(Base):
    """Analysis model for storing contract analysis results"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id', ondelete='CASCADE'), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    version = Column(String(50), nullable=True)
    
    # Analysis results
    form_number = Column(String(100), nullable=True)
    pi_clause = Column(Text, nullable=True)
    ci_clause = Column(Text, nullable=True)
    data_usage_mentioned = Column(Boolean, nullable=True)
    data_limitations_exists = Column(Boolean, nullable=True)
    summary = Column(Text, nullable=True)
    raw_json = Column(JSON, nullable=True)
    
    # Processing metadata
    processed_by = Column(String(100), nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Relationships
    pdf = relationship("PDF", back_populates="analyses")
    clauses = relationship("Clause", back_populates="analysis", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_analyses_pdf_id', 'pdf_id'),
        Index('idx_analyses_version', 'pdf_id', 'version'),
        Index('idx_analyses_date', 'analysis_date'),
    )
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, pdf_id={self.pdf_id}, version='{self.version}')>"

class Clause(Base):
    """Clause model for storing extracted clauses with position information"""
    __tablename__ = 'clauses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False)
    clause_type = Column(String(50), nullable=True)
    clause_text = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    paragraph_index = Column(Integer, nullable=True)
    clause_order = Column(Integer, nullable=True)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="clauses")
    
    # Indexes
    __table_args__ = (
        Index('idx_clauses_analysis_id', 'analysis_id'),
        Index('idx_clauses_type', 'clause_type'),
    )
    
    def __repr__(self):
        return f"<Clause(id={self.id}, type='{self.clause_type}', analysis_id={self.analysis_id})>"

class Feedback(Base):
    """Feedback model for storing user feedback on PDF analyses"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id', ondelete='CASCADE'), nullable=False)
    feedback_date = Column(DateTime, default=datetime.utcnow)
    general_feedback = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Relationships
    pdf = relationship("PDF", back_populates="feedback_entries")
    user = relationship("User", back_populates="feedback_entries")
    
    # Indexes
    __table_args__ = (
        Index('idx_feedback_pdf_id', 'pdf_id'),
        Index('idx_feedback_user_id', 'user_id'),
        Index('idx_feedback_date', 'feedback_date'),
    )
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, pdf_id={self.pdf_id}, user_id={self.user_id})>"

class DatabaseManager:
    """Database manager class for handling connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _get_database_url(self) -> str:
        """Construct database URL from environment variables"""
        
        # Option 1: Use full DATABASE_URL if provided
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
        
        # Option 2: Construct from individual parameters
        db_username = os.getenv('DB_USERNAME', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'password')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'contract_analysis')
        
        return f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def _initialize_database(self):
        """Initialize database connection and session factory"""
        
        try:
            database_url = self._get_database_url()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            print("✅ Database engine and session factory initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize database: {str(e)}")
            raise
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✅ All tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            raise
    
    def drop_tables(self):
        """Drop all tables from the database"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            print("✅ All tables dropped successfully")
        except Exception as e:
            print(f"❌ Error dropping tables: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection test failed: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Database operation functions

def create_user(session_id: str, username: str = None, password_hash: str = None) -> User:
    """Create a new user"""
    with db_manager.get_session() as session:
        user = User(
            username=username,
            password_hash=password_hash,
            session_id=session_id
        )
        session.add(user)
        session.flush()
        session.refresh(user)
        return user

def get_user_by_session(session_id: str) -> Optional[User]:
    """Get user by session ID"""
    with db_manager.get_session() as session:
        return session.query(User).filter(User.session_id == session_id).first()

def create_pdf(pdf_data: Dict[str, Any]) -> PDF:
    """Create a new PDF record"""
    with db_manager.get_session() as session:
        pdf = PDF(**pdf_data)
        session.add(pdf)
        session.flush()
        session.refresh(pdf)
        return pdf

def get_pdf_by_hash(file_hash: str) -> Optional[PDF]:
    """Get PDF by file hash for deduplication"""
    with db_manager.get_session() as session:
        return session.query(PDF).filter(PDF.file_hash == file_hash).first()

def get_pdf_by_id(pdf_id: int) -> Optional[PDF]:
    """Get PDF by ID"""
    with db_manager.get_session() as session:
        return session.query(PDF).filter(PDF.id == pdf_id).first()

def create_analysis(analysis_data: Dict[str, Any]) -> Analysis:
    """Create a new analysis record"""
    with db_manager.get_session() as session:
        analysis = Analysis(**analysis_data)
        session.add(analysis)
        session.flush()
        session.refresh(analysis)
        return analysis

def get_latest_analysis(pdf_id: int) -> Optional[Analysis]:
    """Get latest analysis for a PDF"""
    with db_manager.get_session() as session:
        return session.query(Analysis)\
                     .filter(Analysis.pdf_id == pdf_id)\
                     .order_by(Analysis.analysis_date.desc())\
                     .first()

def create_clauses(clause_list: List[Dict[str, Any]]) -> List[Clause]:
    """Create multiple clause records"""
    with db_manager.get_session() as session:
        clauses = [Clause(**clause_data) for clause_data in clause_list]
        session.add_all(clauses)
        session.flush()
        for clause in clauses:
            session.refresh(clause)
        return clauses

def create_feedback(feedback_data: Dict[str, Any]) -> Feedback:
    """Create a new feedback record"""
    with db_manager.get_session() as session:
        feedback = Feedback(**feedback_data)
        session.add(feedback)
        session.flush()
        session.refresh(feedback)
        return feedback

def get_pdf_with_analyses(pdf_id: int) -> Optional[PDF]:
    """Get PDF with all its analyses"""
    with db_manager.get_session() as session:
        return session.query(PDF)\
                     .filter(PDF.id == pdf_id)\
                     .first()

def get_analysis_with_clauses(analysis_id: int) -> Optional[Analysis]:
    """Get analysis with all its clauses"""
    with db_manager.get_session() as session:
        return session.query(Analysis)\
                     .filter(Analysis.id == analysis_id)\
                     .first()

def update_pdf(pdf_id: int, update_data: Dict[str, Any]) -> Optional[PDF]:
    """Update PDF record"""
    with db_manager.get_session() as session:
        pdf = session.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            for key, value in update_data.items():
                if hasattr(pdf, key):
                    setattr(pdf, key, value)
            session.flush()
            session.refresh(pdf)
        return pdf

def delete_pdf(pdf_id: int) -> bool:
    """Delete PDF and all related records (cascading)"""
    with db_manager.get_session() as session:
        pdf = session.query(PDF).filter(PDF.id == pdf_id).first()
        if pdf:
            session.delete(pdf)
            return True
        return False

def get_user_pdfs(user_id: int, limit: int = 50) -> List[PDF]:
    """Get all PDFs uploaded by a user"""
    with db_manager.get_session() as session:
        return session.query(PDF)\
                     .filter(PDF.uploaded_by == user_id)\
                     .order_by(PDF.upload_date.desc())\
                     .limit(limit)\
                     .all()

def search_pdfs_by_name(name_pattern: str, limit: int = 20) -> List[PDF]:
    """Search PDFs by name pattern"""
    with db_manager.get_session() as session:
        return session.query(PDF)\
                     .filter(PDF.pdf_name.ilike(f'%{name_pattern}%'))\
                     .order_by(PDF.upload_date.desc())\
                     .limit(limit)\
                     .all()

# Utility functions

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash for file content"""
    return hashlib.sha256(content).hexdigest()

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

# Database initialization function
def initialize_database():
    """Initialize database - create tables and test connection"""
    try:
        print("🔌 Testing database connection...")
        
        if not db_manager.test_connection():
            raise Exception("Database connection failed")
        
        print("✅ Database connection successful")
        
        print("🏗️ Creating database tables...")
        db_manager.create_tables()
        
        print("✅ Database initialization complete")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("🧪 Initializing SQLAlchemy database...")
    initialize_database()
    print("✅ Database setup complete")
