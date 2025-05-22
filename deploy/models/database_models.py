# models.py - Enhanced SQLAlchemy models for multi-stage data storage

from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User management for multi-user support"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

class PDF(Base):
    """Stage 1: PDF Parsing and Metadata Storage"""
    __tablename__ = 'pdfs'
    
    # Primary identifiers
    id = Column(Integer, primary_key=True)
    pdf_name = Column(String, nullable=False)
    file_hash = Column(String, unique=True, nullable=False)  # SHA256 for deduplication
    
    # Timestamps
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    
    # PDF Parsing Data (Stage 1)
    layout = Column(String)  # single_column, double_column
    word_count = Column(Integer)
    page_count = Column(Integer)
    parsability = Column(Boolean)
    avg_words_per_page = Column(Float)
    
    # Content Storage
    raw_content = Column(Text)  # Original extracted text
    final_content = Column(Text)  # After obfuscation/processing
    
    # User tracking
    uploaded_by = Column(String)  # Session ID of uploader
    
    # Relationships
    analyses = relationship("Analysis", back_populates="pdf")
    feedbacks = relationship("Feedback", back_populates="pdf")

class Analysis(Base):
    """Stage 2: Contract Analysis Results with Versioning"""
    __tablename__ = 'analyses'
    
    # Primary identifiers
    id = Column(Integer, primary_key=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id'), nullable=False)
    
    # Timestamps
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Versioning for re-runs
    version = Column(Integer, default=1)
    
    # Analysis Results (Stage 2)
    form_number = Column(String)
    pi_clause = Column(String)  # yes/no/missing
    ci_clause = Column(String)  # yes/no/missing
    data_usage_mentioned = Column(String)  # yes/no/missing
    data_limitations_exists = Column(String)  # yes/no/missing
    
    # Raw analysis data
    raw_json = Column(JSON)  # Complete analysis JSON
    summary = Column(Text)
    
    # Processing metadata
    processed_by = Column(String)  # Session ID of processor
    processing_time = Column(Float)  # Time taken in seconds
    
    # Relationships
    pdf = relationship("PDF", back_populates="analyses")
    clauses = relationship("Clause", back_populates="analysis", cascade="all, delete-orphan")

class Clause(Base):
    """Stage 3: Individual Clauses Storage"""
    __tablename__ = 'clauses'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=False)
    
    # Clause data
    clause_type = Column(String)  # pi_clause, ci_clause, data_usage, etc.
    clause_text = Column(Text)
    clause_order = Column(Integer)  # Order in the document
    
    # Relationships
    analysis = relationship("Analysis", back_populates="clauses")

class Feedback(Base):
    """Stage 4: User Feedback Storage"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id'), nullable=False)
    
    # Timestamps
    feedback_date = Column(DateTime, default=datetime.utcnow)
    
    # Feedback content
    form_number_feedback = Column(Text)  # Specific feedback about form number
    general_feedback = Column(Text)  # General feedback about analysis
    
    # User tracking
    user_session_id = Column(String)
    
    # Relationships
    pdf = relationship("PDF", back_populates="feedbacks")

# Batch Processing Management
class BatchJob(Base):
    """Batch Processing Job Management"""
    __tablename__ = 'batch_jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    
    # Job metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Job configuration
    total_files = Column(Integer)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    status = Column(String, default='pending')  # pending, running, completed, failed
    
    # User tracking
    created_by = Column(String)  # Session ID
    
    # Job results
    results_json = Column(JSON)
    error_log = Column(Text)
