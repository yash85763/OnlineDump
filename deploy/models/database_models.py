# models/database_models.py - Complete database models without user obfuscation interface

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
    """
    Stage 1: PDF Parsing and Metadata Storage with Automatic Obfuscation
    
    Stores both original and obfuscated content. Obfuscation happens automatically
    during parsing stage without user interaction.
    """
    __tablename__ = 'pdfs'
    
    # Primary identifiers
    id = Column(Integer, primary_key=True)
    pdf_name = Column(String, nullable=False)
    file_hash = Column(String, unique=True, nullable=False)  # SHA256 for deduplication
    
    # Timestamps
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    
    # PDF Parsing Data (Stage 1) - Based on ORIGINAL content
    layout = Column(String)  # single_column, double_column
    original_word_count = Column(Integer)  # Word count BEFORE obfuscation
    original_page_count = Column(Integer)  # Page count BEFORE obfuscation
    parsability = Column(Boolean)
    
    # PDF Data After Obfuscation - Used for analysis
    final_word_count = Column(Integer)  # Word count AFTER obfuscation
    final_page_count = Column(Integer)  # Page count AFTER obfuscation  
    avg_words_per_page = Column(Float)  # Based on final content
    
    # Content Storage
    raw_content = Column(Text)  # Original extracted text (before obfuscation)
    final_content = Column(Text)  # Content after obfuscation (used for analysis)
    
    # Automatic Obfuscation Data (Internal tracking only)
    obfuscation_applied = Column(Boolean, default=True)  # Always true - obfuscation is automatic
    pages_removed_count = Column(Integer, default=0)  # Number of pages removed by obfuscation
    paragraphs_obfuscated_count = Column(Integer, default=0)  # Number of paragraphs obfuscated
    obfuscation_summary = Column(JSON)  # Internal summary of obfuscation operations
    
    # User tracking
    uploaded_by = Column(String)  # Session ID of uploader
    
    # Relationships
    analyses = relationship("Analysis", back_populates="pdf")
    feedbacks = relationship("Feedback", back_populates="pdf")

class Analysis(Base):
    """
    Stage 2: Contract Analysis Results with Versioning
    
    Analysis is always performed on obfuscated content (final_content).
    """
    __tablename__ = 'analyses'
    
    # Primary identifiers
    id = Column(Integer, primary_key=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id'), nullable=False)
    
    # Timestamps
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Versioning for re-runs
    version = Column(Integer, default=1)
    
    # Analysis Results (Stage 2) - From obfuscated content
    form_number = Column(String)
    pi_clause = Column(String)  # yes/no/missing
    ci_clause = Column(String)  # yes/no/missing
    data_usage_mentioned = Column(String)  # yes/no/missing
    data_limitations_exists = Column(String)  # yes/no/missing
    
    # Analysis content
    summary = Column(Text)
    raw_json = Column(JSON)  # Complete analysis JSON from ContractAnalyzer
    
    # Processing metadata
    processed_by = Column(String)  # Session ID of processor
    processing_time = Column(Float)  # Time taken in seconds
    
    # Relationships
    pdf = relationship("PDF", back_populates="analyses")
    clauses = relationship("Clause", back_populates="analysis", cascade="all, delete-orphan")

class Clause(Base):
    """
    Stage 3: Individual Clauses Storage
    
    Clauses extracted from analysis of obfuscated content.
    """
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
    """
    Stage 4: User Feedback Storage
    
    User feedback about analysis results (which are based on obfuscated content).
    """
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id'), nullable=False)
    
    # Timestamps
    feedback_date = Column(DateTime, default=datetime.utcnow)
    
    # Feedback content (as originally requested)
    form_number_feedback = Column(Text)  # Specific feedback about form number
    general_feedback = Column(Text)  # General feedback about analysis
    
    # Optional rating
    rating = Column(Integer)  # 1-5 rating scale
    
    # User tracking
    user_session_id = Column(String)
    
    # Relationships
    pdf = relationship("PDF", back_populates="feedbacks")

class BatchJob(Base):
    """
    Batch Processing Job Management
    
    Tracks batch processing jobs with automatic obfuscation.
    """
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
    
    # Batch processing summary (including obfuscation stats)
    total_pages_processed = Column(Integer, default=0)
    total_pages_removed = Column(Integer, default=0)  # From obfuscation across all files
    total_paragraphs_obfuscated = Column(Integer, default=0)  # From obfuscation across all files
    
    # User tracking
    created_by = Column(String)  # Session ID
    
    # Job results
    results_json = Column(JSON)
    error_log = Column(Text)

# Optional: System configuration table for obfuscation settings (admin only)
class SystemConfig(Base):
    """
    System-level configuration (admin only - no user interface)
    
    Stores system-wide obfuscation settings that apply to all document processing.
    """
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True)
    config_key = Column(String, unique=True, nullable=False)
    config_value = Column(JSON)  # Store configuration as JSON
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String)  # Admin identifier

# Example system config entries would be:
# config_key: 'obfuscation_page_threshold' 
# config_value: {"multiplier": 1.0, "enabled": true}
#
# config_key: 'obfuscation_keywords'
# config_value: {"keywords": ["confidential", "ssn", ...], "case_sensitive": false}
#
# config_key: 'obfuscation_combinations'  
# config_value: {"combinations": [["personal", "information"], ...]}

# Database initialization function
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_default_system_configs():
    """
    Get default system configurations for obfuscation.
    This would be loaded once during system setup.
    """
    return [
        {
            "config_key": "obfuscation_page_threshold",
            "config_value": {
                "multiplier": 1.0,  # Remove pages with less than average word count
                "enabled": True
            },
            "description": "Page-level obfuscation settings based on word count"
        },
        {
            "config_key": "obfuscation_keywords", 
            "config_value": {
                "keywords": [
                    # Personal Information
                    "ssn", "social security number", "social security",
                    "tax id", "taxpayer id", "ein", "employee id",
                    "driver license", "passport number", "account number",
                    "credit card", "debit card", "bank account", "routing number",
                    
                    # Confidential/Proprietary  
                    "confidential", "proprietary", "trade secret", "internal use only",
                    "restricted", "classified", "private", "sensitive",
                    
                    # Security
                    "password", "api key", "access key", "secret key", "token",
                    
                    # Financial
                    "salary", "compensation", "financial data", "revenue", "profit",
                    
                    # Medical
                    "medical record", "health information", "diagnosis", "treatment"
                ],
                "case_sensitive": False,
                "whole_words_only": True
            },
            "description": "Keywords that trigger paragraph-level obfuscation"
        },
        {
            "config_key": "obfuscation_combinations",
            "config_value": {
                "combinations": [
                    ["personal", "information"],
                    ["personally", "identifiable"], 
                    ["credit", "score"],
                    ["financial", "data"],
                    ["customer", "data"],
                    ["employee", "records"],
                    ["medical", "records"],
                    ["health", "information"],
                    ["confidential", "data"],
                    ["trade", "secret"],
                    ["internal", "document"]
                ]
            },
            "description": "Keyword combinations that trigger paragraph-level obfuscation"
        },
        {
            "config_key": "obfuscation_replacement_text",
            "config_value": {
                "page_replacement": "[PAGE CONTENT REMOVED - Below average word count]",
                "paragraph_replacement": "[PARAGRAPH OBFUSCATED - Contains sensitive keywords]"
            },
            "description": "Replacement text for obfuscated content"
        }
    ]

# Helper functions for working with the models

def get_pdf_processing_stats(db_session, pdf_id: int):
    """Get processing statistics for a PDF including obfuscation impact"""
    
    pdf = db_session.query(PDF).filter_by(id=pdf_id).first()
    if not pdf:
        return None
    
    return {
        "pdf_name": pdf.pdf_name,
        "original_stats": {
            "word_count": pdf.original_word_count,
            "page_count": pdf.original_page_count
        },
        "final_stats": {
            "word_count": pdf.final_word_count, 
            "page_count": pdf.final_page_count,
            "avg_words_per_page": pdf.avg_words_per_page
        },
        "obfuscation_impact": {
            "pages_removed": pdf.pages_removed_count,
            "paragraphs_obfuscated": pdf.paragraphs_obfuscated_count,
            "word_reduction_percentage": ((pdf.original_word_count - pdf.final_word_count) / pdf.original_word_count * 100) if pdf.original_word_count > 0 else 0,
            "page_reduction_percentage": ((pdf.original_page_count - pdf.final_page_count) / pdf.original_page_count * 100) if pdf.original_page_count > 0 else 0
        },
        "processing_metadata": {
            "upload_date": pdf.upload_date,
            "processed_date": pdf.processed_date,
            "uploaded_by": pdf.uploaded_by,
            "layout": pdf.layout,
            "parsability": pdf.parsability
        }
    }

def get_system_obfuscation_config(db_session):
    """Get current system obfuscation configuration"""
    
    configs = {}
    
    # Get all obfuscation-related configs
    obfuscation_configs = db_session.query(SystemConfig).filter(
        SystemConfig.config_key.like('obfuscation_%')
    ).all()
    
    for config in obfuscation_configs:
        configs[config.config_key] = config.config_value
    
    # Return default config if no system config exists
    if not configs:
        defaults = get_default_system_configs()
        return {config["config_key"]: config["config_value"] for config in defaults}
    
    return configs

def update_system_obfuscation_config(db_session, config_key: str, config_value: dict, updated_by: str = "system"):
    """Update system obfuscation configuration (admin only)"""
    
    existing_config = db_session.query(SystemConfig).filter_by(config_key=config_key).first()
    
    if existing_config:
        existing_config.config_value = config_value
        existing_config.updated_at = datetime.utcnow()
        existing_config.updated_by = updated_by
    else:
        new_config = SystemConfig(
            config_key=config_key,
            config_value=config_value,
            updated_by=updated_by,
            description=f"Obfuscation configuration for {config_key}"
        )
        db_session.add(new_config)
    
    db_session.commit()

def get_batch_obfuscation_summary(db_session, batch_job_id: int):
    """Get obfuscation summary for a batch job"""
    
    batch_job = db_session.query(BatchJob).filter_by(id=batch_job_id).first()
    if not batch_job:
        return None
    
    return {
        "batch_job_id": batch_job.job_id,
        "total_files": batch_job.total_files,
        "processed_files": batch_job.processed_files,
        "obfuscation_summary": {
            "total_pages_processed": batch_job.total_pages_processed,
            "total_pages_removed": batch_job.total_pages_removed,
            "total_paragraphs_obfuscated": batch_job.total_paragraphs_obfuscated,
            "average_page_removal_rate": (batch_job.total_pages_removed / batch_job.total_pages_processed * 100) if batch_job.total_pages_processed > 0 else 0
        },
        "job_status": {
            "status": batch_job.status,
            "created_at": batch_job.created_at,
            "completed_at": batch_job.completed_at,
            "created_by": batch_job.created_by
        }
    }