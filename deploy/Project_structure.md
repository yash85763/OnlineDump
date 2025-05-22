# Enhanced Project Structure

```
contract_analyzer/
├── main.py                     # Entry point - Streamlit app
├── config/
│   ├── __init__.py
│   ├── database.py             # Database configuration & connection
│   └── settings.py             # App settings & environment variables
├── models/
│   ├── __init__.py
│   └── database_models.py      # SQLAlchemy ORM models
├── services/
│   ├── __init__.py
│   ├── pdf_service.py          # PDF processing service
│   ├── analysis_service.py     # Contract analysis service
│   ├── feedback_service.py     # Feedback management service
│   ├── batch_service.py        # Batch processing service
│   └── user_service.py         # User session management
├── utils/
│   ├── __init__.py
│   ├── pdf_handler.py          # Enhanced PDFHandler (from ecfr_api_wrapper)
│   ├── hash_utils.py           # File hashing utilities
│   └── validation.py           # Input validation utilities
├── ui/
│   ├── __init__.py
│   ├── pdf_viewer.py           # PDF viewing components
│   ├── analysis_display.py     # Analysis results display
│   ├── feedback_form.py        # Feedback form components
│   └── batch_interface.py      # Batch processing interface
├── static/
│   └── custom.css              # Custom CSS styling
├── tests/
│   ├── test_services.py
│   └── test_models.py
├── requirements.txt
├── .env                        # Environment variables
└── README.md

## Key Components Overview

### 1. Services Layer (Business Logic)
- **pdf_service.py**: Handles PDF processing, deduplication, and storage
- **analysis_service.py**: Manages contract analysis with versioning
- **feedback_service.py**: Handles user feedback storage and retrieval
- **batch_service.py**: Manages batch processing jobs
- **user_service.py**: Session management and user tracking

### 2. UI Layer (Streamlit Components)
- **pdf_viewer.py**: PDF display and navigation
- **analysis_display.py**: Results visualization
- **feedback_form.py**: Feedback collection interface
- **batch_interface.py**: Batch processing UI

### 3. Models Layer (Data Management)
- **database_models.py**: SQLAlchemy ORM models for all tables

### 4. Config Layer (Configuration)
- **database.py**: Database connection and session management
- **settings.py**: Application configuration and environment variables
