You're still getting detached instance errors because the `create_user()` and `get_user_by_session()` functions are returning SQLAlchemy objects instead of dictionaries. Let me write the correct implementations following Fix 2:

## **🔧 Correct Function Implementations**

### **1. create_user() Function**

```python
def create_user(session_id: str, username: str = None, password_hash: str = None) -> dict:
    """Create a new user and return user data as dictionary"""
    with db_manager.get_session() as session:
        user = User(
            username=username,
            password_hash=password_hash,
            session_id=session_id
        )
        session.add(user)
        session.flush()
        session.refresh(user)
        
        # Extract all data within session and return as dict
        return {
            'id': str(user.id),  # Convert UUID to string
            'username': user.username,
            'password_hash': user.password_hash,
            'session_id': user.session_id,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None
        }
```

### **2. get_user_by_session() Function**

```python
def get_user_by_session(session_id: str) -> Optional[dict]:
    """Get user by session ID and return as dictionary"""
    with db_manager.get_session() as session:
        user = session.query(User).filter(User.session_id == session_id).first()
        
        if not user:
            return None
        
        # Extract all data within session and return as dict
        return {
            'id': str(user.id),  # Convert UUID to string
            'username': user.username,
            'password_hash': user.password_hash,
            'session_id': user.session_id,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None,
            
            # Include relationship counts (safe within session)
            'uploaded_pdfs_count': len(user.uploaded_pdfs),
            'feedbacks_count': len(user.feedbacks)
        }
```

### **3. get_user_by_id() Function (you'll need this too)**

```python
def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID and return as dictionary"""
    with db_manager.get_session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            return None
        
        return {
            'id': str(user.id),
            'username': user.username,
            'password_hash': user.password_hash,
            'session_id': user.session_id,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_active': user.last_active.isoformat() if user.last_active else None,
            'uploaded_pdfs_count': len(user.uploaded_pdfs),
            'feedbacks_count': len(user.feedbacks)
        }
```

### **4. All Other Database Functions Need Similar Updates**

**create_pdf():**
```python
def create_pdf(pdf_data: Dict[str, Any]) -> dict:
    """Create a new PDF record and return as dictionary"""
    with db_manager.get_session() as session:
        pdf = PDF(**pdf_data)
        session.add(pdf)
        session.flush()
        session.refresh(pdf)
        
        return {
            'id': pdf.id,
            'pdf_name': pdf.pdf_name,
            'file_hash': pdf.file_hash,
            'upload_date': pdf.upload_date.isoformat() if pdf.upload_date else None,
            'processed_date': pdf.processed_date.isoformat() if pdf.processed_date else None,
            'layout': pdf.layout,
            'original_word_count': pdf.original_word_count,
            'original_page_count': pdf.original_page_count,
            'parsability': pdf.parsability,
            'final_word_count': pdf.final_word_count,
            'final_page_count': pdf.final_page_count,
            'avg_words_per_page': pdf.avg_words_per_page,
            'raw_content': pdf.raw_content,
            'final_content': pdf.final_content,
            'obfuscation_applied': pdf.obfuscation_applied,
            'pages_removed_count': pdf.pages_removed_count,
            'paragraphs_obfuscated_count': pdf.paragraphs_obfuscated_count,
            'obfuscation_summary': pdf.obfuscation_summary,
            'uploaded_by': str(pdf.uploaded_by)
        }
```

**get_pdf_by_hash():**
```python
def get_pdf_by_hash(file_hash: str) -> Optional[dict]:
    """Get PDF by file hash for deduplication"""
    with db_manager.get_session() as session:
        pdf = session.query(PDF).filter(PDF.file_hash == file_hash).first()
        
        if not pdf:
            return None
        
        return {
            'id': pdf.id,
            'pdf_name': pdf.pdf_name,
            'file_hash': pdf.file_hash,
            'upload_date': pdf.upload_date.isoformat() if pdf.upload_date else None,
            'uploaded_by': str(pdf.uploaded_by),
            'original_word_count': pdf.original_word_count,
            'final_word_count': pdf.final_word_count,
            # ... include all fields you need
        }
```

**create_analysis():**
```python
def create_analysis(analysis_data: Dict[str, Any]) -> dict:
    """Create a new analysis record and return as dictionary"""
    with db_manager.get_session() as session:
        analysis = Analysis(**analysis_data)
        session.add(analysis)
        session.flush()
        session.refresh(analysis)
        
        return {
            'id': analysis.id,
            'pdf_id': analysis.pdf_id,
            'analysis_date': analysis.analysis_date.isoformat() if analysis.analysis_date else None,
            'version': analysis.version,
            'form_number': analysis.form_number,
            'pi_clause': analysis.pi_clause,
            'ci_clause': analysis.ci_clause,
            'data_usage_mentioned': analysis.data_usage_mentioned,
            'data_limitations_exists': analysis.data_limitations_exists,
            'summary': analysis.summary,
            'raw_json': analysis.raw_json,
            'processed_by': analysis.processed_by,
            'processing_time': analysis.processing_time
        }
```

## **🧪 Updated Test Functions**

### **TestUserOperations:**

```python
def test_create_user(self):
    """Test user creation"""
    print("\n👤 Testing user creation...")
    
    session_id = generate_session_id()
    user = create_user(
        session_id=session_id,
        username="test_user",
        password_hash="hashed_password_123"
    )
    
    assert user['id'] is not None, "User ID should be auto-generated"
    assert user['session_id'] == session_id, "Session ID should match"
    assert user['username'] == "test_user", "Username should match"
    assert user['created_at'] is not None, "Created timestamp should be set"
    
    print(f"✅ User created with ID: {user['id']}")
    return user

def test_get_user_by_session(self):
    """Test retrieving user by session ID"""
    print("\n🔍 Testing user retrieval by session...")
    
    # Create a test user first
    session_id = generate_session_id()
    created_user = create_user(session_id=session_id, username="session_test_user")
    
    # Retrieve user by session
    retrieved_user = get_user_by_session(session_id)
    
    assert retrieved_user is not None, "User should be found"
    assert retrieved_user['id'] == created_user['id'], "User IDs should match"
    assert retrieved_user['session_id'] == session_id, "Session IDs should match"
    assert retrieved_user['username'] == "session_test_user", "Username should match"
    
    print(f"✅ User retrieved successfully: {retrieved_user['username']}")
    return retrieved_user
```

## **🎯 Key Points**

1. **All functions return dictionaries**, never SQLAlchemy objects
2. **All database access happens within sessions**
3. **UUIDs are converted to strings** for JSON compatibility
4. **Datetime objects are converted to ISO strings**
5. **Relationship counts are included when accessed within sessions**
6. **Tests expect dictionary keys**, not object attributes

This approach completely eliminates detached instance errors because you never pass SQLAlchemy objects outside of session contexts.
