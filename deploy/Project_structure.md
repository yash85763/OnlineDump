The issue is that you've changed the database functions to return dictionaries (Fix 2), but the test code is still expecting SQLAlchemy objects with `.id` attributes. You need to update the tests to work with the new dictionary return format.

## **🔧 What to Update in Tests**

### **1. Update Test Expectations**

**Change from:**
```python
def test_create_user(self):
    user = create_user(session_id=session_id, username="test_user")
    assert user.id is not None  # ❌ user is now a dict
    assert user.session_id == session_id  # ❌ user is now a dict
```

**To:**
```python
def test_create_user(self):
    user = create_user(session_id=session_id, username="test_user")
    assert user['id'] is not None  # ✅ Access dict keys
    assert user['session_id'] == session_id  # ✅ Access dict keys
```

### **2. Update All Test Methods**

Here are the key patterns to update:

**User Tests:**
```python
def test_create_user(self):
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
    session_id = generate_session_id()
    created_user = create_user(session_id=session_id, username="session_test_user")
    
    retrieved_user = get_user_by_session(session_id)
    
    assert retrieved_user is not None, "User should be found"
    assert retrieved_user['id'] == created_user['id'], "User IDs should match"
    assert retrieved_user['session_id'] == session_id, "Session IDs should match"
    
    print(f"✅ User retrieved successfully: {retrieved_user['username']}")
    return retrieved_user
```

**PDF Tests:**
```python
def test_create_pdf(self):
    user = create_user(session_id=generate_session_id(), username="pdf_test_user")
    
    pdf_content = b"Sample PDF content for testing"
    file_hash = generate_file_hash(pdf_content)
    
    pdf_data = {
        'pdf_name': 'test_contract.pdf',
        'file_hash': file_hash,
        'upload_date': datetime.utcnow(),
        'uploaded_by': user['id'],  # ✅ Use dict key
        # ... other fields
    }
    
    pdf = create_pdf(pdf_data)
    
    assert pdf['id'] is not None, "PDF ID should be auto-generated"
    assert pdf['pdf_name'] == 'test_contract.pdf', "PDF name should match"
    assert pdf['uploaded_by'] == user['id'], "Uploaded by should match user ID"
    
    print(f"✅ PDF created with ID: {pdf['id']}")
    return pdf, user
```

**Analysis Tests:**
```python
def test_create_analysis(self):
    pdf, user = TestPDFOperations().test_create_pdf()
    
    analysis_data = {
        'pdf_id': pdf['id'],  # ✅ Use dict key
        'analysis_date': datetime.utcnow(),
        'version': 'v1.0',
        # ... other fields
    }
    
    analysis = create_analysis(analysis_data)
    
    assert analysis['id'] is not None, "Analysis ID should be auto-generated"
    assert analysis['pdf_id'] == pdf['id'], "PDF ID should match"
    
    print(f"✅ Analysis created with ID: {analysis['id']}")
    return analysis, pdf
```

### **3. Update Relationship Tests**

**For tests that check relationships, you need to modify the approach:**

```python
def test_user_pdf_relationship(self):
    user = create_user(session_id=generate_session_id(), username="relationship_test_user")
    
    pdf_data_list = [
        {
            'pdf_name': 'contract1.pdf',
            'file_hash': generate_file_hash(b"content1"),
            'uploaded_by': user['id']  # ✅ Use dict key
        },
        # ... more PDFs
    ]
    
    pdfs = [create_pdf(pdf_data) for pdf_data in pdf_data_list]
    
    # Test relationship by querying the database directly
    user_pdfs = get_user_pdfs(user['id'])  # This should return list of dicts
    
    assert len(user_pdfs) == 3, "User should have 3 PDFs"
    assert all(pdf['uploaded_by'] == user['id'] for pdf in user_pdfs), "All PDFs should belong to user"
    
    print(f"✅ User-PDF relationship working: {len(user_pdfs)} PDFs linked to user")
    return user, pdfs
```

### **4. Update Functions That Return Objects**

**You also need to update your database functions to return dictionaries:**

```python
def get_user_pdfs(user_id: str, limit: int = 50) -> List[dict]:
    """Get all PDFs uploaded by a user"""
    with db_manager.get_session() as session:
        pdfs = session.query(PDF)\
                     .filter(PDF.uploaded_by == user_id)\
                     .order_by(PDF.upload_date.desc())\
                     .limit(limit)\
                     .all()
        
        return [
            {
                'id': pdf.id,
                'pdf_name': pdf.pdf_name,
                'file_hash': pdf.file_hash,
                'upload_date': pdf.upload_date.isoformat(),
                'uploaded_by': str(pdf.uploaded_by),
                'original_word_count': pdf.original_word_count,
                'final_word_count': pdf.final_word_count,
                # ... other fields as needed
            }
            for pdf in pdfs
        ]

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
            'upload_date': pdf.upload_date.isoformat(),
            'uploaded_by': str(pdf.uploaded_by),
            # ... other fields
        }
```

### **5. Update Tests That Use Returned Values**

```python
def test_cascading_deletes(self):
    user = create_user(session_id=generate_session_id(), username="cascade_test_user")
    
    pdf_data = {
        'pdf_name': 'cascade_test.pdf',
        'file_hash': generate_file_hash(b"cascade test content"),
        'uploaded_by': user['id']  # ✅ Use dict key
    }
    pdf = create_pdf(pdf_data)
    
    analysis_data = {
        'pdf_id': pdf['id'],  # ✅ Use dict key
        'version': 'v1.0',
        'summary': 'Test analysis for cascade'
    }
    analysis = create_analysis(analysis_data)
    
    # Delete PDF - should cascade
    success = delete_pdf(pdf['id'])  # ✅ Use dict key
    assert success, "PDF deletion should succeed"
    
    # Verify cascading worked
    with db_manager.get_session() as session:
        assert session.query(PDF).filter(PDF.id == pdf['id']).first() is None
        assert session.query(Analysis).filter(Analysis.id == analysis['id']).first() is None
    
    print("✅ Cascading deletes working correctly")
```

## **🎯 Key Pattern Changes**

1. **Object attributes** → **Dictionary keys**: `user.id` → `user['id']`
2. **Pass IDs not objects**: `pdf_id': pdf['id']` instead of `'pdf_id': pdf.id`
3. **Update function signatures**: Return `List[dict]` instead of `List[Model]`
4. **Test relationship queries**: Use the helper functions that return dicts

This approach maintains the benefits of Fix 2 (no detached instances) while making your tests work with the new dictionary-based return format.
