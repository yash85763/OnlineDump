There are two issues here:

## **🔧 Issue 1: 'dict' object has no attribute 'id'**

The problem is in your test - you're creating a user that returns a dict, but then trying to access `user.id` instead of `user['id']`.

**Fix the test_create_pdf method:**

```python
def test_create_pdf(self):
    """Test PDF creation"""
    print("\n📄 Testing PDF creation...")
    
    # Create a user first
    user = create_user(session_id=generate_session_id(), username="pdf_test_user")
    
    # Sample PDF data
    pdf_content = b"Sample PDF content for testing"
    file_hash = generate_file_hash(pdf_content)
    
    pdf_data = {
        'pdf_name': 'test_contract.pdf',
        'file_hash': file_hash,
        'upload_date': datetime.utcnow(),
        'layout': 'single-column',
        'original_word_count': 1500,
        'original_page_count': 5,
        'parsability': 0.95,
        'final_word_count': 1200,
        'final_page_count': 4,
        'avg_words_per_page': 300.0,
        'raw_content': {"pages": [{"text": "Sample content"}]},
        'final_content': "Processed sample content",
        'obfuscation_applied': True,
        'pages_removed_count': 1,
        'paragraphs_obfuscated_count': 5,
        'obfuscation_summary': {"method": "entity_replacement", "confidence": 0.8},
        'uploaded_by': user['id']  # ✅ Use dict key, not user.id
    }
    
    pdf = create_pdf(pdf_data)
    
    assert pdf['id'] is not None, "PDF ID should be auto-generated"
    assert pdf['pdf_name'] == 'test_contract.pdf', "PDF name should match"
    assert pdf['file_hash'] == file_hash, "File hash should match"
    assert pdf['uploaded_by'] == user['id'], "Uploaded by should match user ID"
    assert pdf['original_word_count'] == 1500, "Original word count should match"
    
    print(f"✅ PDF created with ID: {pdf['id']}")
    return pdf, user
```

## **🔧 Issue 2: Duplicate key violation**

The second test is trying to use the same file hash as the first test. You need to generate unique hashes for each test.

**Fix the test_get_pdf_by_hash method:**

```python
def test_get_pdf_by_hash(self):
    """Test PDF retrieval by hash for deduplication"""
    print("\n🔍 Testing PDF retrieval by hash...")
    
    # Create a user first
    user = create_user(session_id=generate_session_id(), username="hash_test_user")
    
    # Create unique content for this test
    pdf_content = b"Unique PDF content for hash test"
    file_hash = generate_file_hash(pdf_content)
    
    pdf_data = {
        'pdf_name': 'hash_test_contract.pdf',
        'file_hash': file_hash,
        'uploaded_by': user['id']  # ✅ Use dict key
    }
    
    # Create the PDF first
    created_pdf = create_pdf(pdf_data)
    
    # Now retrieve PDF by hash
    retrieved_pdf = get_pdf_by_hash(file_hash)
    
    assert retrieved_pdf is not None, "PDF should be found"
    assert retrieved_pdf['id'] == created_pdf['id'], "PDF IDs should match"
    assert retrieved_pdf['file_hash'] == file_hash, "File hashes should match"
    
    print(f"✅ PDF retrieved successfully by hash")
    return retrieved_pdf
```

**Fix the test_pdf_deduplication method:**

```python
def test_pdf_deduplication(self):
    """Test that duplicate PDFs are detected"""
    print("\n🔄 Testing PDF deduplication...")
    
    # Create first PDF
    user = create_user(session_id=generate_session_id(), username="dedup_test_user")
    
    pdf_content = b"Duplicate test content for deduplication test"
    file_hash = generate_file_hash(pdf_content)
    
    pdf_data = {
        'pdf_name': 'original.pdf',
        'file_hash': file_hash,
        'uploaded_by': user['id']  # ✅ Use dict key
    }
    
    # Create first PDF
    pdf1 = create_pdf(pdf_data)
    
    # Try to create duplicate PDF with same hash
    pdf_data['pdf_name'] = 'duplicate.pdf'
    
    try:
        pdf2 = create_pdf(pdf_data)
        pytest.fail("Should not allow duplicate file hash")
    except Exception as e:
        print(f"✅ Duplicate prevention working: {type(e).__name__}")
```

## **🎯 Alternative: Clean Test State**

To avoid hash collisions between tests, you can also add a setup method that cleans the database between tests:

```python
class TestPDFOperations:
    """Test PDF-related database operations"""
    
    def setup_method(self):
        """Clean database before each test"""
        with db_manager.get_session() as session:
            # Clean up any existing test data
            session.execute(text("DELETE FROM clauses"))
            session.execute(text("DELETE FROM analyses"))
            session.execute(text("DELETE FROM feedback"))
            session.execute(text("DELETE FROM pdfs"))
            session.execute(text("DELETE FROM users"))
    
    def test_create_pdf(self):
        # ... test code
```

Or generate unique content for each test:

```python
import time

def generate_unique_content(base_content: str) -> bytes:
    """Generate unique content with timestamp"""
    timestamp = str(time.time())
    return f"{base_content}_{timestamp}".encode()

# In your tests:
pdf_content = generate_unique_content("Sample PDF content")
```

The key fixes are:
1. **Use `user['id']` instead of `user.id`** in all tests
2. **Generate unique file hashes** for each test to avoid duplicates
3. **Ensure all database functions return dictionaries** consistently
