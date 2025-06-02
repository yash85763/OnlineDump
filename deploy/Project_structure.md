The issue is that multiple tests are calling `TestPDFOperations().test_create_pdf()` which creates PDFs with the same file hash, violating the unique constraint. You need to either clean the database between tests or make each test generate unique data.

## **🔧 Solution 1: Add Database Cleanup (Recommended)**

Add a cleanup method that runs before each test class:

```python
class TestDatabaseSetup:
    def setup_method(self):
        """Clean database before each test"""
        self.cleanup_database()
    
    def cleanup_database(self):
        """Remove all test data"""
        try:
            with db_manager.get_session() as session:
                # Delete in reverse dependency order
                session.execute(text("DELETE FROM clauses"))
                session.execute(text("DELETE FROM feedback"))
                session.execute(text("DELETE FROM analyses"))
                session.execute(text("DELETE FROM pdfs"))
                session.execute(text("DELETE FROM users"))
                print("🧹 Database cleaned for testing")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

# Add cleanup to all test classes
class TestUserOperations:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestPDFOperations:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestAnalysisOperations:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestClauseOperations:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestFeedbackOperations:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestRelationships:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()

class TestDataManipulation:
    def setup_method(self):
        TestDatabaseSetup().cleanup_database()
```

## **🔧 Solution 2: Create Independent Test Data**

Instead of calling other test methods, create fresh data in each test:

```python
class TestAnalysisOperations:
    def create_test_pdf(self, name_suffix=""):
        """Create a test PDF with unique data"""
        user = create_user(
            session_id=generate_session_id(),
            username=f"analysis_test_user{name_suffix}"
        )
        
        pdf_content = f"Analysis test PDF content{name_suffix}".encode()
        file_hash = generate_file_hash(pdf_content)
        
        pdf_data = {
            'pdf_name': f'analysis_test{name_suffix}.pdf',
            'file_hash': file_hash,
            'uploaded_by': user['id']
        }
        
        pdf = create_pdf(pdf_data)
        return pdf, user
    
    def test_create_analysis(self):
        """Test analysis creation"""
        print("\n🔍 Testing analysis creation...")
        
        # Create PDF with unique data for this test
        pdf, user = self.create_test_pdf("_analysis")
        
        analysis_data = {
            'pdf_id': pdf['id'],
            'analysis_date': datetime.utcnow(),
            'version': 'v1.0',
            'form_number': 'FORM-2024-001',
            'summary': 'Test analysis for analysis operations'
        }
        
        analysis = create_analysis(analysis_data)
        
        assert analysis['id'] is not None, "Analysis ID should be auto-generated"
        assert analysis['pdf_id'] == pdf['id'], "PDF ID should match"
        
        print(f"✅ Analysis created with ID: {analysis['id']}")
        return analysis, pdf
```

## **🔧 Solution 3: Generate Unique Hashes**

Add timestamp or random data to make each hash unique:

```python
import time
import random

def generate_unique_file_hash(base_content: str = "test content") -> str:
    """Generate unique file hash with timestamp and random data"""
    timestamp = str(time.time())
    random_num = str(random.randint(1000, 9999))
    unique_content = f"{base_content}_{timestamp}_{random_num}".encode()
    return generate_file_hash(unique_content)

# Use in tests:
def test_create_analysis(self):
    user = create_user(session_id=generate_session_id(), username="analysis_test_user")
    
    pdf_data = {
        'pdf_name': 'analysis_test.pdf',
        'file_hash': generate_unique_file_hash("analysis test"),  # Unique hash
        'uploaded_by': user['id']
    }
    
    pdf = create_pdf(pdf_data)
    # ... rest of test
```

## **🔧 Solution 4: Use Test Fixtures (Most Professional)**

Create reusable test fixtures:

```python
class TestFixtures:
    @staticmethod
    def create_test_user(name_suffix=""):
        return create_user(
            session_id=generate_session_id(),
            username=f"test_user{name_suffix}_{int(time.time())}"
        )
    
    @staticmethod
    def create_test_pdf(user_id, name_suffix=""):
        timestamp = int(time.time())
        pdf_content = f"Test PDF content {name_suffix} {timestamp}".encode()
        
        return create_pdf({
            'pdf_name': f'test{name_suffix}_{timestamp}.pdf',
            'file_hash': generate_file_hash(pdf_content),
            'uploaded_by': user_id,
            'original_word_count': 1000,
            'final_word_count': 800
        })
    
    @staticmethod
    def create_test_analysis(pdf_id, version="v1.0"):
        return create_analysis({
            'pdf_id': pdf_id,
            'version': version,
            'summary': f'Test analysis {int(time.time())}'
        })

# Use in tests:
class TestAnalysisOperations:
    def test_create_analysis(self):
        user = TestFixtures.create_test_user("analysis")
        pdf = TestFixtures.create_test_pdf(user['id'], "analysis")
        analysis = TestFixtures.create_test_analysis(pdf['id'])
        
        assert analysis['id'] is not None
        # ... rest of test
```

## **🎯 Recommended Approach**

Use **Solution 1 (Database Cleanup)** as it's the most straightforward:

1. Add cleanup to each test class
2. Each test runs with a clean database
3. No need to worry about unique data generation
4. Tests are isolated and predictable

This ensures each test starts with a clean state and prevents constraint violations.
