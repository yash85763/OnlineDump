# test_database.py - Comprehensive tests for SQLAlchemy database implementation

import pytest
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys

# Add the database module to path (adjust as needed)
# sys.path.append('/path/to/your/database/module')

# Import database components
from database.models import (
    db_manager, User, PDF, Analysis, Clause, Feedback,
    create_user, get_user_by_session, create_pdf, get_pdf_by_hash,
    get_pdf_by_id, create_analysis, get_latest_analysis, create_clauses,
    create_feedback, get_pdf_with_analyses, get_analysis_with_clauses,
    update_pdf, delete_pdf, get_user_pdfs, search_pdfs_by_name,
    generate_file_hash, generate_session_id, initialize_database
)

class TestDatabaseSetup:
    """Test database connection and table creation"""
    
    def test_database_connection(self):
        """Test basic database connectivity"""
        print("\n🔌 Testing database connection...")
        
        assert db_manager.test_connection(), "Database connection failed"
        print("✅ Database connection successful")
    
    def test_table_creation(self):
        """Test that all tables are created properly"""
        print("\n🏗️ Testing table creation...")
        
        try:
            # Drop and recreate tables for clean test
            db_manager.drop_tables()
            db_manager.create_tables()
            
            # Test that we can query each table (even if empty)
            with db_manager.get_session() as session:
                # Test each table exists and is accessible
                user_count = session.query(User).count()
                pdf_count = session.query(PDF).count()
                analysis_count = session.query(Analysis).count()
                clause_count = session.query(Clause).count()
                feedback_count = session.query(Feedback).count()
                
                print(f"   - Users table: {user_count} records")
                print(f"   - PDFs table: {pdf_count} records")
                print(f"   - Analyses table: {analysis_count} records")
                print(f"   - Clauses table: {clause_count} records")
                print(f"   - Feedback table: {feedback_count} records")
                
            print("✅ All tables created and accessible")
            
        except Exception as e:
            pytest.fail(f"Table creation failed: {str(e)}")

class TestUserOperations:
    """Test user-related database operations"""
    
    def test_create_user(self):
        """Test user creation"""
        print("\n👤 Testing user creation...")
        
        session_id = generate_session_id()
        user = create_user(
            session_id=session_id,
            username="test_user",
            password_hash="hashed_password_123"
        )
        
        assert user.id is not None, "User ID should be auto-generated"
        assert user.session_id == session_id, "Session ID should match"
        assert user.username == "test_user", "Username should match"
        assert user.created_at is not None, "Created timestamp should be set"
        
        print(f"✅ User created with ID: {user.id}")
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
        assert retrieved_user.id == created_user.id, "User IDs should match"
        assert retrieved_user.session_id == session_id, "Session IDs should match"
        
        print(f"✅ User retrieved successfully: {retrieved_user.username}")
        return retrieved_user

class TestPDFOperations:
    """Test PDF-related database operations"""
    
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
            'uploaded_by': user.id
        }
        
        pdf = create_pdf(pdf_data)
        
        assert pdf.id is not None, "PDF ID should be auto-generated"
        assert pdf.pdf_name == 'test_contract.pdf', "PDF name should match"
        assert pdf.file_hash == file_hash, "File hash should match"
        assert pdf.uploaded_by == user.id, "Uploaded by should match user ID"
        assert pdf.original_word_count == 1500, "Original word count should match"
        
        print(f"✅ PDF created with ID: {pdf.id}")
        return pdf, user
    
    def test_get_pdf_by_hash(self):
        """Test PDF retrieval by hash for deduplication"""
        print("\n🔍 Testing PDF retrieval by hash...")
        
        # Create a test PDF first
        pdf, user = self.test_create_pdf()
        
        # Retrieve PDF by hash
        retrieved_pdf = get_pdf_by_hash(pdf.file_hash)
        
        assert retrieved_pdf is not None, "PDF should be found"
        assert retrieved_pdf.id == pdf.id, "PDF IDs should match"
        assert retrieved_pdf.file_hash == pdf.file_hash, "File hashes should match"
        
        print(f"✅ PDF retrieved successfully by hash")
        return retrieved_pdf
    
    def test_pdf_deduplication(self):
        """Test that duplicate PDFs are detected"""
        print("\n🔄 Testing PDF deduplication...")
        
        # Create first PDF
        user = create_user(session_id=generate_session_id(), username="dedup_test_user")
        
        pdf_content = b"Duplicate test content"
        file_hash = generate_file_hash(pdf_content)
        
        pdf_data = {
            'pdf_name': 'original.pdf',
            'file_hash': file_hash,
            'uploaded_by': user.id
        }
        
        # Create first PDF
        pdf1 = create_pdf(pdf_data)
        
        # Try to create duplicate PDF
        pdf_data['pdf_name'] = 'duplicate.pdf'
        
        try:
            pdf2 = create_pdf(pdf_data)
            pytest.fail("Should not allow duplicate file hash")
        except Exception as e:
            print(f"✅ Duplicate prevention working: {type(e).__name__}")

class TestAnalysisOperations:
    """Test analysis-related database operations"""
    
    def test_create_analysis(self):
        """Test analysis creation"""
        print("\n🔍 Testing analysis creation...")
        
        # Create PDF first
        pdf, user = TestPDFOperations().test_create_pdf()
        
        analysis_data = {
            'pdf_id': pdf.id,
            'analysis_date': datetime.utcnow(),
            'version': 'v1.0',
            'form_number': 'FORM-2024-001',
            'pi_clause': 'Personal information clause found in section 3.2',
            'ci_clause': 'Confidential information clause found in section 4.1',
            'data_usage_mentioned': True,
            'data_limitations_exists': True,
            'summary': 'Contract contains standard privacy and confidentiality clauses',
            'raw_json': {
                "analysis_type": "standard",
                "confidence": 0.87,
                "clauses_found": 5
            },
            'processed_by': 'AI_Engine_v2.1',
            'processing_time': 15.6
        }
        
        analysis = create_analysis(analysis_data)
        
        assert analysis.id is not None, "Analysis ID should be auto-generated"
        assert analysis.pdf_id == pdf.id, "PDF ID should match"
        assert analysis.form_number == 'FORM-2024-001', "Form number should match"
        assert analysis.data_usage_mentioned == True, "Data usage flag should match"
        
        print(f"✅ Analysis created with ID: {analysis.id}")
        return analysis, pdf
    
    def test_get_latest_analysis(self):
        """Test retrieving latest analysis"""
        print("\n🔍 Testing latest analysis retrieval...")
        
        # Create multiple analyses
        analysis1, pdf = self.test_create_analysis()
        
        # Create second analysis (newer)
        analysis_data = {
            'pdf_id': pdf.id,
            'version': 'v2.0',
            'summary': 'Updated analysis with improved detection'
        }
        analysis2 = create_analysis(analysis_data)
        
        # Get latest analysis
        latest = get_latest_analysis(pdf.id)
        
        assert latest is not None, "Latest analysis should be found"
        assert latest.id == analysis2.id, "Should return the most recent analysis"
        assert latest.version == 'v2.0', "Should return the newer version"
        
        print(f"✅ Latest analysis retrieved: {latest.version}")
        return latest

class TestClauseOperations:
    """Test clause-related database operations"""
    
    def test_create_clauses(self):
        """Test clause creation"""
        print("\n📝 Testing clause creation...")
        
        # Create analysis first
        analysis, pdf = TestAnalysisOperations().test_create_analysis()
        
        clause_list = [
            {
                'analysis_id': analysis.id,
                'clause_type': 'PI',
                'clause_text': 'Personal information shall be handled in accordance with applicable privacy laws.',
                'page_number': 2,
                'paragraph_index': 3,
                'clause_order': 1
            },
            {
                'analysis_id': analysis.id,
                'clause_type': 'CI',
                'clause_text': 'Confidential information must not be disclosed to third parties.',
                'page_number': 3,
                'paragraph_index': 1,
                'clause_order': 2
            },
            {
                'analysis_id': analysis.id,
                'clause_type': 'DATA_USAGE',
                'clause_text': 'Data may be used for internal analytics and reporting purposes.',
                'page_number': 4,
                'paragraph_index': 2,
                'clause_order': 3
            }
        ]
        
        clauses = create_clauses(clause_list)
        
        assert len(clauses) == 3, "Should create 3 clauses"
        assert all(clause.id is not None for clause in clauses), "All clauses should have IDs"
        assert clauses[0].clause_type == 'PI', "First clause type should be PI"
        assert clauses[1].page_number == 3, "Second clause should be on page 3"
        
        print(f"✅ Created {len(clauses)} clauses successfully")
        return clauses, analysis

class TestFeedbackOperations:
    """Test feedback-related database operations"""
    
    def test_create_feedback(self):
        """Test feedback creation"""
        print("\n💬 Testing feedback creation...")
        
        # Create PDF and user first
        pdf, user = TestPDFOperations().test_create_pdf()
        
        feedback_data = {
            'pdf_id': pdf.id,
            'user_id': user.id,
            'feedback_date': datetime.utcnow(),
            'general_feedback': 'The analysis was very accurate and helpful. The PI clauses were correctly identified.'
        }
        
        feedback = create_feedback(feedback_data)
        
        assert feedback.id is not None, "Feedback ID should be auto-generated"
        assert feedback.pdf_id == pdf.id, "PDF ID should match"
        assert feedback.user_id == user.id, "User ID should match"
        assert "accurate and helpful" in feedback.general_feedback, "Feedback text should match"
        
        print(f"✅ Feedback created with ID: {feedback.id}")
        return feedback, pdf, user

class TestRelationships:
    """Test foreign key relationships and cascading operations"""
    
    def test_user_pdf_relationship(self):
        """Test one-to-many relationship between users and PDFs"""
        print("\n🔗 Testing user-PDF relationships...")
        
        # Create user and multiple PDFs
        user = create_user(session_id=generate_session_id(), username="relationship_test_user")
        
        pdf_data_list = [
            {
                'pdf_name': 'contract1.pdf',
                'file_hash': generate_file_hash(b"content1"),
                'uploaded_by': user.id
            },
            {
                'pdf_name': 'contract2.pdf',
                'file_hash': generate_file_hash(b"content2"),
                'uploaded_by': user.id
            },
            {
                'pdf_name': 'contract3.pdf',
                'file_hash': generate_file_hash(b"content3"),
                'uploaded_by': user.id
            }
        ]
        
        pdfs = [create_pdf(pdf_data) for pdf_data in pdf_data_list]
        
        # Test relationship query
        user_pdfs = get_user_pdfs(user.id)
        
        assert len(user_pdfs) == 3, "User should have 3 PDFs"
        assert all(pdf.uploaded_by == user.id for pdf in user_pdfs), "All PDFs should belong to user"
        
        # Test relationship through SQLAlchemy
        with db_manager.get_session() as session:
            user_obj = session.query(User).filter(User.id == user.id).first()
            assert len(user_obj.uploaded_pdfs) == 3, "User should have 3 uploaded PDFs"
        
        print(f"✅ User-PDF relationship working: {len(user_pdfs)} PDFs linked to user")
        return user, pdfs
    
    def test_pdf_analysis_relationship(self):
        """Test one-to-many relationship between PDFs and analyses"""
        print("\n🔗 Testing PDF-analysis relationships...")
        
        # Create PDF
        pdf, user = TestPDFOperations().test_create_pdf()
        
        # Create multiple analyses for the same PDF
        analysis_data_list = [
            {
                'pdf_id': pdf.id,
                'version': 'v1.0',
                'summary': 'Initial analysis'
            },
            {
                'pdf_id': pdf.id,
                'version': 'v1.1',
                'summary': 'Updated analysis with bug fixes'
            },
            {
                'pdf_id': pdf.id,
                'version': 'v2.0',
                'summary': 'Major revision with new detection methods'
            }
        ]
        
        analyses = [create_analysis(data) for data in analysis_data_list]
        
        # Test relationship
        pdf_with_analyses = get_pdf_with_analyses(pdf.id)
        
        assert pdf_with_analyses is not None, "PDF should be found"
        
        # Test relationship through SQLAlchemy
        with db_manager.get_session() as session:
            pdf_obj = session.query(PDF).filter(PDF.id == pdf.id).first()
            assert len(pdf_obj.analyses) == 3, "PDF should have 3 analyses"
            
            # Check versions are correct
            versions = [analysis.version for analysis in pdf_obj.analyses]
            assert 'v1.0' in versions, "Should contain v1.0"
            assert 'v2.0' in versions, "Should contain v2.0"
        
        print(f"✅ PDF-analysis relationship working: {len(analyses)} analyses linked to PDF")
        return pdf, analyses
    
    def test_analysis_clause_relationship(self):
        """Test one-to-many relationship between analyses and clauses"""
        print("\n🔗 Testing analysis-clause relationships...")
        
        # Create analysis
        analysis, pdf = TestAnalysisOperations().test_create_analysis()
        
        # Create clauses
        clause_list = [
            {
                'analysis_id': analysis.id,
                'clause_type': 'PI',
                'clause_text': 'PI clause 1',
                'clause_order': 1
            },
            {
                'analysis_id': analysis.id,
                'clause_type': 'CI',
                'clause_text': 'CI clause 1',
                'clause_order': 2
            },
            {
                'analysis_id': analysis.id,
                'clause_type': 'DATA_RETENTION',
                'clause_text': 'Data retention clause',
                'clause_order': 3
            }
        ]
        
        clauses = create_clauses(clause_list)
        
        # Test relationship
        analysis_with_clauses = get_analysis_with_clauses(analysis.id)
        
        assert analysis_with_clauses is not None, "Analysis should be found"
        
        # Test relationship through SQLAlchemy
        with db_manager.get_session() as session:
            analysis_obj = session.query(Analysis).filter(Analysis.id == analysis.id).first()
            assert len(analysis_obj.clauses) == 3, "Analysis should have 3 clauses"
            
            # Check clause types
            clause_types = [clause.clause_type for clause in analysis_obj.clauses]
            assert 'PI' in clause_types, "Should contain PI clause"
            assert 'CI' in clause_types, "Should contain CI clause"
            assert 'DATA_RETENTION' in clause_types, "Should contain DATA_RETENTION clause"
        
        print(f"✅ Analysis-clause relationship working: {len(clauses)} clauses linked to analysis")
        return analysis, clauses
    
    def test_cascading_deletes(self):
        """Test that foreign key constraints work with cascading deletes"""
        print("\n🗑️ Testing cascading deletes...")
        
        # Create complete data hierarchy
        user = create_user(session_id=generate_session_id(), username="cascade_test_user")
        
        pdf_data = {
            'pdf_name': 'cascade_test.pdf',
            'file_hash': generate_file_hash(b"cascade test content"),
            'uploaded_by': user.id
        }
        pdf = create_pdf(pdf_data)
        
        analysis_data = {
            'pdf_id': pdf.id,
            'version': 'v1.0',
            'summary': 'Test analysis for cascade'
        }
        analysis = create_analysis(analysis_data)
        
        clause_list = [
            {
                'analysis_id': analysis.id,
                'clause_type': 'TEST',
                'clause_text': 'Test clause for cascade',
                'clause_order': 1
            }
        ]
        clauses = create_clauses(clause_list)
        
        feedback_data = {
            'pdf_id': pdf.id,
            'user_id': user.id,
            'general_feedback': 'Test feedback for cascade'
        }
        feedback = create_feedback(feedback_data)
        
        # Verify everything exists
        with db_manager.get_session() as session:
            assert session.query(PDF).filter(PDF.id == pdf.id).first() is not None
            assert session.query(Analysis).filter(Analysis.id == analysis.id).first() is not None
            assert session.query(Clause).filter(Clause.id == clauses[0].id).first() is not None
            assert session.query(Feedback).filter(Feedback.id == feedback.id).first() is not None
        
        # Delete PDF - should cascade to analyses, clauses, and feedback
        success = delete_pdf(pdf.id)
        assert success, "PDF deletion should succeed"
        
        # Verify cascading worked
        with db_manager.get_session() as session:
            assert session.query(PDF).filter(PDF.id == pdf.id).first() is None
            assert session.query(Analysis).filter(Analysis.id == analysis.id).first() is None
            assert session.query(Clause).filter(Clause.id == clauses[0].id).first() is None
            assert session.query(Feedback).filter(Feedback.id == feedback.id).first() is None
            
            # User should still exist
            assert session.query(User).filter(User.id == user.id).first() is not None
        
        print("✅ Cascading deletes working correctly")

class TestDataManipulation:
    """Test data insertion, updating, and deletion operations"""
    
    def test_update_pdf(self):
        """Test PDF update operations"""
        print("\n✏️ Testing PDF updates...")
        
        # Create PDF
        pdf, user = TestPDFOperations().test_create_pdf()
        
        # Update PDF data
        update_data = {
            'processed_date': datetime.utcnow(),
            'final_word_count': 1100,
            'final_page_count': 3,
            'obfuscation_summary': {
                "method": "advanced_entity_replacement",
                "confidence": 0.92,
                "updated": True
            }
        }
        
        updated_pdf = update_pdf(pdf.id, update_data)
        
        assert updated_pdf is not None, "Updated PDF should be returned"
        assert updated_pdf.processed_date is not None, "Processed date should be set"
        assert updated_pdf.final_word_count == 1100, "Word count should be updated"
        assert updated_pdf.obfuscation_summary['updated'] == True, "JSON field should be updated"
        
        print(f"✅ PDF updated successfully: {updated_pdf.final_word_count} words")
        return updated_pdf
    
    def test_bulk_operations(self):
        """Test bulk data operations"""
        print("\n📦 Testing bulk operations...")
        
        # Create user
        user = create_user(session_id=generate_session_id(), username="bulk_test_user")
        
        # Bulk create PDFs
        pdf_data_list = []
        for i in range(5):
            pdf_data_list.append({
                'pdf_name': f'bulk_contract_{i+1}.pdf',
                'file_hash': generate_file_hash(f"bulk content {i+1}".encode()),
                'uploaded_by': user.id,
                'original_word_count': 1000 + (i * 100),
                'original_page_count': 3 + i
            })
        
        pdfs = []
        for pdf_data in pdf_data_list:
            pdfs.append(create_pdf(pdf_data))
        
        # Verify bulk creation
        user_pdfs = get_user_pdfs(user.id)
        assert len(user_pdfs) == 5, "Should have 5 PDFs"
        
        # Test search functionality
        search_results = search_pdfs_by_name("bulk_contract")
        assert len(search_results) >= 5, "Search should find bulk contracts"
        
        print(f"✅ Bulk operations completed: {len(pdfs)} PDFs created")
        return pdfs, user
    
    def test_complex_queries(self):
        """Test complex database queries"""
        print("\n🔍 Testing complex queries...")
        
        # Create test data
        user = create_user(session_id=generate_session_id(), username="complex_query_user")
        
        # Create PDFs with analyses
        pdf_data = {
            'pdf_name': 'complex_test.pdf',
            'file_hash': generate_file_hash(b"complex test content"),
            'uploaded_by': user.id
        }
        pdf = create_pdf(pdf_data)
        
        analysis_data = {
            'pdf_id': pdf.id,
            'version': 'v1.0',
            'data_usage_mentioned': True,
            'data_limitations_exists': False
        }
        analysis = create_analysis(analysis_data)
        
        # Complex query: Get all PDFs by user with their latest analysis
        with db_manager.get_session() as session:
            from sqlalchemy.orm import joinedload
            
            result = session.query(PDF)\
                           .options(joinedload(PDF.analyses))\
                           .filter(PDF.uploaded_by == user.id)\
                           .first()
            
            assert result is not None, "Complex query should return result"
            assert len(result.analyses) >= 1, "PDF should have analyses loaded"
            assert result.analyses[0].data_usage_mentioned == True, "Analysis data should be correct"
        
        print("✅ Complex queries working correctly")

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_foreign_key_constraints(self):
        """Test foreign key constraint violations"""
        print("\n🚫 Testing foreign key constraints...")
        
        # Try to create PDF with non-existent user
        pdf_data = {
            'pdf_name': 'orphan.pdf',
            'file_hash': generate_file_hash(b"orphan content"),
            'uploaded_by': 99999  # Non-existent user ID
        }
        
        try:
            create_pdf(pdf_data)
            pytest.fail("Should not allow PDF with non-existent user")
        except Exception as e:
            print(f"✅ Foreign key constraint working: {type(e).__name__}")
        
        # Try to create analysis with non-existent PDF
        analysis_data = {
            'pdf_id': 99999,  # Non-existent PDF ID
            'version': 'v1.0'
        }
        
        try:
            create_analysis(analysis_data)
            pytest.fail("Should not allow analysis with non-existent PDF")
        except Exception as e:
            print(f"✅ Foreign key constraint working: {type(e).__name__}")
    
    def test_unique_constraints(self):
        """Test unique constraint violations"""
        print("\n🔒 Testing unique constraints...")
        
        # Test duplicate session ID
        session_id = generate_session_id()
        user1 = create_user(session_id=session_id, username="user1")
        
        try:
            user2 = create_user(session_id=session_id, username="user2")
            pytest.fail("Should not allow duplicate session ID")
        except Exception as e:
            print(f"✅ Unique constraint working for session_id: {type(e).__name__}")
        
        # Test duplicate file hash (already tested in PDF operations)
        print("✅ File hash uniqueness already verified")
    
    def test_null_constraints(self):
        """Test null constraint violations"""
        print("\n❌ Testing null constraints...")
        
        # Try to create user without session_id
        try:
            with db_manager.get_session() as session:
                user = User(username="no_session_user")
                session.add(user)
                session.flush()
            pytest.fail("Should not allow user without session_id")
        except Exception as e:
            print(f"✅ Null constraint working for session_id: {type(e).__name__}")

def run_all_tests():
    """Run all database tests"""
    print("🧪 Starting comprehensive database tests...\n")
    
    try:
        # Initialize database
        print("🏗️ Setting up test database...")
        initialize_database()
        
        # Clean slate for testing
        db_manager.drop_tables()
        db_manager.create_tables()
        
        test_classes = [
            TestDatabaseSetup(),
            TestUserOperations(),
            TestPDFOperations(),
            TestAnalysisOperations(),
            TestClauseOperations(),
            TestFeedbackOperations(),
            TestRelationships(),
            TestDataManipulation(),
            TestErrorHandling()
        ]
        
        # Run all tests
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n{'='*50}")
            print(f"Running {class_name}")
            print(f"{'='*50}")
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    method = getattr(test_class, method_name)
                    method()
                    passed_tests += 1
                except Exception as e:
                    print(f"❌ {method_name} FAILED: {str(e)}")
        
        # Summary
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️ Some tests failed. Check logs above.")
            
    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        raise

def test_performance():
    """Test database performance with larger datasets"""
    print("\n⚡ Testing database performance...")
    
    import time
    
    # Create user
    user = create_user(session_id=generate_session_id(), username="perf_test_user")
    
    # Test bulk PDF creation performance
    start_time = time.time()
    
    for i in range(100):
        pdf_data = {
            'pdf_name': f'performance_test_{i}.pdf',
            'file_hash': generate_file_hash(f"performance content {i}".encode()),
            'uploaded_by': user.id,
            'original_word_count': 1000 + i,
            'original_page_count': 5
        }
        create_pdf(pdf_data)
    
    creation_time = time.time() - start_time
    
    # Test bulk query performance
    start_time = time.time()
    user_pdfs = get_user_pdfs(user.id, limit=100)
    query_time = time.time() - start_time
    
    print(f"✅ Performance test completed:")
    print(f"   - Created 100 PDFs in {creation_time:.2f} seconds")
    print(f"   - Queried 100 PDFs in {query_time:.4f} seconds")
    print(f"   - Average creation time: {(creation_time/100)*1000:.2f} ms per PDF")

if __name__ == "__main__":
    # Set test environment variables if not set
    if not os.getenv('DB_NAME'):
        os.environ['DB_NAME'] = 'contract_analysis_test'
        os.environ['DB_USERNAME'] = 'postgres'
        os.environ['DB_PASSWORD'] = 'password'
        os.environ['DB_HOST'] = 'localhost'
        os.environ['DB_PORT'] = '5432'
    
    try:
        run_all_tests()
        test_performance()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        raise
