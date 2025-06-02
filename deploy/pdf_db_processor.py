# Integration examples for PDF handler and Streamlit

# ===========================
# PDF HANDLER INTEGRATION
# ===========================

# pdf_handler.py - Example integration
import time
from datetime import datetime
from database.helpers import (
    store_pdf_from_upload, update_pdf_processing_data, 
    store_analysis_results, store_extracted_clauses,
    get_pdf_full_details
)

class PDFProcessor:
    """PDF processing with database integration"""
    
    def __init__(self):
        self.processing_start_time = None
    
    def process_uploaded_pdf(self, file_data: bytes, filename: str, user_session_id: str):
        """Complete PDF processing pipeline with database storage"""
        
        # Step 1: Store initial PDF upload
        print(f"📄 Processing uploaded PDF: {filename}")
        
        # Extract basic metadata (your existing code)
        pdf_metadata = self.extract_pdf_metadata(file_data)
        
        # Store in database with safe fallbacks
        pdf_record = store_pdf_from_upload(
            file_data=file_data,
            filename=filename,
            user_session_id=user_session_id,
            pdf_metadata=pdf_metadata
        )
        
        if pdf_record.get('is_duplicate'):
            print(f"📋 PDF is duplicate: {pdf_record['pdf_name']}")
            return pdf_record
        
        if pdf_record.get('error'):
            print(f"❌ Failed to store PDF: {pdf_record['error']}")
            return pdf_record
        
        pdf_id = pdf_record['id']
        
        # Step 2: Process PDF content
        try:
            self.processing_start_time = time.time()
            
            # Your existing PDF processing
            raw_content = self.extract_text_content(file_data)
            processed_content = self.apply_obfuscation(raw_content)
            
            # Calculate final metrics
            final_metrics = self.calculate_final_metrics(processed_content)
            
            # Update database with processing results
            processing_data = {
                'processed_date': datetime.utcnow(),
                'raw_content': raw_content,
                'final_content': processed_content['text'],
                'final_word_count': final_metrics['word_count'],
                'final_page_count': final_metrics['page_count'],
                'obfuscation_applied': True,
                'pages_removed_count': processed_content['pages_removed'],
                'paragraphs_obfuscated_count': processed_content['paragraphs_obfuscated'],
                'obfuscation_summary': processed_content['obfuscation_summary']
            }
            
            success = update_pdf_processing_data(pdf_id, processing_data)
            if not success:
                print(f"⚠️ Warning: Could not update processing data for PDF {pdf_id}")
            
            # Step 3: Perform contract analysis
            analysis_results = self.analyze_contract(processed_content['text'])
            
            # Store analysis results
            analysis_record = store_analysis_results(
                pdf_id=pdf_id,
                analysis_results=analysis_results,
                version="v1.0"
            )
            
            if analysis_record.get('error'):
                print(f"⚠️ Warning: Analysis storage failed: {analysis_record['error']}")
            else:
                analysis_id = analysis_record['id']
                
                # Step 4: Store extracted clauses
                if 'clauses' in analysis_results:
                    clause_records = store_extracted_clauses(
                        analysis_id=analysis_id,
                        clauses=analysis_results['clauses']
                    )
                    
                    successful_clauses = [c for c in clause_records if c.get('status') == 'stored']
                    print(f"📝 Stored {len(successful_clauses)} clauses successfully")
            
            # Return complete processing result
            processing_time = time.time() - self.processing_start_time
            
            return {
                'pdf_id': pdf_id,
                'status': 'completed',
                'processing_time': processing_time,
                'pdf_record': pdf_record,
                'analysis_record': analysis_record,
                'final_metrics': final_metrics,
                'message': f"Successfully processed {filename}"
            }
            
        except Exception as e:
            print(f"❌ Error during PDF processing: {e}")
            return {
                'pdf_id': pdf_id,
                'status': 'failed',
                'error': str(e),
                'pdf_record': pdf_record
            }
    
    def extract_pdf_metadata(self, file_data: bytes) -> dict:
        """Extract basic PDF metadata (your existing logic)"""
        try:
            # Your existing metadata extraction code
            return {
                'layout': 'single-column',  # Detected layout
                'word_count': 1500,         # Initial word count
                'page_count': 5,            # Page count
                'parsability': 0.95         # How well the PDF can be parsed
            }
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}")
            return {}
    
    def extract_text_content(self, file_data: bytes) -> dict:
        """Extract text content from PDF (your existing logic)"""
        try:
            # Your existing text extraction code
            return {
                'pages': [
                    {'page_num': 1, 'text': 'Page 1 content...'},
                    {'page_num': 2, 'text': 'Page 2 content...'}
                ],
                'full_text': 'Combined text from all pages...'
            }
        except Exception as e:
            print(f"⚠️ Text extraction failed: {e}")
            return {'pages': [], 'full_text': ''}
    
    def apply_obfuscation(self, raw_content: dict) -> dict:
        """Apply obfuscation to content (your existing logic)"""
        try:
            # Your existing obfuscation logic
            return {
                'text': 'Obfuscated content...',
                'pages_removed': 1,
                'paragraphs_obfuscated': 5,
                'obfuscation_summary': {
                    'method': 'entity_replacement',
                    'confidence': 0.87,
                    'entities_found': ['PII', 'emails', 'phone_numbers']
                }
            }
        except Exception as e:
            print(f"⚠️ Obfuscation failed: {e}")
            return {
                'text': raw_content.get('full_text', ''),
                'pages_removed': 0,
                'paragraphs_obfuscated': 0,
                'obfuscation_summary': {'error': str(e)}
            }
    
    def calculate_final_metrics(self, processed_content: dict) -> dict:
        """Calculate final metrics after processing"""
        try:
            text = processed_content.get('text', '')
            words = len(text.split()) if text else 0
            
            # Estimate pages (assuming ~250 words per page)
            estimated_pages = max(1, words // 250)
            
            return {
                'word_count': words,
                'page_count': estimated_pages,
                'avg_words_per_page': words / estimated_pages if estimated_pages > 0 else 0
            }
        except Exception as e:
            print(f"⚠️ Metrics calculation failed: {e}")
            return {'word_count': 0, 'page_count': 0, 'avg_words_per_page': 0}
    
    def analyze_contract(self, text: str) -> dict:
        """Perform contract analysis (your existing AI logic)"""
        try:
            # Your existing contract analysis code
            analysis_start = time.time()
            
            # Mock analysis results - replace with your actual analysis
            analysis_results = {
                'form_number': 'FORM-2024-001',
                'pi_clause': 'Personal information clause found in section 3.2',
                'ci_clause': 'Confidential information clause found in section 4.1',
                'data_usage_mentioned': True,
                'data_limitations_exists': True,
                'summary': 'Contract contains standard privacy and confidentiality clauses with data usage provisions.',
                'processed_by': 'AI_Engine_v2.1',
                'processing_time': time.time() - analysis_start,
                'raw_analysis': {
                    'confidence_scores': {'pi_detection': 0.95, 'ci_detection': 0.87},
                    'sections_analyzed': 12,
                    'flags_raised': ['data_usage', 'third_party_sharing']
                },
                'clauses': [
                    {
                        'type': 'PI',
                        'text': 'Personal information shall be handled in accordance with applicable privacy laws.',
                        'page_number': 2,
                        'paragraph_index': 3,
                        'order': 1
                    },
                    {
                        'type': 'CI',
                        'text': 'Confidential information must not be disclosed to third parties.',
                        'page_number': 3,
                        'paragraph_index': 1,
                        'order': 2
                    },
                    {
                        'type': 'DATA_USAGE',
                        'text': 'Data may be used for internal analytics and reporting purposes.',
                        'page_number': 4,
                        'paragraph_index': 2,
                        'order': 3
                    }
                ]
            }
            
            return analysis_results
            
        except Exception as e:
            print(f"⚠️ Contract analysis failed: {e}")
            return {
                'summary': f'Analysis failed: {str(e)}',
                'processed_by': 'AI_Engine_v2.1',
                'processing_time': 0,
                'error': str(e)
            }

# ===========================
# STREAMLIT INTEGRATION
# ===========================

# streamlit_app.py - Example integration
import streamlit as st
from database.helpers import (
    get_or_create_user_by_session, get_user_dashboard_data,
    get_pdf_full_details, store_user_feedback, check_database_health,
    update_user_activity
)

class StreamlitApp:
    """Streamlit app with database integration"""
    
    def __init__(self):
        self.setup_session_state()
        self.setup_user()
    
    def setup_session_state(self):
        """Initialize Streamlit session state"""
        if 'user_session_id' not in st.session_state:
            from database.models import generate_session_id
            st.session_state.user_session_id = generate_session_id()
        
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
    
    def setup_user(self):
        """Setup user for the session"""
        try:
            # Get or create user
            user_data = get_or_create_user_by_session(
                session_id=st.session_state.user_session_id,
                username=f"user_{st.session_state.user_session_id[:8]}"
            )
            
            st.session_state.user_data = user_data
            
            # Update user activity
            update_user_activity(st.session_state.user_session_id)
            
        except Exception as e:
            st.error(f"Error setting up user: {e}")
            st.session_state.user_data = {
                'username': 'Guest User',
                'error': str(e)
            }
    
    def show_dashboard(self):
        """Display user dashboard"""
        st.title("📊 Contract Analysis Dashboard")
        
        # Database health check
        with st.expander("🔍 System Status", expanded=False):
            health = check_database_health()
            if health.get('connection') == 'healthy':
                st.success("✅ Database connection healthy")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Users", health.get('total_users', 0))
                with col2:
                    st.metric("Total PDFs", health.get('total_pdfs', 0))
                with col3:
                    st.metric("Total Analyses", health.get('total_analyses', 0))
            else:
                st.error(f"❌ Database issue: {health.get('error', 'Unknown error')}")
        
        # User info
        user = st.session_state.user_data
        if user and not user.get('error'):
            st.write(f"👋 Welcome back, **{user['username']}**!")
            
            if user.get('is_new'):
                st.info("🎉 This is your first visit! Upload a contract to get started.")
        
        # Get dashboard data
        try:
            dashboard_data = get_user_dashboard_data(st.session_state.user_session_id)
            
            # Display stats
            st.subheader("📈 Your Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Documents Uploaded", 
                    dashboard_data['stats']['total_pdfs']
                )
            
            with col2:
                st.metric(
                    "Documents Processed", 
                    dashboard_data['stats']['processed_pdfs']
                )
            
            with col3:
                st.metric(
                    "Analyses Completed", 
                    dashboard_data['stats']['total_analyses']
                )
            
            with col4:
                st.metric(
                    "Feedback Given", 
                    dashboard_data['stats']['total_feedbacks']
                )
            
            # Recent PDFs
            if dashboard_data['recent_pdfs']:
                st.subheader("📄 Recent Documents")
                
                for pdf in dashboard_data['recent_pdfs']:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{pdf['pdf_name']}**")
                            st.write(f"Uploaded: {pdf['upload_date'][:10]}")
                        
                        with col2:
                            status_color = "🟢" if pdf['status'] == 'processed' else "🟡"
                            st.write(f"{status_color} {pdf['status'].title()}")
                        
                        with col3:
                            if pdf['word_count']:
                                st.write(f"📝 {pdf['word_count']:,} words")
                            if pdf['page_count']:
                                st.write(f"📄 {pdf['page_count']} pages")
                        
                        with col4:
                            if st.button(f"View Details", key=f"view_{pdf['id']}"):
                                st.session_state.selected_pdf_id = pdf['id']
                                st.experimental_rerun()
                        
                        st.divider()
            else:
                st.info("📭 No documents uploaded yet. Use the upload section below to get started!")
        
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
    
    def show_pdf_upload(self):
        """PDF upload interface"""
        st.subheader("📤 Upload Contract Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your contract document for analysis"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing your document..."):
                    try:
                        # Read file data
                        file_data = uploaded_file.read()
                        
                        # Process with PDFProcessor
                        processor = PDFProcessor()
                        result = processor.process_uploaded_pdf(
                            file_data=file_data,
                            filename=uploaded_file.name,
                            user_session_id=st.session_state.user_session_id
                        )
                        
                        # Show results
                        if result.get('status') == 'completed':
                            st.success(f"✅ {result['message']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                            with col2:
                                st.metric("Final Word Count", result['final_metrics']['word_count'])
                            
                            # Store processing result in session for display
                            st.session_state.last_processed_pdf = result['pdf_id']
                            
                            st.balloons()
                            
                        elif result.get('is_duplicate'):
                            st.warning(f"📋 This document was already uploaded: {result['pdf_name']}")
                            st.session_state.last_processed_pdf = result['id']
                            
                        else:
                            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Upload failed: {str(e)}")
    
    def show_pdf_details(self, pdf_id: int):
        """Show detailed PDF information"""
        try:
            pdf_details = get_pdf_full_details(pdf_id)
            
            if pdf_details.get('error'):
                st.error(f"Error loading PDF details: {pdf_details['error']}")
                return
            
            st.title(f"📄 {pdf_details['pdf_name']}")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", pdf_details['status'].title())
            with col2:
                if pdf_details['final_word_count']:
                    st.metric("Words", f"{pdf_details['final_word_count']:,}")
            with col3:
                if pdf_details['final_page_count']:
                    st.metric("Pages", pdf_details['final_page_count'])
            
            # Processing info
            if pdf_details['obfuscation_applied']:
                st.subheader("🔒 Privacy Protection Applied")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pages Removed", pdf_details['pages_removed_count'])
                with col2:
                    st.metric("Paragraphs Obfuscated", pdf_details['paragraphs_obfuscated_count'])
            
            # Analyses
            if pdf_details['analyses']:
                st.subheader("🔍 Analysis Results")
                
                for analysis in pdf_details['analyses']:
                    with st.expander(f"Analysis {analysis['version']} - {analysis['analysis_date'][:10]}", expanded=True):
                        
                        if analysis['summary']:
                            st.write("**Summary:**")
                            st.write(analysis['summary'])
                        
                        # Key findings
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Data Usage Mentioned:**", "✅ Yes" if analysis['data_usage_mentioned'] else "❌ No")
                        with col2:
                            st.write("**Data Limitations Exist:**", "✅ Yes" if analysis['data_limitations_exists'] else "❌ No")
                        
                        # Clauses
                        if analysis['clauses']:
                            st.write("**Extracted Clauses:**")
                            for clause in analysis['clauses']:
                                clause_color = {
                                    'PI': '🔒',
                                    'CI': '🤐', 
                                    'DATA_USAGE': '📊',
                                    'DATA_RETENTION': '📅'
                                }.get(clause['clause_type'], '📄')
                                
                                st.write(f"{clause_color} **{clause['clause_type']}** (Page {clause['page_number']})")
                                st.write(f"_{clause['clause_text']}_")
                                st.write("")
            
            # Feedback section
            st.subheader("💬 Your Feedback")
            
            # Show existing feedback
            if pdf_details['feedbacks']:
                st.write("**Previous Feedback:**")
                for feedback in pdf_details['feedbacks']:
                    st.write(f"_{feedback['general_feedback']}_ - {feedback['feedback_date'][:10]}")
            
            # Add new feedback
            with st.form(f"feedback_form_{pdf_id}"):
                feedback_text = st.text_area("Share your thoughts about this analysis:")
                rating = st.select_slider("Rate the analysis quality:", options=[1, 2, 3, 4, 5], value=3)
                
                if st.form_submit_button("Submit Feedback"):
                    if feedback_text.strip():
                        feedback_result = store_user_feedback(
                            pdf_id=pdf_id,
                            user_session_id=st.session_state.user_session_id,
                            feedback_text=feedback_text,
                            rating=rating
                        )
                        
                        if feedback_result.get('status') == 'stored':
                            st.success("✅ Thank you for your feedback!")
                            st.experimental_rerun()
                        else:
                            st.error(f"❌ Failed to save feedback: {feedback_result.get('error')}")
                    else:
                        st.warning("Please enter some feedback text.")
        
        except Exception as e:
            st.error(f"Error displaying PDF details: {e}")
    
    def run(self):
        """Main Streamlit app runner"""
        st.set_page_config(
            page_title="Contract Analysis Platform",
            page_icon="📄",
            layout="wide"
        )
        
        # Check if viewing specific PDF
        if hasattr(st.session_state, 'selected_pdf_id'):
            self.show_pdf_details(st.session_state.selected_pdf_id)
            
            if st.button("← Back to Dashboard"):
                del st.session_state.selected_pdf_id
                st.experimental_rerun()
        
        # Check if showing recently processed PDF
        elif hasattr(st.session_state, 'last_processed_pdf'):
            self.show_pdf_details(st.session_state.last_processed_pdf)
            
            if st.button("← Back to Dashboard"):
                del st.session_state.last_processed_pdf
                st.experimental_rerun()
        
        # Default dashboard view
        else:
            self.show_dashboard()
            st.divider()
            self.show_pdf_upload()

# ===========================
# USAGE EXAMPLES
# ===========================

if __name__ == "__main__":
    # Example 1: Direct PDF processing
    print("🧪 Testing PDF processing...")
    
    # Simulate file upload
    sample_pdf_data = b"Sample PDF content for testing"
    processor = PDFProcessor()
    
    result = processor.process_uploaded_pdf(
        file_data=sample_pdf_data,
        filename="test_contract.pdf",
        user_session_id="test_session_123"
    )
    
    print(f"Processing result: {result}")
    
    # Example 2: Streamlit app
    # Run with: streamlit run streamlit_app.py
    app = StreamlitApp()
    app.run()
