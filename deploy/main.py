# main.py - Enhanced main application with database integration

import streamlit as st
import os
from datetime import datetime
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Database imports
from config.database import get_session, initialize_database
from models.database_models import PDF, Analysis, User

# Service imports
from services.user_service import UserSessionService
from services.pdf_service import EnhancedPDFService
from services.feedback_service import FeedbackService
from services.batch_service import BatchProcessingService

# UI imports
from ui.analysis_display import render_enhanced_analysis_section
from ui.feedback_form import render_feedback_form
from ui.batch_interface import (
    render_batch_upload_interface, 
    render_batch_history
)

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Contract Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .pdf-viewer {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        height: 85vh;
        overflow-y: auto;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize application with database and session management"""
    
    # Initialize database
    initialize_database()
    
    # Get database session
    if 'db_session' not in st.session_state:
        st.session_state.db_session = get_session()
    
    # Initialize user session
    user_service = UserSessionService(st.session_state.db_session)
    session_id = user_service.get_or_create_session()
    
    # Initialize session state variables
    session_vars = [
        'pdf_files', 'json_data', 'current_pdf', 'search_text',
        'analysis_status', 'selected_pdf', 'highlighted_text',
        'current_batch_job'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var in ['pdf_files', 'json_data', 'analysis_status']:
                st.session_state[var] = {}
            else:
                st.session_state[var] = None

def render_sidebar():
    """Render enhanced sidebar with navigation and controls"""
    
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        
        # Main navigation
        page = st.radio(
            "Select Page",
            ["📄 Document Analysis", "📦 Batch Processing", "📊 Analytics Dashboard"],
            key="main_navigation"
        )
        
        st.divider()
        
        # User session info
        st.markdown("## 👤 Session Info")
        st.info(f"Session ID: {st.session_state.user_session_id[:8]}...")
        
        # Database stats
        db_session = st.session_state.db_session
        total_pdfs = db_session.query(PDF).count()
        total_analyses = db_session.query(Analysis).count()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Total PDFs", total_pdfs)
        with col2:
            st.metric("🔍 Total Analyses", total_analyses)
        
        # Session cleanup button (admin feature)
        st.divider()
        if st.button("🧹 Cleanup Old Sessions", help="Remove inactive sessions older than 24 hours"):
            user_service = UserSessionService(db_session)
            cleaned = user_service.cleanup_inactive_sessions()
            st.success(f"Cleaned up {cleaned} inactive sessions")
        
        return page

def render_document_analysis_page():
    """Render main document analysis page"""
    
    st.markdown('<div class="main-header"><h1>📄 Contract Analysis Platform</h1></div>', 
                unsafe_allow_html=True)
    
    # Create main layout
    col1, col2, col3 = st.columns([25, 40, 35])
    
    # Left column: Document management
    with col1:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📁 Document Management")
        
        # PDF upload section
        st.subheader("⬆️ Upload Documents")
        uploaded_pdfs = st.file_uploader(
            "Upload Contract PDFs",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        # Process uploaded PDFs
        if uploaded_pdfs:
            process_uploaded_pdfs(uploaded_pdfs)
        
        st.divider()
        
        # Available PDFs from database
        render_available_pdfs_grid()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle column: PDF viewer
    with col2:
        st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
        render_pdf_viewer()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column: Analysis results
    with col3:
        render_enhanced_analysis_section(st.session_state.db_session)

def process_uploaded_pdfs(uploaded_pdfs):
    """Process uploaded PDF files"""
    
    pdf_service = EnhancedPDFService(st.session_state.db_session)
    
    for pdf in uploaded_pdfs:
        pdf_name = pdf.name
        
        # Check if already in session
        if pdf_name not in st.session_state.pdf_files:
            pdf_bytes = pdf.getvalue()
            
            # Validate PDF size
            if len(pdf_bytes) > 10 * 1024 * 1024:  # 10MB limit
                st.error(f"❌ {pdf_name} is too large (max 10MB)")
                continue
            
            # Store in session state
            st.session_state.pdf_files[pdf_name] = pdf_bytes
            
            # Process PDF
            with st.spinner(f"Processing {pdf_name}..."):
                success, result_data, pdf_id = pdf_service.process_pdf_pipeline(
                    pdf_bytes, pdf_name, st.session_state.user_session_id
                )
                
                if success:
                    st.session_state.analysis_status[pdf_name] = "✅ Processed"
                    st.success(f"✅ {pdf_name} processed successfully!")
                    
                    # Set as current PDF if none selected
                    if not st.session_state.current_pdf:
                        st.session_state.current_pdf = pdf_name
                        
                else:
                    st.session_state.analysis_status[pdf_name] = f"❌ Failed: {result_data}"
                    st.error(f"❌ Failed to process {pdf_name}")

def render_available_pdfs_grid():
    """Render available PDFs in an interactive grid"""
    
    st.subheader("📋 Available Documents")
    
    db_session = st.session_state.db_session
    pdfs = db_session.query(PDF).order_by(PDF.upload_date.desc()).all()
    
    if not pdfs:
        st.info("No documents available. Upload some PDFs to get started!")
        return
    
    # Create DataFrame for AgGrid
    pdf_data = []
    for pdf in pdfs:
        # Get latest analysis
        latest_analysis = db_session.query(Analysis).filter_by(
            pdf_id=pdf.id
        ).order_by(Analysis.version.desc()).first()
        
        pdf_data.append({
            "PDF Name": pdf.pdf_name,
            "Upload Date": pdf.upload_date.strftime("%Y-%m-%d %H:%M"),
            "Layout": pdf.layout or "Unknown",
            "Word Count": pdf.word_count or 0,
            "Status": "✅ Analyzed" if latest_analysis else "⏳ Pending",
            "Version": latest_analysis.version if latest_analysis else 0
        })
    
    # Configure AgGrid
    pdf_df = pd.DataFrame(pdf_data)
    gb = GridOptionsBuilder.from_dataframe(pdf_df)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    gb.configure_default_column(cellStyle={'fontSize': '14px'})
    gridOptions = gb.build()
    
    # Render grid
    grid_response = AgGrid(
        pdf_df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=400,
        fit_columns_on_grid_load=True,
        theme='streamlit',
        key='pdf_grid'
    )
    
    # Handle selection
    selected_rows = grid_response.get('selected_rows', pd.DataFrame())
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_pdf_name = selected_rows.iloc[0]['PDF Name']
        
        if selected_pdf_name != st.session_state.get('current_pdf'):
            st.session_state.current_pdf = selected_pdf_name
            st.rerun()

def render_pdf_viewer():
    """Render PDF viewer with enhanced features"""
    
    st.header("📖 PDF Viewer")
    
    if not st.session_state.current_pdf:
        st.info("👆 Select a document from the list to view it here")
        return
    
    current_pdf_name = st.session_state.current_pdf
    
    # Get PDF from session state or database
    pdf_bytes = st.session_state.pdf_files.get(current_pdf_name)
    
    if not pdf_bytes:
        # Try to get from database (for PDFs uploaded by other users)
        db_session = st.session_state.db_session
        pdf_record = db_session.query(PDF).filter_by(pdf_name=current_pdf_name).first()
        
        if pdf_record:
            st.info(f"📄 Viewing: {current_pdf_name}")
            st.markdown("*PDF content stored in database - viewer not available for shared files*")
            
            # Show PDF metadata instead
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", pdf_record.word_count or 0)
            with col2:
                st.metric("Pages", pdf_record.page_count or 0)
            with col3:
                st.metric("Layout", pdf_record.layout or "Unknown")
            
            # Show content preview
            if pdf_record.final_content:
                st.subheader("📝 Content Preview")
                preview_text = pdf_record.final_content[:1000] + "..." if len(pdf_record.final_content) > 1000 else pdf_record.final_content
                st.text_area("Content", preview_text, height=300, disabled=True)
        else:
            st.error("PDF not found in database")
        return
    
    # Display PDF using iframe
    st.subheader(f"📄 {current_pdf_name}")
    
    try:
        import base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Search functionality
        search_text = st.text_input(
            "🔍 Search in PDF", 
            value=st.session_state.search_text or "",
            key="pdf_search"
        )
        
        if search_text != st.session_state.search_text:
            st.session_state.search_text = search_text
        
        # PDF display with search
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}'
        
        if search_text:
            import urllib.parse
            encoded_search = urllib.parse.quote(search_text)
            pdf_display += f'#search={encoded_search}'
        
        pdf_display += '" width="100%" height="600px" type="application/pdf"></iframe>'
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        
        # Fallback: offer download
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=current_pdf_name,
            mime="application/pdf"
        )

def render_batch_processing_page():
    """Render batch processing page"""
    
    st.markdown('<div class="main-header"><h1>📦 Batch Processing</h1></div>', 
                unsafe_allow_html=True)
    
    # Create tabs for batch operations
    tab1, tab2 = st.tabs(["🚀 New Batch Job", "📋 Job History"])
    
    with tab1:
        render_batch_upload_interface(st.session_state.db_session)
    
    with tab2:
        render_batch_history(st.session_state.db_session)

def render_analytics_dashboard():
    """Render analytics dashboard"""
    
    st.markdown('<div class="main-header"><h1>📊 Analytics Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    db_session = st.session_state.db_session
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pdfs = db_session.query(PDF).count()
        st.metric("Total PDFs", total_pdfs)
    
    with col2:
        total_analyses = db_session.query(Analysis).count()
        st.metric("Total Analyses", total_analyses)
    
    with col3:
        successful_pdfs = db_session.query(PDF).filter_by(parsability=True).count()
        success_rate = (successful_pdfs / total_pdfs * 100) if total_pdfs > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        from models.database_models import Feedback
        total_feedback = db_session.query(Feedback).count()
        st.metric("Total Feedback", total_feedback)
    
    st.divider()
    
    # Analysis trends
    st.subheader("📊 Analysis Trends")
    
    # PI/CI Clause statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### PI Clause Distribution")
        pi_stats = db_session.query(Analysis.pi_clause, db_session.func.count(Analysis.id)).group_by(Analysis.pi_clause).all()
        
        if pi_stats:
            pi_data = {status: count for status, count in pi_stats}
            st.bar_chart(pi_data)
        else:
            st.info("No PI clause data available")
    
    with col2:
        st.markdown("#### CI Clause Distribution")
        ci_stats = db_session.query(Analysis.ci_clause, db_session.func.count(Analysis.id)).group_by(Analysis.ci_clause).all()
        
        if ci_stats:
            ci_data = {status: count for status, count in ci_stats}
            st.bar_chart(ci_data)
        else:
            st.info("No CI clause data available")
    
    st.divider()
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    
    recent_pdfs = db_session.query(PDF).order_by(PDF.upload_date.desc()).limit(10).all()
    
    if recent_pdfs:
        activity_data = []
        for pdf in recent_pdfs:
            latest_analysis = db_session.query(Analysis).filter_by(
                pdf_id=pdf.id
            ).order_by(Analysis.version.desc()).first()
            
            activity_data.append({
                "PDF Name": pdf.pdf_name,
                "Upload Date": pdf.upload_date.strftime("%Y-%m-%d %H:%M"),
                "Status": "Analyzed" if latest_analysis else "Pending",
                "Form Number": latest_analysis.form_number if latest_analysis else "N/A"
            })
        
        st.dataframe(activity_data, use_container_width=True)
    else:
        st.info("No recent activity")

def main():
    """Main application entry point"""
    
    # Initialize application
    initialize_app()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if selected_page == "📄 Document Analysis":
        render_document_analysis_page()
    elif selected_page == "📦 Batch Processing":
        render_batch_processing_page()
    elif selected_page == "📊 Analytics Dashboard":
        render_analytics_dashboard()

if __name__ == "__main__":
    main()

# config/database.py - Database configuration

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database_models import Base

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://username:password@host:5432/database_name'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_session():
    """Get database session"""
    return SessionLocal()

# utils/hash_utils.py - File hashing utilities

import hashlib

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file content for deduplication"""
    return hashlib.sha256(file_bytes).hexdigest()

def calculate_content_hash(content: str) -> str:
    """Calculate hash of text content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()




#======+============ another update

# main.py - Updated main application showing where to use the deduplication function

import streamlit as st
from services.user_service import process_pdf_with_deduplication, process_pdf_with_rerun
from ui.feedback_form import render_feedback_form, render_feedback_history
from services.feedback_service import FeedbackService

def process_uploaded_pdfs(uploaded_pdfs):
    """
    REPLACE YOUR EXISTING process_pdf FUNCTION WITH THIS
    
    This function now uses the deduplication logic and integrates with the database.
    """
    
    for pdf in uploaded_pdfs:
        pdf_name = pdf.name
        
        # Check if already in session (avoid re-processing)
        if pdf_name not in st.session_state.pdf_files:
            pdf_bytes = pdf.getvalue()
            
            # Validate PDF size (10MB limit)
            if len(pdf_bytes) > 10 * 1024 * 1024:
                st.error(f"❌ {pdf_name} is too large (max 10MB)")
                continue
            
            # Show processing status
            with st.spinner(f"Processing {pdf_name}..."):
                
                # THIS IS WHERE YOUR OLD process_pdf FUNCTION IS REPLACED
                # Use the new deduplication function
                success, result_data, pdf_id = process_pdf_with_deduplication(
                    pdf_bytes=pdf_bytes,
                    pdf_name=pdf_name, 
                    session_id=st.session_state.user_session_id,
                    db_session=st.session_state.db_session
                )
                
                if success:
                    # Success - store in session state for immediate UI use
                    st.session_state.pdf_files[pdf_name] = pdf_bytes
                    st.session_state.json_data[pdf_name] = result_data
                    st.session_state.analysis_status[pdf_name] = "✅ Processed"
                    
                    # Set as current PDF if none selected
                    if not st.session_state.current_pdf:
                        st.session_state.current_pdf = pdf_name
                    
                    # Show success message with details
                    form_number = result_data.get("form_number", "Not identified")
                    st.success(f"✅ {pdf_name} processed successfully! Form: {form_number}")
                    
                else:
                    # Failed - show error
                    error_msg = result_data.get("error", "Unknown error")
                    st.session_state.analysis_status[pdf_name] = f"❌ Failed: {error_msg}"
                    st.error(f"❌ Failed to process {pdf_name}: {error_msg}")

def render_enhanced_analysis_section_with_rerun(db_session):
    """
    ENHANCED version of your analysis display with re-run functionality
    
    This replaces your existing analysis display section.
    """
    
    st.header("📊 Contract Analysis Results")
    
    if not st.session_state.get('current_pdf'):
        st.info("👆 Select a PDF from the list above to view analysis results.")
        return
    
    current_pdf_name = st.session_state.current_pdf
    
    # Get PDF record from database
    from models.database_models import PDF
    pdf_record = db_session.query(PDF).filter_by(pdf_name=current_pdf_name).first()
    
    if not pdf_record:
        st.error("PDF not found in database. Please upload and process the PDF first.")
        return
    
    # Header with re-run button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📄 Analysis: {current_pdf_name}")
    
    with col2:
        # RE-RUN BUTTON - This is the new functionality
        if st.button("🔄 Re-run Analysis", type="secondary", key=f"rerun_{pdf_record.id}"):
            with st.spinner("Re-running analysis..."):
                
                # Use the re-run function
                success, result_data, _ = process_pdf_with_rerun(
                    pdf_id=pdf_record.id,
                    session_id=st.session_state.user_session_id,
                    db_session=db_session
                )
                
                if success:
                    st.success("✅ Analysis re-run completed!")
                    # Update session state with new results
                    st.session_state.json_data[current_pdf_name] = result_data
                    st.rerun()  # Refresh page to show new results
                else:
                    st.error(f"❌ Re-run failed: {result_data.get('error', 'Unknown error')}")
    
    # Get latest analysis for display
    from services.analysis_service import AnalysisService
    analysis_service = AnalysisService(db_session)
    latest_analysis = analysis_service.get_latest_analysis(pdf_record.id)
    
    if not latest_analysis:
        st.warning("No analysis found for this PDF.")
        return
    
    # Version information
    analysis_history = analysis_service.get_analysis_history(pdf_record.id)
    if len(analysis_history) > 1:
        st.info(f"📋 Viewing Version {latest_analysis.version} of {len(analysis_history)} total versions")
        
        # Version selector
        version_options = [
            f"Version {analysis.version} - {analysis.analysis_date.strftime('%Y-%m-%d %H:%M')}"
            for analysis in analysis_history
        ]
        
        selected_version_idx = st.selectbox(
            "Select Version to View",
            options=range(len(version_options)),
            format_func=lambda x: version_options[x],
            key=f"version_selector_{pdf_record.id}"
        )
        
        # Use selected version
        selected_analysis = analysis_history[selected_version_idx]
        display_analysis_results(selected_analysis)
        
        # Show version comparison
        if len(analysis_history) > 1:
            render_version_comparison(analysis_history, selected_version_idx)
    else:
        # Single version
        display_analysis_results(latest_analysis)
    
    st.divider()
    
    # FEEDBACK SECTION - This is new
    st.header("💬 Feedback")
    
    feedback_tab1, feedback_tab2 = st.tabs(["✏️ Provide Feedback", "📋 Feedback History"])
    
    with feedback_tab1:
        render_feedback_form(pdf_record.id, current_pdf_name, db_session)
    
    with feedback_tab2:
        render_feedback_history(pdf_record.id, db_session)

def display_analysis_results(analysis):
    """Display analysis results from database record"""
    
    import json
    analysis_data = json.loads(analysis.raw_json)
    
    # Analysis metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Version", analysis.version)
    
    with col2:
        st.metric("Analysis Date", analysis.analysis_date.strftime('%Y-%m-%d'))
    
    with col3:
        if analysis.processing_time:
            st.metric("Processing Time", f"{analysis.processing_time:.2f}s")
    
    st.divider()
    
    # Form number
    st.subheader("📋 Form Number")
    form_number = analysis.form_number or "Not identified"
    st.markdown(f"<div class='extract-text'>{form_number}</div>", unsafe_allow_html=True)
    
    # Summary
    if analysis.summary:
        st.subheader("📝 Summary")
        st.markdown(f"<div class='extract-text'>{analysis.summary}</div>", unsafe_allow_html=True)
    
    # Contract status
    st.subheader("⚖️ Contract Status")
    
    status_fields = [
        ('Data Usage Mentioned', analysis.data_usage_mentioned),
        ('Data Limitations Exists', analysis.data_limitations_exists),
        ('PI Clause Present', analysis.pi_clause),
        ('CI Clause Present', analysis.ci_clause)
    ]
    
    # Create status grid
    cols = st.columns(2)
    
    for i, (label, value) in enumerate(status_fields):
        with cols[i % 2]:
            # Determine status color and icon
            if str(value).lower() in ['yes', 'true']:
                status_color = "🟢"
            elif str(value).lower() in ['no', 'false']:
                status_color = "🔴"
            else:
                status_color = "🟡"
            
            st.markdown(f"{status_color} **{label}**: {value}")
    
    # Relevant clauses
    st.subheader("📄 Relevant Clauses")
    
    relevant_clauses = analysis_data.get("relevant_clauses", [])
    
    if relevant_clauses:
        for i, clause in enumerate(relevant_clauses, 1):
            with st.expander(f"Clause {i}: {clause.get('type', 'Unknown').title()}"):
                st.markdown("**Type:**")
                st.code(clause.get('type', 'N/A'))
                
                st.markdown("**Content:**")
                st.write(clause.get('text', 'No content available'))
                
                # Search in PDF button
                if st.button(f"🔍 Search in PDF", key=f"search_clause_{analysis.id}_{i}"):
                    st.session_state.search_text = clause.get('text', '')
                    st.success("Search text set! Switch to PDF viewer to see highlighted text.")
    else:
        st.info("No relevant clauses identified in this analysis.")

def render_version_comparison(analysis_history, selected_index):
    """Show comparison between analysis versions"""
    
    if len(analysis_history) < 2:
        return
    
    with st.expander("🔍 Version Comparison"):
        st.markdown("Compare different analysis versions:")
        
        # Create comparison table
        comparison_data = []
        
        fields_to_compare = [
            ('form_number', 'Form Number'),
            ('pi_clause', 'PI Clause'),
            ('ci_clause', 'CI Clause'),
            ('data_usage_mentioned', 'Data Usage Mentioned'),
            ('data_limitations_exists', 'Data Limitations Exists')
        ]
        
        for field_key, field_label in fields_to_compare:
            row = {"Field": field_label}
            
            for analysis in analysis_history:
                version_label = f"V{analysis.version}"
                row[version_label] = getattr(analysis, field_key, 'N/A')
            
            comparison_data.append(row)
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # Highlight recent changes
        if len(analysis_history) >= 2:
            latest = analysis_history[0]
            previous = analysis_history[1]
            
            changes = []
            for field_key, field_label in fields_to_compare:
                latest_val = getattr(latest, field_key, None)
                previous_val = getattr(previous, field_key, None)
                
                if latest_val != previous_val:
                    changes.append(f"**{field_label}**: {previous_val} → {latest_val}")
            
            if changes:
                st.markdown("### 📝 Recent Changes:")
                for change in changes:
                    st.markdown(f"• {change}")
            else:
                st.info("No changes detected between versions.")

# Updated main() function showing where everything integrates

def main():
    """
    UPDATED main function showing integration points
    """
    
    # Initialize application (includes database setup)
    initialize_app()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    if selected_page == "📄 Document Analysis":
        
        # Main layout
        col1, col2, col3 = st.columns([25, 40, 35])
        
        # Left column: Document management
        with col1:
            st.header("📁 Document Management")
            
            # PDF upload section
            st.subheader("⬆️ Upload Documents")
            uploaded_pdfs = st.file_uploader(
                "Upload Contract PDFs",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            
            # INTEGRATION POINT 1: Use the new deduplication function
            if uploaded_pdfs:
                process_uploaded_pdfs(uploaded_pdfs)  # This uses process_pdf_with_deduplication
            
            st.divider()
            
            # Show available PDFs from database
            render_available_pdfs_grid()
        
        # Middle column: PDF viewer (unchanged)
        with col2:
            render_pdf_viewer()
        
        # Right column: Analysis with re-run and feedback
        with col3:
            # INTEGRATION POINT 2: Use enhanced analysis with re-run and feedback
            render_enhanced_analysis_section_with_rerun(st.session_state.db_session)
    
    elif selected_page == "📦 Batch Processing":
        render_batch_processing_page()
    
    elif selected_page == "📊 Analytics Dashboard":
        render_analytics_dashboard()

# File structure summary for clarity:

"""
PROJECT STRUCTURE WITH ALL FILES:

contract_analyzer_platform/
├── main.py                        # ← Updated main app (uses deduplication)
├── services/
│   ├── __init__.py
│   ├── user_service.py            # ← Contains process_pdf_with_deduplication
│   ├── pdf_service.py             # ← Enhanced PDF processing
│   ├── analysis_service.py        # ← Wraps your ContractAnalyzer
│   ├── feedback_service.py        # ← NEW: Feedback management
│   └── batch_service.py           # ← Batch processing
├── ui/
│   ├── __init__.py
│   ├── feedback_form.py           # ← NEW: Feedback UI components
│   ├── analysis_display.py        # ← Enhanced analysis display
│   └── batch_interface.py         # ← Batch processing UI
├── utils/
│   ├── __init__.py
│   ├── contract_analyzer.py       # ← YOUR existing file (minimal changes)
│   ├── pdf_handler.py             # ← From your ecfr_api_wrapper
│   └── hash_utils.py              # ← NEW: File hashing utilities
├── models/
│   ├── __init__.py
│   └── database_models.py         # ← Database schema
├── config/
│   ├── __init__.py
│   └── database.py                # ← Database connection
└── requirements.txt

INTEGRATION POINTS:
1. process_uploaded_pdfs() uses process_pdf_with_deduplication()
2. render_enhanced_analysis_section_with_rerun() uses process_pdf_with_rerun()
3. Both functions integrate with your existing ContractAnalyzer seamlessly
"""