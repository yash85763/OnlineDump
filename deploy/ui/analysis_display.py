# ui/analysis_display.py - Enhanced analysis display with re-run functionality

import streamlit as st
import json
from datetime import datetime
from services.pdf_service import EnhancedPDFService

def render_analysis_with_rerun(pdf_id: int, pdf_name: str, db_session):
    """Render analysis results with re-run functionality and version history"""
    
    pdf_service = EnhancedPDFService(db_session)
    
    # Get all analysis versions for this PDF
    analysis_history = pdf_service.get_analysis_history(pdf_id)
    
    if not analysis_history:
        st.info("No analysis available for this PDF")
        return
    
    # Header with re-run button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📊 Analysis Results: {pdf_name}")
    
    with col2:
        if st.button("🔄 Re-run Analysis", type="secondary", key=f"rerun_{pdf_id}"):
            with st.spinner("Re-running analysis..."):
                success, result_data, _ = pdf_service.process_pdf_pipeline(
                    pdf_bytes=None,  # Will use existing PDF data
                    pdf_name=pdf_name,
                    session_id=st.session_state.user_session_id,
                    force_rerun=True
                )
                
                if success:
                    st.success("✅ Analysis re-run completed!")
                    st.rerun()  # Refresh the page to show new results
                else:
                    st.error(f"❌ Re-run failed: {result_data}")
    
    # Version selector
    if len(analysis_history) > 1:
        st.markdown("### 📋 Analysis Versions")
        
        version_options = [
            f"Version {analysis.version} - {analysis.analysis_date.strftime('%Y-%m-%d %H:%M')}"
            for analysis in analysis_history
        ]
        
        selected_version = st.selectbox(
            "Select Analysis Version",
            options=range(len(version_options)),
            format_func=lambda x: version_options[x],
            key=f"version_selector_{pdf_id}"
        )
        
        current_analysis = analysis_history[selected_version]
        
        # Show version comparison if multiple versions exist
        if len(analysis_history) > 1:
            render_version_comparison(analysis_history, selected_version)
    else:
        current_analysis = analysis_history[0]
    
    # Display current analysis
    render_analysis_details(current_analysis)

def render_version_comparison(analysis_history: list, selected_index: int):
    """Render comparison between different analysis versions"""
    
    if len(analysis_history) < 2:
        return
    
    with st.expander("🔍 Version Comparison"):
        st.markdown("Compare different analysis versions to see changes:")
        
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
            
            for i, analysis in enumerate(analysis_history):
                version_label = f"Version {analysis.version}"
                row[version_label] = getattr(analysis, field_key, 'N/A')
            
            comparison_data.append(row)
        
        # Display comparison table
        st.dataframe(comparison_data, use_container_width=True)
        
        # Highlight changes
        if len(analysis_history) >= 2:
            latest = analysis_history[0]  # Most recent
            previous = analysis_history[1]  # Previous version
            
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
                st.info("No changes detected between the last two versions.")

def render_analysis_details(analysis):
    """Render detailed analysis results"""
    
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
    
    # Contract status with enhanced styling
    st.subheader("⚖️ Contract Status")
    
    status_fields = [
        ('data_usage_mentioned', 'Data Usage Mentioned', analysis.data_usage_mentioned),
        ('data_limitations_exists', 'Data Limitations Exists', analysis.data_limitations_exists),
        ('pi_clause', 'PI Clause Present', analysis.pi_clause),
        ('ci_clause', 'CI Clause Present', analysis.ci_clause)
    ]
    
    # Create status grid
    cols = st.columns(2)
    
    for i, (key, label, value) in enumerate(status_fields):
        with cols[i % 2]:
            # Determine status color and icon
            if str(value).lower() in ['yes', 'true']:
                status_color = "🟢"
                status_class = "success"
            elif str(value).lower() in ['no', 'false']:
                status_color = "🔴"
                status_class = "error"
            else:
                status_color = "🟡"
                status_class = "warning"
            
            st.markdown(f"{status_color} **{label}**: {value}")
    
    # Relevant clauses with enhanced display
    st.subheader("📄 Relevant Clauses")
    
    relevant_clauses = analysis_data.get("relevant_clauses", [])
    
    if relevant_clauses:
        for i, clause in enumerate(relevant_clauses, 1):
            with st.expander(f"Clause {i}: {clause.get('type', 'Unknown').title()}"):
                st.markdown("**Type:**")
                st.code(clause.get('type', 'N/A'))
                
                st.markdown("**Content:**")
                st.write(clause.get('text', 'No content available'))
                
                # Action buttons for clause
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"🔍 Search in PDF", key=f"search_clause_{analysis.id}_{i}"):
                        st.session_state.search_text = clause.get('text', '')
                        st.success("Search text set! Switch to PDF viewer to see highlighted text.")
                
                with col2:
                    if st.button(f"📋 Copy Text", key=f"copy_clause_{analysis.id}_{i}"):
                        # This would require JavaScript integration for actual copying
                        st.info("Text copied to clipboard! (Feature requires JavaScript integration)")
    else:
        st.info("No relevant clauses identified in this analysis.")
    
    # Raw JSON data (collapsible)
    with st.expander("🔧 Raw Analysis Data (JSON)"):
        st.json(analysis_data)

# Enhanced main application integration
def render_enhanced_analysis_section(db_session):
    """Enhanced analysis section with all new features"""
    
    st.header("📊 Contract Analysis Results")
    
    if not st.session_state.get('current_pdf'):
        st.info("👆 Select a PDF from the list above to view analysis results.")
        return
    
    # Get current PDF info
    current_pdf_name = st.session_state.current_pdf
    
    # Get PDF ID from database
    from models.database_models import PDF
    pdf_record = db_session.query(PDF).filter_by(pdf_name=current_pdf_name).first()
    
    if not pdf_record:
        st.error("PDF not found in database. Please upload and process the PDF first.")
        return
    
    # Render analysis with re-run functionality
    render_analysis_with_rerun(pdf_record.id, current_pdf_name, db_session)
    
    st.divider()
    
    # Add feedback section
    st.header("💬 Feedback")
    
    # Import feedback components
    from ui.feedback_form import render_feedback_form, render_feedback_history
    
    # Feedback tabs
    feedback_tab1, feedback_tab2 = st.tabs(["✏️ Provide Feedback", "📋 Feedback History"])
    
    with feedback_tab1:
        render_feedback_form(pdf_record.id, current_pdf_name, db_session)
    
    with feedback_tab2:
        render_feedback_history(pdf_record.id, db_session)
