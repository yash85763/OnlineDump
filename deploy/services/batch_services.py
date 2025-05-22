# services/batch_service.py - Batch processing service

import uuid
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from models.database_models import BatchJob, PDF, Analysis
from services.pdf_service import EnhancedPDFService

class BatchProcessingService:
    """Service for handling batch processing of multiple PDFs"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.pdf_service = EnhancedPDFService(db_session)
    
    def create_batch_job(self, file_list: List[Dict], session_id: str) -> str:
        """Create a new batch processing job"""
        
        job_id = str(uuid.uuid4())
        
        batch_job = BatchJob(
            job_id=job_id,
            created_at=datetime.utcnow(),
            total_files=len(file_list),
            processed_files=0,
            failed_files=0,
            status='pending',
            created_by=session_id,
            results_json=json.dumps([])
        )
        
        self.db_session.add(batch_job)
        self.db_session.commit()
        
        return job_id
    
    def process_batch_async(self, job_id: str, file_list: List[Dict], session_id: str):
        """Process batch of files asynchronously"""
        
        def process_batch_worker():
            """Worker function for batch processing"""
            
            # Update job status to running
            batch_job = self.db_session.query(BatchJob).filter_by(job_id=job_id).first()
            batch_job.status = 'running'
            batch_job.started_at = datetime.utcnow()
            self.db_session.commit()
            
            results = []
            processed_count = 0
            failed_count = 0
            
            try:
                for file_info in file_list:
                    file_name = file_info['name']
                    file_bytes = file_info['bytes']
                    
                    try:
                        # Process individual PDF
                        success, result_data, pdf_id = self.pdf_service.process_pdf_pipeline(
                            file_bytes, file_name, session_id
                        )
                        
                        if success:
                            results.append({
                                'file_name': file_name,
                                'pdf_id': pdf_id,
                                'status': 'success',
                                'form_number': result_data.get('form_number'),
                                'pi_clause': result_data.get('pi_clause'),
                                'ci_clause': result_data.get('ci_clause'),
                                'processed_at': datetime.utcnow().isoformat()
                            })
                            processed_count += 1
                        else:
                            results.append({
                                'file_name': file_name,
                                'status': 'failed',
                                'error': result_data,
                                'processed_at': datetime.utcnow().isoformat()
                            })
                            failed_count += 1
                            
                    except Exception as e:
                        results.append({
                            'file_name': file_name,
                            'status': 'failed',
                            'error': str(e),
                            'processed_at': datetime.utcnow().isoformat()
                        })
                        failed_count += 1
                    
                    # Update progress
                    batch_job = self.db_session.query(BatchJob).filter_by(job_id=job_id).first()
                    batch_job.processed_files = processed_count
                    batch_job.failed_files = failed_count
                    batch_job.results_json = json.dumps(results)
                    self.db_session.commit()
                
                # Mark job as completed
                batch_job = self.db_session.query(BatchJob).filter_by(job_id=job_id).first()
                batch_job.status = 'completed'
                batch_job.completed_at = datetime.utcnow()
                batch_job.results_json = json.dumps(results)
                self.db_session.commit()
                
            except Exception as e:
                # Mark job as failed
                batch_job = self.db_session.query(BatchJob).filter_by(job_id=job_id).first()
                batch_job.status = 'failed'
                batch_job.error_log = str(e)
                batch_job.completed_at = datetime.utcnow()
                self.db_session.commit()
        
        # Start background thread
        thread = threading.Thread(target=process_batch_worker)
        thread.daemon = True
        thread.start()
    
    def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job"""
        
        batch_job = self.db_session.query(BatchJob).filter_by(job_id=job_id).first()
        
        if not batch_job:
            return {"error": "Job not found"}
        
        return {
            "job_id": batch_job.job_id,
            "status": batch_job.status,
            "total_files": batch_job.total_files,
            "processed_files": batch_job.processed_files,
            "failed_files": batch_job.failed_files,
            "created_at": batch_job.created_at.isoformat(),
            "started_at": batch_job.started_at.isoformat() if batch_job.started_at else None,
            "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None,
            "results": json.loads(batch_job.results_json) if batch_job.results_json else [],
            "error_log": batch_job.error_log
        }
    
    def get_user_batch_jobs(self, session_id: str) -> List[Dict]:
        """Get all batch jobs for a user"""
        
        jobs = self.db_session.query(BatchJob).filter_by(
            created_by=session_id
        ).order_by(BatchJob.created_at.desc()).all()
        
        return [
            {
                "job_id": job.job_id,
                "status": job.status,
                "total_files": job.total_files,
                "processed_files": job.processed_files,
                "failed_files": job.failed_files,
                "created_at": job.created_at.isoformat()
            }
            for job in jobs
        ]

# ui/batch_interface.py - Batch processing UI components

import streamlit as st
import time
from services.batch_service import BatchProcessingService

def render_batch_upload_interface(db_session):
    """Render batch upload interface"""
    
    st.subheader("📦 Batch Processing")
    st.markdown("Upload up to 20 PDF files for batch analysis")
    
    # File uploader for batch processing
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        num_files = len(uploaded_files)
        
        if num_files > 20:
            st.error(f"❌ Too many files! Please select maximum 20 files. You selected {num_files} files.")
            return
        
        st.success(f"✅ {num_files} files selected for batch processing")
        
        # Display file list
        with st.expander("📋 Selected Files"):
            for i, file in enumerate(uploaded_files, 1):
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.write(f"{i}. {file.name} ({file_size_mb:.2f} MB)")
        
        # Process batch button
        if st.button("🚀 Start Batch Processing", type="primary"):
            
            # Prepare file data
            file_data = []
            for file in uploaded_files:
                file_data.append({
                    'name': file.name,
                    'bytes': file.getvalue()
                })
            
            # Create and start batch job
            batch_service = BatchProcessingService(db_session)
            job_id = batch_service.create_batch_job(
                file_data, 
                st.session_state.user_session_id
            )
            
            # Start async processing
            batch_service.process_batch_async(
                job_id, 
                file_data, 
                st.session_state.user_session_id
            )
            
            st.session_state.current_batch_job = job_id
            st.success(f"🎯 Batch job started! Job ID: {job_id}")
            
            # Show progress monitoring
            render_batch_progress_monitor(batch_service, job_id)

def render_batch_progress_monitor(batch_service: BatchProcessingService, job_id: str):
    """Render real-time progress monitor for batch job"""
    
    st.subheader("📊 Processing Progress")
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    results_placeholder = st.empty()
    
    # Auto-refresh progress
    refresh_interval = 2  # seconds
    
    while True:
        status = batch_service.get_batch_status(job_id)
        
        if status.get("error"):
            status_placeholder.error(f"❌ {status['error']}")
            break
        
        # Update status
        current_status = status['status']
        total_files = status['total_files']
        processed_files = status['processed_files']
        failed_files = status['failed_files']
        
        # Status indicator
        if current_status == 'pending':
            status_placeholder.info("⏳ Job pending...")
        elif current_status == 'running':
            status_placeholder.info("🔄 Processing files...")
        elif current_status == 'completed':
            status_placeholder.success("✅ Batch processing completed!")
        elif current_status == 'failed':
            status_placeholder.error("❌ Batch processing failed!")
        
        # Progress bar
        if total_files > 0:
            progress = (processed_files + failed_files) / total_files
            progress_placeholder.progress(
                progress, 
                text=f"Progress: {processed_files + failed_files}/{total_files} files processed"
            )
        
        # Results summary
        if processed_files > 0 or failed_files > 0:
            with results_placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Successful", processed_files)
                
                with col2:
                    st.metric("❌ Failed", failed_files)
                
                with col3:
                    st.metric("📊 Total", total_files)
        
        # Break if job is complete
        if current_status in ['completed', 'failed']:
            # Show detailed results
            render_batch_results(status['results'])
            break
        
        # Wait before next update
        time.sleep(refresh_interval)

def render_batch_results(results: list):
    """Render detailed batch processing results"""
    
    if not results:
        return
    
    st.subheader("📋 Detailed Results")
    
    # Success/Failure tabs
    tab1, tab2 = st.tabs(["✅ Successful", "❌ Failed"])
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    with tab1:
        if successful:
            st.success(f"Successfully processed {len(successful)} files")
            
            # Create results table
            success_data = []
            for result in successful:
                success_data.append({
                    "File Name": result['file_name'],
                    "Form Number": result.get('form_number', 'N/A'),
                    "PI Clause": result.get('pi_clause', 'N/A'),
                    "CI Clause": result.get('ci_clause', 'N/A'),
                    "Processed At": result['processed_at']
                })
            
            st.dataframe(success_data, use_container_width=True)
        else:
            st.info("No files were successfully processed.")
    
    with tab2:
        if failed:
            st.error(f"Failed to process {len(failed)} files")
            
            for result in failed:
                with st.expander(f"❌ {result['file_name']}"):
                    st.write(f"**Error:** {result.get('error', 'Unknown error')}")
                    st.write(f"**Time:** {result['processed_at']}")
        else:
            st.success("All files processed successfully!")

def render_batch_history(db_session):
    """Render batch processing history"""
    
    st.subheader("📚 Batch Processing History")
    
    batch_service = BatchProcessingService(db_session)
    user_jobs = batch_service.get_user_batch_jobs(st.session_state.user_session_id)
    
    if user_jobs:
        for job in user_jobs:
            with st.expander(f"Job {job['job_id'][:8]}... - {job['status'].title()}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Files", job['total_files'])
                
                with col2:
                    st.metric("Processed", job['processed_files'])
                
                with col3:
                    st.metric("Failed", job['failed_files'])
                
                st.write(f"**Created:** {job['created_at']}")
                st.write(f"**Status:** {job['status'].title()}")
                
                if st.button(f"View Details", key=f"view_{job['job_id']}"):
                    detailed_status = batch_service.get_batch_status(job['job_id'])
                    render_batch_results(detailed_status.get('results', []))
    else:
        st.info("No batch processing history found.")
