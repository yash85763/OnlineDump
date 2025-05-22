# services/pdf_service.py - Enhanced PDF processing with multi-stage storage

import time
import json
from datetime import datetime
from sqlalchemy.orm import Session
from models.database_models import PDF, Analysis, Clause
from utils.pdf_handler import PDFHandler
from utils.hash_utils import calculate_file_hash

class EnhancedPDFService:
    """Enhanced PDF processing service with multi-stage storage"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.pdf_handler = PDFHandler()
    
    def process_pdf_pipeline(self, pdf_bytes: bytes, pdf_name: str, session_id: str, force_rerun: bool = False):
        """
        Complete PDF processing pipeline with multi-stage storage
        
        Stage 1: PDF Parsing & Metadata Storage
        Stage 2: Contract Analysis & Results Storage  
        Stage 3: Clause Extraction & Storage
        
        Returns: (success, result_data, pdf_id)
        """
        start_time = time.time()
        
        try:
            # STAGE 1: PDF PARSING & DEDUPLICATION
            file_hash = calculate_file_hash(pdf_bytes)
            existing_pdf = self.check_existing_pdf(file_hash)
            
            if existing_pdf and not force_rerun:
                # Check if analysis already exists
                existing_analysis = self.get_latest_analysis(existing_pdf.id)
                if existing_analysis:
                    return True, json.loads(existing_analysis.raw_json), existing_pdf.id
                
                pdf_record = existing_pdf
            else:
                # Process new PDF
                pdf_record = self.stage1_pdf_parsing(pdf_bytes, pdf_name, file_hash, session_id)
                if not pdf_record:
                    return False, "PDF parsing failed", None
            
            # STAGE 2: CONTRACT ANALYSIS
            analysis_record = self.stage2_contract_analysis(
                pdf_record, session_id, force_rerun
            )
            
            if not analysis_record:
                return False, "Contract analysis failed", pdf_record.id
            
            # STAGE 3: CLAUSE EXTRACTION
            self.stage3_clause_extraction(analysis_record)
            
            processing_time = time.time() - start_time
            analysis_record.processing_time = processing_time
            self.db_session.commit()
            
            return True, json.loads(analysis_record.raw_json), pdf_record.id
            
        except Exception as e:
            self.db_session.rollback()
            return False, f"Processing error: {str(e)}", None
    
    def stage1_pdf_parsing(self, pdf_bytes: bytes, pdf_name: str, file_hash: str, session_id: str) -> PDF:
        """Stage 1: PDF parsing and metadata storage"""
        
        # Save PDF temporarily for processing
        temp_path = f"/tmp/{file_hash}.pdf"
        with open(temp_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Process with PDFHandler
        pdf_result = self.pdf_handler.process_pdf(temp_path)
        
        if not pdf_result.get("parsable", False):
            return None
        
        # Calculate metrics
        pages = pdf_result.get("pages", [])
        word_count = sum(len(para.split()) for page in pages for para in page.get("paragraphs", []))
        page_count = len(pages)
        avg_words_per_page = word_count / page_count if page_count > 0 else 0
        
        # Extract content
        raw_content = "\n\n".join(para for page in pages for para in page.get("paragraphs", []))
        
        # TODO: Implement obfuscation logic here
        final_content = raw_content  # Placeholder for obfuscation
        
        # Create PDF record
        pdf_record = PDF(
            pdf_name=pdf_name,
            file_hash=file_hash,
            upload_date=datetime.utcnow(),
            processed_date=datetime.utcnow(),
            layout=pdf_result.get("layout", "unknown"),
            word_count=word_count,
            page_count=page_count,
            parsability=pdf_result.get("parsable", False),
            avg_words_per_page=avg_words_per_page,
            raw_content=raw_content,
            final_content=final_content,
            uploaded_by=session_id
        )
        
        self.db_session.add(pdf_record)
        self.db_session.commit()
        
        return pdf_record
    
    def stage2_contract_analysis(self, pdf_record: PDF, session_id: str, force_rerun: bool = False) -> Analysis:
        """Stage 2: Contract analysis and results storage"""
        
        # Check for existing analysis
        if not force_rerun:
            existing_analysis = self.get_latest_analysis(pdf_record.id)
            if existing_analysis:
                return existing_analysis
        
        # Determine version number
        max_version = self.db_session.query(
            self.db_session.query(Analysis.version).filter_by(pdf_id=pdf_record.id).order_by(Analysis.version.desc()).first()
        )
        new_version = (max_version[0] if max_version else 0) + 1
        
        # Run contract analysis
        # NOTE: This would integrate with your existing ContractAnalyzer
        from contract_analyzer import ContractAnalyzer
        
        analyzer = ContractAnalyzer()
        analysis_result = analyzer.analyze_contract(pdf_record.final_content)
        
        # Create analysis record
        analysis_record = Analysis(
            pdf_id=pdf_record.id,
            analysis_date=datetime.utcnow(),
            version=new_version,
            form_number=analysis_result.get("form_number"),
            pi_clause=analysis_result.get("pi_clause"),
            ci_clause=analysis_result.get("ci_clause"),
            data_usage_mentioned=analysis_result.get("data_usage_mentioned"),
            data_limitations_exists=analysis_result.get("data_limitations_exists"),
            summary=analysis_result.get("summary"),
            raw_json=json.dumps(analysis_result),
            processed_by=session_id
        )
        
        self.db_session.add(analysis_record)
        self.db_session.commit()
        
        return analysis_record
    
    def stage3_clause_extraction(self, analysis_record: Analysis):
        """Stage 3: Extract and store individual clauses"""
        
        analysis_data = json.loads(analysis_record.raw_json)
        relevant_clauses = analysis_data.get("relevant_clauses", [])
        
        for idx, clause_data in enumerate(relevant_clauses):
            clause_record = Clause(
                analysis_id=analysis_record.id,
                clause_type=clause_data.get("type"),
                clause_text=clause_data.get("text"),
                clause_order=idx + 1
            )
            self.db_session.add(clause_record)
        
        self.db_session.commit()
    
    def check_existing_pdf(self, file_hash: str) -> PDF:
        """Check if PDF already exists"""
        return self.db_session.query(PDF).filter_by(file_hash=file_hash).first()
    
    def get_latest_analysis(self, pdf_id: int) -> Analysis:
        """Get latest analysis for a PDF"""
        return self.db_session.query(Analysis).filter_by(pdf_id=pdf_id).order_by(Analysis.version.desc()).first()
    
    def get_analysis_history(self, pdf_id: int) -> list:
        """Get all analysis versions for a PDF"""
        return self.db_session.query(Analysis).filter_by(pdf_id=pdf_id).order_by(Analysis.version.desc()).all()
