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




# second update

# services/pdf_service.py - Updated with obfuscation integration

import time
import json
from datetime import datetime
from sqlalchemy.orm import Session
from models.database_models import PDF, Analysis, Clause
from utils.pdf_handler import PDFHandler
from utils.hash_utils import calculate_file_hash
from utils.obfuscation import ContentObfuscator, ObfuscationConfig, create_default_obfuscation_config

class EnhancedPDFService:
    """Enhanced PDF processing service with obfuscation support"""
    
    def __init__(self, db_session: Session, obfuscation_config: ObfuscationConfig = None):
        self.db_session = db_session
        self.pdf_handler = PDFHandler()
        
        # Initialize obfuscation
        self.obfuscation_config = obfuscation_config or create_default_obfuscation_config()
        self.obfuscator = ContentObfuscator(self.obfuscation_config)
    
    def process_pdf_pipeline(self, pdf_bytes: bytes, pdf_name: str, session_id: str, 
                           force_rerun: bool = False, custom_obfuscation_config: ObfuscationConfig = None):
        """
        Complete PDF processing pipeline with obfuscation support
        
        Stage 1: PDF Parsing & Metadata Storage
        Stage 2: Content Obfuscation (NEW)
        Stage 3: Contract Analysis & Results Storage  
        Stage 4: Clause Extraction & Storage
        
        Returns: (success, result_data, pdf_id)
        """
        start_time = time.time()
        
        try:
            # Use custom obfuscation config if provided
            if custom_obfuscation_config:
                obfuscator = ContentObfuscator(custom_obfuscation_config)
            else:
                obfuscator = self.obfuscator
            
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
                # Process new PDF with obfuscation
                pdf_record = self.stage1_pdf_parsing_with_obfuscation(
                    pdf_bytes, pdf_name, file_hash, session_id, obfuscator
                )
                if not pdf_record:
                    return False, "PDF parsing failed", None
            
            # STAGE 2: CONTRACT ANALYSIS (using obfuscated content)
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
    
    def stage1_pdf_parsing_with_obfuscation(self, pdf_bytes: bytes, pdf_name: str, 
                                          file_hash: str, session_id: str, 
                                          obfuscator: ContentObfuscator) -> PDF:
        """Stage 1: PDF parsing with obfuscation applied"""
        
        # Save PDF temporarily for processing
        import tempfile
        import os
        
        temp_path = f"/tmp/{file_hash}.pdf"
        with open(temp_path, 'wb') as f:
            f.write(pdf_bytes)
        
        try:
            # Process with PDFHandler
            pdf_result = self.pdf_handler.process_pdf(temp_path)
            
            if not pdf_result.get("parsable", False):
                return None
            
            # APPLY OBFUSCATION HERE
            print(f"Applying obfuscation to {pdf_name}...")
            obfuscated_result = obfuscator.obfuscate_content(pdf_result)
            
            # Calculate metrics from obfuscated content
            pages = obfuscated_result.get("pages", [])
            
            # Calculate word counts from obfuscated content
            raw_word_count = sum(len(para.split()) for page in pdf_result.get("pages", []) 
                               for para in page.get("paragraphs", []))
            obfuscated_word_count = sum(len(para.split()) for page in pages 
                                      for para in page.get("paragraphs", []))
            
            page_count = len(pages)
            avg_words_per_page = obfuscated_word_count / page_count if page_count > 0 else 0
            
            # Extract content (both raw and obfuscated)
            raw_content = "\n\n".join(para for page in pdf_result.get("pages", []) 
                                    for para in page.get("paragraphs", []))
            
            obfuscated_content = "\n\n".join(para for page in pages 
                                           for para in page.get("paragraphs", []))
            
            # Get obfuscation report
            obfuscation_report = obfuscator.get_obfuscation_report()
            
            # Create PDF record with obfuscation data
            pdf_record = PDF(
                pdf_name=pdf_name,
                file_hash=file_hash,
                upload_date=datetime.utcnow(),
                processed_date=datetime.utcnow(),
                layout=obfuscated_result.get("layout", "unknown"),
                word_count=obfuscated_word_count,  # Use obfuscated word count
                page_count=page_count,
                parsability=obfuscated_result.get("parsable", False),
                avg_words_per_page=avg_words_per_page,
                raw_content=raw_content,  # Store original content
                final_content=obfuscated_content,  # Store obfuscated content
                uploaded_by=session_id,
                
                # Add obfuscation metadata
                obfuscation_applied=True,
                obfuscation_stats=json.dumps(obfuscation_report),
                original_word_count=raw_word_count
            )
            
            self.db_session.add(pdf_record)
            self.db_session.commit()
            
            # Log obfuscation results
            print(f"Obfuscation completed for {pdf_name}:")
            print(f"  - Pages removed: {obfuscation_report['summary']['pages_removed']}")
            print(f"  - Paragraphs obfuscated: {obfuscation_report['summary']['paragraphs_obfuscated']}")
            print(f"  - Original word count: {raw_word_count}")
            print(f"  - Final word count: {obfuscated_word_count}")
            
            return pdf_record
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def stage2_contract_analysis(self, pdf_record: PDF, session_id: str, force_rerun: bool = False) -> Analysis:
        """Stage 2: Contract analysis using obfuscated content"""
        
        # Check for existing analysis
        if not force_rerun:
            existing_analysis = self.get_latest_analysis(pdf_record.id)
            if existing_analysis:
                return existing_analysis
        
        # Determine version number
        max_version = self.db_session.query(
            self.db_session.func.max(Analysis.version)
        ).filter_by(pdf_id=pdf_record.id).scalar()
        new_version = (max_version or 0) + 1
        
        # Run contract analysis on OBFUSCATED content
        from services.analysis_service import AnalysisService
        analysis_service = AnalysisService(self.db_session)
        
        # Use final_content (obfuscated) for analysis
        content_to_analyze = pdf_record.final_content or pdf_record.raw_content
        
        # Create temporary analysis record to use existing service
        temp_pdf_record = pdf_record
        temp_pdf_record.final_content = content_to_analyze
        
        success, analysis_result = analysis_service.analyze_contract_with_storage(
            temp_pdf_record, session_id, force_rerun
        )
        
        if success:
            # Get the created analysis record
            analysis_record = self.get_latest_analysis(pdf_record.id)
            
            # Add obfuscation note to analysis
            if isinstance(analysis_result, dict):
                analysis_result["obfuscation_applied"] = pdf_record.obfuscation_applied
                analysis_result["content_note"] = "Analysis performed on obfuscated content"
                
                # Update raw_json with obfuscation info
                analysis_record.raw_json = json.dumps(analysis_result)
                self.db_session.commit()
            
            return analysis_record
        else:
            return None
    
    def stage3_clause_extraction(self, analysis_record: Analysis):
        """Stage 3: Extract and store individual clauses (unchanged)"""
        
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
    
    def get_obfuscation_report(self, pdf_id: int) -> Dict[str, Any]:
        """Get obfuscation report for a specific PDF"""
        
        pdf_record = self.db_session.query(PDF).filter_by(id=pdf_id).first()
        
        if not pdf_record or not pdf_record.obfuscation_stats:
            return {"error": "No obfuscation data found"}
        
        try:
            return json.loads(pdf_record.obfuscation_stats)
        except:
            return {"error": "Invalid obfuscation data"}
    
    def reprocess_with_custom_obfuscation(self, pdf_id: int, session_id: str, 
                                        custom_config: ObfuscationConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Reprocess existing PDF with custom obfuscation settings.
        
        Args:
            pdf_id: ID of existing PDF
            session_id: User session ID
            custom_config: Custom obfuscation configuration
            
        Returns:
            Tuple of (success, result_data)
        """
        
        pdf_record = self.db_session.query(PDF).filter_by(id=pdf_id).first()
        
        if not pdf_record:
            return False, {"error": "PDF not found"}
        
        try:
            # Re-process with custom obfuscation
            # Note: This would require re-parsing the original PDF
            # For now, we'll update the existing record with new obfuscation
            
            # Create new obfuscator with custom config
            obfuscator = ContentObfuscator(custom_config)
            
            # Re-parse original content (this is simplified - in practice you'd need the original PDF)
            # For now, we'll work with what we have
            print(f"Reprocessing {pdf_record.pdf_name} with custom obfuscation...")
            
            # This is a placeholder - you'd need to implement full re-parsing
            success, analysis_result = self.stage2_contract_analysis(
                pdf_record, session_id, force_rerun=True
            )
            
            if success:
                return True, analysis_result
            else:
                return False, {"error": "Reprocessing failed"}
                
        except Exception as e:
            return False, {"error": str(e)}


# Updated models/database_models.py - Add obfuscation fields to PDF model

"""
Add these fields to your PDF model in database_models.py:

class PDF(Base):
    # ... existing fields ...
    
    # Obfuscation fields
    obfuscation_applied = Column(Boolean, default=False)
    obfuscation_stats = Column(JSON)  # Store obfuscation report as JSON
    original_word_count = Column(Integer)  # Word count before obfusc
