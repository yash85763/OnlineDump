# services/analysis_service.py - Contract Analysis Service with existing ContractAnalyzer integration

import json
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Tuple
from sqlalchemy.orm import Session

# Import your existing contract analyzer
from utils.contract_analyzer import ContractAnalyzer  # Your existing class
from models.database_models import PDF, Analysis, Clause

class AnalysisService:
    """
    Service wrapper for contract analysis with database integration.
    
    This service wraps your existing ContractAnalyzer class and adds:
    - Database storage integration
    - Version management
    - Error handling and retry logic
    - Performance tracking
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.contract_analyzer = ContractAnalyzer()  # Your existing analyzer
    
    def analyze_contract_with_storage(
        self, 
        pdf_record: PDF, 
        session_id: str, 
        force_rerun: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze contract using your existing ContractAnalyzer and store results.
        
        Args:
            pdf_record: PDF database record
            session_id: User session ID
            force_rerun: Whether to force a new analysis even if one exists
            
        Returns:
            Tuple of (success, analysis_data)
        """
        start_time = time.time()
        
        try:
            # Check for existing analysis
            if not force_rerun:
                existing_analysis = self.get_latest_analysis(pdf_record.id)
                if existing_analysis:
                    return True, json.loads(existing_analysis.raw_json)
            
            # Determine version number for new analysis
            new_version = self.get_next_version(pdf_record.id)
            
            # Create temporary file for your existing analyzer
            # (since your analyzer might expect file paths)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(pdf_record.final_content or pdf_record.raw_content)
                temp_file_path = temp_file.name
            
            try:
                # Use your existing ContractAnalyzer
                # Adjust this call based on your actual ContractAnalyzer interface
                analysis_result = self.contract_analyzer.analyze_contract(
                    pdf_record.final_content or pdf_record.raw_content,
                    output_path=None  # We'll handle storage ourselves
                )
                
                # If your analyzer returns a file path instead of direct results
                # you might need to read the results from the output file
                if isinstance(analysis_result, str) and os.path.exists(analysis_result):
                    with open(analysis_result, 'r') as f:
                        analysis_result = json.load(f)
                
                # Store analysis in database
                analysis_record = self.store_analysis_results(
                    pdf_record.id,
                    analysis_result,
                    new_version,
                    session_id,
                    time.time() - start_time
                )
                
                # Store individual clauses
                self.store_clauses(analysis_record.id, analysis_result)
                
                return True, analysis_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self.db_session.rollback()
            return False, {"error": str(e)}
    
    def store_analysis_results(
        self, 
        pdf_id: int, 
        analysis_data: Dict[str, Any], 
        version: int, 
        session_id: str,
        processing_time: float
    ) -> Analysis:
        """Store analysis results in database"""
        
        analysis_record = Analysis(
            pdf_id=pdf_id,
            analysis_date=datetime.utcnow(),
            version=version,
            
            # Extract key fields from your analyzer results
            # Adjust these field names based on your ContractAnalyzer output
            form_number=analysis_data.get("form_number"),
            pi_clause=analysis_data.get("pi_clause"),
            ci_clause=analysis_data.get("ci_clause"),
            data_usage_mentioned=analysis_data.get("data_usage_mentioned"),
            data_limitations_exists=analysis_data.get("data_limitations_exists"),
            summary=analysis_data.get("summary"),
            
            # Store complete results as JSON
            raw_json=json.dumps(analysis_data),
            processed_by=session_id,
            processing_time=processing_time
        )
        
        self.db_session.add(analysis_record)
        self.db_session.commit()
        
        return analysis_record
    
    def store_clauses(self, analysis_id: int, analysis_data: Dict[str, Any]):
        """Extract and store individual clauses"""
        
        relevant_clauses = analysis_data.get("relevant_clauses", [])
        
        for idx, clause_data in enumerate(relevant_clauses):
            clause_record = Clause(
                analysis_id=analysis_id,
                clause_type=clause_data.get("type"),
                clause_text=clause_data.get("text"),
                clause_order=idx + 1
            )
            self.db_session.add(clause_record)
        
        self.db_session.commit()
    
    def get_latest_analysis(self, pdf_id: int) -> Analysis:
        """Get the latest analysis for a PDF"""
        return self.db_session.query(Analysis).filter_by(
            pdf_id=pdf_id
        ).order_by(Analysis.version.desc()).first()
    
    def get_analysis_history(self, pdf_id: int) -> list:
        """Get all analysis versions for a PDF"""
        return self.db_session.query(Analysis).filter_by(
            pdf_id=pdf_id
        ).order_by(Analysis.version.desc()).all()
    
    def get_next_version(self, pdf_id: int) -> int:
        """Get the next version number for a PDF"""
        max_version = self.db_session.query(
            self.db_session.func.max(Analysis.version)
        ).filter_by(pdf_id=pdf_id).scalar()
        
        return (max_version or 0) + 1
    
    def retry_analysis(self, pdf_id: int, session_id: str, max_retries: int = 3) -> Tuple[bool, Dict[str, Any]]:
        """
        Retry analysis with exponential backoff (useful for API-based analyzers).
        
        This method provides robust retry logic if your ContractAnalyzer
        makes external API calls that might fail.
        """
        
        pdf_record = self.db_session.query(PDF).filter_by(id=pdf_id).first()
        if not pdf_record:
            return False, {"error": "PDF not found"}
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                success, result = self.analyze_contract_with_storage(
                    pdf_record, session_id, force_rerun=True
                )
                
                if success:
                    return True, result
                else:
                    last_error = result
                    
            except Exception as e:
                last_error = {"error": str(e)}
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, etc.
                    time.sleep(wait_time)
        
        return False, last_error or {"error": "Max retries exceeded"}


# Updated services/pdf_service.py - Integration point

class EnhancedPDFService:
    """Enhanced PDF service that integrates with AnalysisService"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.pdf_handler = PDFHandler()
        self.analysis_service = AnalysisService(db_session)  # Add analysis service
    
    def process_pdf_pipeline(self, pdf_bytes: bytes, pdf_name: str, session_id: str, force_rerun: bool = False):
        """Complete PDF processing pipeline with integrated contract analysis"""
        
        try:
            # Stage 1: PDF Parsing (existing logic)
            file_hash = calculate_file_hash(pdf_bytes)
            existing_pdf = self.check_existing_pdf(file_hash)
            
            if existing_pdf and not force_rerun:
                pdf_record = existing_pdf
            else:
                pdf_record = self.stage1_pdf_parsing(pdf_bytes, pdf_name, file_hash, session_id)
                if not pdf_record:
                    return False, "PDF parsing failed", None
            
            # Stage 2: Contract Analysis using your existing analyzer
            success, analysis_result = self.analysis_service.analyze_contract_with_storage(
                pdf_record, session_id, force_rerun
            )
            
            if not success:
                return False, analysis_result.get("error", "Analysis failed"), pdf_record.id
            
            return True, analysis_result, pdf_record.id
            
        except Exception as e:
            self.db_session.rollback()
            return False, f"Processing error: {str(e)}", None


# Example of how to handle your existing ContractAnalyzer in the new structure

"""
File: utils/contract_analyzer.py (Your existing file - minimal changes needed)

# Your existing ContractAnalyzer class stays mostly the same
class ContractAnalyzer:
    def __init__(self):
        # Your existing initialization
        pass
    
    def analyze_contract(self, contract_text, output_path=None):
        # Your existing analysis logic
        
        # Return results directly instead of always writing to file
        # This makes it easier to integrate with the service layer
        results = {
            "form_number": "extracted_form_number",
            "pi_clause": "yes/no/missing",
            "ci_clause": "yes/no/missing", 
            "data_usage_mentioned": "yes/no/missing",
            "relevant_clauses": [
                {"type": "pi_clause", "text": "clause text"},
                {"type": "ci_clause", "text": "clause text"}
            ],
            "summary": "contract summary"
        }
        
        # If output_path is provided, write to file (backward compatibility)
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            return output_path
        
        # Otherwise return results directly (new usage)
        return results
"""