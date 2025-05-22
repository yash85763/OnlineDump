# utils/obfuscation.py - Complete obfuscation system

import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass

@dataclass
class ObfuscationConfig:
    """Configuration for obfuscation settings"""
    
    # Page-level obfuscation (word count based)
    enable_page_filtering: bool = True
    word_count_threshold_multiplier: float = 1.0  # 1.0 = average, 0.8 = 80% of average, etc.
    
    # Paragraph-level obfuscation (keyword based)
    enable_keyword_filtering: bool = True
    obfuscation_keywords: List[str] = None
    keyword_combinations: List[List[str]] = None  # List of keyword combinations (AND logic)
    case_sensitive: bool = False
    whole_words_only: bool = True
    
    # Replacement settings
    page_replacement_text: str = "[PAGE CONTENT REMOVED - Below average word count]"
    paragraph_replacement_text: str = "[PARAGRAPH OBFUSCATED - Contains sensitive keywords]"
    
    # Logging
    enable_logging: bool = True

class ContentObfuscator:
    """
    Handles content obfuscation based on word count and keyword filtering.
    
    Features:
    1. Page-level filtering: Remove pages with word count below average
    2. Paragraph-level filtering: Obfuscate paragraphs containing specific keywords
    3. Keyword combination filtering: Obfuscate paragraphs containing keyword combinations
    4. Configurable replacement text and thresholds
    """
    
    def __init__(self, config: ObfuscationConfig = None):
        self.config = config or ObfuscationConfig()
        
        # Set default keywords if none provided
        if self.config.obfuscation_keywords is None:
            self.config.obfuscation_keywords = [
                # Common sensitive terms - customize as needed
                "confidential", "proprietary", "trade secret", "internal use only",
                "restricted", "classified", "private", "sensitive",
                "ssn", "social security", "tax id", "ein", "account number",
                "credit card", "bank account", "routing number",
                "password", "api key", "token", "secret key"
            ]
        
        # Set default keyword combinations if none provided
        if self.config.keyword_combinations is None:
            self.config.keyword_combinations = [
                ["personal", "information"],
                ["credit", "score"],
                ["financial", "data"],
                ["customer", "data"],
                ["employee", "records"],
                ["medical", "records"],
                ["health", "information"]
            ]
        
        self.obfuscation_stats = {
            "pages_removed": 0,
            "paragraphs_obfuscated": 0,
            "total_pages": 0,
            "total_paragraphs": 0,
            "keyword_matches": [],
            "combination_matches": []
        }
    
    def obfuscate_content(self, pdf_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main obfuscation function that processes PDF content.
        
        Args:
            pdf_result: PDF processing result from PDFHandler
            
        Returns:
            Obfuscated PDF result with statistics
        """
        
        if not pdf_result.get("parsable", False):
            return pdf_result
        
        pages = pdf_result.get("pages", [])
        if not pages:
            return pdf_result
        
        # Reset statistics
        self.obfuscation_stats = {
            "pages_removed": 0,
            "paragraphs_obfuscated": 0,
            "total_pages": len(pages),
            "total_paragraphs": sum(len(page.get("paragraphs", [])) for page in pages),
            "keyword_matches": [],
            "combination_matches": []
        }
        
        # Step 1: Calculate word count statistics
        word_count_stats = self._calculate_word_count_stats(pages)
        
        # Step 2: Apply page-level obfuscation
        if self.config.enable_page_filtering:
            pages = self._apply_page_obfuscation(pages, word_count_stats)
        
        # Step 3: Apply paragraph-level obfuscation
        if self.config.enable_keyword_filtering:
            pages = self._apply_paragraph_obfuscation(pages)
        
        # Update result
        obfuscated_result = pdf_result.copy()
        obfuscated_result["pages"] = pages
        obfuscated_result["obfuscation_applied"] = True
        obfuscated_result["obfuscation_stats"] = self.obfuscation_stats.copy()
        
        if self.config.enable_logging:
            self._log_obfuscation_results()
        
        return obfuscated_result
    
    def _calculate_word_count_stats(self, pages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate word count statistics for pages"""
        
        page_word_counts = []
        
        for page in pages:
            paragraphs = page.get("paragraphs", [])
            page_word_count = sum(len(para.split()) for para in paragraphs)
            page_word_counts.append(page_word_count)
        
        if not page_word_counts:
            return {"average": 0, "min": 0, "max": 0, "threshold": 0}
        
        average_words = sum(page_word_counts) / len(page_word_counts)
        threshold = average_words * self.config.word_count_threshold_multiplier
        
        return {
            "average": average_words,
            "min": min(page_word_counts),
            "max": max(page_word_counts),
            "threshold": threshold,
            "all_counts": page_word_counts
        }
    
    def _apply_page_obfuscation(self, pages: List[Dict[str, Any]], 
                               word_count_stats: Dict[str, float]) -> List[Dict[str, Any]]:
        """Apply page-level obfuscation based on word count"""
        
        threshold = word_count_stats["threshold"]
        obfuscated_pages = []
        
        for page in pages:
            paragraphs = page.get("paragraphs", [])
            page_word_count = sum(len(para.split()) for para in paragraphs)
            
            if page_word_count < threshold:
                # Page below threshold - replace content
                obfuscated_page = page.copy()
                obfuscated_page["paragraphs"] = [self.config.page_replacement_text]
                obfuscated_page["obfuscated"] = True
                obfuscated_page["obfuscation_reason"] = f"Word count {page_word_count} below threshold {threshold:.1f}"
                obfuscated_page["original_word_count"] = page_word_count
                
                obfuscated_pages.append(obfuscated_page)
                self.obfuscation_stats["pages_removed"] += 1
                
                if self.config.enable_logging:
                    print(f"Page {page.get('page_number', 'unknown')} obfuscated: "
                          f"{page_word_count} words < {threshold:.1f} threshold")
            else:
                # Page above threshold - keep as is
                obfuscated_pages.append(page)
        
        return obfuscated_pages
    
    def _apply_paragraph_obfuscation(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply paragraph-level obfuscation based on keywords"""
        
        obfuscated_pages = []
        
        for page in pages:
            # Skip already obfuscated pages
            if page.get("obfuscated", False):
                obfuscated_pages.append(page)
                continue
            
            paragraphs = page.get("paragraphs", [])
            obfuscated_paragraphs = []
            
            for para in paragraphs:
                if self._should_obfuscate_paragraph(para):
                    obfuscated_paragraphs.append(self.config.paragraph_replacement_text)
                    self.obfuscation_stats["paragraphs_obfuscated"] += 1
                else:
                    obfuscated_paragraphs.append(para)
            
            # Create new page with obfuscated paragraphs
            obfuscated_page = page.copy()
            obfuscated_page["paragraphs"] = obfuscated_paragraphs
            obfuscated_pages.append(obfuscated_page)
        
        return obfuscated_pages
    
    def _should_obfuscate_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph should be obfuscated based on keywords"""
        
        if not paragraph or not paragraph.strip():
            return False
        
        # Prepare text for matching
        text = paragraph if self.config.case_sensitive else paragraph.lower()
        
        # Check individual keywords
        for keyword in self.config.obfuscation_keywords:
            search_keyword = keyword if self.config.case_sensitive else keyword.lower()
            
            if self.config.whole_words_only:
                # Use word boundaries to match whole words only
                pattern = r'\b' + re.escape(search_keyword) + r'\b'
                if re.search(pattern, text):
                    self.obfuscation_stats["keyword_matches"].append({
                        "keyword": keyword,
                        "paragraph_preview": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                    })
                    return True
            else:
                # Simple substring match
                if search_keyword in text:
                    self.obfuscation_stats["keyword_matches"].append({
                        "keyword": keyword,
                        "paragraph_preview": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                    })
                    return True
        
        # Check keyword combinations (all keywords in combination must be present)
        for combination in self.config.keyword_combinations:
            if self._check_keyword_combination(text, combination):
                self.obfuscation_stats["combination_matches"].append({
                    "combination": combination,
                    "paragraph_preview": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
                })
                return True
        
        return False
    
    def _check_keyword_combination(self, text: str, combination: List[str]) -> bool:
        """Check if all keywords in combination are present in text"""
        
        for keyword in combination:
            search_keyword = keyword if self.config.case_sensitive else keyword.lower()
            
            if self.config.whole_words_only:
                pattern = r'\b' + re.escape(search_keyword) + r'\b'
                if not re.search(pattern, text):
                    return False
            else:
                if search_keyword not in text:
                    return False
        
        return True
    
    def _log_obfuscation_results(self):
        """Log obfuscation results"""
        
        stats = self.obfuscation_stats
        print(f"\n=== OBFUSCATION RESULTS ===")
        print(f"Total pages: {stats['total_pages']}")
        print(f"Pages removed: {stats['pages_removed']}")
        print(f"Total paragraphs: {stats['total_paragraphs']}")
        print(f"Paragraphs obfuscated: {stats['paragraphs_obfuscated']}")
        
        if stats['keyword_matches']:
            print(f"\nKeyword matches found: {len(stats['keyword_matches'])}")
            for match in stats['keyword_matches'][:5]:  # Show first 5
                print(f"  - '{match['keyword']}' in: {match['paragraph_preview']}")
        
        if stats['combination_matches']:
            print(f"\nCombination matches found: {len(stats['combination_matches'])}")
            for match in stats['combination_matches'][:3]:  # Show first 3
                print(f"  - {match['combination']} in: {match['paragraph_preview']}")
    
    def get_obfuscation_report(self) -> Dict[str, Any]:
        """Get detailed obfuscation report"""
        
        return {
            "summary": {
                "total_pages": self.obfuscation_stats["total_pages"],
                "pages_removed": self.obfuscation_stats["pages_removed"],
                "total_paragraphs": self.obfuscation_stats["total_paragraphs"],
                "paragraphs_obfuscated": self.obfuscation_stats["paragraphs_obfuscated"],
                "page_removal_rate": (self.obfuscation_stats["pages_removed"] / 
                                    max(1, self.obfuscation_stats["total_pages"])) * 100,
                "paragraph_obfuscation_rate": (self.obfuscation_stats["paragraphs_obfuscated"] / 
                                             max(1, self.obfuscation_stats["total_paragraphs"])) * 100
            },
            "keyword_matches": self.obfuscation_stats["keyword_matches"],
            "combination_matches": self.obfuscation_stats["combination_matches"],
            "config": {
                "word_count_threshold_multiplier": self.config.word_count_threshold_multiplier,
                "keywords_count": len(self.config.obfuscation_keywords),
                "combinations_count": len(self.config.keyword_combinations),
                "case_sensitive": self.config.case_sensitive,
                "whole_words_only": self.config.whole_words_only
            }
        }

def create_default_obfuscation_config() -> ObfuscationConfig:
    """Create default obfuscation configuration"""
    
    return ObfuscationConfig(
        enable_page_filtering=True,
        word_count_threshold_multiplier=1.0,  # Pages with less than average word count
        enable_keyword_filtering=True,
        case_sensitive=False,
        whole_words_only=True,
        obfuscation_keywords=[
            # Personal Information
            "ssn", "social security number", "social security", 
            "tax id", "taxpayer id", "ein", "employee id",
            "driver license", "passport number", "account number",
            "credit card", "debit card", "bank account", "routing number",
            
            # Confidential/Proprietary
            "confidential", "proprietary", "trade secret", "internal use only",
            "restricted", "classified", "private", "sensitive",
            "confidential information", "proprietary information",
            
            # Security
            "password", "api key", "access key", "secret key", "token",
            "authentication", "authorization", "login credentials",
            
            # Medical/Health
            "medical record", "health information", "diagnosis", "treatment",
            "hipaa", "phi", "protected health information",
            
            # Financial
            "salary", "compensation", "financial data", "revenue", "profit",
            "budget", "cost", "pricing", "financial information",
            
            # Legal
            "attorney-client", "privileged", "legal advice", "settlement",
            "litigation", "lawsuit", "court", "legal counsel"
        ],
        keyword_combinations=[
            ["personal", "information"],
            ["personally", "identifiable"],
            ["credit", "score"],
            ["financial", "data"],
            ["customer", "data"],
            ["employee", "records"],
            ["medical", "records"],
            ["health", "information"],
            ["confidential", "data"],
            ["trade", "secret"],
            ["proprietary", "technology"],
            ["internal", "document"],
            ["restricted", "access"],
            ["sensitive", "information"]
        ]
    )

def create_custom_obfuscation_config(
    page_threshold_multiplier: float = 1.0,
    custom_keywords: List[str] = None,
    custom_combinations: List[List[str]] = None,
    case_sensitive: bool = False
) -> ObfuscationConfig:
    """
    Create custom obfuscation configuration.
    
    Args:
        page_threshold_multiplier: Multiplier for average word count threshold
        custom_keywords: List of custom keywords to obfuscate
        custom_combinations: List of keyword combinations
        case_sensitive: Whether keyword matching is case sensitive
        
    Returns:
        Custom ObfuscationConfig
    """
    
    config = create_default_obfuscation_config()
    config.word_count_threshold_multiplier = page_threshold_multiplier
    config.case_sensitive = case_sensitive
    
    if custom_keywords:
        config.obfuscation_keywords.extend(custom_keywords)
    
    if custom_combinations:
        config.keyword_combinations.extend(custom_combinations)
    
    return config