"""
Content Obfuscation Module

This module provides functionality for obfuscating PDF content to protect privacy
while maintaining document structure for analysis purposes.

Features:
- Page removal based on configurable probability
- Paragraph obfuscation with content-type awareness
- Maintains minimum document length requirements
- Tracks obfuscation statistics for audit purposes
"""

import random
import re
from typing import List, Dict, Tuple, Any
from datetime import datetime


class ContentObfuscator:
    """Handles content obfuscation for privacy protection"""
    
    def __init__(self, 
                 page_removal_probability: float = 0.15,
                 paragraph_obfuscation_probability: float = 0.25,
                 min_pages_to_keep: int = 3,
                 preserve_structure: bool = True):
        """
        Initialize the obfuscator with configurable parameters.
        
        Args:
            page_removal_probability: Probability of removing a page (0.0-1.0)
            paragraph_obfuscation_probability: Probability of obfuscating a paragraph (0.0-1.0)
            min_pages_to_keep: Minimum number of pages to keep in document
            preserve_structure: Whether to preserve document structure during obfuscation
        """
        self.page_removal_prob = page_removal_probability
        self.paragraph_obfuscation_prob = paragraph_obfuscation_probability
        self.min_pages_to_keep = min_pages_to_keep
        self.preserve_structure = preserve_structure
        
        # Obfuscation templates for different content types
        self.obfuscation_templates = {
            'general': [
                "[CONTENT REDACTED FOR PRIVACY]",
                "[CONFIDENTIAL INFORMATION REMOVED]",
                "[SENSITIVE DATA OBFUSCATED]",
                "[PROPRIETARY CONTENT HIDDEN]",
                "[INFORMATION REDACTED]"
            ],
            'financial': [
                "[FINANCIAL DATA REDACTED]",
                "[MONETARY INFORMATION REMOVED]",
                "[PRICING DETAILS OBFUSCATED]",
                "[COST INFORMATION HIDDEN]",
                "[PAYMENT TERMS REDACTED]"
            ],
            'personal': [
                "[PERSONAL INFORMATION REDACTED]",
                "[INDIVIDUAL DATA REMOVED]",
                "[PRIVATE DETAILS OBFUSCATED]",
                "[PERSONAL IDENTIFIERS HIDDEN]",
                "[CONTACT INFORMATION REDACTED]"
            ],
            'technical': [
                "[TECHNICAL SPECIFICATIONS REDACTED]",
                "[IMPLEMENTATION DETAILS REMOVED]",
                "[SYSTEM INFORMATION OBFUSCATED]",
                "[TECHNICAL DATA HIDDEN]",
                "[CONFIGURATION DETAILS REDACTED]"
            ],
            'legal': [
                "[LEGAL TERMS REDACTED]",
                "[CONTRACTUAL DETAILS REMOVED]",
                "[AGREEMENT TERMS OBFUSCATED]",
                "[LEGAL PROVISIONS HIDDEN]",
                "[CLAUSE DETAILS REDACTED]"
            ],
            'dates': [
                "[DATE INFORMATION REDACTED]",
                "[TIMELINE DETAILS REMOVED]",
                "[SCHEDULING INFO OBFUSCATED]",
                "[TEMPORAL DATA HIDDEN]"
            ]
        }
        
        # Content classification keywords
        self.content_keywords = {
            'financial': [
                'price', 'cost', 'payment', 'fee', 'amount', 'dollar', 'currency',
                'invoice', 'billing', 'revenue', 'profit', 'budget', 'expense',
                'salary', 'wage', 'compensation', 'bonus', 'tax', 'interest'
            ],
            'personal': [
                'name', 'address', 'phone', 'email', 'ssn', 'social security',
                'birth', 'age', 'gender', 'race', 'ethnicity', 'citizen',
                'passport', 'id number', 'license', 'personal', 'individual'
            ],
            'technical': [
                'system', 'software', 'hardware', 'database', 'server',
                'network', 'protocol', 'algorithm', 'api', 'integration',
                'code', 'programming', 'development', 'architecture',
                'configuration', 'implementation', 'deployment'
            ],
            'legal': [
                'contract', 'agreement', 'clause', 'terms', 'conditions',
                'liability', 'indemnify', 'breach', 'termination', 'dispute',
                'jurisdiction', 'governing law', 'arbitration', 'mediation',
                'warranty', 'representation', 'covenant', 'obligation'
            ],
            'dates': [
                'date', 'time', 'deadline', 'schedule', 'calendar', 'year',
                'month', 'day', 'week', 'quarter', 'anniversary', 'expiry',
                'effective', 'commence', 'terminate', 'duration', 'period'
            ]
        }
    
    def classify_content_type(self, text: str) -> str:
        """
        Classify content type based on keywords to apply appropriate obfuscation.
        
        Args:
            text: Text content to classify
            
        Returns:
            Content type classification
        """
        if not text:
            return 'general'
            
        text_lower = text.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.content_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return the category with highest score, default to general
        if not category_scores or max(category_scores.values()) == 0:
            return 'general'
        
        return max(category_scores, key=category_scores.get)
    
    def calculate_obfuscation_length(self, original_text: str, target_ratio: float = 0.7) -> int:
        """
        Calculate appropriate length for obfuscated text to maintain document structure.
        
        Args:
            original_text: Original paragraph text
            target_ratio: Target ratio of obfuscated to original length
            
        Returns:
            Target length for obfuscated text
        """
        original_length = len(original_text)
        return max(50, int(original_length * target_ratio))
    
    def obfuscate_paragraph(self, paragraph: str, preserve_length: bool = None) -> str:
        """
        Obfuscate a single paragraph based on its content type.
        
        Args:
            paragraph: Original paragraph text
            preserve_length: Whether to preserve approximate length (overrides class setting)
            
        Returns:
            Obfuscated paragraph text
        """
        if not paragraph.strip():
            return paragraph
        
        preserve_length = preserve_length if preserve_length is not None else self.preserve_structure
        content_type = self.classify_content_type(paragraph)
        templates = self.obfuscation_templates.get(content_type, self.obfuscation_templates['general'])
        
        # Select a random template
        obfuscation_text = random.choice(templates)
        
        if preserve_length:
            target_length = self.calculate_obfuscation_length(paragraph)
            
            # Add filler text if needed to maintain approximate length
            while len(obfuscation_text) < target_length:
                filler_options = [
                    " [Additional content has been removed to protect confidentiality.]",
                    " [Further details have been redacted for privacy.]",
                    " [Supplementary information has been obfuscated.]",
                    " [Extended content has been hidden for security.]"
                ]
                obfuscation_text += random.choice(filler_options)
                
                # Prevent infinite loop
                if len(obfuscation_text) > target_length * 1.5:
                    break
        
        return obfuscation_text
    
    def should_preserve_page(self, page_content: Dict[str, Any], page_index: int, total_pages: int) -> bool:
        """
        Determine if a page should be preserved based on its importance.
        
        Args:
            page_content: Page content dictionary
            page_index: Current page index (0-based)
            total_pages: Total number of pages in document
            
        Returns:
            True if page should be preserved
        """
        # Always preserve first and last pages (they often contain critical metadata)
        if page_index == 0 or page_index == total_pages - 1:
            return True
        
        # Always preserve if removing would violate minimum pages requirement
        if total_pages <= self.min_pages_to_keep:
            return True
        
        # Check for important content indicators
        paragraphs = page_content.get('paragraphs', [])
        page_text = ' '.join(paragraphs).lower()
        
        # Preserve pages with important legal or structural content
        important_indicators = [
            'signature', 'sign', 'agreement', 'contract', 'terms and conditions',
            'effective date', 'termination', 'governing law', 'jurisdiction',
            'definitions', 'whereas', 'witnesseth', 'in witness whereof',
            'schedule', 'exhibit', 'appendix', 'attachment'
        ]
        
        for indicator in important_indicators:
            if indicator in page_text:
                return True
        
        return False
    
    def should_remove_page(self, page_content: Dict[str, Any], total_pages: int, current_index: int) -> bool:
        """
        Determine if a page should be removed based on various factors.
        
        Args:
            page_content: Page content dictionary
            total_pages: Total number of pages in document
            current_index: Current page index (0-based)
            
        Returns:
            True if page should be removed
        """
        # Check if page should be preserved first
        if self.should_preserve_page(page_content, current_index, total_pages):
            return False
        
        # Never remove if it would leave us with too few pages
        pages_that_would_remain = total_pages - 1
        if pages_that_would_remain < self.min_pages_to_keep:
            return False
        
        # Check page content characteristics
        paragraphs = page_content.get('paragraphs', [])
        
        if not paragraphs:
            # More likely to remove empty pages
            return random.random() < (self.page_removal_prob * 1.8)
        
        total_words = sum(len(p.split()) for p in paragraphs)
        
        # Adjust removal probability based on content density
        if total_words < 50:
            # More likely to remove pages with very little content
            adjusted_prob = self.page_removal_prob * 1.5
        elif total_words > 300:
            # Less likely to remove content-heavy pages
            adjusted_prob = self.page_removal_prob * 0.7
        else:
            # Standard probability for normal pages
            adjusted_prob = self.page_removal_prob
        
        return random.random() < adjusted_prob
    
    def should_obfuscate_paragraph(self, paragraph: str, page_context: Dict[str, Any] = None) -> bool:
        """
        Determine if a paragraph should be obfuscated based on its content and context.
        
        Args:
            paragraph: Paragraph text
            page_context: Optional context about the page containing this paragraph
            
        Returns:
            True if paragraph should be obfuscated
        """
        if not paragraph.strip():
            return False
        
        # Check for structural elements that should typically be preserved
        structural_indicators = [
            'section', 'article', 'clause', 'paragraph', 'subsection',
            'whereas', 'witnesseth', 'definitions', 'terms', 'schedule'
        ]
        
        paragraph_lower = paragraph.lower()
        
        # Less likely to obfuscate structural content
        if any(indicator in paragraph_lower for indicator in structural_indicators):
            adjusted_prob = self.paragraph_obfuscation_prob * 0.5
        else:
            adjusted_prob = self.paragraph_obfuscation_prob
        
        return random.random() < adjusted_prob
    
    def obfuscate_content(self, pages_content: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply obfuscation to the entire document content.
        
        Args:
            pages_content: List of page content dictionaries
            
        Returns:
            Tuple of (obfuscated_pages_content, obfuscation_summary)
        """
        if not pages_content:
            return pages_content, self._create_empty_summary()
        
        obfuscated_pages = []
        pages_removed = 0
        paragraphs_obfuscated = 0
        total_original_paragraphs = 0
        total_original_words = 0
        total_final_words = 0
        
        # Calculate original statistics
        for page in pages_content:
            paragraphs = page.get('paragraphs', [])
            total_original_paragraphs += len(paragraphs)
            total_original_words += sum(len(p.split()) for p in paragraphs)
        
        # Process each page
        for page_index, page in enumerate(pages_content):
            # Decide if this page should be removed
            if self.should_remove_page(page, len(pages_content), page_index):
                pages_removed += 1
                continue  # Skip this page entirely
            
            # Process paragraphs in the page
            original_paragraphs = page.get('paragraphs', [])
            obfuscated_paragraphs = []
            
            for paragraph in original_paragraphs:
                if self.should_obfuscate_paragraph(paragraph, page):
                    obfuscated_paragraph = self.obfuscate_paragraph(paragraph)
                    obfuscated_paragraphs.append(obfuscated_paragraph)
                    paragraphs_obfuscated += 1
                    total_final_words += len(obfuscated_paragraph.split())
                else:
                    obfuscated_paragraphs.append(paragraph)
                    total_final_words += len(paragraph.split())
            
            # Create obfuscated page
            obfuscated_page = page.copy()
            obfuscated_page['paragraphs'] = obfuscated_paragraphs
            obfuscated_page['obfuscation_applied'] = paragraphs_obfuscated > 0
            obfuscated_pages.append(obfuscated_page)
        
        # Create comprehensive obfuscation summary
        obfuscation_summary = self._create_obfuscation_summary(
            pages_content, obfuscated_pages, pages_removed, paragraphs_obfuscated,
            total_original_paragraphs, total_original_words, total_final_words
        )
        
        return obfuscated_pages, obfuscation_summary
    
    def _create_obfuscation_summary(self, original_pages: List[Dict[str, Any]], 
                                  obfuscated_pages: List[Dict[str, Any]],
                                  pages_removed: int, paragraphs_obfuscated: int,
                                  total_original_paragraphs: int, 
                                  total_original_words: int, total_final_words: int) -> Dict[str, Any]:
        """Create a comprehensive summary of obfuscation operations."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'obfuscation_applied': True,
            'pages_removed_count': pages_removed,
            'paragraphs_obfuscated_count': paragraphs_obfuscated,
            'total_original_pages': len(original_pages),
            'total_final_pages': len(obfuscated_pages),
            'total_original_paragraphs': total_original_paragraphs,
            'total_final_paragraphs': sum(len(page.get('paragraphs', [])) for page in obfuscated_pages),
            'total_original_words': total_original_words,
            'total_final_words': total_final_words,
            'obfuscation_rate': paragraphs_obfuscated / max(total_original_paragraphs, 1),
            'page_removal_rate': pages_removed / max(len(original_pages), 1),
            'word_retention_rate': total_final_words / max(total_original_words, 1),
            'methods_applied': {
                'page_removal': pages_removed > 0,
                'paragraph_obfuscation': paragraphs_obfuscated > 0,
                'structure_preservation': self.preserve_structure
            },
            'configuration': {
                'page_removal_probability': self.page_removal_prob,
                'paragraph_obfuscation_probability': self.paragraph_obfuscation_prob,
                'min_pages_to_keep': self.min_pages_to_keep,
                'preserve_structure': self.preserve_structure
            }
        }
    
    def _create_empty_summary(self) -> Dict[str, Any]:
        """Create an empty obfuscation summary for when no content is provided."""
        return {
            'timestamp': datetime.now().isoformat(),
            'obfuscation_applied': False,
            'pages_removed_count': 0,
            'paragraphs_obfuscated_count': 0,
            'total_original_pages': 0,
            'total_final_pages': 0,
            'total_original_paragraphs': 0,
            'total_final_paragraphs': 0,
            'total_original_words': 0,
            'total_final_words': 0,
            'obfuscation_rate': 0.0,
            'page_removal_rate': 0.0,
            'word_retention_rate': 0.0,
            'methods_applied': {
                'page_removal': False,
                'paragraph_obfuscation': False,
                'structure_preservation': self.preserve_structure
            },
            'error': 'No content provided for obfuscation'
        }
    
    def get_obfuscation_stats(self, summary: Dict[str, Any]) -> str:
        """
        Get a human-readable string of obfuscation statistics.
        
        Args:
            summary: Obfuscation summary dictionary
            
        Returns:
            Formatted statistics string
        """
        if not summary.get('obfuscation_applied', False):
            return "No obfuscation applied."
        
        stats = [
            f"Pages: {summary['total_original_pages']} → {summary['total_final_pages']} ({summary['pages_removed_count']} removed)",
            f"Paragraphs: {summary['paragraphs_obfuscated_count']}/{summary['total_original_paragraphs']} obfuscated ({summary['obfuscation_rate']:.1%})",
            f"Words: {summary['total_original_words']:,} → {summary['total_final_words']:,} ({summary['word_retention_rate']:.1%} retained)"
        ]
        
        return " | ".join(stats)


# Utility functions for testing and configuration

def create_light_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with light obfuscation settings."""
    return ContentObfuscator(
        page_removal_probability=0.05,
        paragraph_obfuscation_probability=0.15,
        min_pages_to_keep=5,
        preserve_structure=True
    )

def create_moderate_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with moderate obfuscation settings."""
    return ContentObfuscator(
        page_removal_probability=0.15,
        paragraph_obfuscation_probability=0.25,
        min_pages_to_keep=3,
        preserve_structure=True
    )

def create_heavy_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with heavy obfuscation settings."""
    return ContentObfuscator(
        page_removal_probability=0.25,
        paragraph_obfuscation_probability=0.4,
        min_pages_to_keep=2,
        preserve_structure=False
    )

def test_obfuscation():
    """Test function to demonstrate obfuscation capabilities."""
    
    # Sample content for testing
    sample_pages = [
        {
            'page_number': 1,
            'paragraphs': [
                'This is a confidential agreement between parties.',
                'The payment amount shall be $50,000 annually.',
                'Personal information including John Doe\'s address will be protected.'
            ],
            'layout': 'single_column'
        },
        {
            'page_number': 2,
            'paragraphs': [
                'Technical specifications include API integration.',
                'The system shall process data according to protocols.',
                'Database configuration will be managed by IT department.'
            ],
            'layout': 'single_column'
        }
    ]
    
    # Test different obfuscation levels
    obfuscators = {
        'Light': create_light_obfuscator(),
        'Moderate': create_moderate_obfuscator(),
        'Heavy': create_heavy_obfuscator()
    }
    
    print("🧪 Testing Content Obfuscation...")
    print("=" * 50)
    
    for level, obfuscator in obfuscators.items():
        print(f"\n{level} Obfuscation:")
        print("-" * 20)
        
        obfuscated_pages, summary = obfuscator.obfuscate_content(sample_pages)
        
        print(f"Original pages: {len(sample_pages)}, Final pages: {len(obfuscated_pages)}")
        print(f"Statistics: {obfuscator.get_obfuscation_stats(summary)}")
        
        if obfuscated_pages:
            print(f"Sample obfuscated content: {obfuscated_pages[0]['paragraphs'][0][:100]}...")


if __name__ == "__main__":
    test_obfuscation()
