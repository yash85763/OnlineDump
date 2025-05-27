"""
Content Obfuscation Module

This module provides functionality for obfuscating PDF content to protect privacy
while maintaining document structure for analysis purposes.

Features:
- Average word count based page removal (primary method)
- Probability-based page removal (legacy method)
- Hybrid approach combining both methods
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
                 obfuscation_method: str = "average_word_count",
                 page_removal_probability: float = 0.15,
                 paragraph_obfuscation_probability: float = 0.25,
                 min_pages_to_keep: int = 3,
                 preserve_structure: bool = True,
                 word_count_threshold_multiplier: float = 1.0):
        """
        Initialize the obfuscator with configurable parameters.
        
        Args:
            obfuscation_method: Method to use for obfuscation ("average_word_count", "probability", "hybrid")
            page_removal_probability: Probability of removing a page (0.0-1.0) - used in probability method
            paragraph_obfuscation_probability: Probability of obfuscating a paragraph (0.0-1.0)
            min_pages_to_keep: Minimum number of pages to keep in document
            preserve_structure: Whether to preserve document structure during obfuscation
            word_count_threshold_multiplier: Multiplier for average word count threshold (1.0 = exactly average)
        """
        self.obfuscation_method = obfuscation_method
        self.page_removal_prob = page_removal_probability
        self.paragraph_obfuscation_prob = paragraph_obfuscation_probability
        self.min_pages_to_keep = min_pages_to_keep
        self.preserve_structure = preserve_structure
        self.word_count_threshold_multiplier = word_count_threshold_multiplier
        
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
    
    def calculate_page_word_count(self, page_content: Dict[str, Any]) -> int:
        """
        Calculate the word count for a single page.
        
        Args:
            page_content: Page content dictionary
            
        Returns:
            Word count for the page
        """
        paragraphs = page_content.get('paragraphs', [])
        return sum(len(paragraph.split()) for paragraph in paragraphs)
    
    def calculate_average_word_count(self, pages_content: List[Dict[str, Any]]) -> float:
        """
        Calculate the average word count per page across all pages.
        
        Args:
            pages_content: List of page content dictionaries
            
        Returns:
            Average word count per page
        """
        if not pages_content:
            return 0.0
        
        total_words = 0
        for page in pages_content:
            total_words += self.calculate_page_word_count(page)
        
        return total_words / len(pages_content)
    
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
    
    def should_remove_page_by_word_count(self, page_content: Dict[str, Any], 
                                       average_word_count: float, 
                                       page_index: int, total_pages: int) -> bool:
        """
        Determine if a page should be removed based on word count compared to average.
        
        Args:
            page_content: Page content dictionary
            average_word_count: Average word count per page for the document
            page_index: Current page index (0-based)
            total_pages: Total number of pages in document
            
        Returns:
            True if page should be removed
        """
        # Check if page should be preserved first (critical pages)
        if self.should_preserve_page(page_content, page_index, total_pages):
            return False
        
        # Never remove if it would leave us with too few pages
        if total_pages <= self.min_pages_to_keep:
            return False
        
        # Calculate page word count
        page_word_count = self.calculate_page_word_count(page_content)
        
        # Calculate threshold (average * multiplier)
        threshold = average_word_count * self.word_count_threshold_multiplier
        
        # Remove page if word count is below threshold
        return page_word_count < threshold
    
    def should_remove_page_by_probability(self, page_content: Dict[str, Any], total_pages: int, current_index: int) -> bool:
        """
        Determine if a page should be removed based on probability (original method).
        
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
    
    def should_remove_page(self, page_content: Dict[str, Any], total_pages: int, current_index: int, 
                         average_word_count: float = None) -> bool:
        """
        Determine if a page should be removed based on the selected obfuscation method.
        
        Args:
            page_content: Page content dictionary
            total_pages: Total number of pages in document
            current_index: Current page index (0-based)
            average_word_count: Average word count per page (required for average_word_count method)
            
        Returns:
            True if page should be removed
        """
        if self.obfuscation_method == "average_word_count":
            if average_word_count is None:
                raise ValueError("average_word_count is required when using average_word_count method")
            return self.should_remove_page_by_word_count(
                page_content, average_word_count, current_index, total_pages
            )
        
        elif self.obfuscation_method == "probability":
            return self.should_remove_page_by_probability(page_content, total_pages, current_index)
        
        elif self.obfuscation_method == "hybrid":
            # Use both methods - remove if either condition is met
            word_count_removal = False
            if average_word_count is not None:
                word_count_removal = self.should_remove_page_by_word_count(
                    page_content, average_word_count, current_index, total_pages
                )
            
            probability_removal = self.should_remove_page_by_probability(
                page_content, total_pages, current_index
            )
            
            return word_count_removal or probability_removal
        
        else:
            raise ValueError(f"Unknown obfuscation method: {self.obfuscation_method}")
    
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
        
        # Calculate average word count for word count-based methods
        average_word_count = None
        if self.obfuscation_method in ["average_word_count", "hybrid"]:
            average_word_count = self.calculate_average_word_count(pages_content)
        
        obfuscated_pages = []
        pages_removed = 0
        paragraphs_obfuscated = 0
        total_original_paragraphs = 0
        total_original_words = 0
        total_final_words = 0
        
        # Store page word counts for summary
        page_word_counts = []
        removed_page_word_counts = []
        
        # Calculate original statistics
        for page in pages_content:
            paragraphs = page.get('paragraphs', [])
            page_word_count = sum(len(p.split()) for p in paragraphs)
            page_word_counts.append(page_word_count)
            total_original_paragraphs += len(paragraphs)
            total_original_words += page_word_count
        
        # Process each page
        for page_index, page in enumerate(pages_content):
            # Decide if this page should be removed
            should_remove = self.should_remove_page(
                page, len(pages_content), page_index, average_word_count
            )
            
            if should_remove:
                pages_removed += 1
                removed_page_word_counts.append(page_word_counts[page_index])
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
            total_original_paragraphs, total_original_words, total_final_words,
            average_word_count, page_word_counts, removed_page_word_counts
        )
        
        return obfuscated_pages, obfuscation_summary
    
    def _create_obfuscation_summary(self, original_pages: List[Dict[str, Any]], 
                                  obfuscated_pages: List[Dict[str, Any]],
                                  pages_removed: int, paragraphs_obfuscated: int,
                                  total_original_paragraphs: int, 
                                  total_original_words: int, total_final_words: int,
                                  average_word_count: float = None,
                                  page_word_counts: List[int] = None,
                                  removed_page_word_counts: List[int] = None) -> Dict[str, Any]:
        """Create a comprehensive summary of obfuscation operations."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'obfuscation_applied': True,
            'obfuscation_method': self.obfuscation_method,
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
                'obfuscation_method': self.obfuscation_method,
                'page_removal_probability': self.page_removal_prob,
                'paragraph_obfuscation_probability': self.paragraph_obfuscation_prob,
                'min_pages_to_keep': self.min_pages_to_keep,
                'preserve_structure': self.preserve_structure,
                'word_count_threshold_multiplier': self.word_count_threshold_multiplier
            }
        }
        
        # Add word count specific information if using word count method
        if self.obfuscation_method in ["average_word_count", "hybrid"] and average_word_count is not None:
            summary['word_count_analysis'] = {
                'average_word_count_per_page': average_word_count,
                'word_count_threshold': average_word_count * self.word_count_threshold_multiplier,
                'min_page_word_count': min(page_word_counts) if page_word_counts else 0,
                'max_page_word_count': max(page_word_counts) if page_word_counts else 0,
                'removed_pages_word_counts': removed_page_word_counts or [],
                'avg_removed_page_word_count': sum(removed_page_word_counts) / max(len(removed_page_word_counts), 1) if removed_page_word_counts else 0
            }
        
        return summary
    
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

def create_average_word_count_obfuscator(word_count_threshold_multiplier: float = 1.0) -> ContentObfuscator:
    """Create an obfuscator that uses average word count method."""
    return ContentObfuscator(
        obfuscation_method="average_word_count",
        min_pages_to_keep=3,
        preserve_structure=True,
        word_count_threshold_multiplier=word_count_threshold_multiplier
    )

def create_light_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with light obfuscation settings using probability method."""
    return ContentObfuscator(
        obfuscation_method="probability",
        page_removal_probability=0.05,
        paragraph_obfuscation_probability=0.15,
        min_pages_to_keep=5,
        preserve_structure=True
    )

def create_moderate_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with moderate obfuscation settings using probability method."""
    return ContentObfuscator(
        obfuscation_method="probability",
        page_removal_probability=0.15,
        paragraph_obfuscation_probability=0.25,
        min_pages_to_keep=3,
        preserve_structure=True
    )

def create_heavy_obfuscator() -> ContentObfuscator:
    """Create an obfuscator with heavy obfuscation settings using probability method."""
    return ContentObfuscator(
        obfuscation_method="probability",
        page_removal_probability=0.25,
        paragraph_obfuscation_probability=0.4,
        min_pages_to_keep=2,
        preserve_structure=False
    )

def create_hybrid_obfuscator(word_count_threshold_multiplier: float = 1.0) -> ContentObfuscator:
    """Create an obfuscator that uses both average word count and probability methods."""
    return ContentObfuscator(
        obfuscation_method="hybrid",
        page_removal_probability=0.15,
        paragraph_obfuscation_probability=0.25,
        min_pages_to_keep=3,
        preserve_structure=True,
        word_count_threshold_multiplier=word_count_threshold_multiplier
    )

def test_obfuscation():
    """Test function to demonstrate obfuscation capabilities."""
    
    # Sample content for testing with varying word counts
    sample_pages = [
        {
            'page_number': 1,
            'paragraphs': [
                'This is a confidential agreement between parties involving multiple stakeholders and comprehensive terms.',
                'The payment amount shall be $50,000 annually with quarterly reviews and adjustments.',
                'Personal information including John Doe\'s address, contact details, and identification will be protected under strict confidentiality measures.'
            ],
            'layout': 'single_column'
        },
        {
            'page_number': 2,
            'paragraphs': [
                'Short page.'  # This page has very few words
            ],
            'layout': 'single_column'
        },
        {
            'page_number': 3,
            'paragraphs': [
                'Technical specifications include comprehensive API integration with multiple endpoints.',
                'The system shall process data according to established protocols and security standards.',
                'Database configuration will be managed by IT department with regular backups and monitoring.'
            ],
            'layout': 'single_column'
        },
        {
            'page_number': 4,
            'paragraphs': [
                'Brief content here.'  # Another short page
            ],
            'layout': 'single_column'
        }
    ]
    
    # Test different obfuscation methods
    obfuscators = {
        'Average Word Count (1.0x)': create_average_word_count_obfuscator(1.0),
        'Average Word Count (0.8x)': create_average_word_count_obfuscator(0.8),
        'Average Word Count (0.5x)': create_average_word_count_obfuscator(0.5),
        'Probability (Light)': create_light_obfuscator(),
        'Probability (Moderate)': create_moderate_obfuscator(),
        'Hybrid': create_hybrid_obfuscator(0.8)
    }
    
    print("🧪 Testing Content Obfuscation with Different Methods...")
    print("=" * 60)
    
    # Calculate and display page word counts first
    print("\n📊 Original Document Analysis:")
    print("-" * 30)
    total_words = 0
    for i, page in enumerate(sample_pages):
        page_words = sum(len(p.split()) for p in page['paragraphs'])
        total_words += page_words
        print(f"Page {i+1}: {page_words} words")
    
    avg_words = total_words / len(sample_pages)
    print(f"Average words per page: {avg_words:.1f}")
    print(f"Total pages: {len(sample_pages)}")
    print(f"Total words: {total_words}")
    
    for level, obfuscator in obfuscators.items():
        print(f"\n{level}:")
        print("-" * (len(level) + 1))
        
        obfuscated_pages, summary = obfuscator.obfuscate_content(sample_pages)
        
        print(f"Method: {summary.get('obfuscation_method', 'N/A')}")
        print(f"Original pages: {len(sample_pages)}, Final pages: {len(obfuscated_pages)}")
        print(f"Pages removed: {summary.get('pages_removed_count', 0)}")
        print(f"Statistics: {obfuscator.get_obfuscation_stats(summary)}")
        
        # Show word count analysis if available
        if 'word_count_analysis' in summary:
            wc_analysis = summary['word_count_analysis']
            print(f"Word count threshold: {wc_analysis['word_count_threshold']:.1f}")
            print(f"Removed pages had word counts: {wc_analysis['removed_pages_word_counts']}")
        
        if obfuscated_pages:
            print(f"Sample remaining content: {obfuscated_pages[0]['paragraphs'][0][:80]}...")


if __name__ == "__main__":
    test_obfuscation()
