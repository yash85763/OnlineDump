"""
Content Obfuscation Module

This module provides functionality for obfuscating PDF content to protect privacy
while maintaining document structure for analysis purposes.

Primary Feature:
- Average word count based page removal
- Pages with word count below document average are removed
- No paragraph obfuscation is applied to remaining pages
- Tracks obfuscation statistics for audit purposes
"""

import random
import re
from typing import List, Dict, Tuple, Any
from datetime import datetime


class ContentObfuscator:
    """Handles content obfuscation for privacy protection using average word count method"""
    
    def __init__(self, 
                 min_pages_to_keep: int = 3,
                 word_count_threshold_multiplier: float = 1.0):
        """
        Initialize the obfuscator with configurable parameters.
        
        Args:
            min_pages_to_keep: Minimum number of pages to keep in document
            word_count_threshold_multiplier: Multiplier for average word count threshold (1.0 = exactly average)
        """
        self.min_pages_to_keep = min_pages_to_keep
        self.word_count_threshold_multiplier = word_count_threshold_multiplier
    
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
    
    def should_remove_page(self, page_content: Dict[str, Any], total_pages: int, current_index: int, 
                         average_word_count: float) -> bool:
        """
        Determine if a page should be removed based on word count compared to average.
        
        Args:
            page_content: Page content dictionary
            total_pages: Total number of pages in document
            current_index: Current page index (0-based)
            average_word_count: Average word count per page for the document
            
        Returns:
            True if page should be removed
        """
        # Check if page should be preserved first (critical pages)
        if self.should_preserve_page(page_content, current_index, total_pages):
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
    
    def obfuscate_content(self, pages_content: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply obfuscation to the entire document content using average word count method.
        
        Note: When using average word count method, only page removal is applied.
        No paragraph obfuscation is performed on the remaining pages.
        
        Args:
            pages_content: List of page content dictionaries
            
        Returns:
            Tuple of (remaining_pages_after_removal, obfuscation_summary)
            Note: Returns only the pages that remain after page removal (no paragraph changes)
        """
        if not pages_content:
            return pages_content, self._create_empty_summary()
        
        # Calculate average word count
        average_word_count = self.calculate_average_word_count(pages_content)
        
        remaining_pages = []  # Pages that remain after removal
        pages_removed = 0
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
            # Decide if this page should be removed based on average word count
            should_remove = self.should_remove_page(
                page, len(pages_content), page_index, average_word_count
            )
            
            if should_remove:
                pages_removed += 1
                removed_page_word_counts.append(page_word_counts[page_index])
                continue  # Skip this page entirely - it's removed from the final document
            
            # Get original paragraphs from the page
            original_paragraphs = page.get('paragraphs', [])
            
            # Create the final version of this page (no paragraph obfuscation)
            final_page = page.copy()
            final_page['paragraphs'] = original_paragraphs  # Keep original paragraphs unchanged
            final_page['obfuscation_applied'] = False  # No paragraph obfuscation applied
            final_page['page_removed'] = False  # This page was kept
            remaining_pages.append(final_page)
            
            # Count words in original paragraphs for final statistics
            for paragraph in original_paragraphs:
                total_final_words += len(paragraph.split())
        
        # Create comprehensive obfuscation summary
        obfuscation_summary = self._create_obfuscation_summary(
            pages_content, remaining_pages, pages_removed, 
            total_original_paragraphs, total_original_words, total_final_words,
            average_word_count, page_word_counts, removed_page_word_counts
        )
        
        return remaining_pages, obfuscation_summary
    
    def _create_obfuscation_summary(self, original_pages: List[Dict[str, Any]], 
                                  remaining_pages: List[Dict[str, Any]],
                                  pages_removed: int, 
                                  total_original_paragraphs: int, 
                                  total_original_words: int, total_final_words: int,
                                  average_word_count: float,
                                  page_word_counts: List[int],
                                  removed_page_word_counts: List[int]) -> Dict[str, Any]:
        """Create a comprehensive summary of obfuscation operations."""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'obfuscation_applied': True,
            'obfuscation_method': 'average_word_count',
            'pages_removed_count': pages_removed,
            'paragraphs_obfuscated_count': 0,  # No paragraph obfuscation applied
            'total_original_pages': len(original_pages),
            'total_final_pages': len(remaining_pages),  # Pages that remain after removal
            'total_original_paragraphs': total_original_paragraphs,
            'total_final_paragraphs': sum(len(page.get('paragraphs', [])) for page in remaining_pages),
            'total_original_words': total_original_words,
            'total_final_words': total_final_words,
            'obfuscation_rate': 0.0,  # No paragraph obfuscation
            'page_removal_rate': pages_removed / max(len(original_pages), 1),
            'word_retention_rate': total_final_words / max(total_original_words, 1),
            'methods_applied': {
                'page_removal': pages_removed > 0,
                'paragraph_obfuscation': False,  # Not applied
                'structure_preservation': True
            },
            'configuration': {
                'obfuscation_method': 'average_word_count',
                'min_pages_to_keep': self.min_pages_to_keep,
                'word_count_threshold_multiplier': self.word_count_threshold_multiplier
            }
        }
        
        # Add word count analysis information
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
                'paragraph_obfuscation': False
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
            f"Words: {summary['total_original_words']:,} → {summary['total_final_words']:,} ({summary['word_retention_rate']:.1%} retained)"
        ]
        
        return " | ".join(stats)


# Utility functions for configuration

def create_average_word_count_obfuscator(word_count_threshold_multiplier: float = 1.0) -> ContentObfuscator:
    """
    Create an obfuscator that uses average word count method.
    
    Args:
        word_count_threshold_multiplier: Multiplier for average word count threshold
    
    Returns:
        ContentObfuscator configured for average word count method
    """
    return ContentObfuscator(
        min_pages_to_keep=3,
        word_count_threshold_multiplier=word_count_threshold_multiplier
    )


def test_obfuscation():
    """Test function to demonstrate average word count obfuscation capabilities."""
    
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
    
    # Test different threshold multipliers for average word count method
    test_configs = {
        'Strict (1.0x average)': create_average_word_count_obfuscator(1.0),
        'Moderate (0.8x average)': create_average_word_count_obfuscator(0.8),
        'Lenient (0.5x average)': create_average_word_count_obfuscator(0.5),
        'Very Lenient (0.3x average)': create_average_word_count_obfuscator(0.3)
    }
    
    print("🧪 Testing Average Word Count Obfuscation...")
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
    
    for config_name, obfuscator in test_configs.items():
        print(f"\n{config_name}:")
        print("-" * (len(config_name) + 1))
        
        remaining_pages, summary = obfuscator.obfuscate_content(sample_pages)
        
        print(f"Original pages: {len(sample_pages)}, Remaining pages: {len(remaining_pages)}")
        print(f"Pages removed: {summary.get('pages_removed_count', 0)}")
        print(f"Statistics: {obfuscator.get_obfuscation_stats(summary)}")
        
        # Show word count analysis
        if 'word_count_analysis' in summary:
            wc_analysis = summary['word_count_analysis']
            print(f"Word count threshold: {wc_analysis['word_count_threshold']:.1f}")
            print(f"Removed pages had word counts: {wc_analysis['removed_pages_word_counts']}")
        
        # Show which pages remain
        remaining_page_numbers = [page.get('page_number', 'Unknown') for page in remaining_pages]
        print(f"Remaining page numbers: {remaining_page_numbers}")
        
        if remaining_pages:
            print(f"Sample remaining content: {remaining_pages[0]['paragraphs'][0][:80]}...")


if __name__ == "__main__":
    test_obfuscation()
