import pdfplumber
import re
import json
import argparse
import os
from typing import List, Dict, Optional, Tuple, Any
import statistics
from collections import defaultdict


class ContractParser:
    def __init__(self, pdf_path: str, header_height_percent: float = 0.1, footer_height_percent: float = 0.1):
        """
        Initialize the contract parser
        
        Args:
            pdf_path: Path to the PDF file
            header_height_percent: Percentage of page height to consider as header (0.0-1.0)
            footer_height_percent: Percentage of page height to consider as footer (0.0-1.0)
        """
        self.pdf_path = pdf_path
        self.header_height_percent = header_height_percent
        self.footer_height_percent = footer_height_percent
        self.pdf = None
        self.column_boundaries = []
        self.heading_patterns = [
            # Common heading patterns in contracts
            r'^(?:\d+\.)+\s+',          # Numbered headings like "1.2.3 "
            r'^(?:[A-Z]\.)+\s+',        # Lettered headings like "A.B. "
            r'^(?:[ivxlcdm]+\.)\s+',    # Roman numeral headings like "iv. "
            r'^(?:[IVXLCDM]+\.)\s+',    # Upper Roman numeral headings like "IV. "
            r'^(?:\d+\))\s+',           # Numbered parenthesis like "1) "
            r'^(?:[a-z]\))\s+',         # Lettered parenthesis like "a) "
            r'^ARTICLE\s+\d+',          # "ARTICLE X"
            r'^Section\s+\d+',          # "Section X"
            r'^SECTION\s+\d+',          # "SECTION X"
            r'^EXHIBIT\s+[A-Z]',        # "EXHIBIT X"
            r'^SCHEDULE\s+[A-Z\d]',     # "SCHEDULE X"
            r'^APPENDIX\s+[A-Z\d]',     # "APPENDIX X"
        ]
        self.font_stats = {}
        self.hierarchical_structure = []

    def open_pdf(self):
        """Open the PDF file with pdfplumber"""
        self.pdf = pdfplumber.open(self.pdf_path)
        print(f"Opened PDF: {self.pdf_path} ({len(self.pdf.pages)} pages)")

    def close_pdf(self):
        """Close the PDF file"""
        if self.pdf:
            self.pdf.close()

    def detect_columns(self, page) -> List[Tuple[float, float]]:
        """
        Detect columns in a page based on text distribution
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List of (start_x, end_x) tuples for each column
        """
        # Extract words with their bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        if not words:
            return [(0, page.width)]
        
        # Analyze x-coordinates distribution
        x_starts = [word["x0"] for word in words]
        x_ends = [word["x1"] for word in words]
        
        # Create x-density histogram
        hist_buckets = 100
        bucket_size = page.width / hist_buckets
        density = [0] * hist_buckets
        
        for word in words:
            start_bucket = int(word["x0"] / bucket_size)
            end_bucket = min(int(word["x1"] / bucket_size), hist_buckets - 1)
            for i in range(start_bucket, end_bucket + 1):
                density[i] += 1
        
        # Find gaps (potential column separators)
        gaps = []
        in_gap = False
        gap_start = 0
        min_gap_width = page.width * 0.05  # 5% of page width
        
        for i in range(hist_buckets):
            if density[i] == 0 and not in_gap:
                in_gap = True
                gap_start = i * bucket_size
            elif density[i] > 0 and in_gap:
                in_gap = False
                gap_end = i * bucket_size
                if gap_end - gap_start >= min_gap_width:
                    gaps.append((gap_start, gap_end))
        
        if not gaps:
            # No gaps found, assume single column
            return [(0, page.width)]
        elif len(gaps) == 1:
            # One gap found, assume two columns
            return [(0, gaps[0][0]), (gaps[0][1], page.width)]
        else:
            # Multiple gaps, use the most significant one
            widest_gap = max(gaps, key=lambda g: g[1] - g[0])
            return [(0, widest_gap[0]), (widest_gap[1], page.width)]

    def get_content_area(self, page) -> Tuple[float, float, float, float]:
        """
        Get content area excluding headers and footers
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Tuple of (x0, y0, x1, y1) for content area
        """
        header_height = page.height * self.header_height_percent
        footer_start = page.height * (1 - self.footer_height_percent)
        
        return (0, header_height, page.width, footer_start)

    def extract_text_with_properties(self, page) -> List[Dict]:
        """
        Extract text with font, size and position properties
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List of dictionaries with text and its properties
        """
        content_area = self.get_content_area(page)
        column_boundaries = self.detect_columns(page)
        
        # Extract characters with their properties
        chars = page.chars
        if not chars:
            return []
        
        # Filter out characters in header/footer
        content_chars = [
            c for c in chars
            if c['y0'] >= content_area[1] and c['y1'] <= content_area[3]
        ]
        
        if not content_chars:
            return []
        
        # Group characters into lines
        y_tolerance = 3  # Adjust based on line spacing
        lines = defaultdict(list)
        
        for char in content_chars:
            # Round y-coordinate to group characters on the same line
            y_key = round(char['y0'] / y_tolerance) * y_tolerance
            lines[y_key].append(char)
        
        # Sort lines by y-coordinate
        sorted_lines = sorted(lines.items(), key=lambda x: x[0])
        
        # Group lines into text blocks with properties
        text_blocks = []
        current_block = None
        
        for y, line_chars in sorted_lines:
            # Sort characters by x-coordinate
            line_chars.sort(key=lambda c: c['x0'])
            
            # Assign characters to columns
            col_chars = [[] for _ in range(len(column_boundaries))]
            
            for char in line_chars:
                for col_idx, (col_start, col_end) in enumerate(column_boundaries):
                    if col_start <= char['x0'] < col_end:
                        col_chars[col_idx].append(char)
                        break
            
            # Process each column's characters
            for col_idx, chars in enumerate(col_chars):
                if not chars:
                    continue
                
                # Extract properties from the first character
                first_char = chars[0]
                col_start, col_end = column_boundaries[col_idx]
                
                # Build text and collect font statistics
                text = ''.join(c['text'] for c in chars)
                if not text.strip():
                    continue
                
                # Get font properties
                font_name = first_char.get('fontname', '')
                font_size = first_char.get('size', 0)
                
                # Update font statistics
                if font_name not in self.font_stats:
                    self.font_stats[font_name] = {'sizes': [], 'count': 0}
                self.font_stats[font_name]['sizes'].append(font_size)
                self.font_stats[font_name]['count'] += 1
                
                # Create text block
                text_block = {
                    'text': text,
                    'page': page.page_number,
                    'column': col_idx,
                    'y0': min(c['y0'] for c in chars),
                    'y1': max(c['y1'] for c in chars),
                    'x0': min(c['x0'] for c in chars),
                    'x1': max(c['x1'] for c in chars),
                    'font_name': font_name,
                    'font_size': font_size,
                    'bold': 'bold' in font_name.lower() or 'heavy' in font_name.lower(),
                    'italic': 'italic' in font_name.lower() or 'oblique' in font_name.lower(),
                    'margin_left': min(c['x0'] for c in chars) - col_start,
                }
                
                text_blocks.append(text_block)
        
        return text_blocks

    def analyze_font_statistics(self):
        """
        Analyze font statistics to help identify headings
        
        Returns:
            Dictionary with font statistics
        """
        print("Analyzing font statistics...")
        
        for font_name, stats in self.font_stats.items():
            if stats['sizes']:
                stats['median_size'] = statistics.median(stats['sizes'])
                stats['mode_size'] = statistics.mode(stats['sizes'])
                stats['max_size'] = max(stats['sizes'])
                stats['min_size'] = min(stats['sizes'])
                stats['avg_size'] = sum(stats['sizes']) / len(stats['sizes'])
        
        # Find most common font (likely body text)
        body_font = max(self.font_stats.items(), key=lambda x: x[1]['count'])
        self.body_font_name = body_font[0]
        self.body_font_size = body_font[1]['mode_size']
        
        print(f"Identified body font: {self.body_font_name}, size: {self.body_font_size}")
        return self.font_stats

    def is_heading(self, text_block: Dict) -> bool:
        """
        Check if a text block is likely a heading
        
        Args:
            text_block: Text block dictionary
            
        Returns:
            True if the text block is likely a heading, False otherwise
        """
        text = text_block['text'].strip()
        
        # Empty text is not a heading
        if not text:
            return False
        
        # Check against heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check font properties
        font_size = text_block['font_size']
        font_name = text_block['font_name']
        is_bold = text_block['bold']
        
        # If font is significantly larger than body text, it's likely a heading
        if font_size > self.body_font_size * 1.1:
            return True
        
        # If font is bold and not body font, it's likely a heading
        if is_bold and font_name != self.body_font_name:
            return True
            
        # If text is all caps and not too long, it might be a heading
        if text.isupper() and len(text) < 100:
            return True
            
        # If line has different left margin, it might be a heading
        if text_block['margin_left'] > 20:  # Adjust based on document
            return True
        
        return False

    def get_heading_level(self, text_block: Dict) -> int:
        """
        Determine the heading level for a heading text block
        
        Args:
            text_block: Text block dictionary
            
        Returns:
            Heading level (1, 2, 3, etc.)
        """
        text = text_block['text'].strip()
        
        # Check numbered headings
        numbered_match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?', text)
        if numbered_match:
            # Count non-None groups to determine level
            level = sum(1 for g in numbered_match.groups() if g is not None)
            return level
        
        # Check for specific heading indicators
        if re.match(r'^ARTICLE|^SECTION', text, re.IGNORECASE):
            return 1
        
        if re.match(r'^EXHIBIT|^SCHEDULE|^APPENDIX', text, re.IGNORECASE):
            return 1
        
        # Use font size to estimate level
        font_size = text_block['font_size']
        font_stats = self.font_stats.get(text_block['font_name'], {})
        max_size = font_stats.get('max_size', self.body_font_size * 1.5)
        
        # Map font size to levels
        if font_size >= max_size * 0.9:
            return 1
        elif font_size >= max_size * 0.8:
            return 2
        elif font_size >= max_size * 0.7:
            return 3
        else:
            return 4

    def build_hierarchical_structure(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Build hierarchical structure from text blocks
        
        Args:
            text_blocks: List of text block dictionaries
            
        Returns:
            Hierarchical structure of the document
        """
        hierarchy = []
        current_path = [None] * 10  # Max 10 levels of hierarchy
        
        current_content = []
        
        for block in text_blocks:
            if self.is_heading(block):
                # If we have content pending, add it to the current section
                if current_content and any(current_path):
                    # Find the current section
                    current_section = hierarchy
                    for level in range(10):
                        if current_path[level] is None:
                            break
                        for section in current_section:
                            if section.get('id') == current_path[level]:
                                if 'content' not in section:
                                    section['content'] = ""
                                section['content'] += ' '.join([b['text'] for b in current_content])
                                if not section['content'].endswith(' '):
                                    section['content'] += ' '
                                current_section = section.get('subsections', [])
                                break
                    
                    current_content = []
                
                # Process the heading
                level = self.get_heading_level(block)
                heading_text = block.get('text', '').strip()
                
                # Reset lower levels in the path
                for i in range(level, 10):
                    current_path[i] = None
                
                # Create section ID
                section_id = f"section_{block['page']}_{block['y0']}_{level}"
                current_path[level-1] = section_id
                
                # Create section object
                section = {
                    'id': section_id,
                    'level': level,
                    'heading': heading_text,
                    'page': block['page'],
                    'subsections': []
                }
                
                # Add to the hierarchy at the appropriate level
                if level == 1:
                    hierarchy.append(section)
                else:
                    # Find parent section
                    parent_level = level - 1
                    parent_section = None
                    
                    # Navigate to the correct place in the hierarchy
                    current_section = hierarchy
                    for l in range(parent_level):
                        parent_id = current_path[l]
                        if parent_id is None:
                            break
                        
                        found = False
                        for s in current_section:
                            if s.get('id') == parent_id:
                                current_section = s.get('subsections', [])
                                found = True
                                break
                        
                        if not found:
                            break
                    
                    # Add to the last level we reached
                    current_section.append(section)
            else:
                # Regular content, add to current content list
                current_content.append(block)
        
        # Add any remaining content to the last section
        if current_content and any(current_path):
            # Find the current section
            current_section = hierarchy
            for level in range(10):
                if current_path[level] is None:
                    break
                for section in current_section:
                    if section.get('id') == current_path[level]:
                        if 'content' not in section:
                            section['content'] = ""
                        section['content'] += ' '.join([b['text'] for b in current_content])
                        current_section = section.get('subsections', [])
                        break
        
        return hierarchy

    def parse_pdf(self) -> Dict:
        """
        Parse the PDF and generate hierarchical structure
        
        Returns:
            Dictionary with document metadata and hierarchical structure
        """
        try:
            self.open_pdf()
            
            all_text_blocks = []
            
            # Process each page
            for page_idx, page in enumerate(self.pdf.pages):
                print(f"Processing page {page_idx + 1}/{len(self.pdf.pages)}")
                
                # Extract text with properties
                text_blocks = self.extract_text_with_properties(page)
                all_text_blocks.extend(text_blocks)
            
            # Analyze font statistics to identify headings
            font_stats = self.analyze_font_statistics()
            
            # Build hierarchical structure
            hierarchy = self.build_hierarchical_structure(all_text_blocks)
            
            # Create document metadata
            metadata = {
                'filename': os.path.basename(self.pdf_path),
                'pages': len(self.pdf.pages),
                'title': os.path.splitext(os.path.basename(self.pdf_path))[0],
            }
            
            # Create result object
            result = {
                'metadata': metadata,
                'structure': hierarchy
            }
            
            return result
            
        finally:
            self.close_pdf()

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and fixing common issues
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unnecessary line breaks
        text = text.replace('\n', ' ')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

def parse_contract(pdf_path: str, output_file: Optional[str] = None) -> Dict:
    """
    Parse a contract PDF and generate hierarchical structure
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the parsed structure (optional)
        
    Returns:
        Dictionary with document metadata and hierarchical structure
    """
    parser = ContractParser(pdf_path)
    result = parser.parse_pdf()
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Structure saved to {output_file}")
    
    return result

def print_section(section, indent=0):
    """Print a section with proper indentation"""
    print(" " * indent + f"- {section.get('heading', 'Untitled Section')} (Level {section.get('level', 0)}, Page {section.get('page', 0)})")
    
    content = section.get('content', '')
    if content:
        # Print just the first 100 characters of content
        print(" " * (indent + 2) + f"Content: {content[:100]}..." if len(content) > 100 else content)
    
    for subsection in section.get('subsections', []):
        print_section(subsection, indent + 4)

def main():
    """Main function to run the contract parser from command line"""
    parser = argparse.ArgumentParser(description='Parse contract PDFs into structured JSON')
    parser.add_argument('pdf_path', help='Path to the contract PDF file')
    parser.add_argument('--output', help='Path to save the structured JSON')
    parser.add_argument('--visualize', action='store_true', help='Visualize the structure')
    
    args = parser.parse_args()
    
    # Parse contract
    result = parse_contract(args.pdf_path, args.output)
    
    # Visualize if requested
    if args.visualize:
        print("\nDocument Structure:")
        print(f"Title: {result['metadata']['title']}")
        print(f"Pages: {result['metadata']['pages']}")
        print("\nHierarchy:")
        
        for section in result['structure']:
            print_section(section)

if __name__ == "__main__":
    main()