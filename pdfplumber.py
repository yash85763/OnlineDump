import os
import re
import json
import pdfplumber
from typing import List, Dict, Tuple
import statistics
from collections import defaultdict

class ContractParser:
    def __init__(self, pdf_path: str):
        """
        Initialize the contract parser
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.pdf = None
        self.is_multi_column = False
        self.page_layouts = {}
        self.font_stats = {}
        
        # Common heading patterns in contracts
        self.heading_patterns = [
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

    def parse(self) -> Dict:
        """
        Parse the PDF and extract structured content
        
        Returns:
            Dictionary with hierarchical document structure
        """
        try:
            # Open the PDF
            self.pdf = pdfplumber.open(self.pdf_path)
            print(f"Processing: {os.path.basename(self.pdf_path)} ({len(self.pdf.pages)} pages)")
            
            # Detect column layout
            self.detect_layout()
            
            # Extract all text blocks with properties
            all_text_blocks = self.extract_all_text_blocks()
            
            # Analyze font statistics to help identify headings
            self.analyze_font_statistics()
            
            # Build hierarchical structure
            structure = self.build_hierarchical_structure(all_text_blocks)
            
            # Create result with metadata
            result = {
                'metadata': {
                    'filename': os.path.basename(self.pdf_path),
                    'pages': len(self.pdf.pages),
                    'layout': 'multi-column' if self.is_multi_column else 'single-column',
                },
                'structure': structure
            }
            
            return result
            
        finally:
            # Close the PDF
            if self.pdf:
                self.pdf.close()

    def detect_layout(self, sample_pages=5):
        """
        Detect if the document has single or multiple columns
        
        Args:
            sample_pages: Number of pages to sample for detection
        """
        column_votes = []
        
        # Sample pages throughout the document
        num_pages = len(self.pdf.pages)
        pages_to_analyze = min(sample_pages, num_pages)
        
        # Sample evenly distributed pages
        step = max(1, num_pages // pages_to_analyze)
        indices = [i * step for i in range(pages_to_analyze)]
        
        for page_idx in indices:
            if page_idx >= num_pages:
                continue
                
            page = self.pdf.pages[page_idx]
            
            # Get text distribution
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True
            )
            
            if not words:
                continue
            
            # Create horizontal density profile
            page_width = page.width
            histogram_bins = 100
            bin_size = page_width / histogram_bins
            density_profile = [0] * histogram_bins
            
            for word in words:
                start_bin = int(word['x0'] / bin_size)
                end_bin = min(int(word['x1'] / bin_size), histogram_bins - 1)
                for bin_idx in range(start_bin, end_bin + 1):
                    density_profile[bin_idx] += 1
            
            # Look for a significant gap in the middle region
            middle_start = histogram_bins // 3
            middle_end = 2 * histogram_bins // 3
            middle_region = density_profile[middle_start:middle_end]
            
            if not middle_region:
                continue
                
            # Check if there's a significant drop in density in the middle
            max_density = max(density_profile)
            min_middle = min(middle_region)
            
            # If there's a significant empty area in the middle section, 
            # it's likely a two-column layout
            gap_threshold = 0.2  # Consider it a gap if density drops below 20% of max
            has_middle_gap = min_middle < gap_threshold * max_density
            
            # Find the minimum position
            if has_middle_gap:
                mid_point = middle_start + middle_region.index(min_middle)
                gap_start = mid_point
                gap_end = mid_point
                
                # Expand gap boundaries
                while gap_start > 0 and density_profile[gap_start] < gap_threshold * max_density:
                    gap_start -= 1
                
                while gap_end < histogram_bins - 1 and density_profile[gap_end] < gap_threshold * max_density:
                    gap_end += 1
                
                # Store column boundaries for this page
                col1_end = gap_start * bin_size
                col2_start = gap_end * bin_size
                
                self.page_layouts[page_idx] = {
                    'columns': [(0, col1_end), (col2_start, page_width)]
                }
            else:
                # Single column
                self.page_layouts[page_idx] = {
                    'columns': [(0, page_width)]
                }
            
            column_votes.append(has_middle_gap)
        
        # Determine overall layout by majority vote
        self.is_multi_column = sum(column_votes) > len(column_votes) / 2
        
        print(f"Detected layout: {'Multi-column' if self.is_multi_column else 'Single-column'}")

    def get_column_boundaries(self, page_idx: int) -> List[Tuple[float, float]]:
        """
        Get column boundaries for a specific page
        
        Args:
            page_idx: Page index
            
        Returns:
            List of (start_x, end_x) tuples for each column
        """
        # If we have stored layout for this page, use it
        if page_idx in self.page_layouts:
            return self.page_layouts[page_idx]['columns']
        
        # Otherwise determine based on overall document layout
        page = self.pdf.pages[page_idx]
        
        if self.is_multi_column:
            # Default multi-column layout
            mid_x = page.width / 2
            margin = page.width * 0.1
            return [(0, mid_x - margin/2), (mid_x + margin/2, page.width)]
        else:
            # Single column
            return [(0, page.width)]

    def extract_text_blocks(self, page_idx: int) -> List[Dict]:
        """
        Extract text blocks with properties from a page
        
        Args:
            page_idx: Page index
            
        Returns:
            List of text blocks with properties
        """
        page = self.pdf.pages[page_idx]
        column_boundaries = self.get_column_boundaries(page_idx)
        
        # Calculate header and footer areas (10% of page height)
        header_height = page.height * 0.1
        footer_start = page.height * 0.9
        
        # Extract characters
        chars = page.chars
        if not chars:
            return []
        
        # Filter out header and footer
        content_chars = [
            c for c in chars
            if c['y0'] >= header_height and c['y1'] <= footer_start
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
        
        # Process lines into text blocks
        text_blocks = []
        
        for y, line_chars in sorted_lines:
            # Sort characters by x-coordinate
            line_chars.sort(key=lambda c: c['x0'])
            
            # Assign characters to columns
            for col_idx, (col_start, col_end) in enumerate(column_boundaries):
                # Get characters in this column
                col_chars = [c for c in line_chars if col_start <= c['x0'] < col_end]
                
                if not col_chars:
                    continue
                
                # Extract text and properties
                text = ''.join(c['text'] for c in col_chars)
                if not text.strip():
                    continue
                
                # Get font properties from first character
                first_char = col_chars[0]
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
                    'y0': min(c['y0'] for c in col_chars),
                    'y1': max(c['y1'] for c in col_chars),
                    'x0': min(c['x0'] for c in col_chars),
                    'x1': max(c['x1'] for c in col_chars),
                    'font_name': font_name,
                    'font_size': font_size,
                    'bold': 'bold' in font_name.lower() or 'heavy' in font_name.lower(),
                    'italic': 'italic' in font_name.lower() or 'oblique' in font_name.lower(),
                    'margin_left': min(c['x0'] for c in col_chars) - col_start,
                }
                
                text_blocks.append(text_block)
        
        return text_blocks

    def extract_all_text_blocks(self) -> List[Dict]:
        """
        Extract text blocks from all pages
        
        Returns:
            List of text blocks with properties
        """
        all_blocks = []
        
        for page_idx in range(len(self.pdf.pages)):
            blocks = self.extract_text_blocks(page_idx)
            all_blocks.extend(blocks)
            
            # Print progress
            if (page_idx + 1) % 10 == 0 or page_idx + 1 == len(self.pdf.pages):
                print(f"Processed {page_idx + 1}/{len(self.pdf.pages)} pages")
        
        return all_blocks

    def analyze_font_statistics(self):
        """
        Analyze font statistics to help identify headings
        """
        for font_name, stats in self.font_stats.items():
            if stats['sizes']:
                stats['median_size'] = statistics.median(stats['sizes'])
                try:
                    stats['mode_size'] = statistics.mode(stats['sizes'])
                except statistics.StatisticsError:
                    # If no unique mode, use median
                    stats['mode_size'] = stats['median_size']
                stats['max_size'] = max(stats['sizes'])
                stats['min_size'] = min(stats['sizes'])
                stats['avg_size'] = sum(stats['sizes']) / len(stats['sizes'])
        
        # Find most common font (likely body text)
        if self.font_stats:
            body_font = max(self.font_stats.items(), key=lambda x: x[1]['count'])
            self.body_font_name = body_font[0]
            self.body_font_size = body_font[1].get('mode_size', 0)
        else:
            self.body_font_name = ""
            self.body_font_size = 0

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
        font_size = text_block.get('font_size', 0)
        font_name = text_block.get('font_name', '')
        is_bold = text_block.get('bold', False)
        
        # If body font size is 0, use a default threshold
        size_threshold = self.body_font_size * 1.1 if self.body_font_size > 0 else 12
        
        # If font is significantly larger than body text, it's likely a heading
        if font_size > size_threshold:
            return True
        
        # If font is bold and not body font, it's likely a heading
        if is_bold and font_name != self.body_font_name:
            return True
            
        # If text is all caps and not too long, it might be a heading
        if text.isupper() and len(text) < 100:
            return True
            
        # If line has different left margin, it might be a heading
        if text_block.get('margin_left', 0) > 20:  # Adjust based on document
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
        
        # Check numbered headings (e.g., 1.2.3)
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
        font_size = text_block.get('font_size', 0)
        font_stats = self.font_stats.get(text_block.get('font_name', ''), {})
        max_size = font_stats.get('max_size', self.body_font_size * 1.5 if self.body_font_size > 0 else 12)
        
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


def process_folder(input_folder: str, output_folder: str = None):
    """
    Process all PDFs in a folder and save as structured JSON
    
    Args:
        input_folder: Path to folder containing PDFs
        output_folder: Path to save JSON output (defaults to input folder)
    """
    if output_folder is None:
        output_folder = input_folder
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.json")
        
        print(f"\nProcessing: {pdf_file}")
        try:
            # Parse the PDF
            parser = ContractParser(pdf_path)
            result = parser.parse()
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Saved structure to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")


def main():
    """
    Main function with hardcoded paths
    """
    # EDIT THESE PATHS
    input_folder = "/path/to/contracts"  # <-- CHANGE THIS to your input folder path
    output_folder = "/path/to/output"    # <-- CHANGE THIS to your output folder path (or set to None to use same as input)
    
    # Process all PDFs in the folder
    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()