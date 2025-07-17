import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Any

# PDF processing imports
try:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox, LTPage
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("Warning: pdfminer.six not available. Install with: pip install pdfminer.six")

class PDFParser:
    """
    Clean PDF parser for extracting text content from PDF documents, ignoring image/graph captions and headings.
    """
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 min_words_per_paragraph: int = 5):
        """
        Initialize the PDF parser with configurable parameters.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric to total characters (0.0-1.0)
            paragraph_spacing_threshold: Max vertical spacing between text blocks (points)
            min_words_per_paragraph: Minimum words for standalone paragraphs
        """
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer.six is required for PDF processing")
            
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.min_words_per_paragraph = min_words_per_paragraph
        
        # Configure pdfminer parameters
        self.layout_params = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True
        )

    def is_likely_caption_or_heading(self, text: str, prev_text: str = None, next_text: str = None) -> bool:
        """
        Determine if a text block is likely a caption or heading for images/graphs.
        
        Args:
            text: The text to evaluate
            prev_text: Previous text block for context (optional)
            next_text: Next text block for context (optional)
            
        Returns:
            True if the text is likely a caption or heading, False otherwise
        """
        text = text.strip()
        if not text:
            return True  # Empty text is ignored
        
        # Common caption/heading indicators
        caption_indicators = [
            r'^(Figure|Fig\.|Table|Graph|Chart|Image|Diagram)\s*\d+[\.:]',
            r'^\d+\.\s+',  # e.g., "1. Title"
            r'^(Caption|Source|Note):',
            r'^\([a-zA-Z]\)',  # e.g., "(a) Description"
        ]
        
        # Check for caption/heading patterns
        if len(text.split()) < 10:
            for pattern in caption_indicators:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        # Check if text is in all caps or title case (common for headings)
        if text.isupper() or text.istitle():
            return True
        
        # Check if text is significantly shorter than surrounding text
        if prev_text and next_text:
            if len(text.split()) < 5 and (len(prev_text.split()) > 10 or len(next_text.split()) > 10):
                return True
        
        return False

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file and extract structured text content, ignoring captions/headings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing the parsed content and metadata
        """
        try:
            # Step 1: Extract raw content and validate quality
            pages_data, is_parsable, quality_info = self._extract_pdf_content(pdf_path)
            
            if not is_parsable:
                return {
                    "filename": os.path.basename(pdf_path),
                    "success": False,
                    "error": quality_info,
                    "pages": []
                }
            
            # Step 2: Determine document layout
            layout_type = self._determine_layout(pages_data)
            
            # Step 3: Extract and organize paragraphs
            structured_pages = self._extract_paragraphs(pages_data, layout_type)
            
            # Step 4: Create result structure
            result = {
                "filename": os.path.basename(pdf_path),
                "success": True,
                "layout": layout_type,
                "total_pages": len(structured_pages),
                "quality_info": quality_info,
                "pages": structured_pages
            }
            
            return result
            
        except Exception as e:
            return {
                "filename": os.path.basename(pdf_path),
                "success": False,
                "error": f"Error processing PDF: {str(e)}",
                "pages": []
            }

    def _extract_pdf_content(self, pdf_path: str) -> Tuple[List[Dict], bool, str]:
        """
        Extract raw content from PDF and validate quality.
        
        Returns:
            Tuple of (pages_data, is_parsable, quality_info)
        """
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(resource_manager, laparams=self.layout_params)
        interpreter = PDFPageInterpreter(resource_manager, device)
        
        pages_data = []
        total_text = ""
        
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)  # Initialize document directly with parser
            
            if not document.is_extractable:
                return [], False, "Document is encrypted or not extractable"
            
            # Extract content from each page
            for page_num, page in enumerate(PDFPage.create_pages(document)):
                interpreter.process_page(page)
                layout = device.get_result()
                
                # Extract text blocks from page
                text_blocks = []
                page_text = ""
                
                for i, element in enumerate(layout):
                    if isinstance(element, LTTextBox):
                        text_content = element.get_text().strip()
                        if text_content:
                            # Get previous and next text for context
                            prev_text = layout[i-1].get_text().strip() if i > 0 and isinstance(layout[i-1], LTTextBox) else None
                            next_text = layout[i+1].get_text().strip() if i < len(layout)-1 and isinstance(layout[i+1], LTTextBox) else None
                            
                            # Skip likely captions or headings
                            if not self.is_likely_caption_or_heading(text_content, prev_text, next_text):
                                text_blocks.append({
                                    'x0': element.x0,
                                    'y0': element.y0,
                                    'x1': element.x1,
                                    'y1': element.y1,
                                    'text': text_content
                                })
                                page_text += text_content + " "
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'width': layout.width if hasattr(layout, 'width') else 612,
                    'height': layout.height if hasattr(layout, 'height') else 792,
                    'text_blocks': text_blocks,
                    'raw_text': page_text.strip()
                })
                
                total_text += page_text
        
        # Validate extracted content
        if not total_text.strip():
            return pages_data, False, "No text extracted - PDF may need OCR"
        
        # Calculate quality metrics
        total_chars = len(total_text)
        alphanumeric_chars = sum(1 for char in total_text if char.isalnum())
        quality_ratio = alphanumeric_chars / total_chars if total_chars > 0 else 0
        
        # Check average content per page
        avg_chars_per_page = total_chars / len(pages_data) if pages_data else 0
        
        if avg_chars_per_page < 100:
            return pages_data, False, f"Insufficient content ({avg_chars_per_page:.1f} chars/page)"
        
        if quality_ratio < self.min_quality_ratio:
            return pages_data, False, f"Low quality text (ratio: {quality_ratio:.2f})"
        
        quality_info = f"Quality ratio: {quality_ratio:.2f}, Avg chars/page: {avg_chars_per_page:.1f}"
        return pages_data, True, quality_info

    def _determine_layout(self, pages_data: List[Dict]) -> str:
        """
        Determine if the document uses single or double column layout.
        """
        x_coordinates = []
        sample_pages = min(3, len(pages_data))
        
        for page_idx in range(sample_pages):
            page = pages_data[page_idx]
            for block in page.get('text_blocks', []):
                x_center = (block['x0'] + block['x1']) / 2
                x_coordinates.append(x_center)
        
        if len(x_coordinates) < 5:
            return "single_column"
        
        try:
            X = np.array(x_coordinates).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            centers = kmeans.cluster_centers_.flatten()
            counts = np.bincount(labels)
            
            center_distance = abs(centers[0] - centers[1])
            page_width = pages_data[0]['width'] if pages_data else 612
            
            if (center_distance > page_width * 0.3 and 
                min(counts) > len(x_coordinates) * 0.2):
                return "double_column"
            
        except Exception:
            pass
        
        return "single_column"

    def _extract_paragraphs(self, pages_data: List[Dict], layout_type: str) -> List[Dict]:
        """
        Extract and organize paragraphs from pages, handling cross-page continuity.
        """
        structured_pages = []
        previous_paragraph = None
        
        for page in pages_data:
            page_number = page['page_number']
            page_height = page['height']
            text_blocks = page['text_blocks']
            
            if not text_blocks:
                structured_pages.append({
                    "page_number": page_number,
                    "paragraphs": [],
                    "layout": layout_type
                })
                continue
            
            sorted_blocks = self._sort_blocks_by_position(text_blocks, page_height, layout_type)
            paragraphs = self._group_blocks_into_paragraphs(sorted_blocks)
            
            if previous_paragraph:
                prev_text, should_continue = previous_paragraph
                if should_continue and paragraphs:
                    paragraphs[0] = prev_text + " " + paragraphs[0]
                elif should_continue:
                    paragraphs.insert(0, prev_text)
                previous_paragraph = None
            
            if paragraphs:
                last_para = paragraphs[-1]
                should_continue = self._should_continue_paragraph(last_para)
                if should_continue:
                    previous_paragraph = (last_para, True)
                    paragraphs = paragraphs[:-1]
            
            structured_pages.append({
                "page_number": page_number,
                "paragraphs": paragraphs,
                "layout": layout_type,
                "paragraph_count": len(paragraphs)
            })
        
        if previous_paragraph:
            last_page = structured_pages[-1]
            last_page["paragraphs"].append(previous_paragraph[0])
            last_page["paragraph_count"] += 1
        
        return structured_pages

    def _sort_blocks_by_position(self, text_blocks: List[Dict], page_height: float, layout_type: str) -> List[Dict]:
        """
        Sort text blocks by their position on the page.
        """
        if layout_type == "double_column":
            page_width = max(block['x1'] for block in text_blocks) if text_blocks else 612
            midpoint = page_width / 2
            
            left_blocks = [b for b in text_blocks if (b['x0'] + b['x1']) / 2 < midpoint]
            right_blocks = [b for b in text_blocks if (b['x0'] + b['x1']) / 2 >= midpoint]
            
            left_blocks.sort(key=lambda b: page_height - b['y1'])
            right_blocks.sort(key=lambda b: page_height - b['y1'])
            
            return left_blocks + right_blocks
        else:
            return sorted(text_blocks, key=lambda b: (page_height - b['y1'], b['x0']))

    def _group_blocks_into_paragraphs(self, sorted_blocks: List[Dict]) -> List[str]:
        """
        Group text blocks into paragraphs based on spacing and content.
        """
        if not sorted_blocks:
            return []
        
        paragraphs = []
        current_paragraph = sorted_blocks[0]['text']
        
        for i in range(1, len(sorted_blocks)):
            current_block = sorted_blocks[i]
            previous_block = sorted_blocks[i-1]
            
            vertical_spacing = abs(current_block['y0'] - previous_block['y1'])
            
            if vertical_spacing <= self.paragraph_spacing_threshold:
                current_paragraph += " " + current_block['text']
            else:
                paragraphs.append(self._clean_paragraph_text(current_paragraph))
                current_paragraph = current_block['text']
        
        if current_paragraph:
            paragraphs.append(self._clean_paragraph_text(current_paragraph))
        
        return self._join_short_paragraphs(paragraphs)

    def _clean_paragraph_text(self, text: str) -> str:
        """
        Clean and normalize paragraph text.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        return text.strip()

    def _join_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Join paragraphs that are too short or don't end with punctuation.
        """
        if not paragraphs:
            return []
        
        result = []
        current_paragraph = paragraphs[0]
        
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            
            word_count = len(current_paragraph.split())
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', current_paragraph.strip()))
            
            if word_count < self.min_words_per_paragraph or not ends_with_punctuation:
                current_paragraph += " " + next_paragraph
            else:
                result.append(current_paragraph)
                current_paragraph = next_paragraph
        
        if current_paragraph:
            result.append(current_paragraph)
        
        return result

    def _should_continue_paragraph(self, paragraph: str) -> bool:
        """
        Determine if a paragraph should continue on the next page.
        """
        word_count = len(paragraph.split())
        ends_with_punctuation = bool(re.search(r'[.!?:;]$', paragraph.strip()))
        
        return word_count < self.min_words_per_paragraph or not ends_with_punctuation

    def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
        """
        Save the parsed result to a JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_path

def parse_single_pdf(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to parse a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save JSON output
    
    Returns:
        Dictionary with parsed content
    """
    parser = PDFParser()
    result = parser.parse_pdf(pdf_path)
    
    if output_path:
        parser.save_to_json(result, output_path)
        print(f"Saved results to: {output_path}")
    
    return result

def parse_pdf_directory(input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Parse all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save JSON results
    
    Returns:
        List of results for each processed PDF
    """
    parser = PDFParser()
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for filename in pdf_files:
        pdf_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        
        print(f"Processing: {filename}")
        result = parser.parse_pdf(pdf_path)
        parser.save_to_json(result, output_path)
        
        results.append(result)
        
        if result["success"]:
            total_paragraphs = sum(page["paragraph_count"] for page in result["pages"])
            print(f"  ✓ Extracted {total_paragraphs} paragraphs from {result['total_pages']} pages")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse PDF files, ignoring captions and headings.")
    parser.add_argument("input", help="Path to a PDF file or directory containing PDFs")
    parser.add_argument("-o", "--output", help="Path to save JSON output or output directory", default=None)
    args = parser.parse_args()

    if os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        result = parse_single_pdf(args.input, args.output)
        if result["success"]:
            print(f"Successfully parsed {result['filename']}")
            print(f"Layout: {result['layout']}")
            print(f"Total pages: {result['total_pages']}")
            total_paragraphs = sum(page["paragraph_count"] for page in result["pages"])
            print(f"Total paragraphs: {total_paragraphs}")
        else:
            print(f"Failed to parse: {result['error']}")
    elif os.path.isdir(args.input):
        if not args.output:
            print("Error: Output directory must be specified for directory input")
        else:
            results = parse_pdf_directory(args.input, args.output)
            print(f"Processed {len(results)} PDF files")
    else:
        print("Error: Input must be a PDF file or a directory")
