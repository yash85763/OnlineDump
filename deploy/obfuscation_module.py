"""
Enhanced PDF Handling Module with Database Integration

This module provides functionality for extracting information from OCR'd PDF contracts,
applying obfuscation techniques, and storing the results in a PostgreSQL database.

Features:
- PDF parsability check to ensure quality
- Layout analysis (single or double column)
- Paragraph extraction with cross-page and cross-column continuity handling
- Content obfuscation integration
- Database storage of original and obfuscated content
- File hash-based deduplication
"""

import os
import json
import re
import io
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from datetime import datetime

# Import pdfminer.six components
try:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTPage
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    raise ImportError("pdfminer.six is required. Install it with 'pip install pdfminer.six'")

# Import obfuscation module
try:
    from obfuscation import ContentObfuscator, create_average_word_count_obfuscator
    OBFUSCATION_AVAILABLE = True
except ImportError:
    OBFUSCATION_AVAILABLE = False
    print("Warning: Obfuscation module not available. Obfuscation will be disabled.")

# Import database configuration
try:
    from config.database import store_pdf_data, get_pdf_by_hash, check_database_connection
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Warning: Database configuration not available. Data will not be stored.")


class EnhancedPDFHandler:
    """Enhanced PDF handler with database integration and obfuscation"""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 page_continuity_threshold: float = 0.1,
                 min_words_threshold: int = 5,
                 enable_obfuscation: bool = True,
                 obfuscation_config: Dict[str, Any] = None,
                 enable_database: bool = True):
        """
        Initialize the enhanced PDF handler.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric chars to total chars
            paragraph_spacing_threshold: Max vertical spacing between text blocks
            page_continuity_threshold: Percentage of page height to check for continuation
            min_words_threshold: Minimum number of words for standalone paragraph
            enable_obfuscation: Whether to apply content obfuscation
            obfuscation_config: Configuration for obfuscation parameters
            enable_database: Whether to store results in database
        """
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        self.min_words_threshold = min_words_threshold
        self.enable_obfuscation = enable_obfuscation and OBFUSCATION_AVAILABLE
        self.enable_database = enable_database and DATABASE_AVAILABLE
        
        # Set up pdfminer configuration
        self.laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True
        )
        
        # Initialize obfuscator if enabled
        if self.enable_obfuscation:
            if obfuscation_config:
                self.obfuscator = ContentObfuscator(**obfuscation_config)
            else:
                # Use average word count method as default
                self.obfuscator = create_average_word_count_obfuscator()
        else:
            self.obfuscator = None
            
        # Check database connection if enabled
        if self.enable_database:
            try:
                self.database_connected = check_database_connection()
                if not self.database_connected:
                    print("Warning: Database connection failed. Results will not be stored.")
                    self.enable_database = False
            except Exception as e:
                print(f"Warning: Database connection error: {str(e)}")
                self.enable_database = False
                self.database_connected = False
        else:
            self.database_connected = False
    
    def calculate_file_hash(self, pdf_bytes: bytes) -> str:
        """Calculate SHA-256 hash of PDF file for deduplication"""
        return hashlib.sha256(pdf_bytes).hexdigest()
    
    def process_pdf_with_database(self, pdf_path: str = None, pdf_bytes: bytes = None, 
                                pdf_name: str = None, uploaded_by: str = "system") -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline and store in database.
        
        Args:
            pdf_path: Path to the PDF file (optional if pdf_bytes provided)
            pdf_bytes: PDF file bytes (optional if pdf_path provided)
            pdf_name: Name of the PDF file (required if using pdf_bytes)
            uploaded_by: Identifier for who uploaded the file
            
        Returns:
            Dictionary with processing results and database IDs
        """
        try:
            # Validate input parameters
            if pdf_bytes is None and pdf_path is None:
                return {
                    "success": False,
                    "error": "Either pdf_path or pdf_bytes must be provided",
                    "parsable": False
                }
            
            # Read PDF bytes if not provided
            if pdf_bytes is None:
                if not os.path.exists(pdf_path):
                    return {
                        "success": False,
                        "error": f"PDF file not found: {pdf_path}",
                        "parsable": False
                    }
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                pdf_name = os.path.basename(pdf_path)
            elif pdf_name is None:
                pdf_name = os.path.basename(pdf_path) if pdf_path else "uploaded_file.pdf"
            
            # Calculate file hash for deduplication
            file_hash = self.calculate_file_hash(pdf_bytes)
            
            # Check if file already exists in database
            if self.enable_database:
                try:
                    existing_pdf = get_pdf_by_hash(file_hash)
                    if existing_pdf:
                        return {
                            "success": True,
                            "message": "File already exists in database",
                            "pdf_id": existing_pdf["id"],
                            "duplicate": True,
                            "existing_record": existing_pdf
                        }
                except Exception as e:
                    print(f"Warning: Could not check for existing PDF: {str(e)}")
            
            # Extract PDF content and check parsability
            pages_data, is_parsable, quality_info = self.extract_pdf_content_from_bytes(pdf_bytes)
            
            if not is_parsable:
                return {
                    "success": False,
                    "error": quality_info,
                    "parsable": False,
                    "filename": pdf_name
                }
            
            # Determine the layout
            layout_type = self.determine_layout(pages_data)
            
            # Parse content into paragraphs
            original_pages_content = self.parse_paragraphs(pages_data)
            
            # Calculate original content metrics
            original_word_count = sum(
                len(paragraph.split()) 
                for page in original_pages_content 
                for paragraph in page.get('paragraphs', [])
            )
            original_page_count = len(original_pages_content)
            
            # Store original content as text
            raw_content = self.pages_to_text(original_pages_content)
            
            # Apply obfuscation if enabled
            if self.enable_obfuscation and self.obfuscator:
                final_pages_content, obfuscation_summary = self.obfuscator.obfuscate_content(original_pages_content)
                obfuscation_applied = True
            else:
                final_pages_content = original_pages_content.copy()
                obfuscation_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'obfuscation_applied': False,
                    'pages_removed_count': 0,
                    'paragraphs_obfuscated_count': 0,
                    'total_original_pages': original_page_count,
                    'total_final_pages': original_page_count,
                    'methods_applied': {
                        'page_removal': False,
                        'paragraph_obfuscation': False
                    }
                }
                obfuscation_applied = False
            
            # Calculate final content metrics
            final_word_count = sum(
                len(paragraph.split()) 
                for page in final_pages_content 
                for paragraph in page.get('paragraphs', [])
            )
            final_page_count = len(final_pages_content)
            avg_words_per_page = final_word_count / max(final_page_count, 1)
            
            # Store final content as text (obfuscated content)
            final_content = self.pages_to_text(final_pages_content)
            
            # Prepare data for database storage
            pdf_data = {
                'pdf_name': pdf_name,
                'file_hash': file_hash,
                'upload_date': datetime.now(),
                'processed_date': datetime.now(),
                'layout': layout_type,
                'original_word_count': original_word_count,
                'original_page_count': original_page_count,
                'parsability': True,
                'final_word_count': final_word_count,
                'final_page_count': final_page_count,
                'avg_words_per_page': avg_words_per_page,
                'raw_content': raw_content,
                'final_content': final_content,  # This contains the obfuscated content (remaining pages only)
                'obfuscation_applied': obfuscation_applied,
                'pages_removed_count': obfuscation_summary.get('pages_removed_count', 0),
                'paragraphs_obfuscated_count': obfuscation_summary.get('paragraphs_obfuscated_count', 0),
                'obfuscation_summary': obfuscation_summary,
                'uploaded_by': uploaded_by
            }
            
            # Store in database if available
            pdf_id = None
            if self.enable_database:
                try:
                    pdf_id = store_pdf_data(pdf_data)
                except Exception as e:
                    print(f"Warning: Failed to store PDF data in database: {str(e)}")
            
            # Return comprehensive result
            result = {
                "success": True,
                "pdf_id": pdf_id,
                "filename": pdf_name,
                "file_hash": file_hash,
                "parsable": True,
                "layout": layout_type,
                "original_metrics": {
                    "word_count": original_word_count,
                    "page_count": original_page_count
                },
                "final_metrics": {
                    "word_count": final_word_count,
                    "page_count": final_page_count,
                    "avg_words_per_page": avg_words_per_page
                },
                "obfuscation_summary": obfuscation_summary,
                "pages": final_pages_content,  # Return obfuscated content
                "raw_pages": original_pages_content if not self.enable_obfuscation else None,  # Only return if no obfuscation
                "database_stored": pdf_id is not None
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing PDF: {str(e)}",
                "parsable": False,
                "filename": pdf_name if 'pdf_name' in locals() else "unknown"
            }
    
    def extract_pdf_content_from_bytes(self, pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Extract content from PDF bytes and check parsability.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Tuple of (pages_data, is_parsable, quality_info)
        """
        # Initialize required pdfminer objects
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(resource_manager, laparams=self.laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        
        pages_data = []
        total_text = ""
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            parser = PDFParser(pdf_file)
            document = PDFDocument(parser)
            
            # Check if document is empty or encrypted
            if not document.is_extractable:
                return [], False, "Document is encrypted or not extractable"
            
            # Extract content from each page
            for page_num, page in enumerate(PDFPage.create_pages(document)):
                interpreter.process_page(page)
                layout = device.get_result()
                
                # Extract textboxes
                text_boxes = []
                page_text = ""
                
                # Get page dimensions
                page_width = layout.width if hasattr(layout, 'width') else 612
                page_height = layout.height if hasattr(layout, 'height') else 792
                
                for element in layout:
                    if isinstance(element, LTTextBox):
                        box_text = element.get_text().strip()
                        if box_text:
                            text_boxes.append({
                                'x0': element.x0,
                                'y0': element.y0,
                                'x1': element.x1,
                                'y1': element.y1,
                                'text': box_text
                            })
                            page_text += box_text + " "
                
                # Add page data to the results
                pages_data.append({
                    'page_num': page_num + 1,  # 1-based page numbering
                    'width': page_width,
                    'height': page_height,
                    'text_boxes': text_boxes,
                    'text': page_text.strip()
                })
                
                # Accumulate text for quality check
                total_text += page_text
        
        except Exception as e:
            return [], False, f"Error parsing PDF: {str(e)}"
        
        # If no text was extracted, the PDF might not be OCR'd or has issues
        if not total_text.strip():
            return pages_data, False, "No text extracted from PDF. The PDF might need OCR processing."
        
        # Count alphanumeric characters vs. total characters
        total_chars = len(total_text)
        alpha_chars = sum(1 for char in total_text if char.isalnum())
        
        # Calculate quality ratio
        quality_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        # Check if text length is reasonable for the number of pages
        if pages_data:
            avg_chars_per_page = total_chars / len(pages_data)
            if avg_chars_per_page < 100:
                return pages_data, False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
        # Check quality ratio against threshold
        if quality_ratio < self.min_quality_ratio:
            return pages_data, False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
        quality_info = f"PDF is parsable with quality ratio {quality_ratio:.2f}"
        return pages_data, True, quality_info
    
    def pages_to_text(self, pages_content: List[Dict[str, Any]]) -> str:
        """
        Convert pages content to plain text format.
        
        Args:
            pages_content: List of page content dictionaries
            
        Returns:
            Plain text representation of the content
        """
        text_parts = []
        for page in pages_content:
            page_paragraphs = page.get('paragraphs', [])
            if page_paragraphs:
                text_parts.extend(page_paragraphs)
        
        return '\n\n'.join(text_parts)
    
    def determine_layout(self, pages_data: List[Dict[str, Any]]) -> str:
        """Determine if the PDF has a single or double column layout."""
        x_coordinates = []
        num_pages_to_check = min(3, len(pages_data))
        
        for page_idx in range(num_pages_to_check):
            page = pages_data[page_idx]
            text_boxes = page.get('text_boxes', [])
            
            for block in text_boxes:
                x_mid = (block['x0'] + block['x1']) / 2
                x_coordinates.append(x_mid)
        
        if len(x_coordinates) < 5:
            return "single_column"
        
        try:
            X = np.array(x_coordinates).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
            centers = kmeans.cluster_centers_.flatten()
            counts = np.bincount(kmeans.labels_)
            
            center_distance = abs(centers[0] - centers[1])
            
            if pages_data and 'width' in pages_data[0]:
                page_width = pages_data[0]['width']
            else:
                page_width = 612
            
            if (center_distance > page_width * 0.3 and 
                    min(counts) > len(x_coordinates) * 0.15):
                return "double_column"
            else:
                return "single_column"
                
        except Exception:
            return "single_column"
    
    def parse_paragraphs(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse PDF content into paragraphs with cross-page continuity."""
        pages_content = []
        last_paragraph_info = None
        
        for page in pages_data:
            page_num = page['page_num']
            page_height = page.get('height', 792)
            page_width = page.get('width', 612)
            
            blocks = self.convert_to_blocks(page.get('text_boxes', []))
            
            if not blocks:
                pages_content.append({
                    "page_number": page_num,
                    "paragraphs": []
                })
                continue
            
            # Determine if double column layout
            is_double_column = False
            midpoint = page_width / 2
            
            if len(blocks) >= 3:
                try:
                    x_centers = [(block[0] + block[2]) / 2 for block in blocks]
                    X = np.array(x_centers).reshape(-1, 1)
                    
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
                    centers = kmeans.cluster_centers_.flatten()
                    counts = np.bincount(kmeans.labels_)
                    
                    column_centers = sorted(centers)
                    
                    if len(column_centers) > 1:
                        center_distance = abs(column_centers[0] - column_centers[1])
                        if (center_distance > page_width * 0.3 and 
                                min(counts) > len(x_centers) * 0.15):
                            is_double_column = True
                            midpoint = (column_centers[0] + column_centers[1]) / 2
                except Exception:
                    # If clustering fails, assume single column
                    pass
            
            # Process content blocks
            content_blocks = blocks  # Simplified - you may want to add header/footer filtering
            content_paragraphs = []
            
            if is_double_column and content_blocks:
                left_column = []
                right_column = []
                
                for block in content_blocks:
                    block_center_x = (block[0] + block[2]) / 2
                    if block_center_x < midpoint:
                        left_column.append(block)
                    else:
                        right_column.append(block)
                
                left_column.sort(key=lambda b: page_height - b[3])
                right_column.sort(key=lambda b: page_height - b[3])
                
                left_paragraphs = self._process_blocks_into_paragraphs(left_column)
                right_paragraphs = self._process_blocks_into_paragraphs(right_column)
                
                all_column_paragraphs = left_paragraphs + right_paragraphs
                content_paragraphs = self.process_sequential_paragraphs(all_column_paragraphs)
            elif content_blocks:
                content_blocks.sort(key=lambda b: page_height - b[3])
                raw_paragraphs = self._process_blocks_into_paragraphs(content_blocks)
                content_paragraphs = self.process_sequential_paragraphs(raw_paragraphs)
            
            all_paragraphs = []
            
            # Handle cross-page paragraph continuity
            if last_paragraph_info and content_paragraphs:
                prev_text, ends_with_punctuation, word_count = last_paragraph_info
                
                if not ends_with_punctuation or word_count < self.min_words_threshold:
                    if content_paragraphs:
                        first_content_para = content_paragraphs[0]
                        joined_paragraph = prev_text + " " + first_content_para
                        content_paragraphs[0] = joined_paragraph
                else:
                    all_paragraphs.append(prev_text)
                
                last_paragraph_info = None
            
            all_paragraphs.extend(content_paragraphs)
            
            # Check last paragraph for potential continuation
            if content_paragraphs:
                last_para = content_paragraphs[-1]
                ends_with_punctuation = bool(re.search(r'[.!?:;], last_para.strip()))
                word_count = len(last_para.split())
                
                if not ends_with_punctuation or word_count < self.min_words_threshold:
                    last_paragraph_info = (last_para, ends_with_punctuation, word_count)
                    all_paragraphs.pop()
            
            pages_content.append({
                "page_number": page_num,
                "paragraphs": all_paragraphs,
                "layout": "double_column" if is_double_column else "single_column"
            })
        
        # Add final paragraph if needed
        if last_paragraph_info:
            last_page = pages_content[-1]
            last_page["paragraphs"].append(last_paragraph_info[0])
        
        return pages_content
    
    def convert_to_blocks(self, text_boxes: List[Dict[str, Any]]) -> List[List]:
        """Convert pdfminer text boxes to block format."""
        blocks = []
        for box in text_boxes:
            block = [box['x0'], box['y0'], box['x1'], box['y1'], box['text']]
            blocks.append(block)
        return blocks
    
    def _process_blocks_into_paragraphs(self, blocks):
        """Process blocks into initial paragraphs based on vertical spacing."""
        paragraphs = []
        current_paragraph = ""
        
        for i, block in enumerate(blocks):
            text = block[4]
            
            if not text.strip():
                continue
            
            if not current_paragraph:
                current_paragraph = text
            else:
                if i > 0:
                    prev_block = blocks[i-1]
                    prev_bottom = prev_block[3]
                    current_top = block[1]
                    spacing = abs(current_top - prev_bottom)
                    
                    if spacing <= self.paragraph_spacing_threshold:
                        current_paragraph += " " + text
                    else:
                        paragraphs.append(current_paragraph)
                        current_paragraph = text
                else:
                    current_paragraph = text
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        return paragraphs
    
    def process_sequential_paragraphs(self, paragraphs):
        """
        Process a list of paragraphs sequentially, joining paragraphs that:
        1. Don't end with punctuation, OR
        2. Have fewer than min_words_threshold words
        """
        if not paragraphs:
            return []
        
        result_paragraphs = []
        current_paragraph = paragraphs[0]
        
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            
            word_count = len(current_paragraph.split())
            ends_with_punctuation = bool(re.search(r'[.!?:;], current_paragraph.strip()))
            
            if not ends_with_punctuation or word_count < self.min_words_threshold:
                current_paragraph += " " + next_paragraph
            else:
                result_paragraphs.append(current_paragraph)
                current_paragraph = next_paragraph
        
        if current_paragraph:
            result_paragraphs.append(current_paragraph)
        
        return result_paragraphs
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by handling common PDF extraction issues."""
        if not text:
            return ""
            
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Handle hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Replace single newlines within sentences with spaces
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Replace any remaining newlines with proper paragraph breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Trim extra whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text


# Utility functions for batch processing and integration

def process_single_pdf_from_streamlit(pdf_name: str, 
                                    pdf_bytes: bytes,
                                    enable_obfuscation: bool = True,
                                    obfuscation_config: Dict[str, Any] = None,
                                    uploaded_by: str = "streamlit_user") -> Dict[str, Any]:
    """
    Process a single PDF from Streamlit upload.
    
    Args:
        pdf_name: Name of the PDF file
        pdf_bytes: PDF file content as bytes
        enable_obfuscation: Whether to apply obfuscation
        obfuscation_config: Custom obfuscation configuration
        uploaded_by: User identifier
        
    Returns:
        Processing result dictionary
    """
    try:
        # Initialize handler with appropriate settings
        handler = EnhancedPDFHandler(
            enable_obfuscation=enable_obfuscation,
            obfuscation_config=obfuscation_config,
            enable_database=True
        )
        
        # Process the PDF
        result = handler.process_pdf_with_database(
            pdf_bytes=pdf_bytes,
            pdf_name=pdf_name,
            uploaded_by=uploaded_by
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "filename": pdf_name,
            "error": f"Failed to process PDF: {str(e)}",
            "parsable": False
        }


def process_pdf_batch(pdf_files: List[Tuple[str, bytes]], 
                     uploaded_by: str = "batch_system",
                     enable_obfuscation: bool = True,
                     obfuscation_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process multiple PDF files in batch.
    
    Args:
        pdf_files: List of tuples (filename, pdf_bytes)
        uploaded_by: Identifier for who uploaded the files
        enable_obfuscation: Whether to apply obfuscation
        obfuscation_config: Configuration for obfuscation
        
    Returns:
        List of processing results
    """
    handler = EnhancedPDFHandler(
        enable_obfuscation=enable_obfuscation,
        obfuscation_config=obfuscation_config,
        enable_database=True
    )
    
    results = []
    
    for filename, pdf_bytes in pdf_files:
        try:
            result = handler.process_pdf_with_database(
                pdf_bytes=pdf_bytes,
                pdf_name=filename,
                uploaded_by=uploaded_by
            )
            results.append(result)
            
        except Exception as e:
            results.append({
                "success": False,
                "filename": filename,
                "error": f"Failed to process {filename}: {str(e)}"
            })
    
    return results


def get_processing_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of batch processing results.
    
    Args:
        results: List of processing results
        
    Returns:
        Summary statistics
    """
    total_files = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total_files - successful
    duplicates = sum(1 for r in results if r.get('duplicate', False))
    stored_in_db = sum(1 for r in results if r.get('database_stored', False))
    
    # Aggregate obfuscation statistics
    total_pages_removed = 0
    total_paragraphs_obfuscated = 0
    total_original_pages = 0
    total_final_pages = 0
    total_original_words = 0
    total_final_words = 0
    
    for result in results:
        if result.get('success') and not result.get('duplicate'):
            obf_summary = result.get('obfuscation_summary', {})
            total_pages_removed += obf_summary.get('pages_removed_count', 0)
            total_paragraphs_obfuscated += obf_summary.get('paragraphs_obfuscated_count', 0)
            
            original_metrics = result.get('original_metrics', {})
            final_metrics = result.get('final_metrics', {})
            total_original_pages += original_metrics.get('page_count', 0)
            total_final_pages += final_metrics.get('page_count', 0)
            total_original_words += original_metrics.get('word_count', 0)
            total_final_words += final_metrics.get('word_count', 0)
    
    return {
        'total_files': total_files,
        'successful': successful,
        'failed': failed,
        'duplicates': duplicates,
        'stored_in_database': stored_in_db,
        'success_rate': successful / max(total_files, 1),
        'database_storage_rate': stored_in_db / max(total_files, 1),
        'obfuscation_stats': {
            'total_pages_removed': total_pages_removed,
            'total_paragraphs_obfuscated': total_paragraphs_obfuscated,
            'total_original_pages': total_original_pages,
            'total_final_pages': total_final_pages,
            'total_original_words': total_original_words,
            'total_final_words': total_final_words,
            'page_removal_rate': total_pages_removed / max(total_original_pages, 1),
            'word_retention_rate': total_final_words / max(total_original_words, 1)
        }
    }


# Example usage and testing functions

def test_enhanced_pdf_handler():
    """Test function to demonstrate enhanced PDF handler capabilities."""
    
    print("🧪 Testing Enhanced PDF Handler with Database Integration...")
    print("=" * 60)
    
    # Test configuration with average word count method
    test_config = {
        'enable_obfuscation': True,
        'obfuscation_config': {
            'obfuscation_method': 'average_word_count',
            'word_count_threshold_multiplier': 1.0,
            'min_pages_to_keep': 2,
            'paragraph_obfuscation_probability': 0.2
        },
        'enable_database': True
    }
    
    try:
        handler = EnhancedPDFHandler(**test_config)
        
        print("✅ PDF Handler initialized successfully")
        print(f"✅ Obfuscation enabled: {handler.enable_obfuscation}")
        print(f"✅ Database integration: {handler.enable_database}")
        print(f"✅ Database connected: {handler.database_connected}")
        
        if handler.obfuscator:
            print(f"✅ Obfuscator configuration:")
            print(f"   - Obfuscation method: {handler.obfuscator.obfuscation_method}")
            print(f"   - Word count threshold multiplier: {handler.obfuscator.word_count_threshold_multiplier}")
            print(f"   - Paragraph obfuscation probability: {handler.obfuscator.paragraph_obfuscation_prob}")
            print(f"   - Minimum pages to keep: {handler.obfuscator.min_pages_to_keep}")
        
        return handler
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the enhanced PDF handler
    handler = test_enhanced_pdf_handler()
    
    if handler:
        print("\n🎉 Enhanced PDF Handler is ready for use!")
        print("\nExample usage:")
        print("```python")
        print("# Using default average word count method")
        print("result = handler.process_pdf_with_database(")
        print("    pdf_path='path/to/contract.pdf',")
        print("    uploaded_by='test_user'")
        print(")")
        print("")
        print("# Using custom word count threshold (remove pages with < 80% of average)")
        print("custom_config = {")
        print("    'obfuscation_method': 'average_word_count',")
        print("    'word_count_threshold_multiplier': 0.8")
        print("}")
        print("handler_custom = EnhancedPDFHandler(")
        print("    enable_obfuscation=True,")
        print("    obfuscation_config=custom_config")
        print(")")
        print("```")
