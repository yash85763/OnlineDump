"""
Enhanced PDF Handling Module for Contract Analysis

This module provides functionality for extracting information from OCR'd PDF contracts,
determining their layout structure, parsing content into paragraphs, and storing the
results in a structured JSON format.

Features:
- PDF parsability check to ensure quality
- Layout analysis (single or double column)
- Paragraph extraction with cross-page and cross-column continuity handling
- Short paragraph handling (fewer than 5 words)
- Punctuation-based paragraph joining
- JSON storage with optional embedding generation (Sentence Transformers or Azure OpenAI)
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans

# Try importing pdfplumber instead of PyMuPDF
try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required. Install it with 'pip install pdfplumber'")

# Optional: Import libraries for embeddings
try:
    # For sentence-transformers
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# For Azure OpenAI embeddings
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# Check if any embedding method is available
EMBEDDINGS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE or AZURE_OPENAI_AVAILABLE


class PDFHandler:
    """Main class for handling PDF extraction and processing."""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 page_continuity_threshold: float = 0.1,
                 min_words_threshold: int = 5,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_provider: str = "sentence_transformers",
                 azure_openai_config: Dict[str, str] = None):
        """
        Initialize the PDF handler with configurable thresholds.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric chars to total chars (default 0.5)
            paragraph_spacing_threshold: Max vertical spacing between text blocks to be considered
                                         part of the same paragraph, in points (default 10)
            page_continuity_threshold: Percentage of page height to check for paragraph 
                                       continuation across pages (default 0.1 or 10%)
            min_words_threshold: Minimum number of words for a paragraph to be considered standalone (default 5)
            embedding_model: Name of the embedding model to use
                            - For sentence_transformers: model name like "all-MiniLM-L6-v2"
                            - For azure_openai: deployment name for the embedding model
            embedding_provider: Which embedding provider to use, either "sentence_transformers" or "azure_openai"
            azure_openai_config: Configuration for Azure OpenAI, required if using "azure_openai" provider:
                                - api_key: Azure OpenAI API key
                                - azure_endpoint: Azure OpenAI endpoint URL
                                - api_version: API version (e.g., "2023-05-15")
        """
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        self.min_words_threshold = min_words_threshold
        self.embedding_provider = embedding_provider
        self.azure_openai_config = azure_openai_config or {}
        
        # Initialize the embedding model if available
        self.embedding_model = None
        self.azure_client = None
        
        if embedding_provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Initialized Sentence Transformers model: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load Sentence Transformers model: {e}")
        
        elif embedding_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
            try:
                # Check if we have all the required config parameters
                required_params = ["api_key", "azure_endpoint", "api_version"]
                if not all(param in self.azure_openai_config for param in required_params):
                    missing = [p for p in required_params if p not in self.azure_openai_config]
                    raise ValueError(f"Missing required Azure OpenAI config parameters: {missing}")
                
                self.azure_client = AzureOpenAI(
                    api_key=self.azure_openai_config["api_key"],
                    api_version=self.azure_openai_config["api_version"],
                    azure_endpoint=self.azure_openai_config["azure_endpoint"]
                )
                self.embedding_model = embedding_model  # Store the deployment name
                print(f"Initialized Azure OpenAI client with deployment: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not initialize Azure OpenAI client: {e}")
    
    def process_pdf(self, pdf_path: str, generate_embeddings: bool = False) -> Dict[str, Any]:
        """
        Process a PDF file through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            generate_embeddings: Whether to generate embeddings for paragraphs
            
        Returns:
            Dictionary with the processed content or an error message
        """
        try:
            # Open the PDF with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Check if the PDF is parsable
                is_parsable, quality_info = self.check_parsability(pdf)
                
                if not is_parsable:
                    return {
                        "filename": os.path.basename(pdf_path),
                        "parsable": False,
                        "error": quality_info
                    }
                
                # Determine the layout
                layout_type = self.determine_layout(pdf)
                
                # Parse content into paragraphs
                pages_content = self.parse_paragraphs(pdf)
                
                # Generate JSON structure
                result = {
                    "filename": os.path.basename(pdf_path),
                    "parsable": True,
                    "layout": layout_type,
                    "pages": pages_content
                }
                
                # Optionally generate embeddings
                if generate_embeddings and (
                    (self.embedding_provider == "sentence_transformers" and self.embedding_model is not None) or
                    (self.embedding_provider == "azure_openai" and self.azure_client is not None)):
                    result["embeddings"] = self.generate_embeddings(pages_content)
                
                return result
            
        except Exception as e:
            return {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": f"Error processing PDF: {str(e)}"
            }
    
    def check_parsability(self, pdf) -> Tuple[bool, str]:
        """
        Check if a PDF is parsable by extracting text and assessing quality.
        
        Args:
            pdf: pdfplumber PDF object
            
        Returns:
            Tuple of (is_parsable, message)
        """
        total_text = ""
        total_chars = 0
        alpha_chars = 0
        
        # Extract all text from the document
        for page in pdf.pages:
            text = page.extract_text() or ""
            total_text += text
        
        # If no text was extracted, the PDF might not be OCR'd or has issues
        if not total_text.strip():
            return False, "No text extracted from PDF. The PDF might need OCR processing."
        
        # Count alphanumeric characters vs. total characters
        total_chars = len(total_text)
        alpha_chars = sum(1 for char in total_text if char.isalnum())
        
        # Calculate quality ratio
        if total_chars > 0:
            quality_ratio = alpha_chars / total_chars
        else:
            quality_ratio = 0
        
        # Check if text length is reasonable for the number of pages
        avg_chars_per_page = total_chars / len(pdf.pages)
        if avg_chars_per_page < 100:  # Arbitrary threshold, adjust as needed
            return False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
        # Check quality ratio against threshold
        if quality_ratio < self.min_quality_ratio:
            return False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
        return True, f"PDF is parsable with quality ratio {quality_ratio:.2f}"
    
    def determine_layout(self, pdf) -> str:
        """
        Determine if the PDF has a single or double column layout.
        
        Args:
            pdf: pdfplumber PDF object
            
        Returns:
            String indicating layout type: "single_column" or "double_column"
        """
        # Collect x-coordinates of text characters from multiple pages
        x_coordinates = []
        
        # Sample a few pages for efficiency (first 3 pages or all if less)
        num_pages_to_check = min(3, len(pdf.pages))
        
        for page_num in range(num_pages_to_check):
            page = pdf.pages[page_num]
            chars = page.chars
            
            # If no characters on page, skip
            if not chars:
                continue
                
            # Extract x-coordinates of the characters
            for char in chars:
                x_mid = (char['x0'] + char['x1']) / 2
                x_coordinates.append(x_mid)
        
        # If we don't have enough data points for clustering, assume single column
        if len(x_coordinates) < 5:
            return "single_column"
        
        # Use K-means clustering to identify column structure
        try:
            # Reshape for sklearn
            X = np.array(x_coordinates).reshape(-1, 1)
            
            # Try clustering with k=2 (assuming either 1 or 2 columns)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            centers = kmeans.cluster_centers_.flatten()
            counts = np.bincount(kmeans.labels_)
            
            # Calculate the distance between cluster centers
            center_distance = abs(centers[0] - centers[1])
            
            # Get page width from the first page
            page = pdf.pages[0]
            page_width = page.width
            
            # If the centers are far apart (relative to page width) and both clusters have 
            # a significant number of blocks, classify as double column
            if (center_distance > page_width * 0.3 and 
                    min(counts) > len(x_coordinates) * 0.15):
                return "double_column"
            else:
                return "single_column"
                
        except Exception:
            # If clustering fails, default to single column
            return "single_column"
    
    def extract_blocks_from_page(self, page):
        """
        Extract text blocks from a page using pdfplumber.
        
        In pdfplumber, we need to work with the words or characters and group them.
        This function converts pdfplumber's word objects to a format similar to
        PyMuPDF's blocks for compatibility with the rest of the code.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List of blocks in format [x0, y0, x1, y1, text]
        """
        # First try to extract words, which is faster and usually works well for normal documents
        words = page.extract_words(
            x_tolerance=3,  # Adjust based on your document's characteristics
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        if not words:
            # If no words extracted, try using characters as a fallback
            # This can be helpful for poorly OCR'd documents
            chars = page.chars
            if not chars:
                return []
                
            # Group characters into words based on proximity
            x_tolerance = 2  # Space between characters in a word
            y_tolerance = 2  # Vertical alignment tolerance
            
            # Sort chars by y-position first, then x-position
            chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
            
            words = []
            if chars:
                current_word = [chars[0]]
                
                for i in range(1, len(chars)):
                    prev_char = chars[i-1]
                    curr_char = chars[i]
                    
                    # If chars are on the same line and close horizontally
                    if (abs(curr_char['top'] - prev_char['top']) <= y_tolerance and
                            curr_char['x0'] - prev_char['x1'] <= x_tolerance):
                        current_word.append(curr_char)
                    else:
                        # Complete the current word
                        if current_word:
                            x0 = min(c['x0'] for c in current_word)
                            top = min(c['top'] for c in current_word)
                            x1 = max(c['x1'] for c in current_word)
                            bottom = max(c['bottom'] for c in current_word)
                            text = ''.join(c['text'] for c in current_word)
                            
                            words.append({
                                'x0': x0,
                                'top': top,
                                'x1': x1,
                                'bottom': bottom,
                                'text': text
                            })
                        
                        # Start a new word
                        current_word = [curr_char]
                
                # Add the last word
                if current_word:
                    x0 = min(c['x0'] for c in current_word)
                    top = min(c['top'] for c in current_word)
                    x1 = max(c['x1'] for c in current_word)
                    bottom = max(c['bottom'] for c in current_word)
                    text = ''.join(c['text'] for c in current_word)
                    
                    words.append({
                        'x0': x0,
                        'top': top,
                        'x1': x1,
                        'bottom': bottom,
                        'text': text
                    })
        
        if not words:
            return []
        
        # Group words into lines based on y-position
        y_tolerance = 5  # Adjust as needed
        lines = []
        
        # Sort words by y-position first, then x-position
        words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        current_line = [words[0]]
        
        for i in range(1, len(words)):
            prev_word = words[i-1]
            curr_word = words[i]
            
            # If y-positions are similar, add to current line
            if abs(curr_word['top'] - prev_word['top']) <= y_tolerance:
                current_line.append(curr_word)
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [curr_word]
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Convert lines to blocks with format [x0, y0, x1, y1, text]
        blocks = []
        for line in lines:
            if not line:
                continue
                
            # Sort words by x-position to ensure correct order
            line.sort(key=lambda w: w['x0'])
            
            # Calculate line boundaries
            x0 = min(w['x0'] for w in line)
            y0 = min(w['top'] for w in line)
            x1 = max(w['x1'] for w in line)
            y1 = max(w['bottom'] for w in line)
            
            # Combine text with spaces
            text = " ".join(w['text'] for w in line)
            
            blocks.append([x0, y0, x1, y1, text])
        
        return blocks
    
    def parse_paragraphs(self, pdf) -> List[Dict[str, Any]]:
        """
        Parse PDF content into paragraphs, handling cross-page and cross-column continuity.
        
        Args:
            pdf: pdfplumber PDF object
            
        Returns:
            List of dictionaries, each containing page number and paragraphs
        """
        pages_content = []
        last_paragraph_info = None  # Store info about the last paragraph of the previous page
        
        for page_num, page in enumerate(pdf.pages):
            page_height = page.height
            page_width = page.width
            
            # Extract text blocks using pdfplumber
            blocks = self.extract_blocks_from_page(page)
            
            # If no blocks, skip this page
            if not blocks:
                pages_content.append({
                    "page_number": page_num + 1,
                    "paragraphs": []
                })
                continue
            
            # Determine if the page has a double column layout
            is_double_column = False
            midpoint = page_width / 2
            
            # If enough blocks, try to detect columns
            if len(blocks) >= 3:
                # Get x-coordinates of blocks (middle of each block)
                x_centers = [(block[0] + block[2]) / 2 for block in blocks]
                X = np.array(x_centers).reshape(-1, 1)
                
                # Try clustering with k=2 (assuming either 1 or 2 columns)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                centers = kmeans.cluster_centers_.flatten()
                counts = np.bincount(kmeans.labels_)
                
                # Sort centers from left to right
                column_centers = sorted(centers)
                
                # If centers are far apart and both clusters have blocks, it's likely double column
                if len(column_centers) > 1:
                    center_distance = abs(column_centers[0] - column_centers[1])
                    if (center_distance > page_width * 0.3 and 
                            min(counts) > len(x_centers) * 0.15):
                        is_double_column = True
                        midpoint = (column_centers[0] + column_centers[1]) / 2
            
            # Define header and footer boundaries
            header_boundary = page_height * 0.1  # Top 10% of page
            footer_boundary = page_height * 0.9  # Bottom 10% of page
            
            # Separate blocks into header, footer, and main content
            header_blocks = []
            footer_blocks = []
            content_blocks = []
            
            for block in blocks:
                # Block coordinates
                y0 = block[1]  # Top of block
                y1 = block[3]  # Bottom of block
                
                if y0 < header_boundary:
                    header_blocks.append(block)
                elif y1 > footer_boundary:
                    footer_blocks.append(block)
                else:
                    content_blocks.append(block)
            
            # Process headers (if needed)
            header_paragraphs = []
            if header_blocks:
                header_blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by y, then x
                raw_header_paragraphs = self._process_blocks_into_paragraphs(header_blocks)
                header_paragraphs = self.process_sequential_paragraphs(raw_header_paragraphs)
            
            # Process footer blocks (typically excluded)
            footer_paragraphs = []
            if footer_blocks:
                footer_blocks.sort(key=lambda b: (b[1], b[0]))
                raw_footer_paragraphs = self._process_blocks_into_paragraphs(footer_blocks)
                footer_paragraphs = self.process_sequential_paragraphs(raw_footer_paragraphs)
            
            # Process content blocks based on layout
            content_paragraphs = []
            
            if is_double_column and content_blocks:
                # Separate into left and right columns
                left_column = []
                right_column = []
                
                for block in content_blocks:
                    block_center_x = (block[0] + block[2]) / 2
                    
                    if block_center_x < midpoint:
                        left_column.append(block)
                    else:
                        right_column.append(block)
                
                # Sort each column by y-coordinate (top to bottom)
                left_column.sort(key=lambda b: b[1])
                right_column.sort(key=lambda b: b[1])
                
                # Process each column into initial paragraphs
                left_paragraphs = self._process_blocks_into_paragraphs(left_column)
                right_paragraphs = self._process_blocks_into_paragraphs(right_column)
                
                # Combine paragraphs from both columns and process sequentially
                all_column_paragraphs = left_paragraphs + right_paragraphs
                content_paragraphs = self.process_sequential_paragraphs(all_column_paragraphs)
            elif content_blocks:
                # Single column - sort all blocks by y-coordinate
                content_blocks.sort(key=lambda b: b[1])
                raw_paragraphs = self._process_blocks_into_paragraphs(content_blocks)
                content_paragraphs = self.process_sequential_paragraphs(raw_paragraphs)
            
            # Combine paragraphs in the correct order
            all_paragraphs = []
            
            # Add headers first
            all_paragraphs.extend(header_paragraphs)
            
            # Handle cross-page paragraph continuity
            if last_paragraph_info and content_paragraphs:
                prev_text, ends_with_punctuation, word_count = last_paragraph_info
                
                # Join if previous paragraph doesn't end with punctuation or is very short
                if not ends_with_punctuation or word_count < self.min_words_threshold:
                    if content_paragraphs:
                        first_content_para = content_paragraphs[0]
                        joined_paragraph = prev_text + " " + first_content_para
                        content_paragraphs[0] = joined_paragraph
                else:
                    # Add previous paragraph as separate
                    all_paragraphs.append(prev_text)
                
                # Reset previous paragraph info
                last_paragraph_info = None
            
            # Add content paragraphs
            all_paragraphs.extend(content_paragraphs)
            
            # Add footers (typically excluded in most applications)
            # Uncomment if you want to include footers
            # all_paragraphs.extend(footer_paragraphs)
            
            # Check last paragraph for potential continuation to next page
            if content_paragraphs:
                last_para = content_paragraphs[-1]
                ends_with_punctuation = bool(re.search(r'[.!?:;]$', last_para.strip()))
                word_count = len(last_para.split())
                
                # If doesn't end with punctuation or is very short, might continue on next page
                if not ends_with_punctuation or word_count < self.min_words_threshold:
                    last_paragraph_info = (last_para, ends_with_punctuation, word_count)
                    # Remove from current page as we'll carry it to the next
                    all_paragraphs.pop()
            
            # Add processed paragraphs to result
            pages_content.append({
                "page_number": page_num + 1,  # 1-based page numbering
                "paragraphs": all_paragraphs,
                "layout": "double_column" if is_double_column else "single_column"
            })
        
        # If there's still a paragraph carried over at the end of the document, add it
        if last_paragraph_info:
            last_page = pages_content[-1]
            last_page["paragraphs"].append(last_paragraph_info[0])
        
        return pages_content
    
    def _process_blocks_into_paragraphs(self, blocks):
        """
        Internal method to process blocks into initial paragraphs based on vertical spacing.
        
        Args:
            blocks: List of text blocks with position information
            
        Returns:
            List of paragraph texts
        """
        paragraphs = []
        current_paragraph = ""
        
        for i, block in enumerate(blocks):
            text = block[4]
            
            if not text.strip():
                continue
            
            # If we're starting a new paragraph
            if not current_paragraph:
                current_paragraph = text
            else:
                # Check spacing between blocks
                if i > 0:
                    prev_block = blocks[i-1]
                    prev_bottom = prev_block[3]  # y1 (bottom)
                    current_top = block[1]       # y0 (top)
                    
                    spacing = current_top - prev_bottom
                    
                    # If spacing is small, consider it part of the same paragraph
                    if spacing <= self.paragraph_spacing_threshold:
                        current_paragraph += " " + text
                    else:
                        # End current paragraph and start a new one
                        paragraphs.append(current_paragraph)
                        current_paragraph = text
                else:
                    current_paragraph = text
        
        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        return paragraphs
    
    def process_sequential_paragraphs(self, paragraphs):
        """
        Process a list of paragraphs sequentially, joining paragraphs that:
        1. Don't end with punctuation, OR
        2. Have fewer than min_words_threshold words
        
        Args:
            paragraphs: List of paragraphs to process
            
        Returns:
            List of processed paragraphs with appropriate joins
        """
        if not paragraphs:
            return []
        
        result_paragraphs = []
        current_paragraph = paragraphs[0]
        
        # Process paragraphs sequentially
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            
            # Check if current paragraph doesn't end with punctuation or is very short
            word_count = len(current_paragraph.split())
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', current_paragraph.strip()))
            
            # Join if either condition is met
            if not ends_with_punctuation or word_count < self.min_words_threshold:
                # Join with the next paragraph
                current_paragraph += " " + next_paragraph
            else:
                # Current paragraph is complete, add to results and move to next
                result_paragraphs.append(current_paragraph)
                current_paragraph = next_paragraph
        
        # Add the final paragraph
        if current_paragraph:
            result_paragraphs.append(current_paragraph)
        
        return result_paragraphs
    
    def generate_embeddings(self, pages_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for each paragraph using the configured embedding provider.
        
        Args:
            pages_content: List of dictionaries containing page content
            
        Returns:
            Dictionary with paragraph indices and their embeddings
        """
        if not EMBEDDINGS_AVAILABLE or (
            self.embedding_provider == "sentence_transformers" and self.embedding_model is None) or (
            self.embedding_provider == "azure_openai" and self.azure_client is None):
            return {"error": "Embedding generation not available"}
        
        all_paragraphs = []
        paragraph_indices = []
        
        # Collect all paragraphs and their indices
        for page_idx, page in enumerate(pages_content):
            for para_idx, paragraph in enumerate(page["paragraphs"]):
                all_paragraphs.append(paragraph)
                paragraph_indices.append((page_idx, para_idx))
        
        # Generate embeddings based on the selected provider
        try:
            embeddings = None
            
            if self.embedding_provider == "sentence_transformers":
                # Use Sentence Transformers
                embeddings = self.embedding_model.encode(all_paragraphs)
                
                # Create a mapping of indices to embeddings
                embedding_map = {}
                for (page_idx, para_idx), embedding in zip(paragraph_indices, embeddings):
                    if page_idx not in embedding_map:
                        embedding_map[page_idx] = {}
                    embedding_map[page_idx][para_idx] = embedding.tolist()
                
                return embedding_map
                
            elif self.embedding_provider == "azure_openai":
                # Use Azure OpenAI API
                embedding_map = {}
                
                # Process in batches to avoid token limits (max 16 texts per request)
                batch_size = 16
                for i in range(0, len(all_paragraphs), batch_size):
                    batch = all_paragraphs[i:i+batch_size]
                    batch_indices = paragraph_indices[i:i+batch_size]
                    
                    # Get embeddings from Azure OpenAI
                    response = self.azure_client.embeddings.create(
                        input=batch,
                        model=self.embedding_model  # This should be the deployment name
                    )
                    
                    # Extract embeddings from response and map to paragraphs
                    for j, embedding_data in enumerate(response.data):
                        page_idx, para_idx = batch_indices[j]
                        
                        if page_idx not in embedding_map:
                            embedding_map[page_idx] = {}
                        
                        embedding_map[page_idx][para_idx] = embedding_data.embedding
                
                return embedding_map
            
            else:
                return {"error": f"Unknown embedding provider: {self.embedding_provider}"}
            
        except Exception as e:
            return {"error": f"Error generating embeddings: {str(e)}"}
    
    def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
        """
        Save the processed result to a JSON file.
        
        Args:
            result: Dictionary with the processed content
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return output_path


def process_directory(input_dir: str, output_dir: str, generate_embeddings: bool = False) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save JSON results
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        List of results for each processed PDF
    """
    # Create PDF handler
    handler = PDFHandler()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Process each PDF file
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
            print(f"Processing {filename}...")
            result = handler.process_pdf(pdf_path, generate_embeddings)
            
            # Save result to JSON
            handler.save_to_json(result, output_path)
            
            results.append(result)
    
    return results


if __name__ == "__main__":
    # Hardcoded file paths
    input_path = "contracts/sample_contract.pdf"  # Path to a single PDF file
    output_path = "extracted/sample_contract.json"  # Path for the output JSON
    
    # Whether to generate embeddings
    generate_embeddings = True
    
    # Choose embedding provider: "sentence_transformers" or "azure_openai"
    embedding_provider = "azure_openai"
    
    # For Azure OpenAI embeddings, provide these configuration details
    azure_openai_config = {
        "api_key": "your-azure-openai-api-key",
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2023-05-15"
    }
    
    # Process a single file
    if os.path.isfile(input_path):
        # Initialize the handler with appropriate configuration
        if embedding_provider == "azure_openai":
            handler = PDFHandler(
                embedding_provider=embedding_provider,
                embedding_model="text-embedding-ada-002",  # Use your deployment name here
                azure_openai_config=azure_openai_config
            )
        else:
            # Default to sentence_transformers
            handler = PDFHandler(embedding_provider="sentence_transformers")
            
        result = handler.process_pdf(input_path, generate_embeddings)
        handler.save_to_json(result, output_path)
        print(f"Processed {input_path} and saved to {output_path}")
