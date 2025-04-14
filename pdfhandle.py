"""
PDF Handling Module for Contract Analysis

This module provides functionality for extracting information from OCR'd PDF contracts,
determining their layout structure, parsing content into paragraphs, and storing the
results in a structured JSON format.

Features:
- PDF parsability check to ensure quality
- Layout analysis (single or double column)
- Paragraph extraction with cross-page continuity handling
- JSON storage with optional embedding generation
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans

# Try importing fitz (PyMuPDF)
try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required. Install it with 'pip install pymupdf'")

# Optional: Import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class PDFHandler:
    """Main class for handling PDF extraction and processing."""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 page_continuity_threshold: float = 0.1,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PDF handler with configurable thresholds.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric chars to total chars (default 0.5)
            paragraph_spacing_threshold: Max vertical spacing between text blocks to be considered
                                         part of the same paragraph, in points (default 10)
            page_continuity_threshold: Percentage of page height to check for paragraph 
                                       continuation across pages (default 0.1 or 10%)
            embedding_model: Name of the sentence-transformer model to use for embeddings
        """
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        
        # Initialize the embedding model if available
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
    
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
            # Check if the PDF is parsable
            doc = fitz.open(pdf_path)
            is_parsable, quality_info = self.check_parsability(doc)
            
            if not is_parsable:
                return {
                    "filename": os.path.basename(pdf_path),
                    "parsable": False,
                    "error": quality_info
                }
            
            # Determine the layout
            layout_type = self.determine_layout(doc)
            
            # Parse content into paragraphs
            pages_content = self.parse_paragraphs(doc)
            
            # Generate JSON structure
            result = {
                "filename": os.path.basename(pdf_path),
                "parsable": True,
                "layout": layout_type,
                "pages": pages_content
            }
            
            # Optionally generate embeddings
            if generate_embeddings and self.embedding_model is not None:
                result["embeddings"] = self.generate_embeddings(pages_content)
            
            return result
            
        except Exception as e:
            return {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": f"Error processing PDF: {str(e)}"
            }
    
    def check_parsability(self, doc: fitz.Document) -> Tuple[bool, str]:
        """
        Check if a PDF is parsable by extracting text and assessing quality.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Tuple of (is_parsable, message)
        """
        total_text = ""
        total_chars = 0
        alpha_chars = 0
        
        # Extract all text from the document
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
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
        avg_chars_per_page = total_chars / len(doc)
        if avg_chars_per_page < 100:  # Arbitrary threshold, adjust as needed
            return False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
        # Check quality ratio against threshold
        if quality_ratio < self.min_quality_ratio:
            return False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
        return True, f"PDF is parsable with quality ratio {quality_ratio:.2f}"
    
    def determine_layout(self, doc: fitz.Document) -> str:
        """
        Determine if the PDF has a single or double column layout.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            String indicating layout type: "single_column" or "double_column"
        """
        # Collect x-coordinates of text blocks from multiple pages
        x_coordinates = []
        
        # Sample a few pages for efficiency (first 3 pages or all if less)
        num_pages_to_check = min(3, len(doc))
        
        for page_num in range(num_pages_to_check):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            
            # Extract x-coordinates of the blocks (middle point of the block)
            for block in blocks:
                x_mid = (block[0] + block[2]) / 2  # (x0 + x2) / 2
                x_coordinates.append(x_mid)
        
        # If we don't have enough blocks for clustering, assume single column
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
            page = doc.load_page(0)
            page_width = page.rect.width
            
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
    
    def parse_paragraphs(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Parse PDF content into paragraphs, handling cross-page continuity.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of dictionaries, each containing page number and paragraphs
        """
        pages_content = []
        last_block_info = None  # Store info about the last block of the previous page
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_height = page.rect.height
            
            # Extract text blocks
            blocks = page.get_text("blocks")
            
            # Sort blocks by y-coordinate (top to bottom)
            blocks.sort(key=lambda b: b[1])  # Sort by y0 (top coordinate)
            
            current_paragraphs = []
            current_paragraph = ""
            
            # Check if we need to continue a paragraph from the previous page
            if last_block_info and blocks:
                prev_text, prev_has_end_punctuation = last_block_info
                
                # Get first block of current page
                first_block = blocks[0]
                first_block_text = first_block[4]
                
                # Check if first block starts with a lowercase letter (potential continuation)
                first_char = first_block_text.strip()[0] if first_block_text.strip() else ""
                is_lowercase_start = first_char.islower() if first_char.isalpha() else False
                
                # If the previous block didn't end with punctuation and the current starts lowercase,
                # consider it a continuation
                if not prev_has_end_punctuation and is_lowercase_start:
                    current_paragraph = prev_text + " " + first_block_text
                    blocks = blocks[1:]  # Remove the first block as it's been processed
                else:
                    # Add the previous paragraph as a separate paragraph
                    current_paragraphs.append(prev_text)
            
            # Process remaining blocks
            for i, block in enumerate(blocks):
                text = block[4]
                
                if not text.strip():
                    continue
                
                # If we're starting a new paragraph
                if not current_paragraph:
                    current_paragraph = text
                else:
                    # Check vertical spacing between blocks
                    prev_block = blocks[i-1]
                    prev_bottom = prev_block[3]  # y1 (bottom)
                    current_top = block[1]      # y0 (top)
                    
                    spacing = current_top - prev_bottom
                    
                    # If spacing is small, consider it part of the same paragraph
                    if spacing <= self.paragraph_spacing_threshold:
                        current_paragraph += " " + text
                    else:
                        # End current paragraph and start a new one
                        current_paragraphs.append(current_paragraph)
                        current_paragraph = text
            
            # Add the last paragraph if it exists
            if current_paragraph:
                current_paragraphs.append(current_paragraph)
            
            # Store info about the last block for potential cross-page continuity
            if blocks:
                last_block = blocks[-1]
                last_text = last_block[4]
                last_bottom = last_block[3]  # y1 (bottom)
                
                # Check if the last block is near the bottom of the page
                is_near_bottom = (page_height - last_bottom) < (page_height * self.page_continuity_threshold)
                
                # Check if the last block ends with punctuation
                has_end_punctuation = bool(re.search(r'[.!?;:]$', last_text.strip()))
                
                # If it's near the bottom and doesn't end with punctuation, 
                # it might continue on the next page
                if is_near_bottom and not has_end_punctuation:
                    # Remove the last paragraph as we'll carry it to the next page
                    if current_paragraphs:
                        last_paragraph = current_paragraphs.pop()
                        last_block_info = (last_paragraph, has_end_punctuation)
                else:
                    last_block_info = None
            else:
                last_block_info = None
            
            # Add page content to the result
            pages_content.append({
                "page_number": page_num + 1,  # 1-based page numbering
                "paragraphs": current_paragraphs
            })
        
        return pages_content
    
    def generate_embeddings(self, pages_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for each paragraph using sentence-transformers.
        
        Args:
            pages_content: List of dictionaries containing page content
            
        Returns:
            Dictionary with paragraph indices and their embeddings
        """
        if not EMBEDDINGS_AVAILABLE or self.embedding_model is None:
            return {"error": "Embedding generation not available"}
        
        all_paragraphs = []
        paragraph_indices = []
        
        # Collect all paragraphs and their indices
        for page_idx, page in enumerate(pages_content):
            for para_idx, paragraph in enumerate(page["paragraphs"]):
                all_paragraphs.append(paragraph)
                paragraph_indices.append((page_idx, para_idx))
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(all_paragraphs)
            
            # Create a mapping of indices to embeddings
            embedding_map = {}
            for (page_idx, para_idx), embedding in zip(paragraph_indices, embeddings):
                if page_idx not in embedding_map:
                    embedding_map[page_idx] = {}
                embedding_map[page_idx][para_idx] = embedding.tolist()
            
            return embedding_map
            
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
    # Hardcoded file paths instead of command line arguments
    input_path = "contracts/sample_contract.pdf"  # Path to a single PDF file
    output_path = "extracted/sample_contract.json"  # Path for the output JSON
    
    # Alternative directory paths for batch processing
    input_directory = "contracts/"  # Directory containing PDF files
    output_directory = "extracted/"  # Directory for output JSON files
    
    # Whether to generate embeddings
    generate_embeddings = False
    
    # Process a single file
    if os.path.isfile(input_path):
        handler = PDFHandler()
        result = handler.process_pdf(input_path, generate_embeddings)
        handler.save_to_json(result, output_path)
        print(f"Processed {input_path} and saved to {output_path}")
    
    # Uncomment the following lines to process a directory instead
    # if os.path.isdir(input_directory):
    #     results = process_directory(input_directory, output_directory, generate_embeddings)
    #     print(f"Processed {len(results)} PDF files from {input_directory} and saved to {output_directory}")
    #
    # else:
    #     print(f"Input path does not exist")




# -----------------------------WITHOUT PYMUPDF --------------------------------------------
# """
# PDF Handling Module for Contract Analysis

# This module provides functionality for extracting information from OCR'd PDF contracts,
# determining their layout structure, parsing content into paragraphs, and storing the
# results in a structured JSON format. This implementation uses PyPDF2, PDFPlumber, and 
# PDFMiner.six instead of PyMuPDF.

# Features:
# - PDF parsability check to ensure quality
# - Layout analysis (single or double column)
# - Paragraph extraction with cross-page continuity handling
# - JSON storage with optional embedding generation
# """

# import os
# import json
# import re
# import io
# import numpy as np
# from typing import List, Dict, Tuple, Optional, Any
# from sklearn.cluster import KMeans
# from collections import defaultdict

# # Required libraries
# try:
#     import PyPDF2
#     from pdfminer.high_level import extract_pages
#     from pdfminer.layout import LTTextContainer, LTTextBox, LTTextLine, LAParams
#     import pdfplumber
# except ImportError:
#     raise ImportError("Required libraries missing. Install with: 'pip install PyPDF2 pdfminer.six pdfplumber'")

# # Optional: Import libraries for embeddings
# try:
#     # For sentence-transformers
#     from sentence_transformers import SentenceTransformer
#     import faiss
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False

# # For Azure OpenAI embeddings
# try:
#     from openai import AzureOpenAI
#     AZURE_OPENAI_AVAILABLE = True
# except ImportError:
#     AZURE_OPENAI_AVAILABLE = False

# # Check if any embedding method is available
# EMBEDDINGS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE or AZURE_OPENAI_AVAILABLE


# class PDFHandler:
#     """Main class for handling PDF extraction and processing."""
    
#     def __init__(self, 
#                  min_quality_ratio: float = 0.5,
#                  paragraph_spacing_threshold: int = 10,
#                  page_continuity_threshold: float = 0.1,
#                  embedding_model: str = "all-MiniLM-L6-v2",
#                  embedding_provider: str = "sentence_transformers",
#                  azure_openai_config: Dict[str, str] = None):
#         """
#         Initialize the PDF handler with configurable thresholds.
        
#         Args:
#             min_quality_ratio: Minimum ratio of alphanumeric chars to total chars (default 0.5)
#             paragraph_spacing_threshold: Max vertical spacing between text blocks to be considered
#                                          part of the same paragraph, in points (default 10)
#             page_continuity_threshold: Percentage of page height to check for paragraph 
#                                        continuation across pages (default 0.1 or 10%)
#             embedding_model: Name of the embedding model to use
#                             - For sentence_transformers: model name like "all-MiniLM-L6-v2"
#                             - For azure_openai: deployment name for the embedding model
#             embedding_provider: Which embedding provider to use, either "sentence_transformers" or "azure_openai"
#             azure_openai_config: Configuration for Azure OpenAI, required if using "azure_openai" provider:
#                                 - api_key: Azure OpenAI API key
#                                 - azure_endpoint: Azure OpenAI endpoint URL
#                                 - api_version: API version (e.g., "2023-05-15")
#         """
#         self.min_quality_ratio = min_quality_ratio
#         self.paragraph_spacing_threshold = paragraph_spacing_threshold
#         self.page_continuity_threshold = page_continuity_threshold
#         self.embedding_provider = embedding_provider
#         self.azure_openai_config = azure_openai_config or {}
        
#         # Initialize the embedding model if available
#         self.embedding_model = None
#         self.azure_client = None
        
#         if embedding_provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
#             try:
#                 self.embedding_model = SentenceTransformer(embedding_model)
#                 print(f"Initialized Sentence Transformers model: {embedding_model}")
#             except Exception as e:
#                 print(f"Warning: Could not load Sentence Transformers model: {e}")
        
#         elif embedding_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
#             try:
#                 # Check if we have all the required config parameters
#                 required_params = ["api_key", "azure_endpoint", "api_version"]
#                 if not all(param in self.azure_openai_config for param in required_params):
#                     missing = [p for p in required_params if p not in self.azure_openai_config]
#                     raise ValueError(f"Missing required Azure OpenAI config parameters: {missing}")
                
#                 self.azure_client = AzureOpenAI(
#                     api_key=self.azure_openai_config["api_key"],
#                     api_version=self.azure_openai_config["api_version"],
#                     azure_endpoint=self.azure_openai_config["azure_endpoint"]
#                 )
#                 self.embedding_model = embedding_model  # Store the deployment name
#                 print(f"Initialized Azure OpenAI client with deployment: {embedding_model}")
#             except Exception as e:
#                 print(f"Warning: Could not initialize Azure OpenAI client: {e}")
    
#     def process_pdf(self, pdf_path: str, generate_embeddings: bool = False) -> Dict[str, Any]:
#         """
#         Process a PDF file through the complete pipeline.
        
#         Args:
#             pdf_path: Path to the PDF file
#             generate_embeddings: Whether to generate embeddings for paragraphs
            
#         Returns:
#             Dictionary with the processed content or an error message
#         """
#         try:
#             # Check if the PDF is parsable
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 is_parsable, quality_info = self.check_parsability(reader)
                
#                 if not is_parsable:
#                     return {
#                         "filename": os.path.basename(pdf_path),
#                         "parsable": False,
#                         "error": quality_info
#                     }
                
#                 # Determine the layout
#                 layout_type = self.determine_layout(pdf_path)
                
#                 # Parse content into paragraphs
#                 pages_content = self.parse_paragraphs(pdf_path)
                
#                 # Generate JSON structure
#                 result = {
#                     "filename": os.path.basename(pdf_path),
#                     "parsable": True,
#                     "layout": layout_type,
#                     "pages": pages_content
#                 }
                
#                 # Optionally generate embeddings
#                 if generate_embeddings and self.embedding_model is not None:
#                     result["embeddings"] = self.generate_embeddings(pages_content)
                
#                 return result
            
#         except Exception as e:
#             return {
#                 "filename": os.path.basename(pdf_path),
#                 "parsable": False,
#                 "error": f"Error processing PDF: {str(e)}"
#             }
    
#     def check_parsability(self, reader: PyPDF2.PdfReader) -> Tuple[bool, str]:
#         """
#         Check if a PDF is parsable by extracting text and assessing quality.
        
#         Args:
#             reader: PyPDF2 PdfReader object
            
#         Returns:
#             Tuple of (is_parsable, message)
#         """
#         total_text = ""
#         total_chars = 0
#         alpha_chars = 0
        
#         # Extract all text from the document
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text = page.extract_text() or ""
#             total_text += text
        
#         # If no text was extracted, the PDF might not be OCR'd or has issues
#         if not total_text.strip():
#             return False, "No text extracted from PDF. The PDF might need OCR processing."
        
#         # Count alphanumeric characters vs. total characters
#         total_chars = len(total_text)
#         alpha_chars = sum(1 for char in total_text if char.isalnum())
        
#         # Calculate quality ratio
#         if total_chars > 0:
#             quality_ratio = alpha_chars / total_chars
#         else:
#             quality_ratio = 0
        
#         # Check if text length is reasonable for the number of pages
#         avg_chars_per_page = total_chars / len(reader.pages)
#         if avg_chars_per_page < 100:  # Arbitrary threshold, adjust as needed
#             return False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
#         # Check quality ratio against threshold
#         if quality_ratio < self.min_quality_ratio:
#             return False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
#         return True, f"PDF is parsable with quality ratio {quality_ratio:.2f}"
    
#     def determine_layout(self, pdf_path: str) -> str:
#         """
#         Determine if the PDF has a single or double column layout using PDFMiner.
        
#         Args:
#             pdf_path: Path to the PDF file
            
#         Returns:
#             String indicating layout type: "single_column" or "double_column"
#         """
#         # Using PDFMiner to extract text elements with position information
#         x_coordinates = []
#         page_width = 0
        
#         # Sample up to 3 pages for efficiency
#         page_count = 0
#         max_pages = 3
        
#         with open(pdf_path, 'rb') as file:
#             # Get page dimensions from PDFPlumber for the first page
#             with pdfplumber.open(pdf_path) as pdf:
#                 if len(pdf.pages) > 0:
#                     first_page = pdf.pages[0]
#                     page_width = first_page.width
            
#             # Get text box positions from PDFMiner
#             laparams = LAParams()
#             for page_layout in extract_pages(file, maxpages=max_pages, laparams=laparams):
#                 page_count += 1
                
#                 # Extract positions of text boxes
#                 for element in page_layout:
#                     if isinstance(element, LTTextBox):
#                         # Calculate center x-coordinate of the text box
#                         x_mid = (element.x0 + element.x1) / 2
#                         x_coordinates.append(x_mid)
                
#                 # Stop after max_pages
#                 if page_count >= max_pages:
#                     break
        
#         # If we don't have enough blocks for clustering, assume single column
#         if len(x_coordinates) < 5:
#             return "single_column"
        
#         # Use K-means clustering to identify column structure
#         try:
#             # Reshape for sklearn
#             X = np.array(x_coordinates).reshape(-1, 1)
            
#             # Try clustering with k=2 (assuming either 1 or 2 columns)
#             kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#             centers = kmeans.cluster_centers_.flatten()
#             counts = np.bincount(kmeans.labels_)
            
#             # Calculate the distance between cluster centers
#             center_distance = abs(centers[0] - centers[1])
            
#             # If the centers are far apart (relative to page width) and both clusters have 
#             # a significant number of blocks, classify as double column
#             if (center_distance > page_width * 0.3 and 
#                     min(counts) > len(x_coordinates) * 0.15):
#                 return "double_column"
#             else:
#                 return "single_column"
                
#         except Exception:
#             # If clustering fails, default to single column
#             return "single_column"
    
#     def parse_paragraphs(self, pdf_path: str) -> List[Dict[str, Any]]:
#         """
#         Parse PDF content into paragraphs using PDFMiner, handling cross-page continuity.
        
#         Args:
#             pdf_path: Path to the PDF file
            
#         Returns:
#             List of dictionaries, each containing page number and paragraphs
#         """
#         pages_content = []
#         last_block_info = None  # Store info about the last block of the previous page
        
#         # Use PDFPlumber to get page dimensions
#         page_heights = []
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_heights.append(page.height)
        
#         # Use PDFMiner for detailed text extraction with positioning
#         with open(pdf_path, 'rb') as file:
#             laparams = LAParams()
            
#             # First pass: extract all text elements with positions
#             all_pages_elements = []
#             for page_layout in extract_pages(file, laparams=laparams):
#                 page_elements = []
                
#                 for element in page_layout:
#                     if isinstance(element, LTTextContainer):
#                         # Store text and position info
#                         text = element.get_text().strip()
#                         if text:
#                             page_elements.append({
#                                 'text': text,
#                                 'x0': element.x0,
#                                 'y0': element.y0,  # In PDFMiner, y0 is bottom, not top
#                                 'x1': element.x1,
#                                 'y1': element.y1,
#                                 'height': element.height,
#                                 'width': element.width
#                             })
                
#                 # Sort by y-coordinate (bottom to top in PDFMiner)
#                 # We need to reverse it to get top to bottom
#                 page_elements.sort(key=lambda e: -e['y0'])
#                 all_pages_elements.append(page_elements)
        
#         # Process each page
#         for page_num, page_elements in enumerate(all_pages_elements):
#             current_paragraphs = []
#             current_paragraph = ""
            
#             # Get page height
#             page_height = page_heights[page_num] if page_num < len(page_heights) else 792  # Default to US Letter
            
#             # Check if we need to continue a paragraph from the previous page
#             if last_block_info and page_elements:
#                 prev_text, prev_has_end_punctuation = last_block_info
                
#                 # Get first block of current page
#                 first_block = page_elements[0]
#                 first_block_text = first_block['text']
                
#                 # Check if first block starts with a lowercase letter (potential continuation)
#                 first_char = first_block_text.strip()[0] if first_block_text.strip() else ""
#                 is_lowercase_start = first_char.islower() if first_char.isalpha() else False
                
#                 # If the previous block didn't end with punctuation and the current starts lowercase,
#                 # consider it a continuation
#                 if not prev_has_end_punctuation and is_lowercase_start:
#                     current_paragraph = prev_text + " " + first_block_text
#                     page_elements = page_elements[1:]  # Remove the first block as it's been processed
#                 else:
#                     # Add the previous paragraph as a separate paragraph
#                     current_paragraphs.append(prev_text)
            
#             # Process remaining blocks
#             for i, element in enumerate(page_elements):
#                 text = element['text']
                
#                 if not text.strip():
#                     continue
                
#                 # If we're starting a new paragraph
#                 if not current_paragraph:
#                     current_paragraph = text
#                 else:
#                     # Check vertical spacing between elements
#                     if i > 0:
#                         prev_element = page_elements[i-1]
                        
#                         # In PDFMiner, y0 is the bottom of the element
#                         # Calculate spacing as the difference between the bottom of the current and the top of the previous
#                         # Note: Since we sorted from top to bottom, we use curr_bottom - prev_top
#                         prev_top = prev_element['y1']
#                         curr_bottom = element['y0']
                        
#                         # PDFMiner coordinates are from bottom, so spacing is prev_top - curr_bottom
#                         spacing = prev_top - curr_bottom
                        
#                         # If spacing is small, consider it part of the same paragraph
#                         if spacing <= self.paragraph_spacing_threshold:
#                             current_paragraph += " " + text
#                         else:
#                             # End current paragraph and start a new one
#                             current_paragraphs.append(current_paragraph)
#                             current_paragraph = text
#                     else:
#                         # First element on the page (and not a continuation)
#                         current_paragraph = text
            
#             # Add the last paragraph if it exists
#             if current_paragraph:
#                 current_paragraphs.append(current_paragraph)
            
#             # Store info about the last block for potential cross-page continuity
#             if page_elements:
#                 last_element = page_elements[-1]
#                 last_text = last_element['text']
                
#                 # Check if the last element is near the bottom of the page
#                 # In PDFMiner, y0 is the bottom of the text
#                 last_bottom = last_element['y0']
#                 is_near_bottom = last_bottom < (page_height * self.page_continuity_threshold)
                
#                 # Check if the last block ends with punctuation
#                 has_end_punctuation = bool(re.search(r'[.!?;:]$', last_text.strip()))
                
#                 # If it's near the bottom and doesn't end with punctuation, 
#                 # it might continue on the next page
#                 if is_near_bottom and not has_end_punctuation:
#                     # Remove the last paragraph as we'll carry it to the next page
#                     if current_paragraphs:
#                         last_paragraph = current_paragraphs.pop()
#                         last_block_info = (last_paragraph, has_end_punctuation)
#                 else:
#                     last_block_info = None
#             else:
#                 last_block_info = None
            
#             # Add page content to the result
#             pages_content.append({
#                 "page_number": page_num + 1,  # 1-based page numbering
#                 "paragraphs": current_paragraphs
#             })
        
#         return pages_content
    
#     def generate_embeddings(self, pages_content: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Generate embeddings for each paragraph using the configured embedding provider.
        
#         Args:
#             pages_content: List of dictionaries containing page content
            
#         Returns:
#             Dictionary with paragraph indices and their embeddings
#         """
#         if not EMBEDDINGS_AVAILABLE or (
#             self.embedding_provider == "sentence_transformers" and self.embedding_model is None) or (
#             self.embedding_provider == "azure_openai" and self.azure_client is None):
#             return {"error": "Embedding generation not available"}
        
#         all_paragraphs = []
#         paragraph_indices = []
        
#         # Collect all paragraphs and their indices
#         for page_idx, page in enumerate(pages_content):
#             for para_idx, paragraph in enumerate(page["paragraphs"]):
#                 all_paragraphs.append(paragraph)
#                 paragraph_indices.append((page_idx, para_idx))
        
#         # Generate embeddings based on the selected provider
#         try:
#             embeddings = None
            
#             if self.embedding_provider == "sentence_transformers":
#                 # Use Sentence Transformers
#                 embeddings = self.embedding_model.encode(all_paragraphs)
                
#                 # Create a mapping of indices to embeddings
#                 embedding_map = {}
#                 for (page_idx, para_idx), embedding in zip(paragraph_indices, embeddings):
#                     if page_idx not in embedding_map:
#                         embedding_map[page_idx] = {}
#                     embedding_map[page_idx][para_idx] = embedding.tolist()
                
#                 return embedding_map
                
#             elif self.embedding_provider == "azure_openai":
#                 # Use Azure OpenAI API
#                 embedding_map = {}
                
#                 # Process in batches to avoid token limits (max 16 texts per request)
#                 batch_size = 16
#                 for i in range(0, len(all_paragraphs), batch_size):
#                     batch = all_paragraphs[i:i+batch_size]
#                     batch_indices = paragraph_indices[i:i+batch_size]
                    
#                     # Get embeddings from Azure OpenAI
#                     response = self.azure_client.embeddings.create(
#                         input=batch,
#                         model=self.embedding_model  # This should be the deployment name
#                     )
                    
#                     # Extract embeddings from response and map to paragraphs
#                     for j, embedding_data in enumerate(response.data):
#                         page_idx, para_idx = batch_indices[j]
                        
#                         if page_idx not in embedding_map:
#                             embedding_map[page_idx] = {}
                        
#                         embedding_map[page_idx][para_idx] = embedding_data.embedding
                
#                 return embedding_map
            
#             else:
#                 return {"error": f"Unknown embedding provider: {self.embedding_provider}"}
            
#         except Exception as e:
#             return {"error": f"Error generating embeddings: {str(e)}"}
    
#     def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
#         """
#         Save the processed result to a JSON file.
        
#         Args:
#             result: Dictionary with the processed content
#             output_path: Path to save the JSON file
            
#         Returns:
#             Path to the saved JSON file
#         """
#         # Create output directory if it doesn't exist
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         # Save to JSON
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, ensure_ascii=False, indent=2)
        
#         return output_path


# def process_directory(input_dir: str, output_dir: str, generate_embeddings: bool = False) -> List[Dict[str, Any]]:
#     """
#     Process all PDF files in a directory.
    
#     Args:
#         input_dir: Directory containing PDF files
#         output_dir: Directory to save JSON results
#         generate_embeddings: Whether to generate embeddings
        
#     Returns:
#         List of results for each processed PDF
#     """
#     # Create PDF handler
#     handler = PDFHandler()
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     results = []
    
#     # Process each PDF file
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith('.pdf'):
#             pdf_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
#             print(f"Processing {filename}...")
#             result = handler.process_pdf(pdf_path, generate_embeddings)
            
#             # Save result to JSON
#             handler.save_to_json(result, output_path)
            
#             results.append(result)
    
#     return results


# if __name__ == "__main__":
#     # Hardcoded file paths instead of command line arguments
#     input_path = "contracts/sample_contract.pdf"  # Path to a single PDF file
#     output_path = "extracted/sample_contract.json"  # Path for the output JSON
    
#     # Alternative directory paths for batch processing
#     input_directory = "contracts/"  # Directory containing PDF files
#     output_directory = "extracted/"  # Directory for output JSON files
    
#     # Whether to generate embeddings
#     generate_embeddings = True
    
#     # Choose embedding provider: "sentence_transformers" or "azure_openai"
#     embedding_provider = "azure_openai"
    
#     # For Azure OpenAI embeddings, provide these configuration details
#     azure_openai_config = {
#         "api_key": "your-azure-openai-api-key",
#         "azure_endpoint": "https://your-resource-name.openai.azure.com/",
#         "api_version": "2023-05-15"
#     }
    
#     # Process a single file
#     if os.path.isfile(input_path):
#         # Initialize the handler with appropriate configuration
#         if embedding_provider == "azure_openai":
#             handler = PDFHandler(
#                 embedding_provider=embedding_provider,
#                 embedding_model="text-embedding-ada-002",  # Use your deployment name here
#                 azure_openai_config=azure_openai_config
#             )
#         else:
#             # Default to sentence_transformers
#             handler = PDFHandler(embedding_provider="sentence_transformers")
            
#         result = handler.process_pdf(input_path, generate_embeddings)
#         handler.save_to_json(result, output_path)
#         print(f"Processed {input_path} and saved to {output_path}")
    
#     # Uncomment the following lines to process a directory instead
#     # if os.path.isdir(input_directory):
#     #     # Use the same handler initialization as above
#     #     if embedding_provider == "azure_openai":
#     #         handler = PDFHandler(
#     #             embedding_provider=embedding_provider,
#     #             embedding_model="text-embedding-ada-002",
#     #             azure_openai_config=azure_openai_config
#     #         )
#     #     else:
#     #         handler = PDFHandler(embedding_provider="sentence_transformers")
#     #         
#     #     # Custom directory processing with the configured handler
#     #     results = []
#     #     for filename in os.listdir(input_directory):
#     #         if filename.lower().endswith('.pdf'):
#     #             pdf_path = os.path.join(input_directory, filename)
#     #             output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")
#     #             
#     #             print(f"Processing {filename}...")
#     #             result = handler.process_pdf(pdf_path, generate_embeddings)
#     #             
#     #             # Save result to JSON
#     #             handler.save_to_json(result, output_path)
#     #             
#     #             results.append(result)
#     #     
#     #     print(f"Processed {len(results)} PDF files from {input_directory} and saved to {output_directory}")
#     #
#     # else:
#     #     print(f"Input path does not exist")
