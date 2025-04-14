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
