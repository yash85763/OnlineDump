"""
Enhanced PDF Handling Module for Contract Analysis

This module provides functionality for extracting information from OCR'd PDF contracts,
determining their layout structure, parsing content into paragraphs, and storing the
results in a structured JSON format.

Features:
- PDF parsability check to ensure quality
- Layout analysis (single or double column) with support for mixed layouts on a single page
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

# Try importing pdfplumber
try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required. Install it with 'pip install pdfplumber'")

# Optional: Import libraries for embeddings
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

EMBEDDINGS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE or AZURE_OPENAI_AVAILABLE


class PDFHandler:
    """Main class for handling PDF extraction and processing."""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 page_continuity_threshold: float = 0.1,
                 min_words_threshold: int = 5,
                 region_spacing_threshold: int = 50,
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
            region_spacing_threshold: Max vertical spacing to group blocks into regions for layout analysis (default 50)
            embedding_model: Name of the embedding model to use
                            - For sentence_transformers: model name like "all-MiniLM-L6-v2"
                            - For azure_openai: deployment name for the embedding model
            embedding_provider: Which embedding provider to use, either "sentence_transformers" or "azure_openai"
            azure_openai_config: Configuration for Azure OpenAI, required if using "azure_openai" provider
        """
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        self.min_words_threshold = min_words_threshold
        self.region_spacing_threshold = region_spacing_threshold
        self.embedding_provider = embedding_provider
        self.azure_openai_config = azure_openai_config or {}
        
        self.embedding_model = None
        self.azure_client = None
        
        if embedding_provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Initialized Sentence Transformers model: {encoding_model}")
            except Exception as e:
                print(f"Warning: Could not load Sentence Transformers model: {e}")
        
        elif embedding_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
            try:
                required_params = ["api_key", "azure_endpoint", "api_version"]
                if not all(param in self.azure_openai_config for param in required_params):
                    missing = [p for p in required_params if p not in self.azure_openai_config]
                    raise ValueError(f"Missing required Azure OpenAI config parameters: {missing}")
                
                self.azure_client = AzureOpenAI(
                    api_key=self.azure_openai_config["api_key"],
                    api_version=self.azure_openai_config["api_version"],
                    azure_endpoint=self.azure_openai_config["azure_endpoint"]
                )
                self.embedding_model = embedding_model
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
            with pdfplumber.open(pdf_path) as doc:
                is_parsable, quality_info = self.check_parsability(doc)
                
                if not is_parsable:
                    return {
                        "filename": os.path.basename(pdf_path),
                        "parsable": False,
                        "error": quality_info
                    }
                
                # Layout is now determined per-page in parse_paragraphs
                pages_content = self.parse_paragraphs(doc)
                
                result = {
                    "filename": os.path.basename(pdf_path),
                    "parsable": True,
                    "pages": pages_content
                }
                
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
    
    def check_parsability(self, doc: pdfplumber.PDF) -> Tuple[bool, str]:
        """
        Check if a PDF is parsable by extracting text and assessing quality.
        
        Args:
            doc: pdfplumber PDF object
            
        Returns:
            Tuple of (is_parsable, message)
        """
        total_text = ""
        total_chars = 0
        alpha_chars = 0
        
        for page in doc.pages:
            text = page.extract_text()
            if text:
                total_text += text + "\n"
        
        if not total_text.strip():
            return False, "No text extracted from PDF. The PDF might need OCR processing."
        
        total_chars = len(total_text)
        alpha_chars = sum(1 for char in total_text if char.isalnum())
        
        if total_chars > 0:
            quality_ratio = alpha_chars / total_chars
        else:
            quality_ratio = 0
        
        avg_chars_per_page = total_chars / len(doc.pages) if doc.pages else 0
        if avg_chars_per_page < 100:
            return False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
        if quality_ratio < self.min_quality_ratio:
            return False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
        return True, f"PDF is parsable with quality ratio {quality_ratio:.2f}"
    
    def determine_layout(self, blocks: List[Dict], page_width: float) -> str:
        """
        Determine if a group of blocks has a single or double column layout.
        
        Args:
            blocks: List of block dictionaries with x0, x1 coordinates
            page_width: Width of the page
            
        Returns:
            String indicating layout type: "single_column" or "double_column"
        """
        x_coordinates = [(block['x0'] + block['x1']) / 2 for block in blocks]
        
        if len(x_coordinates) < 5:
            return "single_column"
        
        try:
            X = np.array(x_coordinates).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            centers = kmeans.cluster_centers_.flatten()
            counts = np.bincount(kmeans.labels_)
            
            center_distance = abs(centers[0] - centers[1])
            
            if (center_distance > page_width * 0.3 and 
                    min(counts) > len(x_coordinates) * 0.15):
                return "double_column"
            else:
                return "single_column"
                
        except Exception:
            return "single_column"
    
    def _segment_page_into_regions(self, blocks: List[Dict]) -> List[List[Dict]]:
        """
        Segment a page's blocks into regions based on vertical spacing.
        
        Args:
            blocks: List of block dictionaries with y0, y1 coordinates
            
        Returns:
            List of regions, where each region is a list of blocks
        """
        if not blocks:
            return []
        
        blocks = sorted(blocks, key=lambda b: b['y0'])
        regions = []
        current_region = [blocks[0]]
        last_y1 = blocks[0]['y1']
        
        for block in blocks[1:]:
            spacing = block['y0'] - last_y1
            if spacing <= self.region_spacing_threshold:
                current_region.append(block)
            else:
                regions.append(current_region)
                current_region = [block]
            last_y1 = block['y1']
        
        if current_region:
            regions.append(current_region)
        
        return regions
    
    def parse_paragraphs(self, doc: pdfplumber.PDF) -> List[Dict[str, Any]]:
        """
        Parse PDF content into paragraphs, handling cross-page and cross-column continuity
        and mixed single/double-column layouts within a page.
        
        Args:
            doc: pdfplumber PDF object
            
        Returns:
            List of dictionaries, each containing page number and paragraphs
        """
        pages_content = []
        last_paragraph_info = None
        
        for page_num, page in enumerate(doc.pages):
            page_height = page.height
            page_width = page.width
            
            text_objects = page.extract_words(keep_blank_chars=True)
            
            if not text_objects:
                pages_content.append({
                    "page_number": page_num + 1,
                    "paragraphs": [],
                    "layout": "single_column"
                })
                continue
            
            blocks = self._group_words_into_blocks(text_objects)
            
            header_boundary = page_height * 0.1
            footer_boundary = page_height * 0.9
            
            header_blocks = []
            footer_blocks = []
            content_blocks = []
            
            for block in blocks:
                y0 = block['y0']
                y1 = block['y1']
                
                if y0 < header_boundary:
                    header_blocks.append(block)
                elif y1 > footer_boundary:
                    footer_blocks.append(block)
                else:
                    content_blocks.append(block)
            
            header_paragraphs = []
            if header_blocks:
                header_blocks.sort(key=lambda b: (b['y0'], b['x0']))
                raw_header_paragraphs = self._process_blocks_into_paragraphs(header_blocks)
                header_paragraphs = self.process_sequential_paragraphs(raw_header_paragraphs)
            
            footer_paragraphs = []
            if footer_blocks:
                footer_blocks.sort(key=lambda b: (b['y0'], b['x0']))
                raw_footer_paragraphs = self._process_blocks_into_paragraphs(footer_blocks)
                footer_paragraphs = self.process_sequential_paragraphs(raw_footer_paragraphs)
            
            # Segment content blocks into regions
            regions = self._segment_page_into_regions(content_blocks)
            content_paragraphs = []
            region_layouts = []
            
            for region_blocks in regions:
                if not region_blocks:
                    continue
                
                # Determine layout for this region
                region_layout = self.determine_layout(region_blocks, page_width)
                region_layouts.append(region_layout)
                
                if region_layout == "double_column":
                    midpoint = page_width / 2
                    x_centers = [(b['x0'] + b['x1']) / 2 for b in region_blocks]
                    if x_centers:
                        X = np.array(x_centers).reshape(-1, 1)
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                        centers = sorted(kmeans.cluster_centers_.flatten())
                        midpoint = (centers[0] + centers[1]) / 2
                    
                    left_column = []
                    right_column = []
                    
                    for block in region_blocks:
                        block_center_x = (block['x0'] + block['x1']) / 2
                        if block_center_x < midpoint:
                            left_column.append(block)
                        else:
                            right_column.append(block)
                    
                    left_column.sort(key=lambda b: b['y0'])
                    right_column.sort(key=lambda b: b['y0'])
                    
                    left_paragraphs = self._process_blocks_into_paragraphs(left_column)
                    right_paragraphs = self._process_blocks_into_paragraphs(right_column)
                    
                    region_paragraphs = left_paragraphs + right_paragraphs
                    region_paragraphs = self.process_sequential_paragraphs(region_paragraphs)
                else:
                    region_blocks.sort(key=lambda b: b['y0'])
                    region_paragraphs = self._process_blocks_into_paragraphs(region_blocks)
                    region_paragraphs = self.process_sequential_paragraphs(region_paragraphs)
                
                content_paragraphs.extend(region_paragraphs)
            
            all_paragraphs = []
            all_paragraphs.extend(header_paragraphs)
            
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
            
            if content_paragraphs:
                last_para = content_paragraphs[-1]
                ends_with_punctuation = bool(re.search(r'[.!?:;]$', last_para.strip()))
                word_count = len(last_para.split())
                
                if not ends_with_punctuation or word_count < self.min_words_threshold:
                    last_paragraph_info = (last_para, ends_with_punctuation, word_count)
                    all_paragraphs.pop()
            
            # Determine overall page layout based on regions
            page_layout = "mixed" if len(set(region_layouts)) > 1 else (region_layouts[0] if region_layouts else "single_column")
            
            pages_content.append({
                "page_number": page_num + 1,
                "paragraphs": all_paragraphs,
                "layout": page_layout,
                "region_layouts": region_layouts
            })
        
        if last_paragraph_info:
            last_page = pages_content[-1]
            last_page["paragraphs"].append(last_paragraph_info[0])
        
        return pages_content
    
    def _group_words_into_blocks(self, words):
        """
        Group individual words into blocks based on proximity.
        
        Args:
            words: List of word dictionaries from pdfplumber
            
        Returns:
            List of block dictionaries with text and coordinates
        """
        blocks = []
        current_block = None
        last_y1 = None
        
        words = sorted(words, key=lambda w: (w['y0'], w['x0']))
        
        for word in words:
            if not word['text'].strip():
                continue
                
            if current_block is None:
                current_block = {
                    'x0': word['x0'],
                    'y0': word['y0'],
                    'x1': word['x1'],
                    'y1': word['y1'],
                    'text': word['text']
                }
                last_y1 = word['y1']
            else:
                spacing = word['y0'] - last_y1
                if spacing <= self.paragraph_spacing_threshold:
                    current_block['text'] += " " + word['text']
                    current_block['x0'] = min(current_block['x0'], word['x0'])
                    current_block['y0'] = min(current_block['y0'], word['y0'])
                    current_block['x1'] = max(current_block['x1'], word['x1'])
                    current_block['y1'] = max(current_block['y1'], word['y1'])
                else:
                    blocks.append(current_block)
                    current_block = {
                        'x0': word['x0'],
                        'y0': word['y0'],
                        'x1': word['x1'],
                        'y1': word['y1'],
                        'text': word['text']
                    }
                last_y1 = word['y1']
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _process_blocks_into_paragraphs(self, blocks):
        """
        Process blocks into initial paragraphs based on vertical spacing.
        
        Args:
            blocks: List of text blocks with position information
            
        Returns:
            List of paragraph texts
        """
        paragraphs = []
        current_paragraph = ""
        
        for i, block in enumerate(blocks):
            text = block['text']
            if not text.strip():
                continue
            
            if not current_paragraph:
                current_paragraph = text
            else:
                if i > 0:
                    prev_block = blocks[i-1]
                    prev_bottom = prev_block['y1']
                    current_top = block['y0']
                    spacing = current_top - prev_bottom
                    
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
        
        Args:
            paragraphs: List of paragraphs to process
            
        Returns:
            List of processed paragraphs with appropriate joins
        """
        if not paragraphs:
            return []
        
        result_paragraphs = []
        current_paragraph = paragraphs[0]
        
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            word_count = len(current_paragraph.split())
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', current_paragraph.strip()))
            
            if not ends_with_punctuation or word_count < self.min_words_threshold:
                current_paragraph += " " + next_paragraph
            else:
                result_paragraphs.append(current_paragraph)
                current_paragraph = next_paragraph
        
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
        
        for page_idx, page in enumerate(pages_content):
            for para_idx, paragraph in enumerate(page["paragraphs"]):
                all_paragraphs.append(paragraph)
                paragraph_indices.append((page_idx, para_idx))
        
        try:
            embeddings = None
            
            if self.embedding_provider == "sentence_transformers":
                embeddings = self.embedding_model.encode(all_paragraphs)
                embedding_map = {}
                for (page_idx, para_idx), embedding in zip(paragraph_indices, embeddings):
                    if page_idx not in embedding_map:
                        embedding_map[page_idx] = {}
                    embedding_map[page_idx][para_idx] = embedding.tolist()
                
                return embedding_map
                
            elif self.embedding_provider == "azure_openai":
                embedding_map = {}
                batch_size = 16
                for i in range(0, len(all_paragraphs), batch_size):
                    batch = all_paragraphs[i:i+batch_size]
                    batch_indices = paragraph_indices[i:i+batch_size]
                    
                    response = self.azure_client.embeddings.create(
                        input=batch,
                        model=self.embedding_model
                    )
                    
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    handler = PDFHandler()
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
            print(f"Processing {filename}...")
            result = handler.process_pdf(pdf_path, generate_embeddings)
            handler.save_to_json(result, output_path)
            results.append(result)
    
    return results


if __name__ == "__main__":
    input_path = "contracts/sample_contract.pdf"
    output_path = "extracted/sample_contract.json"
    generate_embeddings = True
    embedding_provider = "azure_openai"
    
    azure_openai_config = {
        "api_key": "your-azure-openai-api-key",
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2023-05-15"
    }
    
    if os.path.isfile(input_path):
        if embedding_provider == "azure_openai":
            handler = PDFHandler(
                embedding_provider=embedding_provider,
                embedding_model="text-embedding-ada-002",
                azure_openai_config=azure_openai_config
            )
        else:
            handler = PDFHandler(embedding_provider="sentence_transformers")
            
        result = handler.process_pdf(input_path, generate_embeddings)
        handler.save_to_json(result, output_path)
        print(f"Processed {input_path} and saved to {output_path}")
