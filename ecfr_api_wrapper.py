"""
Enhanced PDF Handling Module for Contract Analysis

This module provides functionality for extracting information from OCR'd PDF contracts,
determining their layout structure, parsing content into paragraphs, and storing the
results in structured JSON or text format.

Features:
- PDF parsability check to ensure quality
- Layout analysis (single or double column)
- Paragraph extraction with cross-page and cross-column continuity handling
- Short paragraph handling (fewer than 5 words)
- Punctuation-based paragraph joining
- JSON and text storage with optional embedding generation
"""

import os
import json
import re
import io
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# [Existing imports remain unchanged]
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

class OutputFormatter(ABC):
    """Abstract base class for output formatters."""
    
    @abstractmethod
    def format_output(self, result: Dict[str, Any], output_path: Optional[str] = None) -> Any:
        pass

class JSONFormatter(OutputFormatter):
    """Formatter for JSON output."""
    
    def format_output(self, result: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

class TextFormatter(OutputFormatter):
    """Formatter for text output."""
    
    def format_output(self, result: Dict[str, Any], output_path: Optional[str] = None) -> str:
        text_output = []
        for page in result.get("pages", []):
            for paragraph in page.get("paragraphs", []):
                cleaned_para = self._clean_paragraph(paragraph)
                if cleaned_para:
                    text_output.append(cleaned_para)
        
        formatted_text = "\n\n".join(text_output)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
        
        return formatted_text
    
    @staticmethod
    def _clean_paragraph(paragraph: str) -> str:
        """Clean a single paragraph for text output."""
        paragraph = re.sub(r'\s+', ' ', paragraph.strip())
        return paragraph

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
        # [Existing initialization code remains unchanged]
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        self.min_words_threshold = min_words_threshold
        self.embedding_provider = embedding_provider
        self.azure_openai_config = azure_openai_config or {}
        
        self.laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True
        )
        
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
        
        # Initialize formatters
        self.formatters = {
            'json': JSONFormatter(),
            'txt': TextFormatter()
        }

    @contextmanager
    def _process_context(self, pdf_path: str, generate_embeddings: bool):
        """Context manager for processing PDF content."""
        try:
            pages_data, is_parsable, quality_info = self.extract_pdf_content(pdf_path)
            
            if not is_parsable:
                yield {
                    "filename": os.path.basename(pdf_path),
                    "parsable": False,
                    "error": quality_info
                }
                return
            
            layout_type = self.determine_layout(pages_data)
            pages_content = self.parse_paragraphs(pages_data)
            
            result = {
                "filename": os.path.basename(pdf_path),
                "parsable": True,
                "layout": layout_type,
                "pages": pages_content
            }
            
            if generate_embeddings and (
                (self.embedding_provider == "sentence_transformers" and self.embedding_model is not None) or
                (self.embedding_provider == "azure_openai" and self.azure_client is not None)):
                result["embeddings"] = self.generate_embeddings(pages_content)
            
            yield result
            
        except Exception as e:
            yield {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": f"Error processing PDF: {str(e)}"
            }

    class ProcessPDF:
        """Inner class for method chaining to select output format."""
        
        def __init__(self, handler: 'PDFHandler', pdf_path: str, generate_embeddings: bool):
            self.handler = handler
            self.pdf_path = pdf_path
            self.generate_embeddings = generate_embeddings
            self.result = None
        
        def __enter__(self):
            self.context = self.handler._process_context(self.pdf_path, self.generate_embeddings)
            self.result = next(self.context)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
            """Return JSON format or save to JSON file."""
            return self.handler.formatters['json'].format_output(self.result, output_path)
        
        def txt(self, output_path: Optional[str] = None) -> str:
            """Return text format or save to text file."""
            return self.handler.formatters['txt'].format_output(self.result, output_path)

    def process_pdf(self, pdf_path: str, generate_embeddings: bool = False) -> 'ProcessPDF':
        """
        Process a PDF file and return a ProcessPDF object for format selection.
        
        Args:
            pdf_path: Path to the PDF file
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            ProcessPDF object allowing .json() or .txt() method calls
        """
        return self.ProcessPDF(self, pdf_path, generate_embeddings)

    # [Existing methods remain unchanged: extract_pdf_content, determine_layout, convert_to_blocks,
    # parse_paragraphs, _process_blocks_into_paragraphs, process_sequential_paragraphs,
    # clean_text, generate_embeddings]

    def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
        """Deprecated: Use process_pdf().json(output_path) instead."""
        return self.formatters['json'].format_output(result, output_path)

# [Existing process_directory function remains unchanged]

if __name__ == "__main__":
    # Hardcoded file paths
    input_path = "contracts/sample_contract.pdf"
    output_json_path = "extracted/sample_contract.json"
    output_txt_path = "extracted/sample_contract.txt"
    
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
            
        # Example usage of new functionality
        with handler.process_pdf(input_path, generate_embeddings) as processor:
            # Get JSON output and save to file
            json_result = processor.json(output_json_path)
            print(f"Processed {input_path} and saved JSON to {output_json_path}")
            
            # Get text output and save to file
            text_result = processor.txt(output_txt_path)
            print(f"Processed {input_path} and saved text to {output_txt_path}")
            
            # Use results directly
            print(f"JSON result sample: {json_result.get('filename')}")
            print(f"Text result sample (first 100 chars): {text_result[:100]}")
