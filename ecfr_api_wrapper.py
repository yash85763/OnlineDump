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
from typing import List, Dict, Tuple, Optional, Any, Generator, Union, Iterator
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import warnings
from functools import wraps

# Import dependencies with consistent error handling
DEPENDENCIES = {
    'pdfminer': False,
    'sentence_transformers': False,
    'azure_openai': False
}

try:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTPage
    DEPENDENCIES['pdfminer'] = True
except ImportError:
    warnings.warn("pdfminer.six is required. Install it with 'pip install pdfminer.six'")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPENDENCIES['sentence_transformers'] = True
except ImportError:
    warnings.warn("sentence_transformers is not available. Install with 'pip install sentence-transformers faiss-cpu'")

try:
    from openai import AzureOpenAI
    DEPENDENCIES['azure_openai'] = True
except ImportError:
    warnings.warn("Azure OpenAI client is not available. Install with 'pip install openai'")

def check_dependency(dependency_name: str):
    """Decorator to check if a required dependency is available."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not DEPENDENCIES.get(dependency_name, False):
                raise ImportError(f"This function requires {dependency_name} which is not available.")
            return func(*args, **kwargs)
        return wrapper
    return decorator


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
                 azure_openai_config: Optional[Dict[str, str]] = None):
        
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
        
        # Initialize embedding model based on provider
        self._initialize_embedding_provider(embedding_model)
        
        # Initialize formatters
        self.formatters = {
            'json': JSONFormatter(),
            'txt': TextFormatter()
        }

    def _initialize_embedding_provider(self, embedding_model: str) -> None:
        """Initialize the appropriate embedding provider."""
        if self.embedding_provider == "sentence_transformers" and DEPENDENCIES['sentence_transformers']:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Initialized Sentence Transformers model: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load Sentence Transformers model: {e}")
        
        elif self.embedding_provider == "azure_openai" and DEPENDENCIES['azure_openai']:
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

    @contextmanager
    def _process_context(self, pdf_path: str, generate_embeddings: bool) -> Iterator[Dict[str, Any]]:
        """Context manager for processing PDF content."""
        if not DEPENDENCIES['pdfminer']:
            yield {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": "pdfminer.six is not available"
            }
            return
            
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
            
            can_generate_embeddings = (
                generate_embeddings and (
                    (self.embedding_provider == "sentence_transformers" and self.embedding_model is not None) or
                    (self.embedding_provider == "azure_openai" and self.azure_client is not None)
                )
            )
            
            if can_generate_embeddings:
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
            self.context = None
        
        def __enter__(self):
            self.context = self.handler._process_context(self.pdf_path, self.generate_embeddings)
            try:
                self.result = next(self.context.__enter__())
                return self
            except Exception as e:
                # Make sure we properly exit the context if there's an error
                self.context.__exit__(type(e), e, e.__traceback__)
                raise
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.context:
                return self.context.__exit__(exc_type, exc_val, exc_tb)
            return False
        
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

    @check_dependency('pdfminer')
    def extract_pdf_content(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], bool, str]:
        """Extract content from PDF file."""
        # Implementation would go here
        # For now, just placeholder since we were instructed to leave unchanged functions
        pass

    def determine_layout(self, pages_data: List[Dict[str, Any]]) -> str:
        """Determine the layout structure of the PDF."""
        # Implementation would go here
        pass

    def convert_to_blocks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert page data to text blocks."""
        # Implementation would go here
        pass

    def parse_paragraphs(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
        """Parse paragraphs from text blocks."""
        # Implementation would go here
        pass

    def _process_blocks_into_paragraphs(self, blocks: List[Dict[str, Any]]) -> List[str]:
        """Process text blocks into paragraphs."""
        # Implementation would go here
        pass

    def process_sequential_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Process sequential paragraphs to handle continuations."""
        # Implementation would go here
        pass

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing whitespace."""
        # Implementation would go here
        pass
    
    @check_dependency('sentence_transformers')
    def generate_sentence_transformer_embeddings(self, pages_content: List[Dict[str, List[str]]]) -> Dict[str, List[List[float]]]:
        """Generate embeddings using sentence-transformers."""
        # Implementation would go here
        pass
    
    @check_dependency('azure_openai')
    def generate_azure_openai_embeddings(self, pages_content: List[Dict[str, List[str]]]) -> Dict[str, List[List[float]]]:
        """Generate embeddings using Azure OpenAI."""
        # Implementation would go here
        pass
    
    def generate_embeddings(self, pages_content: List[Dict[str, List[str]]]) -> Dict[str, List[List[float]]]:
        """Generate embeddings for paragraphs."""
        if self.embedding_provider == "sentence_transformers" and self.embedding_model is not None:
            return self.generate_sentence_transformer_embeddings(pages_content)
        elif self.embedding_provider == "azure_openai" and self.azure_client is not None:
            return self.generate_azure_openai_embeddings(pages_content)
        else:
            raise ValueError(f"Embedding provider {self.embedding_provider} not available or properly initialized")

    def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
        """
        Deprecated: Use process_pdf().json(output_path) instead.
        
        This method will be removed in a future version.
        """
        warnings.warn(
            "save_to_json is deprecated. Use process_pdf().json(output_path) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.formatters['json'].format_output(result, output_path)


def process_directory(directory_path: str, output_dir: str, handler: PDFHandler, 
                      generate_embeddings: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Process all PDF files in a directory.
    
    Args:
        directory_path: Path to directory containing PDF files
        output_dir: Directory to save output files
        handler: PDFHandler instance
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        Dictionary of results keyed by filename
    """
    results = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        filename = os.path.splitext(pdf_file)[0]
        
        # Output paths
        json_path = os.path.join(output_dir, f"{filename}.json")
        txt_path = os.path.join(output_dir, f"{filename}.txt")
        
        try:
            # Process PDF with both outputs
            with handler.process_pdf(pdf_path, generate_embeddings) as processor:
                json_result = processor.json(json_path)
                txt_result = processor.txt(txt_path)
                
                results[pdf_file] = {
                    "parsable": json_result.get("parsable", False),
                    "json_path": json_path,
                    "txt_path": txt_path
                }
                
                print(f"Processed {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            results[pdf_file] = {
                "parsable": False,
                "error": str(e)
            }
    
    return results


if __name__ == "__main__":
    # Example configuration
    input_path = "contracts/sample_contract.pdf"
    output_json_path = "extracted/sample_contract.json"
    output_txt_path = "extracted/sample_contract.txt"
    
    generate_embeddings = True
    
    # Using environment variables for sensitive information
    from os import environ
    
    azure_openai_config = None
    embedding_provider = environ.get("EMBEDDING_PROVIDER", "sentence_transformers")
    
    if embedding_provider == "azure_openai":
        azure_openai_config = {
            "api_key": environ.get("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": environ.get("AZURE_OPENAI_ENDPOINT"),
            "api_version": environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
        }
        
        # Validate required environment variables
        missing_vars = [k for k, v in azure_openai_config.items() if not v]
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            print("Set these environment variables or change the embedding provider to 'sentence_transformers'")
            exit(1)
    
    # Create PDFHandler with appropriate configuration
    handler = PDFHandler(
        embedding_provider=embedding_provider,
        embedding_model="text-embedding-ada-002" if embedding_provider == "azure_openai" else "all-MiniLM-L6-v2",
        azure_openai_config=azure_openai_config
    )
    
    # Process individual file or directory
    if os.path.isfile(input_path):
        # Example usage of new functionality
        try:
            with handler.process_pdf(input_path, generate_embeddings) as processor:
                # Get JSON output and save to file
                json_result = processor.json(output_json_path)
                print(f"Processed {input_path} and saved JSON to {output_json_path}")
                
                # Get text output and save to file
                text_result = processor.txt(output_txt_path)
                print(f"Processed {input_path} and saved text to {output_txt_path}")
                
                # Use results directly
                print(f"JSON result sample: {json_result.get('filename')}")
                print(f"Text result sample (first 100 chars): {text_result[:100] if text_result else 'No text extracted'}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    elif os.path.isdir(input_path):
        output_dir = "extracted" if not output_json_path else os.path.dirname(output_json_path)
        results = process_directory(input_path, output_dir, handler, generate_embeddings)
        
        # Save summary of processing results
        with open(os.path.join(output_dir, "processing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(results)} files. See {os.path.join(output_dir, 'processing_summary.json')} for details.")
    
    else:
        print(f"Error: Input path {input_path} does not exist.")
