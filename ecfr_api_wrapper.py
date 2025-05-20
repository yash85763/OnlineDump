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

    def _process_pdf(self, pdf_path: str, generate_embeddings: bool) -> Dict[str, Any]:
        """
        Process a PDF file and return the extraction result.
        
        Args:
            pdf_path: Path to the PDF file
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Dictionary with extraction results
        """
        if not DEPENDENCIES['pdfminer']:
            return {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": "pdfminer.six is not available"
            }
        
        # Filter PDFMiner warnings about CropBox
        warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
            
        try:
            try:
                pages_data, is_parsable, quality_info = self.extract_pdf_content(pdf_path)
            except Exception as e:
                # Handle specific PDF parsing errors
                if "generator didn't stop after throw()" in str(e) or "CropBox missing" in str(e):
                    return {
                        "filename": os.path.basename(pdf_path),
                        "parsable": False,
                        "error": "PDF structure error: Invalid page structure or missing CropBox"
                    }
                else:
                    # Re-raise other exceptions
                    raise
            
            if not is_parsable:
                return {
                    "filename": os.path.basename(pdf_path),
                    "parsable": False,
                    "error": quality_info
                }
            
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
            
            return result
            
        except Exception as e:
            return {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": f"Error processing PDF: {str(e)}"
            }

    class ProcessPDF:
        """Class for output formatting selection."""
        
        def __init__(self, handler: 'PDFHandler', result: Dict[str, Any]):
            self.handler = handler
            self.result = result
        
        def json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
            """Return JSON format or save to JSON file."""
            return self.handler.formatters['json'].format_output(self.result, output_path)
        
        def txt(self, output_path: Optional[str] = None) -> str:
            """Return text format or save to text file."""
            return self.handler.formatters['txt'].format_output(self.result, output_path)
        
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
        result = self._process_pdf(pdf_path, generate_embeddings)
        return self.ProcessPDF(self, result)

    @check_dependency('pdfminer')
    def extract_pdf_content(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Extract content from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing:
              - List of page data dictionaries
              - Boolean indicating if the PDF is parsable
              - Quality information or error message
        """
        try:
            # Filter warnings about CropBox
            warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
            
            with open(pdf_path, 'rb') as file:
                # Set up PDFMiner components
                parser = PDFParser(file)
                document = PDFDocument(parser)
                
                # Check if the document can be decrypted
                if not document.is_extractable:
                    return [], False, "PDF is encrypted or not extractable"
                
                # Create components for extracting layouts
                rsrcmgr = PDFResourceManager()
                device = PDFPageAggregator(rsrcmgr, laparams=self.laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                
                pages_data = []
                
                # Process each page
                for page_num, page in enumerate(PDFPage.create_pages(document), 1):
                    try:
                        interpreter.process_page(page)
                        layout = device.get_result()
                        
                        # Extract text elements
                        text_elements = self._extract_text_elements(layout)
                        
                        if text_elements:
                            pages_data.append({
                                'page_num': page_num,
                                'layout': layout,
                                'text_elements': text_elements
                            })
                    except Exception as page_error:
                        # Log the error but continue processing other pages
                        print(f"Error processing page {page_num}: {page_error}")
                        continue
                
                # Check if we extracted enough content
                total_text = sum(len(''.join(element.get('text', ''))) 
                                for page in pages_data 
                                for element in page.get('text_elements', []))
                
                if not pages_data:
                    return [], False, "No pages could be parsed from the PDF"
                
                if total_text < 100:  # Arbitrary threshold for minimum text content
                    return pages_data, False, f"Not enough text content extracted (only {total_text} characters)"
                
                return pages_data, True, "OK"
                
        except Exception as e:
            if "generator didn't stop after throw()" in str(e):
                return [], False, "PDF structure error: Check PDF integrity"
            return [], False, f"Error extracting PDF content: {str(e)}"
    
    def _extract_text_elements(self, layout: LTPage) -> List[Dict[str, Any]]:
        """
        Extract text elements from a page layout.
        
        Args:
            layout: PDFMiner layout object
            
        Returns:
            List of text element dictionaries
        """
        text_elements = []
        
        # Process layout elements
        for element in layout:
            if isinstance(element, LTTextBox):
                # Extract position info
                x0, y0, x1, y1 = element.bbox
                text = element.get_text()
                
                # Normalize text
                text = self.clean_text(text) if hasattr(self, 'clean_text') else text
                
                if text.strip():
                    text_elements.append({
                        'type': 'textbox',
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'text': text
                    })
        
        # Sort by vertical position (top to bottom)
        text_elements.sort(key=lambda e: -e['y0'])
        
        return text_elements

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
            processor = handler.process_pdf(pdf_path, generate_embeddings)
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
            processor = handler.process_pdf(input_path, generate_embeddings)
            
            # Check if PDF was parsable
            if not processor.result.get('parsable', False):
                error_msg = processor.result.get('error', 'Unknown error')
                print(f"Error processing {input_path}: {error_msg}")
            else:
                # Get JSON output and save to file
                json_result = processor.json(output_json_path)
                print(f"Processed {input_path} and saved JSON to {output_json_path}")
                
                # Get text output and save to file
                text_result = processor.txt(output_txt_path)
                print(f"Processed {input_path} and saved text to {output_txt_path}")
                
                # Use results directly
                print(f"JSON result sample: {json_result.get('filename')}")
                text_sample = text_result[:100] if text_result else 'No text extracted'
                print(f"Text result sample (first 100 chars): {text_sample}")
        except Exception as e:
            import traceback
            print(f"Error processing {input_path}: {e}")
            print(traceback.format_exc())
    
    elif os.path.isdir(input_path):
        output_dir = "extracted" if not output_json_path else os.path.dirname(output_json_path)
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing directory {input_path}...")
        results = process_directory(input_path, output_dir, handler, generate_embeddings)
        
        # Count successful and failed files
        success_count = sum(1 for r in results.values() if r.get('parsable', False))
        failure_count = len(results) - success_count
        
        # Save summary of processing results
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(results)} files: {success_count} successful, {failure_count} failed.")
        print(f"See {summary_path} for details.")
    
    else:
        print(f"Error: Input path {input_path} does not exist.")
