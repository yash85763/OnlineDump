“””
Complete OCR Hybrid PDF Parser

A comprehensive PDF processing module that uses OCR as primary method with pdfminer.six fallback.
Extracts paragraphs from HTML, handles disclaimer detection, and distributes content across actual PDF pages.

Features:

- OCR API integration with timeout (20 seconds)
- Automatic fallback to pdfminer.six if OCR fails
- Smart disclaimer detection and removal
- Real PDF page counting using pdfminer
- Paragraph extraction from HTML <p> tags only
- Even distribution across actual pages
- Optional embeddings generation
  “””

import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans

# PDF processing imports

try:
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTImage, LTFigure
from pdfminer.high_level import extract_pages
PDFMINER_AVAILABLE = True
except ImportError:
PDFMINER_AVAILABLE = False
raise ImportError(“pdfminer.six is required. Install with: pip install pdfminer.six”)

# Optional embeddings support

try:
from sentence_transformers import SentenceTransformer
EMBEDDINGS_AVAILABLE = True
except ImportError:
EMBEDDINGS_AVAILABLE = False
print(“Warning: sentence-transformers not available. Install with: pip install sentence-transformers”)

class PDFParser:
“””
Clean PDF parser for extracting and structuring text content from PDF documents.
This is the fallback parser using pdfminer.six only.
“””

```
def __init__(self, 
             min_quality_ratio: float = 0.5,
             paragraph_spacing_threshold: int = 10,
             min_words_per_paragraph: int = 5,
             embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Initialize the PDF parser with configurable parameters.
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
    
    # Initialize embedding model if available
    self.embedding_model = None
    if EMBEDDINGS_AVAILABLE:
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")

def parse_pdf(self, pdf_path: str, generate_embeddings: bool = False) -> Dict[str, Any]:
    """
    Parse a PDF file and extract structured content.
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
        
        # Step 5: Generate embeddings if requested
        if generate_embeddings and self.embedding_model is not None:
            result["embeddings"] = self._generate_embeddings(structured_pages)
        
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
    """
    pages_data = []
    total_text = ""
    
    try:
        # Use the high-level API first
        page_count = 0
        for page_layout in extract_pages(pdf_path):
            page_count += 1
            
            # Extract text blocks from page
            text_blocks = []
            page_text = ""
            
            # Get page dimensions
            page_width = getattr(page_layout, 'width', 612)
            page_height = getattr(page_layout, 'height', 792)
            
            # Extract text elements
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    text_content = element.get_text().strip()
                    if text_content:
                        text_blocks.append({
                            'x0': element.x0,
                            'y0': element.y0,
                            'x1': element.x1,
                            'y1': element.y1,
                            'text': text_content
                        })
                        page_text += text_content + " "
            
            pages_data.append({
                'page_number': page_count,
                'width': page_width,
                'height': page_height,
                'text_blocks': text_blocks,
                'raw_text': page_text.strip()
            })
            
            total_text += page_text
            
    except Exception as e:
        return [], False, f"Error extracting PDF content: {str(e)}"
    
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
    # Collect x-coordinates from first few pages
    x_coordinates = []
    sample_pages = min(3, len(pages_data))
    
    for page_idx in range(sample_pages):
        page = pages_data[page_idx]
        for block in page.get('text_blocks', []):
            # Use center x-coordinate of each block
            x_center = (block['x0'] + block['x1']) / 2
            x_coordinates.append(x_center)
    
    if len(x_coordinates) < 5:
        return "single_column"
    
    # Use clustering to detect columns
    try:
        X = np.array(x_coordinates).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        centers = kmeans.cluster_centers_.flatten()
        counts = np.bincount(labels)
        
        # Check if centers are well-separated and both have significant content
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
    Extract and organize paragraphs from pages.
    """
    structured_pages = []
    
    for page in pages_data:
        page_number = page['page_number']
        text_blocks = page['text_blocks']
        
        if not text_blocks:
            structured_pages.append({
                "page_number": page_number,
                "paragraphs": [],
                "paragraph_count": 0,
                "layout": layout_type
            })
            continue
        
        # Sort blocks by position
        sorted_blocks = sorted(text_blocks, 
                             key=lambda b: (page['height'] - b['y1'], b['x0']))
        
        # Group blocks into paragraphs
        paragraphs = []
        current_paragraph = ""
        
        for i, block in enumerate(sorted_blocks):
            text = block['text']
            if not text.strip():
                continue
            
            if not current_paragraph:
                current_paragraph = text
            else:
                # Simple paragraph detection based on spacing
                current_paragraph += " " + text
        
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        
        structured_pages.append({
            "page_number": page_number,
            "paragraphs": paragraphs,
            "paragraph_count": len(paragraphs),
            "layout": layout_type
        })
    
    return structured_pages

def _generate_embeddings(self, structured_pages: List[Dict]) -> Dict[str, Any]:
    """
    Generate embeddings for all paragraphs using sentence transformers.
    """
    if not self.embedding_model:
        return {"error": "No embedding model available"}
    
    try:
        # Collect all paragraphs
        all_paragraphs = []
        paragraph_map = []
        
        for page_idx, page in enumerate(structured_pages):
            for para_idx, paragraph in enumerate(page["paragraphs"]):
                all_paragraphs.append(paragraph)
                paragraph_map.append((page_idx, para_idx))
        
        if not all_paragraphs:
            return {"error": "No paragraphs found for embedding"}
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(all_paragraphs)
        
        # Create nested dictionary structure
        embedding_dict = {}
        for (page_idx, para_idx), embedding in zip(paragraph_map, embeddings):
            if page_idx not in embedding_dict:
                embedding_dict[page_idx] = {}
            embedding_dict[page_idx][para_idx] = embedding.tolist()
        
        return {
            "model": getattr(self.embedding_model, '_model_name', 'unknown'),
            "dimensions": len(embeddings[0]) if len(embeddings) > 0 else 0,
            "total_paragraphs": len(all_paragraphs),
            "embeddings": embedding_dict
        }
        
    except Exception as e:
        return {"error": f"Error generating embeddings: {str(e)}"}

def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
    """
    Save the parsed result to a JSON file.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_path
```

class OCRHybridParser:
“””
Hybrid parser that tries OCR first, falls back to pdfminer if OCR fails or takes too long
“””

```
def __init__(self, 
             ocr_api_url: str,
             ocr_api_headers: dict = None,
             ocr_timeout: int = 20,
             disclaimer_keywords: List[str] = None,
             fallback_parser: Any = None):
    """
    Initialize OCR hybrid parser
    
    Args:
        ocr_api_url: URL endpoint for your OCR API
        ocr_api_headers: Headers for OCR API requests
        ocr_timeout: Timeout in seconds for OCR processing
        disclaimer_keywords: Keywords to identify disclaimer pages (legacy - not used with new logic)
        fallback_parser: Instance of your existing PDFParser class
    """
    self.ocr_api_url = ocr_api_url
    self.ocr_api_headers = ocr_api_headers or {}
    self.ocr_timeout = ocr_timeout
    self.disclaimer_keywords = disclaimer_keywords or [
        'disclaimer', 'confidential', 'proprietary', 'terms and conditions',
        'legal notice', 'copyright notice', 'privacy policy'
    ]
    self.fallback_parser = fallback_parser

def parse_pdf(self, pdf_path: str, generate_embeddings: bool = False) -> Dict[str, Any]:
    """
    Main parsing method - tries OCR first, falls back to pdfminer
    """
    try:
        # Attempt OCR parsing with timeout
        ocr_result = self._try_ocr_parsing(pdf_path, generate_embeddings)
        
        if ocr_result and ocr_result.get("success", False):
            ocr_result["parsing_method"] = "OCR"
            return ocr_result
        else:
            print(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"OCR parsing failed with exception: {str(e)}")
    
    # Fallback to existing pdfminer parser
    print("Falling back to pdfminer parsing...")
    if self.fallback_parser:
        fallback_result = self.fallback_parser.parse_pdf(pdf_path, generate_embeddings)
        fallback_result["parsing_method"] = "pdfminer_fallback"
        return fallback_result
    else:
        return {
            "filename": os.path.basename(pdf_path),
            "success": False,
            "error": "OCR failed and no fallback parser available",
            "parsing_method": "none"
        }

def _try_ocr_parsing(self, pdf_path: str, generate_embeddings: bool) -> Dict[str, Any]:
    """
    Attempt OCR parsing with timeout
    """
    def ocr_worker():
        return self._ocr_parse_pdf(pdf_path, generate_embeddings)
    
    # Use thread with timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ocr_worker)
        try:
            result = future.result(timeout=self.ocr_timeout)
            return result
        except TimeoutError:
            return {
                "success": False,
                "error": f"OCR parsing exceeded {self.ocr_timeout} seconds timeout"
            }

def _ocr_parse_pdf(self, pdf_path: str, generate_embeddings: bool) -> Dict[str, Any]:
    """
    Parse PDF using OCR API and process resulting HTML
    """
    try:
        # Step 1: Send PDF to OCR API
        html_content = self._call_ocr_api(pdf_path)
        
        if not html_content:
            return {
                "success": False,
                "error": "OCR API returned empty content"
            }
        
        # Step 2: Save HTML locally (optional, for debugging)
        html_path = pdf_path.replace('.pdf', '_ocr.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Step 3: Parse HTML and extract paragraphs (includes disclaimer filtering)
        # Pass pdf_path to get actual page count
        pages_content = self._parse_html_content(html_content, pdf_path)
        
        if not pages_content:
            return {
                "success": False,
                "error": "No content extracted from HTML"
            }
        
        # Step 4: Structure result
        result = {
            "filename": os.path.basename(pdf_path),
            "success": True,
            "layout": "ocr_extracted",
            "total_pages": len(pages_content),
            "actual_pdf_pages": self._get_pdf_page_count(pdf_path),
            "pages": pages_content,
            "html_path": html_path
        }
        
        # Step 5: Generate embeddings if requested
        if generate_embeddings and self.fallback_parser and self.fallback_parser.embedding_model:
            result["embeddings"] = self.fallback_parser._generate_embeddings(pages_content)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"OCR parsing error: {str(e)}"
        }

def _call_ocr_api(self, pdf_path: str) -> Optional[str]:
    """
    Call your custom OCR API
    Modify this method to match your API's requirements
    """
    try:
        # Prepare file for upload
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            
            # Make API request
            response = requests.post(
                self.ocr_api_url,
                files=files,
                headers=self.ocr_api_headers,
                timeout=self.ocr_timeout
            )
        
        if response.status_code == 200:
            # Assuming API returns HTML content directly
            # Modify this based on your API's response format
            return response.text
        else:
            print(f"OCR API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"OCR API call failed: {str(e)}")
        return None

def _get_pdf_page_count(self, pdf_path: str) -> int:
    """
    Get the actual number of pages in the PDF using only pdfminer
    """
    try:
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            
            # Check if document is accessible
            if not document.is_extractable:
                print("PDF is encrypted or not extractable")
                return 1  # Default to 1 page
            
            # Count pages
            page_count = 0
            for page in PDFPage.create_pages(document):
                page_count += 1
            
            print(f"PDF page count using pdfminer: {page_count}")
            return page_count
            
    except Exception as e:
        print(f"Error counting PDF pages with pdfminer: {e}")
        # Alternative method using pdfminer's high-level API
        try:
            page_count = 0
            for page_layout in extract_pages(pdf_path):
                page_count += 1
            
            print(f"PDF page count using extract_pages: {page_count}")
            return page_count
            
        except Exception as e2:
            print(f"Error with extract_pages method: {e2}")
            # Final fallback: try to read PDF info
            try:
                return self._count_pages_fallback(pdf_path)
            except:
                print("All page counting methods failed, defaulting to 10 pages")
                return 10  # Default assumption

def _count_pages_fallback(self, pdf_path: str) -> int:
    """
    Fallback method to count pages using pdfminer document info
    """
    try:
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            
            # Try to get page count from document info
            if hasattr(document, 'catalog') and document.catalog:
                if 'Pages' in document.catalog:
                    pages_ref = document.catalog['Pages']
                    pages_obj = pages_ref.resolve()
                    if 'Count' in pages_obj:
                        count = pages_obj['Count']
                        if hasattr(count, 'resolve'):
                            count = count.resolve()
                        return int(count)
            
            # If that doesn't work, manually count by iterating
            page_count = sum(1 for _ in PDFPage.create_pages(document))
            return page_count
            
    except Exception as e:
        print(f"Fallback page counting failed: {e}")
        return 1

def _parse_html_content(self, html_content: str, pdf_path: str = None) -> List[Dict[str, Any]]:
    """
    Parse HTML content and extract paragraphs, stopping at disclaimer section
    Distribute paragraphs across actual PDF page count
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get all paragraph elements in document order
    all_paragraphs = soup.find_all('p')
    
    paragraphs_before_disclaimer = []
    disclaimer_found = False
    
    for p in all_paragraphs:
        # Check if this paragraph contains the disclaimer marker
        if self._is_disclaimer_paragraph(p):
            disclaimer_found = True
            print(f"Found disclaimer marker, stopping content extraction. Total paragraphs before disclaimer: {len(paragraphs_before_disclaimer)}")
            break
        
        # Extract text from paragraph
        text = p.get_text(strip=True)
        if text and len(text.split()) >= 3:  # Filter out very short paragraphs
            paragraphs_before_disclaimer.append(text)
    
    # If no disclaimer found, use all paragraphs
    if not disclaimer_found:
        print("No disclaimer marker found, using all content")
    
    # Get actual page count from PDF
    if pdf_path:
        actual_page_count = self._get_pdf_page_count(pdf_path)
        print(f"PDF has {actual_page_count} pages, distributing {len(paragraphs_before_disclaimer)} paragraphs")
    else:
        actual_page_count = max(1, len(paragraphs_before_disclaimer) // 10)  # Fallback
    
    # Distribute paragraphs across actual pages
    pages_content = []
    
    if len(paragraphs_before_disclaimer) == 0:
        return pages_content
    
    # Calculate paragraphs per page
    paragraphs_per_page = max(1, len(paragraphs_before_disclaimer) // actual_page_count)
    remainder = len(paragraphs_before_disclaimer) % actual_page_count
    
    current_index = 0
    
    for page_num in range(1, actual_page_count + 1):
        # Calculate how many paragraphs for this page
        # Distribute remainder paragraphs across first few pages
        current_page_paragraph_count = paragraphs_per_page
        if remainder > 0:
            current_page_paragraph_count += 1
            remainder -= 1
        
        # Get paragraphs for this page
        end_index = min(current_index + current_page_paragraph_count, len(paragraphs_before_disclaimer))
        page_paragraphs = paragraphs_before_disclaimer[current_index:end_index]
        
        if page_paragraphs:  # Only add pages with content
            pages_content.append({
                "page_number": page_num,
                "paragraphs": page_paragraphs,
                "paragraph_count": len(page_paragraphs),
                "layout": "ocr_extracted"
            })
        
        current_index = end_index
        
        # Break if we've used all paragraphs
        if current_index >= len(paragraphs_before_disclaimer):
            break
    
    return pages_content

def _is_disclaimer_paragraph(self, paragraph_element) -> bool:
    """
    Check if a paragraph contains the disclaimer marker
    Looking for: <p> <span class='font1' style='font-weight:bold;'> Disclaimers … </span> </p>
    """
    try:
        # Check if paragraph contains a span with the specific class and style
        spans = paragraph_element.find_all('span', class_='font1')
        
        for span in spans:
            # Check if span has font-weight:bold style
            style = span.get('style', '')
            if 'font-weight:bold' in style:
                # Check if the text content contains "disclaimer" (case insensitive)
                span_text = span.get_text(strip=True).lower()
                if 'disclaimer' in span_text:
                    return True
        
        # Alternative check: look for any bold text containing "disclaimer"
        # in case the HTML structure is slightly different
        paragraph_text = paragraph_element.get_text(strip=True).lower()
        if 'disclaimer' in paragraph_text:
            # Check if it's in a bold span or strong tag
            bold_elements = paragraph_element.find_all(['span', 'strong', 'b'])
            for element in bold_elements:
                element_text = element.get_text(strip=True).lower()
                if 'disclaimer' in element_text:
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error checking disclaimer paragraph: {e}")
        return False

def save_to_json(self, result: Dict[str, Any], output_path: str) -> str:
    """
    Save the parsed result to a JSON file.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_path
```

# Factory function to create hybrid parser

def create_hybrid_parser(ocr_api_url: str,
ocr_api_headers: dict = None,
ocr_timeout: int = 20,
disclaimer_keywords: List[str] = None) -> OCRHybridParser:
“””
Factory function to create a hybrid parser with fallback
“””
# Create fallback parser instance
fallback_parser = PDFParser()

```
# Create hybrid parser
hybrid_parser = OCRHybridParser(
    ocr_api_url=ocr_api_url,
    ocr_api_headers=ocr_api_headers,
    ocr_timeout=ocr_timeout,
    disclaimer_keywords=disclaimer_keywords,
    fallback_parser=fallback_parser
)

return hybrid_parser
```

# Convenience functions for easy usage

def parse_single_pdf_hybrid(pdf_path: str,
ocr_api_url: str,
ocr_api_headers: dict = None,
output_path: str = None,
generate_embeddings: bool = False) -> Dict[str, Any]:
“””
Convenience function to parse a single PDF file with OCR hybrid approach.
“””
parser = create_hybrid_parser(
ocr_api_url=ocr_api_url,
ocr_api_headers=ocr_api_headers
)

```
result = parser.parse_pdf(pdf_path, generate_embeddings)

if output_path:
    parser.save_to_json(result, output_path)
    print(f"Saved results to: {output_path}")

return result
```

def parse_pdf_directory_hybrid(input_dir: str,
output_dir: str,
ocr_api_url: str,
ocr_api_headers: dict = None,
generate_embeddings: bool = False) -> List[Dict[str, Any]]:
“””
Parse all PDF files in a directory using OCR hybrid approach.
“””
parser = create_hybrid_parser(
ocr_api_url=ocr_api_url,
ocr_api_headers=ocr_api_headers
)

```
os.makedirs(output_dir, exist_ok=True)

results = []
pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

for filename in pdf_files:
    pdf_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    
    print(f"Processing: {filename}")
    result = parser.parse_pdf(pdf_path, generate_embeddings)
    parser.save_to_json(result, output_path)
    
    results.append(result)
    
    # Print summary
    if result["success"]:
        method = result.get("parsing_method", "unknown")
        total_paragraphs = sum(page["paragraph_count"] for page in result["pages"])
        print(f"  ✓ [{method}] Extracted {total_paragraphs} paragraphs from {result['total_pages']} pages")
    else:
        print(f"  ✗ Failed: {result['error']}")

return results
```

# Example usage

if **name** == “**main**”:
# Configuration for your OCR API
ocr_config = {
“ocr_api_url”: “https://your-ocr-api-endpoint.com/process”,
“ocr_api_headers”: {
“Authorization”: “Bearer your-api-key”,
# Don’t include Content-Type for multipart/form-data - requests will set it
},
“ocr_timeout”: 20
}

```
# Example 1: Parse a single PDF
pdf_file = "sample_contract.pdf"
json_output = "output/sample_contract.
```