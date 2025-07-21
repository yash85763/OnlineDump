# Additional imports to add at the top of your existing file

import time
import requests
from bs4 import BeautifulSoup
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Add this new class - insert after your existing PDFParser class

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
        disclaimer_keywords: Keywords to identify disclaimer pages
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
        
        # Step 3: Parse HTML and extract paragraphs
        pages_content = self._parse_html_content(html_content)
        
        # Step 4: Filter out disclaimer pages
        filtered_pages = self._remove_disclaimer_pages(pages_content)
        
        if not filtered_pages:
            return {
                "success": False,
                "error": "No content remaining after disclaimer filtering"
            }
        
        # Step 5: Structure result
        result = {
            "filename": os.path.basename(pdf_path),
            "success": True,
            "layout": "ocr_extracted",
            "total_pages": len(filtered_pages),
            "pages": filtered_pages,
            "html_path": html_path
        }
        
        # Step 6: Generate embeddings if requested
        if generate_embeddings and self.fallback_parser and self.fallback_parser.embedding_model:
            result["embeddings"] = self.fallback_parser._generate_embeddings(filtered_pages)
        
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

def _parse_html_content(self, html_content: str) -> List[Dict[str, Any]]:
    """
    Parse HTML content and extract paragraphs by page
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    pages_content = []
    
    # Find page divisions - adjust selectors based on your OCR output format
    # Common patterns: div.page, div[data-page], .page-container, etc.
    pages = soup.find_all(['div'], class_=lambda x: x and 'page' in x.lower()) or \
            soup.find_all(['div'], attrs={'data-page': True}) or \
            [soup]  # Fallback: treat entire HTML as one page
    
    for page_num, page_element in enumerate(pages, 1):
        # Extract all paragraph elements
        paragraphs = page_element.find_all('p')
        
        page_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text.split()) >= 3:  # Filter out very short paragraphs
                page_paragraphs.append(text)
        
        if page_paragraphs:  # Only add pages with content
            pages_content.append({
                "page_number": page_num,
                "paragraphs": page_paragraphs,
                "paragraph_count": len(page_paragraphs),
                "layout": "ocr_extracted"
            })
    
    return pages_content

def _remove_disclaimer_pages(self, pages_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove pages that contain disclaimer content
    """
    filtered_pages = []
    
    for page in pages_content:
        page_text = " ".join(page["paragraphs"]).lower()
        
        # Check if page contains disclaimer keywords
        has_disclaimer = any(keyword.lower() in page_text 
                           for keyword in self.disclaimer_keywords)
        
        if not has_disclaimer:
            filtered_pages.append(page)
        else:
            print(f"Removed page {page['page_number']} (contains disclaimer content)")
    
    # Renumber pages after filtering
    for i, page in enumerate(filtered_pages, 1):
        page["page_number"] = i
    
    return filtered_pages
```

# Add this convenience function

def create_hybrid_parser(ocr_api_url: str,
ocr_api_headers: dict = None,
ocr_timeout: int = 20,
disclaimer_keywords: List[str] = None) -> OCRHybridParser:
“””
Factory function to create a hybrid parser with fallback
“””
# Create fallback parser instance
fallback_parser = PDFParser()  # Your existing parser class

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

# Add this to your main/example usage section

def example_hybrid_usage():
“””
Example of how to use the hybrid parser
“””
# Configuration for your OCR API
ocr_config = {
“ocr_api_url”: “https://your-ocr-api-endpoint.com/process”,
“ocr_api_headers”: {
“Authorization”: “Bearer your-api-key”,
“Content-Type”: “multipart/form-data”
},
“ocr_timeout”: 20,
“disclaimer_keywords”: [
‘disclaimer’, ‘confidential’, ‘proprietary’,
‘terms and conditions’, ‘legal notice’
]
}

```
# Create hybrid parser
parser = create_hybrid_parser(**ocr_config)

# Parse PDF
pdf_file = "sample_contract.pdf"
result = parser.parse_pdf(pdf_file, generate_embeddings=True)

if result["success"]:
    print(f"Parsing method: {result['parsing_method']}")
    print(f"Total pages: {result['total_pages']}")
    
    total_paragraphs = sum(page["paragraph_count"] for page in result["pages"])
    print(f"Total paragraphs: {total_paragraphs}")
    
    # Save to JSON
    output_path = "output/parsed_content.json"
    if hasattr(parser.fallback_parser, 'save_to_json'):
        parser.fallback_parser.save_to_json(result, output_path)
    
else:
    print(f"Parsing failed: {result['error']}")
```

# Modification instructions for your existing PDFParser class:

“””
MODIFICATION INSTRUCTIONS:

1. Add the imports at the top of your existing file:
- import time
- import requests
- from bs4 import BeautifulSoup
- import threading
- from concurrent.futures import ThreadPoolExecutor, TimeoutError
1. Install required packages:
   pip install requests beautifulsoup4
1. Add the OCRHybridParser class after your existing PDFParser class
1. Update your main usage example to use create_hybrid_parser() instead of direct PDFParser instantiation
1. Customize the _call_ocr_api() method to match your API’s specific requirements:
- Update the request format
- Modify response parsing
- Adjust headers and authentication
1. Adjust the HTML parsing selectors in _parse_html_content() based on your OCR output format:
- Update the page detection logic
- Modify paragraph extraction if needed
1. Customize disclaimer_keywords list based on your specific requirements

USAGE:
Replace your existing parser instantiation:
# OLD
parser = PDFParser()

```
# NEW  
parser = create_hybrid_parser(
    ocr_api_url="your-api-url",
    ocr_api_headers={"Authorization": "Bearer token"},
    ocr_timeout=20
)
```

“””