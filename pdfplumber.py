import pdfplumber
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

def extract_text_from_pdf(pdf_path: str, skip_header_footer: bool = True, skip_first_page: bool = False) -> str:
    """
    Extract text from PDF file, optionally skipping headers and footers and the first page.
    
    Args:
        pdf_path: Path to the PDF file
        skip_header_footer: Whether to skip headers and footers
        skip_first_page: Whether to skip the first page of the PDF
        
    Returns:
        String containing all extracted text from the PDF
    """
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Skip the first page if requested
            if skip_first_page and i == 0:
                continue
                
            # Extract text with or without cropping
            if skip_header_footer:
                # Get page dimensions
                height = page.height
                width = page.width
                
                # Crop to exclude headers and footers (adjust these values as needed)
                # Typically header is top 10% and footer is bottom 10%
                crop_box = (0, height * 0.1, width, height * 0.9)
                cropped_page = page.crop(crop_box)
                page_text = cropped_page.extract_text()
            else:
                page_text = page.extract_text()
                
            if page_text:
                full_text += page_text + "\n"
    
    return full_text
    
def clean_cid_characters(text):
    # Replace specific CIDs with appropriate characters
    text = re.sub(r'\(cid:9\)', '\t', text)  # Replace tab CID with actual tab
    text = re.sub(r'\(cid:13\)', '\n', text)  # Replace return CID with newline
    # Replace any remaining CIDs with spaces
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    return text

def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text.
    
    Args:
        text: The full text extracted from the PDF
        
    Returns:
        List of paragraphs
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into paragraphs based on double newlines or significant spacing
    # This regex pattern might need adjustment based on your specific PDF structure
    paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
    
    # Filter out empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs

def paragraphs_to_json(paragraphs: List[str], output_path: Optional[str] = None) -> str:
    """
    Convert paragraphs to JSON.
    
    Args:
        paragraphs: List of paragraph strings
        output_path: Optional path to save the JSON file
        
    Returns:
        JSON string representation of paragraphs
    """
    # Create a simple structure with paragraph index and text
    formatted_structure = [
        {
            "paragraph_id": i + 1,
            "text": para
        }
        for i, para in enumerate(paragraphs)
    ]
    
    # Convert to JSON
    json_str = json.dumps(formatted_structure, indent=2, ensure_ascii=False)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    return json_str

def main():
    parser = argparse.ArgumentParser(description='Extract paragraphs from a PDF file')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--include-header-footer', action='store_true', 
                        help='Include header and footer in extraction')
    parser.add_argument('--skip-first-page', action='store_true',
                        help='Skip the first page of the PDF')
    
    args = parser.parse_args()
    
    # Extract text from PDF
    full_text = extract_text_from_pdf(
        args.pdf_path, 
        skip_header_footer=not args.include_header_footer,
        skip_first_page=args.skip_first_page
    )
    
    # Extract paragraphs
    paragraphs = extract_paragraphs(full_text)
    
    # Convert to JSON and save
    json_str = paragraphs_to_json(paragraphs, args.output)
    
    if args.output:
        print(f"Paragraphs saved to {args.output}")
        print(f"Total paragraphs extracted: {len(paragraphs)}")
    else:
        print(json_str)

if __name__ == "__main__":
    main()