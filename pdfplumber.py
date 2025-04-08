import pdfplumber
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

def extract_text_from_pdf(pdf_path: str, skip_header_footer: bool = True) -> List[str]:
    """
    Extract text from PDF file, optionally skipping headers and footers.
    
    Args:
        pdf_path: Path to the PDF file
        skip_header_footer: Whether to skip headers and footers
        
    Returns:
        List of strings, each representing a page's text content
    """
    text_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
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
                text_content.append(page_text)
    
    return text_content

def identify_headings_and_content(text_content: List[str]) -> Dict[str, str]:
    """
    Identify headings and associated content from extracted text.
    
    Args:
        text_content: List of strings, each representing a page's text content
        
    Returns:
        Dictionary with headings as keys and their content as values
    """
    # Join all pages into a single string
    full_text = "\n".join(text_content)
    
    # Pattern to match headings: number(s) followed by a dot, space, and text
    heading_pattern = r'\n\s*(\d+\.(?:\d+\.)*)\s+(.*?)\n'
    
    # Find all headings
    headings = re.findall(heading_pattern, full_text)
    
    # Dictionary to store heading structure
    structure = {}
    
    if not headings:
        return structure
    
    # Iterate through headings
    for i in range(len(headings)):
        heading_num, heading_text = headings[i]
        heading_key = f"{heading_num} {heading_text}"
        
        # Find start position of this heading
        heading_pos = full_text.find(f"\n{heading_num} {heading_text}\n")
        
        # Find the start position of the next heading (if any)
        if i < len(headings) - 1:
            next_heading_num, next_heading_text = headings[i + 1]
            next_heading_pos = full_text.find(f"\n{next_heading_num} {next_heading_text}\n")
            # Extract content between headings
            content = full_text[heading_pos + len(f"\n{heading_num} {heading_text}\n"):next_heading_pos].strip()
        else:
            # If this is the last heading, extract content until the end
            content = full_text[heading_pos + len(f"\n{heading_num} {heading_text}\n"):].strip()
        
        structure[heading_key] = content
    
    return structure

def structure_to_json(structure: Dict[str, str], output_path: Optional[str] = None) -> str:
    """
    Convert the heading-content structure to JSON.
    
    Args:
        structure: Dictionary with headings as keys and their content as values
        output_path: Optional path to save the JSON file
        
    Returns:
        JSON string representation of the structure
    """
    # Create a structured format with separate heading number, title, and content
    formatted_structure = []
    
    for heading, content in structure.items():
        # Extract heading number and text
        match = re.match(r'(\d+\.(?:\d+\.)*)\s+(.*)', heading)
        if match:
            number, title = match.groups()
            formatted_structure.append({
                "heading_number": number.strip(),
                "heading_title": title.strip(),
                "content": content
            })
    
    # Convert to JSON
    json_str = json.dumps(formatted_structure, indent=2, ensure_ascii=False)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    return json_str

def main():
    parser = argparse.ArgumentParser(description='Extract structured content from a PDF file')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--include-header-footer', action='store_true', 
                        help='Include header and footer in extraction')
    
    args = parser.parse_args()
    
    # Extract text from PDF
    text_content = extract_text_from_pdf(
        args.pdf_path, 
        skip_header_footer=not args.include_header_footer
    )
    
    # Identify headings and content
    structure = identify_headings_and_content(text_content)
    
    # Convert to JSON and save
    json_str = structure_to_json(structure, args.output)
    
    if args.output:
        print(f"Structured content saved to {args.output}")
    else:
        print(json_str)

if __name__ == "__main__":
    main()