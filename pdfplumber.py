import pdfplumber
import re
import json
import os
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

def clean_text(text: str) -> str:
    """
    Clean the text by normalizing whitespace and replacing CID characters.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    # Standardize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Keep (cid:9) for heading detection but replace other CID characters
    text = re.sub(r'\(cid:(?!9\))\d+\)', ' ', text)
    
    # Normalize non-significant whitespace (not at beginning of lines)
    text = re.sub(r'(?<!^)[ \t]+', ' ', text, flags=re.MULTILINE)
    
    return text

def identify_headings_and_content(text: str) -> List[Dict[str, str]]:
    """
    Identify headings and associated content based on:
    1. Numerical patterns followed by (cid:9)
    2. Lines that have an integer followed by a dot and fewer than 5 words
    
    Args:
        text: Cleaned text from the PDF
        
    Returns:
        List of dictionaries containing heading and content information
    """
    # Process text to prepare for pattern matching
    processed_text = re.sub(r'\(cid:9\)', ' ', text)
    
    # Two patterns for heading detection:
    # 1. Number(s) followed by a dot and space, with fewer than 5 words until next newline
    # 2. Number(s) followed by a dot and space at the beginning of a line
    heading_pattern = r'(?:^|\n)(\d+(?:\.\d+)*\.)\s+((?:\S+\s+){0,4}\S+)(?=\n|$)'
    
    # Find all potential headings
    heading_matches = list(re.finditer(heading_pattern, processed_text))
    
    # If no headings found, return the entire text as a single content block
    if not heading_matches:
        return [{"heading": "", "heading_number": "", "heading_title": "", "content": processed_text.strip()}]
    
    # Prepare the structure to hold the results
    sections = []
    
    # Process each heading and its content
    for i, match in enumerate(heading_matches):
        heading_num = match.group(1)  # The number part (e.g., "1." or "1.1.")
        heading_title = match.group(2).strip()  # The heading text (limited to 5 words max)
        full_heading = f"{heading_num} {heading_title}"
        start_pos = match.start()
        
        # Find where this section ends (start of next heading or end of text)
        if i < len(heading_matches) - 1:
            end_pos = heading_matches[i + 1].start()
        else:
            end_pos = len(processed_text)
        
        # Find the content start (after this heading)
        content_start = processed_text.find('\n', start_pos)
        if content_start > -1 and content_start < end_pos:
            content = processed_text[content_start + 1:end_pos].strip()
        else:
            content = ""
        
        sections.append({
            "heading": full_heading,
            "heading_number": heading_num,
            "heading_title": heading_title,
            "content": content
        })
    
    # Check if there's content before the first heading
    if heading_matches[0].start() > 0:
        preamble = processed_text[:heading_matches[0].start()].strip()
        if preamble:
            sections.insert(0, {
                "heading": "",
                "heading_number": "",
                "heading_title": "",
                "content": preamble
            })
    
    return sections

def sections_to_json(sections: List[Dict[str, str]], output_path: Optional[str] = None) -> str:
    """
    Convert sections to JSON.
    
    Args:
        sections: List of dictionaries containing heading and content
        output_path: Optional path to save the JSON file
        
    Returns:
        JSON string representation of sections
    """
    # Format the sections for output
    formatted_sections = []
    
    for i, section in enumerate(sections):
        section_data = {
            "section_id": i + 1,
            "heading_number": section.get("heading_number", ""),
            "heading_title": section.get("heading_title", ""),
            "content": section["content"]
        }
        formatted_sections.append(section_data)
    
    # Convert to JSON
    json_str = json.dumps(formatted_sections, indent=2, ensure_ascii=False)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    return json_str

def parse_pdf_to_json(pdf_path: str, output_path: str, skip_header_footer: bool = True, skip_first_page: bool = False) -> None:
    """
    Main function to parse a PDF into a structured JSON file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the output JSON file
        skip_header_footer: Whether to skip headers and footers
        skip_first_page: Whether to skip the first page of the PDF
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    # Extract text from PDF
    full_text = extract_text_from_pdf(
        pdf_path, 
        skip_header_footer=skip_header_footer,
        skip_first_page=skip_first_page
    )
    
    # Clean the text
    cleaned_text = clean_text(full_text)
    
    # Identify headings and content
    sections = identify_headings_and_content(cleaned_text)
    
    # Convert to JSON and save
    sections_to_json(sections, output_path)
    
    print(f"Structured content saved to {output_path}")
    print(f"Total sections extracted: {len(sections)}")

# Example usage in the main function
def main():
    # Define file paths directly in the code
    pdf_path = "document.pdf"  # Replace with your actual PDF path
    output_path = "document_structured.json"  # Replace with your desired output path
    
    # Optional parameters
    skip_header_footer = True
    skip_first_page = False
    
    # Process the PDF
    parse_pdf_to_json(
        pdf_path=pdf_path,
        output_path=output_path, 
        skip_header_footer=skip_header_footer,
        skip_first_page=skip_first_page
    )

if __name__ == "__main__":
    main()