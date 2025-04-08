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
    
    
    
    
    
#-----------------------
import pdfplumber
import re
import json
import os
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

def extract_bold_text(pdf_path: str) -> Dict[int, List[Dict[str, any]]]:
    """
    Extract all bold text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with page numbers as keys and lists of bold text objects as values
    """
    bold_text_by_page = defaultdict(list)
    
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        # Process each page
        for page_num, page in enumerate(pdf.pages, 1):
            # Get all text characters with their properties
            chars = page.chars
            
            # If no characters on the page, continue to next page
            if not chars:
                continue
            
            # Identify bold text by font properties
            # Look for fonts with "Bold" in their name or with fontweight > 400
            bold_chars = []
            for char in chars:
                # Check if the font name contains "Bold" (case insensitive)
                is_bold_by_name = False
                if "fontname" in char:
                    is_bold_by_name = bool(re.search(r'bold|heavy|black', char["fontname"], re.IGNORECASE))
                
                # Check if the font weight is greater than 400 (normal weight)
                is_bold_by_weight = False
                if "fontweight" in char:
                    is_bold_by_weight = char["fontweight"] > 400
                
                # If either condition is true, consider it bold
                if is_bold_by_name or is_bold_by_weight:
                    bold_chars.append(char)
            
            # Group adjacent bold characters into words
            if bold_chars:
                bold_words = group_chars_into_words(bold_chars)
                for word in bold_words:
                    word_data = {
                        "text": word["text"],
                        "x0": word["x0"],
                        "y0": word["y0"],
                        "x1": word["x1"],
                        "y1": word["y1"],
                        "font": word.get("fontname", "Unknown")
                    }
                    bold_text_by_page[page_num].append(word_data)
    
    return dict(bold_text_by_page)

def group_chars_into_words(chars: List[Dict]) -> List[Dict]:
    """
    Group adjacent characters into words.
    
    Args:
        chars: List of character dictionaries with positions and text
        
    Returns:
        List of word dictionaries with text and bounding box
    """
    # Sort characters by position (top to bottom, left to right)
    sorted_chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    
    words = []
    current_word = None
    
    for char in sorted_chars:
        # Skip space characters when building words
        if char["text"].isspace():
            continue
            
        # Start a new word if this is the first character or not close to previous
        if current_word is None:
            current_word = {
                "text": char["text"],
                "x0": char["x0"],
                "y0": char["top"],
                "x1": char["x1"],
                "y1": char["bottom"],
                "fontname": char.get("fontname", "Unknown")
            }
        else:
            # Check if character is part of current word (horizontal proximity)
            # This threshold may need adjustment based on your PDF
            distance_threshold = 2 * char["width"]  # Adjust this threshold as needed
            
            if (char["x0"] - current_word["x1"] <= distance_threshold and 
                abs(char["top"] - current_word["y0"]) < 2):  # Characters on same line
                # Append to current word
                current_word["text"] += char["text"]
                current_word["x1"] = char["x1"]
                # Update y extents if needed
                current_word["y0"] = min(current_word["y0"], char["top"])
                current_word["y1"] = max(current_word["y1"], char["bottom"])
            else:
                # Finish current word and start a new one
                words.append(current_word)
                current_word = {
                    "text": char["text"],
                    "x0": char["x0"],
                    "y0": char["top"],
                    "x1": char["x1"],
                    "y1": char["bottom"],
                    "fontname": char.get("fontname", "Unknown")
                }
    
    # Add the last word if it exists
    if current_word:
        words.append(current_word)
    
    return words

def get_font_information(pdf_path: str) -> Dict[str, List[str]]:
    """
    Get information about all fonts used in the PDF.
    This helps identify which fonts are used for bold text.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary mapping font names to page numbers where they are used
    """
    font_pages = defaultdict(list)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract unique fonts from chars
            if page.chars:
                fonts = {char.get("fontname", "Unknown") for char in page.chars}
                for font in fonts:
                    font_pages[font].append(page_num)
    
    return dict(font_pages)

def extract_bold_words_in_context(pdf_path: str, context_words: int = 3) -> Dict[int, List[Dict]]:
    """
    Extract bold words with surrounding context.
    
    Args:
        pdf_path: Path to the PDF file
        context_words: Number of words to include before and after the bold text
        
    Returns:
        Dictionary with page numbers as keys and lists of context objects as values
    """
    bold_context_by_page = defaultdict(list)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # First get all the text and words on the page
            text = page.extract_text()
            if not text:
                continue
                
            # Get bold text on this page
            bold_text = extract_bold_text(pdf_path).get(page_num, [])
            if not bold_text:
                continue
                
            # Get all words on the page
            all_words = text.split()
            
            # For each bold word/phrase, find it in the text and extract context
            for bold_item in bold_text:
                bold_word = bold_item["text"]
                
                # Find the word in the text (simple search)
                try:
                    word_index = all_words.index(bold_word)
                    
                    # Calculate context range
                    start_index = max(0, word_index - context_words)
                    end_index = min(len(all_words), word_index + context_words + 1)
                    
                    # Extract context
                    context_before = " ".join(all_words[start_index:word_index])
                    context_after = " ".join(all_words[word_index+1:end_index])
                    
                    context_data = {
                        "bold_text": bold_word,
                        "context_before": context_before,
                        "context_after": context_after,
                        "full_context": f"{context_before} {bold_word} {context_after}"
                    }
                    bold_context_by_page[page_num].append(context_data)
                except ValueError:
                    # Word might be part of a larger phrase, or may contain special characters
                    # This is a simple approach - a more sophisticated text matching would be needed
                    # for complex cases
                    pass
    
    return dict(bold_context_by_page)

def save_to_json(data: Dict, output_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_path}")

def extract_and_save_bold_text(pdf_path: str, output_path: str, include_context: bool = False) -> None:
    """
    Extract bold text from PDF and save to JSON.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the output JSON file
        include_context: Whether to include surrounding context for bold text
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    # Get font information to help troubleshoot
    font_info = get_font_information(pdf_path)
    print(f"Found {len(font_info)} different fonts in the PDF")
    
    # Extract bold text
    bold_text = extract_bold_text(pdf_path)
    
    # Create output dictionary
    result = {
        "pdf_filename": os.path.basename(pdf_path),
        "fonts_used": font_info,
        "bold_text_by_page": bold_text
    }
    
    # Add context if requested
    if include_context:
        context_data = extract_bold_words_in_context(pdf_path)
        result["bold_text_with_context"] = context_data
    
    # Count total bold words found
    total_bold_words = sum(len(words) for words in bold_text.values())
    print(f"Found {total_bold_words} bold words/phrases across {len(bold_text)} pages")
    
    # Save to JSON
    save_to_json(result, output_path)

def main():
    # Set the PDF path and output JSON path
    pdf_path = "document.pdf"  # Replace with your actual PDF path
    output_path = "bold_text.json"  # Replace with your desired output path
    include_context = True  # Set to True to include surrounding context for bold text
    
    # Extract bold text and save to JSON
    extract_and_save_bold_text(
        pdf_path=pdf_path,
        output_path=output_path,
        include_context=include_context
    )

if __name__ == "__main__":
    main()
