"""
Enhanced Double-Column PDF Parser using PyMuPDF

This script demonstrates how to properly extract text from double-columned PDFs
like research papers, ensuring that:
1. Columns are processed in the correct order
2. Headers and footers are properly handled (optionally ignored)
3. Paragraphs that span across pages are correctly joined
"""

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Any, Optional
import os
import re


def parse_double_column_pdf(
    pdf_path: str,
    include_headers: bool = True,
    include_footers: bool = False,
    header_height_percentage: float = 0.1,
    footer_height_percentage: float = 0.1,
    paragraph_spacing_threshold: int = 10
) -> Dict[str, Any]:
    """
    Parse a double-column PDF correctly with enhanced handling of columns, headers, footers,
    and cross-page paragraph continuity.
    
    Args:
        pdf_path: Path to the PDF file
        include_headers: Whether to include headers in the output (default True)
        include_footers: Whether to include footers in the output (default False)
        header_height_percentage: Percentage of page height considered as header area (default 10%)
        footer_height_percentage: Percentage of page height considered as footer area (default 10%)
        paragraph_spacing_threshold: Max spacing between blocks to be considered same paragraph (default 10)
        
    Returns:
        Dictionary with parsed content by page
    """
    result = {
        "filename": os.path.basename(pdf_path),
        "pages": []
    }
    
    # Open the document
    doc = fitz.open(pdf_path)
    
    # Previous paragraph info for handling cross-page continuity
    prev_paragraph_info = None  # (text, ends_with_period)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Define header and footer boundaries
        header_boundary = page_height * header_height_percentage
        footer_boundary = page_height * (1 - footer_height_percentage)
        
        # 1. Extract text blocks
        blocks = page.get_text("blocks")
        
        # If no blocks, skip this page
        if not blocks:
            result["pages"].append({
                "page_number": page_num + 1,
                "paragraphs": []
            })
            continue
        
        # 2. Separate blocks into header, footer, and main content
        header_blocks = []
        footer_blocks = []
        content_blocks = []
        
        for block in blocks:
            # Block coordinates
            x0, y0, x1, y1, text = block[:5]
            
            # Determine if block is in header, footer, or main content
            if y0 < header_boundary:
                header_blocks.append(block)
            elif y1 > footer_boundary:
                footer_blocks.append(block)
            else:
                content_blocks.append(block)
        
        # 3. Identify columns in the main content area using x-coordinates
        if content_blocks:
            x_centers = [(block[0] + block[2]) / 2 for block in content_blocks]
            X = np.array(x_centers).reshape(-1, 1)
            
            # Skip clustering if too few blocks
            is_double_column = False
            column_centers = [page_width / 2]  # Default for single column
            
            if len(content_blocks) >= 3:
                # Try to identify 2 columns with K-means
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
                centers = kmeans.cluster_centers_.flatten()
                
                # Sort centers from left to right
                column_centers = sorted(centers)
                
                # Check if this is likely a double-column layout
                if len(column_centers) > 1:
                    # If centers are separated by at least 20% of page width
                    col_distance = abs(column_centers[1] - column_centers[0])
                    if col_distance > (page_width * 0.2):
                        is_double_column = True
            
            # Get the midpoint between columns
            if len(column_centers) > 1:
                midpoint = (column_centers[0] + column_centers[1]) / 2
            else:
                midpoint = page_width / 2
            
            # 4. Group blocks by column and sort within columns
            left_column = []
            right_column = []
            
            for block in content_blocks:
                block_center_x = (block[0] + block[2]) / 2
                
                if block_center_x < midpoint:
                    left_column.append(block)
                else:
                    right_column.append(block)
            
            # Sort each column by y-coordinate (top to bottom)
            left_column.sort(key=lambda b: b[1])
            right_column.sort(key=lambda b: b[1])
            
            # 5. Process column blocks into paragraphs
            # Function to extract paragraphs from a column
            def process_column(column_blocks):
                paragraphs = []
                current_paragraph = ""
                
                for i, block in enumerate(column_blocks):
                    text = block[4]
                    
                    if not text.strip():
                        continue
                    
                    # If we're starting a new paragraph
                    if not current_paragraph:
                        current_paragraph = text
                    else:
                        # Check spacing between blocks
                        prev_block = column_blocks[i-1]
                        prev_bottom = prev_block[3]  # y1 (bottom)
                        current_top = block[1]       # y0 (top)
                        
                        spacing = current_top - prev_bottom
                        
                        # If spacing is small, consider it part of the same paragraph
                        if spacing <= paragraph_spacing_threshold:
                            current_paragraph += " " + text
                        else:
                            # End current paragraph and start a new one
                            paragraphs.append(current_paragraph)
                            current_paragraph = text
                
                # Add the last paragraph if it exists
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    
                return paragraphs
            
            # Process each column
            left_paragraphs = process_column(left_column)
            right_paragraphs = process_column(right_column)
            
            # 6. Handle header blocks
            header_paragraphs = []
            if include_headers and header_blocks:
                # Sort header blocks from top to bottom, then left to right
                header_blocks.sort(key=lambda b: (b[1], b[0]))
                header_paragraphs = process_column(header_blocks)
            
            # 7. Handle footer blocks
            footer_paragraphs = []
            if include_footers and footer_blocks:
                # Sort footer blocks from top to bottom, then left to right
                footer_blocks.sort(key=lambda b: (b[1], b[0]))
                footer_paragraphs = process_column(footer_blocks)
            
            # 8. Build the final paragraphs list based on layout
            paragraphs = []
            
            # Add headers first if included
            if include_headers:
                paragraphs.extend(header_paragraphs)
            
            # Add main content based on column layout
            if is_double_column:
                # For double-column layout, first process left column, then right column
                paragraphs.extend(left_paragraphs)
                paragraphs.extend(right_paragraphs)
            else:
                # For single-column, combine all content blocks and sort by y-coordinate
                all_content_blocks = sorted(content_blocks, key=lambda b: b[1])
                content_paragraphs = process_column(all_content_blocks)
                paragraphs.extend(content_paragraphs)
            
            # Add footers last if included
            if include_footers:
                paragraphs.extend(footer_paragraphs)
            
            # 9. Handle cross-page paragraph continuity
            if paragraphs:
                # Check if we need to join with a paragraph from the previous page
                if prev_paragraph_info and paragraphs:
                    prev_text, ends_with_period = prev_paragraph_info
                    
                    # If previous paragraph doesn't end with a period and current page has paragraphs
                    if not ends_with_period:
                        # Get the first paragraph from this page
                        # Skip headers if they're included (headers shouldn't be joined with previous page content)
                        first_content_idx = len(header_paragraphs) if include_headers else 0
                        
                        if len(paragraphs) > first_content_idx:
                            first_content_paragraph = paragraphs[first_content_idx]
                            
                            # Check if this paragraph starts with a lowercase letter (likely continuation)
                            if first_content_paragraph and first_content_paragraph.strip():
                                first_char = first_content_paragraph.strip()[0]
                                is_lowercase_start = first_char.islower() if first_char.isalpha() else False
                                
                                if is_lowercase_start:
                                    # Join with previous paragraph and replace in the paragraphs list
                                    joined_paragraph = prev_text + " " + first_content_paragraph
                                    paragraphs[first_content_idx] = joined_paragraph
                                    prev_paragraph_info = None  # Reset after joining
                                else:
                                    # If not joining, still include the previous paragraph
                                    paragraphs.insert(first_content_idx, prev_text)
                                    prev_paragraph_info = None
                        else:
                            # If there are no content paragraphs, still include the previous paragraph
                            paragraphs.append(prev_text)
                            prev_paragraph_info = None
                
                # Check the last paragraph of current page for potential continuation
                last_idx = len(paragraphs) - 1
                if last_idx >= 0:
                    # Skip footers when checking for continuation
                    if include_footers and footer_paragraphs and last_idx >= len(paragraphs) - len(footer_paragraphs):
                        last_content_idx = last_idx - len(footer_paragraphs)
                        if last_content_idx >= 0:
                            last_paragraph = paragraphs[last_content_idx]
                        else:
                            last_paragraph = ""
                    else:
                        last_paragraph = paragraphs[last_idx]
                    
                    # Check if the paragraph ends with a period, question mark, exclamation mark, etc.
                    ends_with_period = bool(re.search(r'[.!?:;]$', last_paragraph.strip()))
                    
                    # Store for next page if it doesn't end with a sentence-ending punctuation
                    if not ends_with_period:
                        prev_paragraph_info = (last_paragraph, ends_with_period)
                        
                        # Remove this paragraph as we'll carry it to the next page
                        if include_footers and footer_paragraphs and last_idx >= len(paragraphs) - len(footer_paragraphs):
                            paragraphs.pop(last_content_idx)
                        else:
                            paragraphs.pop(last_idx)
        else:
            # If there are no content blocks, just process headers and footers
            paragraphs = []
            
            if include_headers and header_blocks:
                header_blocks.sort(key=lambda b: (b[1], b[0]))
                header_paragraphs = [block[4] for block in header_blocks]
                paragraphs.extend(header_paragraphs)
            
            if include_footers and footer_blocks:
                footer_blocks.sort(key=lambda b: (b[1], b[0]))
                footer_paragraphs = [block[4] for block in footer_blocks]
                paragraphs.extend(footer_paragraphs)
        
        # Add to result
        result["pages"].append({
            "page_number": page_num + 1,
            "paragraphs": paragraphs,
            "layout": "double_column" if is_double_column else "single_column"
        })
    
    # If there's still a paragraph carried over at the end of the document, add it
    if prev_paragraph_info:
        last_page = result["pages"][-1]
        last_page["paragraphs"].append(prev_paragraph_info[0])
    
    return result


def extract_text_from_pdf(
    pdf_path: str,
    include_headers: bool = True,
    include_footers: bool = False
) -> str:
    """
    Extract text from a PDF file, handling double-column layout and cross-page continuity.
    Returns the full text as a single string.
    
    Args:
        pdf_path: Path to the PDF file
        include_headers: Whether to include headers in the output
        include_footers: Whether to include footers in the output
        
    Returns:
        Extracted text as a single string
    """
    result = parse_double_column_pdf(
        pdf_path,
        include_headers=include_headers,
        include_footers=include_footers
    )
    
    all_text = []
    for page in result["pages"]:
        all_text.extend(page["paragraphs"])
    
    return "\n\n".join(all_text)


def main():
    # Example usage
    pdf_path = "path/to/your/research_paper.pdf"
    
    # Parse PDF with default settings (include headers, exclude footers)
    result = parse_double_column_pdf(
        pdf_path,
        include_headers=True,
        include_footers=False
    )
    
    # Print the first few paragraphs from each page
    for page in result["pages"]:
        print(f"Page {page['page_number']} - Layout: {page.get('layout', 'unknown')}")
        for i, para in enumerate(page["paragraphs"][:3]):  # First 3 paragraphs
            print(f"  Paragraph {i+1}: {para[:100]}...")  # First 100 chars
        print()
    
    # Alternative: Extract as plain text
    full_text = extract_text_from_pdf(pdf_path)
    print(f"Total extracted text length: {len(full_text)} characters")


if __name__ == "__main__":
    main()
