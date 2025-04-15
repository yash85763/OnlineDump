"""
Enhanced Double-Column PDF Parser using PyMuPDF

This script provides functionality for extracting text from double-columned PDFs
like research papers, ensuring:
1. Columns are processed in the correct order
2. Headers and footers are properly handled (optionally ignored)
3. Paragraphs that span across columns are correctly joined
4. Paragraphs that span across pages are correctly joined
5. Short paragraphs (less than 5 words) are joined with adjacent paragraphs
6. Any paragraph not ending with punctuation is joined with the next paragraph
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
    paragraph_spacing_threshold: int = 10,
    min_words_threshold: int = 5
) -> Dict[str, Any]:
    """
    Parse a double-column PDF correctly with enhanced handling of columns, headers, footers,
    cross-page paragraph continuity, and short paragraphs.
    
    Args:
        pdf_path: Path to the PDF file
        include_headers: Whether to include headers in the output (default True)
        include_footers: Whether to include footers in the output (default False)
        header_height_percentage: Percentage of page height considered as header area (default 10%)
        footer_height_percentage: Percentage of page height considered as footer area (default 10%)
        paragraph_spacing_threshold: Max spacing between blocks to be considered same paragraph (default 10)
        min_words_threshold: Minimum number of words for a paragraph to be considered standalone (default 5)
        
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
    prev_paragraph_info = None  # Will be a tuple: (text, word_count)
    
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
        
        # 3. Process headers and footers
        header_paragraphs = []
        if include_headers and header_blocks:
            # Sort header blocks from top to bottom, then left to right
            header_blocks.sort(key=lambda b: (b[1], b[0]))
            header_paragraphs = process_blocks_into_paragraphs(header_blocks, paragraph_spacing_threshold)
        
        footer_paragraphs = []
        if include_footers and footer_blocks:
            # Sort footer blocks from top to bottom, then left to right
            footer_blocks.sort(key=lambda b: (b[1], b[0]))
            footer_paragraphs = process_blocks_into_paragraphs(footer_blocks, paragraph_spacing_threshold)
        
        # 4. Process main content
        content_paragraphs = []
        is_double_column = False
        
        if content_blocks:
            # Identify columns using x-coordinates
            x_centers = [(block[0] + block[2]) / 2 for block in content_blocks]
            X = np.array(x_centers).reshape(-1, 1)
            
            # Skip clustering if too few blocks
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
            
            # Get the midpoint between columns for separation
            if len(column_centers) > 1:
                midpoint = (column_centers[0] + column_centers[1]) / 2
            else:
                midpoint = page_width / 2
            
            if is_double_column:
                # Separate into left and right columns
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
                
                # Process each column into paragraphs
                left_paragraphs = process_blocks_into_paragraphs(left_column, paragraph_spacing_threshold)
                right_paragraphs = process_blocks_into_paragraphs(right_column, paragraph_spacing_threshold)
                
                # Process all paragraphs from both columns sequentially
                all_column_paragraphs = left_paragraphs + right_paragraphs
                content_paragraphs = process_sequential_paragraphs(all_column_paragraphs, min_words_threshold)
            else:
                # Single column - sort all blocks by y-coordinate
                content_blocks.sort(key=lambda b: b[1])
                content_paragraphs_raw = process_blocks_into_paragraphs(content_blocks, paragraph_spacing_threshold)
                content_paragraphs = process_sequential_paragraphs(content_paragraphs_raw, min_words_threshold)
        
        # 5. Combine all paragraphs in the correct order
        all_paragraphs = []
        
        # Add headers first if included
        if header_paragraphs:
            all_paragraphs.extend(header_paragraphs)
        
        # Handle cross-page paragraph continuity
        if prev_paragraph_info and content_paragraphs:
            prev_text, prev_word_count = prev_paragraph_info
            
            # Always join if:
            # 1. The previous paragraph was very short (fewer than min_words_threshold words), OR
            # 2. The previous paragraph doesn't end with punctuation
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', prev_text.strip()))
            
            if prev_word_count < min_words_threshold or not ends_with_punctuation:
                first_content_para = content_paragraphs[0]
                joined_paragraph = prev_text + " " + first_content_para
                content_paragraphs[0] = joined_paragraph
                prev_paragraph_info = None
            else:
                # If not joining, still include the previous paragraph
                all_paragraphs.append(prev_text)
                prev_paragraph_info = None
        
        # Add main content
        all_paragraphs.extend(content_paragraphs)
        
        # Add footers last if included
        if footer_paragraphs:
            all_paragraphs.extend(footer_paragraphs)
        
        # Check the last paragraph for potential continuation to next page
        if content_paragraphs:
            last_para = content_paragraphs[-1]
            word_count = len(last_para.split())
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', last_para.strip()))
            
            # If it's a very short paragraph OR doesn't end with punctuation,
            # it might continue on the next page
            if word_count < min_words_threshold or not ends_with_punctuation:
                prev_paragraph_info = (last_para, word_count)
                
                # Remove from current page as we'll carry it forward
                all_paragraphs.pop()
        
        # Add processed paragraphs to result
        result["pages"].append({
            "page_number": page_num + 1,
            "paragraphs": all_paragraphs,
            "layout": "double_column" if is_double_column else "single_column"
        })
    
    # If there's still a paragraph carried over at the end of the document, add it
    if prev_paragraph_info:
        last_page = result["pages"][-1]
        last_page["paragraphs"].append(prev_paragraph_info[0])
    
    return result


def process_blocks_into_paragraphs(blocks, paragraph_spacing_threshold):
    """
    Process a list of text blocks into paragraphs based on vertical spacing.
    
    Args:
        blocks: List of text blocks with position information
        paragraph_spacing_threshold: Maximum spacing to consider blocks part of the same paragraph
        
    Returns:
        List of paragraph texts
    """
    paragraphs = []
    current_paragraph = ""
    
    for i, block in enumerate(blocks):
        text = block[4]
        
        if not text.strip():
            continue
        
        # If we're starting a new paragraph
        if not current_paragraph:
            current_paragraph = text
        else:
            # Check spacing between blocks
            if i > 0:
                prev_block = blocks[i-1]
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
            else:
                current_paragraph = text
    
    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append(current_paragraph)
        
    return paragraphs


def process_sequential_paragraphs(paragraphs, min_words_threshold=5):
    """
    Process a list of paragraphs sequentially, joining paragraphs that:
    1. Don't end with punctuation, OR
    2. Have fewer than min_words_threshold words
    
    Args:
        paragraphs: List of paragraphs to process
        min_words_threshold: Minimum word count to be considered a standalone paragraph
        
    Returns:
        List of processed paragraphs with appropriate joins
    """
    if not paragraphs:
        return []
    
    result_paragraphs = []
    current_paragraph = paragraphs[0]
    
    # Process paragraphs sequentially
    for i in range(1, len(paragraphs)):
        next_paragraph = paragraphs[i]
        
        # Check if current paragraph doesn't end with punctuation or is very short
        word_count = len(current_paragraph.split())
        ends_with_punctuation = bool(re.search(r'[.!?:;]$', current_paragraph.strip()))
        
        # Join if either condition is met
        if not ends_with_punctuation or word_count < min_words_threshold:
            # Join with the next paragraph
            current_paragraph += " " + next_paragraph
        else:
            # Current paragraph is complete, add to results and move to next
            result_paragraphs.append(current_paragraph)
            current_paragraph = next_paragraph
    
    # Add the final paragraph
    if current_paragraph:
        result_paragraphs.append(current_paragraph)
    
    return result_paragraphs


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
