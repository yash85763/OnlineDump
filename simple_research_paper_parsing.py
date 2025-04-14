"""
Double-Column PDF Parser using PyMuPDF

This script demonstrates how to properly extract text from double-columned PDFs
like research papers, ensuring that columns are processed in the correct order.
"""

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Any
import os


def parse_double_column_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Parse a double-column PDF correctly, handling column separation.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with parsed content by page
    """
    result = {
        "filename": os.path.basename(pdf_path),
        "pages": []
    }
    
    # Open the document
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 1. Extract text blocks
        blocks = page.get_text("blocks")
        
        # If no blocks, skip this page
        if not blocks:
            result["pages"].append({
                "page_number": page_num + 1,
                "paragraphs": []
            })
            continue
        
        # 2. Identify columns using the x-coordinates of blocks
        x_centers = [(block[0] + block[2]) / 2 for block in blocks]
        X = np.array(x_centers).reshape(-1, 1)
        
        # Skip clustering if too few blocks
        if len(blocks) < 3:
            # Sort blocks by y-coordinate (top to bottom)
            sorted_blocks = sorted(blocks, key=lambda b: b[1])
            paragraphs = [block[4] for block in sorted_blocks]
            
            result["pages"].append({
                "page_number": page_num + 1,
                "paragraphs": paragraphs
            })
            continue
        
        # Try to identify 2 columns with K-means
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        centers = kmeans.cluster_centers_.flatten()
        
        # Sort centers from left to right
        column_centers = sorted(centers)
        
        # Get the midpoint between columns
        if len(column_centers) > 1:
            midpoint = (column_centers[0] + column_centers[1]) / 2
        else:
            # If only one cluster was found, use the page midpoint
            midpoint = page_width / 2
        
        # Check if this is likely a double-column layout
        is_double_column = False
        if len(column_centers) > 1:
            # If centers are separated by at least 20% of page width
            col_distance = abs(column_centers[1] - column_centers[0])
            if col_distance > (page_width * 0.2):
                is_double_column = True
        
        # 3. Group blocks by column and sort within columns
        left_column = []
        right_column = []
        
        for block in blocks:
            block_center_x = (block[0] + block[2]) / 2
            
            if block_center_x < midpoint:
                left_column.append(block)
            else:
                right_column.append(block)
        
        # Sort each column by y-coordinate (top to bottom)
        left_column.sort(key=lambda b: b[1])
        right_column.sort(key=lambda b: b[1])
        
        # 4. Extract paragraphs from each column
        left_paragraphs = []
        right_paragraphs = []
        
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
                    paragraph_spacing_threshold = 10  # Adjust as needed
                    
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
        
        # 5. Build the final paragraphs list based on layout
        if is_double_column:
            # For double-column layout, first process left column, then right column
            paragraphs = left_paragraphs + right_paragraphs
        else:
            # For single-column layout, sort all blocks by y-coordinate
            sorted_blocks = sorted(blocks, key=lambda b: b[1])
            paragraphs = process_column(sorted_blocks)
        
        # Add to result
        result["pages"].append({
            "page_number": page_num + 1,
            "paragraphs": paragraphs,
            "layout": "double_column" if is_double_column else "single_column"
        })
        
    return result


def main():
    # Example usage
    pdf_path = "path/to/your/research_paper.pdf"
    result = parse_double_column_pdf(pdf_path)
    
    # Print the first few paragraphs from each page
    for page in result["pages"]:
        print(f"Page {page['page_number']} - Layout: {page.get('layout', 'unknown')}")
        for i, para in enumerate(page["paragraphs"][:3]):  # First 3 paragraphs
            print(f"  Paragraph {i+1}: {para[:100]}...")  # First 100 chars
        print()


if __name__ == "__main__":
    main()
