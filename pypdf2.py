import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def visualize_pdf_extraction(pdf_path, page_num=0, resolution=150):
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        
        # Get page dimensions
        pdf_width = page.width
        pdf_height = page.height
        
        # Extract the page as an image with specified resolution
        img = page.to_image(resolution=resolution).original
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Calculate scaling factors
        x_scale = img_width / pdf_width
        y_scale = img_height / pdf_height
        
        # Convert PIL Image to numpy array for matplotlib
        img_array = np.array(img)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(img_array)
        
        # Extract words with bounding boxes
        words = page.extract_words()
        
        # Draw bounding boxes around each word with proper scaling
        for word in words:
            # Scale the coordinates to match the image
            x0 = word['x0'] * x_scale
            top = word['top'] * y_scale
            x1 = word['x1'] * x_scale
            bottom = word['bottom'] * y_scale
            
            width = x1 - x0
            height = bottom - top
            
            # Create a rectangle patch
            rect = patches.Rectangle((x0, top), width, height, 
                                    linewidth=1, edgecolor='r', facecolor='none', alpha=0.7)
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
        
        # Show the extracted text
        all_text = page.extract_text()
        print("Extracted Text:")
        print(all_text)
            
        plt.title(f"PDF Content Extraction - Page {page_num+1}")
        plt.tight_layout()
        plt.savefig(f"pdf_extraction_page_{page_num+1}.png", dpi=300)
        plt.show()
        
        return all_text, words

# Usage
pdf_path = "your_document.pdf"
text, word_data = visualize_pdf_extraction(pdf_path)
```

If you're still having issues with alignment, here's a more robust approach using PyMuPDF (fitz) which typically handles coordinate mapping more precisely:

```python
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_with_pymupdf(pdf_path, page_num=0, zoom=2):
    # Open the document
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Set a higher resolution for rendering
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    
    # Get the image
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:  # RGBA
        img = img[:, :, :3]  # Convert RGBA to RGB
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img)
    
    # Get the words with their bounding boxes (more precise than blocks)
    word_list = page.get_text("words")
    
    # Draw bounding boxes around each word with proper scaling
    for word in word_list:
        x0, y0, x1, y1, text, block_no, line_no, word_no = word
        
        # Scale coordinates to match the rendered image
        x0 = x0 * zoom
        y0 = y0 * zoom
        x1 = x1 * zoom
        y1 = y1 * zoom
        
        width = x1 - x0
        height = y1 - y0
        
        # Create a rectangle patch
        rect = patches.Rectangle((x0, y0), width, height, 
                                linewidth=1, edgecolor='r', facecolor='none', alpha=0.7)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    plt.title(f"PDF Content Extraction with PyMuPDF - Page {page_num+1}")
    plt.tight_layout()
    plt.savefig(f"pymupdf_extraction_page_{page_num+1}.png", dpi=300)
    plt.show()
    
    # Extract all text
    all_text = page.get_text()
    print("Extracted Text:")
    print(all_text)
    
    return all_text, word_list

# Usage
pdf_path = "your_document.pdf"
text, words = visualize_with_pymupdf(pdf_path)