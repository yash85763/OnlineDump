import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def visualize_pdf_extraction(pdf_path, page_num=0):
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        
        # Extract the page as an image
        img = page.to_image(resolution=150).original
        
        # Convert PIL Image to numpy array for matplotlib
        img_array = np.array(img)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(img_array)
        
        # Extract words with bounding boxes
        words = page.extract_words()
        
        # Draw bounding boxes around each word
        for word in words:
            x0, top, x1, bottom = word['x0'], word['top'], word['x1'], word['bottom']
            width = x1 - x0
            height = bottom - top
            
            # Create a rectangle patch
            rect = patches.Rectangle((x0, top), width, height, 
                                     linewidth=1, edgecolor='r', facecolor='none')
            
            # Add the rectangle to the plot
            ax.add_patch(rect)
            
        # Show the extracted text below the boxes
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

If you specifically need to use PyPDF2, here's an alternative approach combining PyPDF2 with pdf2image and pytesseract for OCR and visualization:

```python
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_pdf_with_pypdf2(pdf_path, page_num=0):
    # Extract text using PyPDF2
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num]
    pypdf2_text = page.extract_text()
    print("PyPDF2 Extracted Text:")
    print(pypdf2_text)
    
    # Convert PDF to image
    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
    img = images[0]
    
    # Use pytesseract for OCR with bounding boxes
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(12, 16))
    plt.imshow(np.array(img))
    
    # Draw boxes around each detected text area
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            plt.gca().add_patch(
                patches.Rectangle((x, y), w, h, 
                                 linewidth=1, edgecolor='r', facecolor='none')
            )
            
    plt.title(f"PDF Content Extraction with PyPDF2 + OCR - Page {page_num+1}")
    plt.tight_layout()
    plt.savefig(f"pypdf2_extraction_page_{page_num+1}.png", dpi=300)
    plt.show()
    
    return pypdf2_text, ocr_data

# Usage
pdf_path = "your_document.pdf" 
text, ocr_data = visualize_pdf_with_pypdf2(pdf_path)