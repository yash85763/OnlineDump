import pdfplumber
from pathlib import Path

def filter_pages_by_word_count(input_pdf_path, output_txt_path, min_word_count=100):
    """
    Parse a PDF using pdfplumber, save content of pages with word count > average,
    excluding pages with < min_word_count.
    
    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_txt_path (str): Path to the output text file.
        min_word_count (int): Minimum word count to consider a page for averaging (default: 100).
    """
    # Open the PDF
    with pdfplumber.open(input_pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # Calculate word count for each page and store text
        word_counts = []
        page_texts = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""  # Handle pages with no text
            text = text.strip()
            word_count = len(text.split())
            word_counts.append(word_count)
            page_texts.append(text)
        
        # Filter pages with word count >= min_word_count for averaging
        valid_word_counts = [wc for wc in word_counts if wc >= min_word_count]
        
        # Compute average word count for valid pages
        if valid_word_counts:
            avg_word_count = sum(valid_word_counts) / len(valid_word_counts)
        else:
            avg_word_count = float('inf')  # No valid pages, no output
        
        # Save content of pages with word count >= min_word_count and > average
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            for page_num, (word_count, text) in enumerate(zip(word_counts, page_texts)):
                if word_count >= min_word_count and word_count > avg_word_count:
                    txt_file.write(f"--- Page {page_num + 1} (Word Count: {word_count}) ---\n")
                    txt_file.write(text)
                    txt_file.write("\n\n")
    
    print(f"Average word count (for pages with >= {min_word_count} words): {avg_word_count:.2f}")
    print(f"Content saved to {output_txt_path}")

def process_multiple_pdfs(input_dir, output_dir, min_word_count=100):
    """
    Process all PDFs in a directory, filtering pages by word count.
    
    Args:
        input_dir (str): Directory containing input PDFs.
        output_dir (str): Directory to save output text files.
        min_word_count (int): Minimum word count to consider a page.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for pdf_path in input_dir.glob("*.pdf"):
        output_txt_path = output_dir / f"{pdf_path.stem}_filtered.txt"
        filter_pages_by_word_count(pdf_path, output_txt_path, min_word_count)

# Example usage
if __name__ == "__main__":
    # Single PDF
    input_pdf = "contract.pdf"
    output_txt = "filtered_content.txt"
    filter_pages_by_word_count(input_pdf, output_txt, min_word_count=100)
    
    # Multiple PDFs
    input_dir = "contracts"
    output_dir = "filtered_texts"
    process_multiple_pdfs(input_dir, output_dir, min_word_count=100)