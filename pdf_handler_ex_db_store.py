# Database Schema: The pdf_metadata table stores:

#     id: Auto-incrementing primary key.
#     filename: Name of the PDF file.
#     layout: Layout type ("single_column" or "double_column").
#     num_pages: Number of pages in the PDF.
#     total_word_count: Total number of words across all paragraphs.
#     avg_word_count_per_page: Average word count per page.
#     raw_text: Full text content, with paragraphs joined by double newlines.

# Functionality:

#     The create_database function sets up a SQLite database and creates the pdf_metadata table if it doesn't exist.
#     The store_pdf_data function processes the PDF using PDFHandler, extracts the required metadata, and inserts it into the database.
#     The raw text is collected by joining paragraphs per page with spaces and pages with double newlines, consistent with the text file output format.

# Usage: Run the script with a valid PDF path and database path. The script checks if the PDF exists, processes it, and stores the metadata in the SQLite database.


import sqlite3
import os
from pdfhandle import PDFHandler

def create_database(db_path: str):
    """Create a SQLite database and table for storing PDF metadata."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table for PDF metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            layout TEXT,
            num_pages INTEGER,
            total_word_count INTEGER,
            avg_word_count_per_page REAL,
            raw_text TEXT
        )
    """)
    
    conn.commit()
    return conn, cursor

def store_pdf_data(pdf_path: str, db_path: str, generate_embeddings: bool = False):
    """
    Process a PDF and store its metadata in a SQLite database.
    
    Args:
        pdf_path: Path to the PDF file
        db_path: Path to the SQLite database
        generate_embeddings: Whether to generate embeddings (optional)
    """
    # Initialize PDF handler
    handler = PDFHandler()
    
    # Process the PDF
    result = handler.process_pdf(pdf_path, generate_embeddings)
    
    if not result.get("parsable"):
        print(f"Error processing {pdf_path}: {result.get('error')}")
        return
    
    # Extract metadata
    filename = result["filename"]
    layout = result["layout"]
    num_pages = len(result["pages"])
    
    # Collect all text and count words
    raw_text = []
    total_word_count = 0
    
    for page in result["pages"]:
        page_text = " ".join(page["paragraphs"])
        raw_text.append(page_text)
        total_word_count += len(page_text.split())
    
    raw_text = "\n\n".join(raw_text)
    avg_word_count_per_page = total_word_count / num_pages if num_pages > 0 else 0
    
    # Connect to database
    conn, cursor = create_database(db_path)
    
    # Insert data into the table
    cursor.execute("""
        INSERT INTO pdf_metadata (filename, layout, num_pages, total_word_count, 
                                avg_word_count_per_page, raw_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, layout, num_pages, total_word_count, 
          avg_word_count_per_page, raw_text))
    
    conn.commit()
    print(f"Stored metadata for {filename} in {db_path}")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    # Example usage
    pdf_path = "contracts/sample_contract.pdf"
    db_path = "pdf_database.db"
    
    if os.path.isfile(pdf_path):
        store_pdf_data(pdf_path, db_path, generate_embeddings=False)
    else:
        print(f"PDF file not found: {pdf_path}")
