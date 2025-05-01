Here's the code that needs to be added to your `PDFHandler` class to save parsed content to a text file and handle file naming using the PDF's stem:

```python
from pathlib import Path  # Add this import at the top with other imports

# Add this new method to the PDFHandler class:
def save_to_txt(self, pages_content: List[Dict[str, Any]], output_path: str) -> str:
    """
    Save the parsed content to a text file.
    
    Args:
        pages_content: List of dictionaries containing page content
        output_path: Path to save the text file
        
    Returns:
        Path to the saved text file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write content to text file
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages_content:
            f.write(f"=== Page {page['page_number']} ===\n")
            f.write(f"Layout: {page.get('layout', 'unknown')}\n")
            f.write("\n")
            
            for paragraph in page["paragraphs"]:
                f.write(paragraph)
                f.write("\n\n")
            
            f.write("\n")
    
    return output_path

# Modify the process_pdf method to include optional txt saving:
def process_pdf(self, pdf_path: str, generate_embeddings: bool = False, save_txt: bool = True) -> Dict[str, Any]:
    """
    Process a PDF file through the complete pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        generate_embeddings: Whether to generate embeddings for paragraphs
        save_txt: Whether to save the parsed content to a txt file
        
    Returns:
        Dictionary with the processed content or an error message
    """
    try:
        # Check if the PDF is parsable
        doc = fitz.open(pdf_path)
        is_parsable, quality_info = self.check_parsability(doc)
        
        if not is_parsable:
            return {
                "filename": os.path.basename(pdf_path),
                "parsable": False,
                "error": quality_info
            }
        
        # Determine the layout
        layout_type = self.determine_layout(doc)
        
        # Parse content into paragraphs
        pages_content = self.parse_paragraphs(doc)
        
        # Generate JSON structure
        result = {
            "filename": os.path.basename(pdf_path),
            "parsable": True,
            "layout": layout_type,
            "pages": pages_content
        }
        
        # Optionally generate embeddings
        if generate_embeddings and (
            (self.embedding_provider == "sentence_transformers" and self.embedding_model is not None) or
            (self.embedding_provider == "azure_openai" and self.azure_client is not None)):
            result["embeddings"] = self.generate_embeddings(pages_content)
        
        # Optionally save to txt file
        if save_txt and is_parsable:
            # Use PDF's stem for the output filename
            pdf_path_obj = Path(pdf_path)
            txt_output_path = pdf_path_obj.parent / f"{pdf_path_obj.stem}.txt"
            self.save_to_txt(pages_content, str(txt_output_path))
            result["txt_saved"] = str(txt_output_path)
        
        return result
        
    except Exception as e:
        return {
            "filename": os.path.basename(pdf_path),
            "parsable": False,
            "error": f"Error processing PDF: {str(e)}"
        }

# Modify the process_directory function to use the PDF stem for output filenames:
def process_directory(input_dir: str, output_dir: str, generate_embeddings: bool = False, save_txt: bool = True) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save results
        generate_embeddings: Whether to generate embeddings
        save_txt: Whether to save content to txt files
        
    Returns:
        List of results for each processed PDF
    """
    # Create PDF handler
    handler = PDFHandler()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Process each PDF file
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            
            # Use the stem of the PDF file for output names
            pdf_stem = Path(filename).stem
            
            json_output_path = os.path.join(output_dir, f"{pdf_stem}.json")
            txt_output_path = os.path.join(output_dir, f"{pdf_stem}.txt")
            
            print(f"Processing {filename}...")
            result = handler.process_pdf(pdf_path, generate_embeddings, save_txt)
            
            # Save result to JSON
            handler.save_to_json(result, json_output_path)
            
            # If result is successful and save_txt is True, save to txt
            if result.get("parsable", False) and save_txt:
                handler.save_to_txt(result["pages"], txt_output_path)
                result["txt_saved"] = txt_output_path
            
            results.append(result)
    
    return results

# Update the main execution block to handle file naming:
if __name__ == "__main__":
    # Hardcoded file paths
    input_path = "contracts/sample_contract.pdf"  # Path to a single PDF file
    
    # Use Path to get the stem and create output paths
    pdf_path_obj = Path(input_path)
    pdf_stem = pdf_path_obj.stem
    
    output_json_path = f"extracted/{pdf_stem}.json"  # Path for the output JSON
    output_txt_path = f"extracted/{pdf_stem}.txt"    # Path for the output TXT
    
    # Whether to generate embeddings
    generate_embeddings = True
    save_txt = True
    
    # Choose embedding provider: "sentence_transformers" or "azure_openai"
    embedding_provider = "azure_openai"
    
    # For Azure OpenAI embeddings, provide these configuration details
    azure_openai_config = {
        "api_key": "your-azure-openai-api-key",
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2023-05-15"
    }
    
    # Process a single file
    if os.path.isfile(input_path):
        # Initialize the handler with appropriate configuration
        if embedding_provider == "azure_openai":
            handler = PDFHandler(
                embedding_provider=embedding_provider,
                embedding_model="text-embedding-ada-002",  # Use your deployment name here
                azure_openai_config=azure_openai_config
            )
        else:
            # Default to sentence_transformers
            handler = PDFHandler(embedding_provider="sentence_transformers")
            
        result = handler.process_pdf(input_path, generate_embeddings, save_txt)
        handler.save_to_json(result, output_json_path)
        
        if result.get("parsable", False) and save_txt:
            handler.save_to_txt(result["pages"], output_txt_path)
            print(f"Processed {input_path} and saved to {output_json_path} and {output_txt_path}")
        else:
            print(f"Processed {input_path} and saved to {output_json_path}")
```

The key changes include:
1. Add a `Path` import at the top
2. Add a new `save_to_txt()` method to save parsed content to text files
3. Modify `process_pdf()` to accept a `save_txt` parameter and automatically save to txt
4. Modify `process_directory()` to use the PDF stem for naming output files
5. Update the main execution block to properly handle file naming using `Path.stem`