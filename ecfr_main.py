"""
Example usage of the PDFHandler with different language detection methods
"""

import os
from ecfr_api_wrapper import PDFHandler


def process_pdf_with_custom_detection(pdf_path, output_folder, language_detection="basic"):
    """
    Process a PDF file with custom language detection method
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save output files
        language_detection: Language detection method ('spacy', 'nltk', 'basic')
    
    Returns:
        Result of processing
    """
    print(f"Processing PDF with {language_detection} language detection...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a PDF handler with the chosen language detection method
    pdf_handler = PDFHandler(
        min_quality_ratio=0.5,
        paragraph_spacing_threshold=10,
        min_words_threshold=5,
        lang_detection_method=language_detection,
        min_english_ratio=0.4  # Allow slightly lower English content (40% threshold)
    )
    
    # Process the PDF
    result = pdf_handler.process_pdf(pdf_path)
    
    # Check parsability
    if not result.get("parsable", False):
        print(f"PDF parsing failed: {result.get('error', 'Unknown error')}")
        return result
    
    # Save results
    pdf_name = os.path.basename(pdf_path)
    file_stem = os.path.splitext(pdf_name)[0]
    
    # Save JSON output
    json_output_path = os.path.join(output_folder, f"{file_stem}.json")
    pdf_handler.save_to_json(result, json_output_path)
    
    # Save text output
    txt_output_path = os.path.join(output_folder, f"{file_stem}.txt")
    pdf_handler.save_to_txt(result, txt_output_path)
    
    print(f"PDF processed successfully. Results saved to {output_folder}")
    return result


def compare_language_detection_methods(pdf_path):
    """
    Compare different language detection methods on the same PDF
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary with results from each method
    """
    methods = ["basic", "nltk", "spacy"]
    results = {}
    
    for method in methods:
        # Create a PDF handler with the current method
        pdf_handler = PDFHandler(lang_detection_method=method)
        
        # Extract content with the current method
        pages_data, is_parsable, quality_info = pdf_handler.extract_pdf_content(pdf_path)
        
        # Store results
        results[method] = {
            "is_parsable": is_parsable,
            "quality_info": quality_info,
            "page_count": len(pages_data)
        }
        
        print(f"Method: {method}")
        print(f"  Parsable: {is_parsable}")
        print(f"  Quality: {quality_info}")
        print(f"  Pages: {len(pages_data)}")
        print("-" * 50)
    
    return results


def override_detection_method_example(pdf_path):
    """
    Example showing how to override detection method for a specific call
    
    Args:
        pdf_path: Path to the PDF file
    """
    # Create handler with basic detection as default
    pdf_handler = PDFHandler(lang_detection_method="basic")
    
    print("Using default (basic) detection method:")
    pages_data, is_parsable, quality_info = pdf_handler.extract_pdf_content(pdf_path)
    print(f"  Parsable: {is_parsable}")
    print(f"  Quality: {quality_info}")
    
    # Override with spaCy for a specific call
    print("\nOverriding with spaCy detection method:")
    pages_data, is_parsable, quality_info = pdf_handler.extract_pdf_content(
        pdf_path, 
        lang_detection_method="spacy"
    )
    print(f"  Parsable: {is_parsable}")
    print(f"  Quality: {quality_info}")
    
    # Override with NLTK for a specific call
    print("\nOverriding with NLTK detection method:")
    pages_data, is_parsable, quality_info = pdf_handler.extract_pdf_content(
        pdf_path, 
        lang_detection_method="nltk"
    )
    print(f"  Parsable: {is_parsable}")
    print(f"  Quality: {quality_info}")


if __name__ == "__main__":
    # Example paths
    pdf_path = "contracts/sample_contract.pdf"
    output_folder = "extracted"
    
    # Process PDF with different detection methods
    process_pdf_with_custom_detection(pdf_path, output_folder, "spacy")
    
    # Compare all methods
    print("\nComparing all detection methods:")
    compare_language_detection_methods(pdf_path)
    
    # Example of method override
    print("\nMethod override example:")
    override_detection_method_example(pdf_path)