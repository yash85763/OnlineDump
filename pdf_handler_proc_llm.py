# Functionality:
#         The process_pdf_to_llm function processes the PDF using PDFHandler to extract paragraphs.
#         All paragraphs are cleaned using the clean_text method and joined with double newlines to form a single string.
#         The text is sent to an Azure OpenAI LLM (e.g., GPT-4) with a prompt requesting a summary of the contract text.
#         The response includes the filename, the LLM's summary, and the number of tokens used.
#     Azure OpenAI Integration: Uses the same Azure OpenAI configuration format as the PDFHandler class for consistency. The model deployment name (e.g., "gpt-4") is specified, and the prompt is tailored for summarization.
#     Error Handling: Checks for PDF parsability and handles API call errors, returning appropriate error messages.
#     Usage: Run the script with a valid PDF path and Azure OpenAI configuration. The script processes the PDF and prints the LLM's summary and token usage.

# Notes

#     Dependencies: Ensure pdfminer.six and openai are installed (pip install pdfminer.six openai). For embeddings, install sentence-transformers if needed.
#     Azure OpenAI Configuration: Replace placeholder values (your-azure-openai-api-key, https://your-resource-name.openai.azure.com/) with actual credentials.
#     Database: The SQLite example uses a simple schema. For production, you might want to add indexes or additional fields (e.g., timestamps, embedding storage).
#     LLM Token Limits: Large PDFs may exceed token limits for some LLMs. You may need to chunk the text or summarize per page for very long documents.
#     Text File Output: The save_to_txt method ensures paragraphs are separated by double newlines, matching the requested format.


import os
from pdfhandle import PDFHandler
from openai import AzureOpenAI

def process_pdf_to_llm(pdf_path: str, azure_openai_config: Dict[str, str], model: str = "gpt-4"):
    """
    Process a PDF and send its content to an Azure OpenAI LLM for processing.
    
    Args:
        pdf_path: Path to the PDF file
        azure_openai_config: Configuration for Azure OpenAI
                            - api_key: Azure OpenAI API key
                            - azure_endpoint: Azure OpenAI endpoint URL
                            - api_version: API version
        model: Azure OpenAI model deployment name (default: "gpt-4")
    
    Returns:
        Response from the LLM or error message
    """
    # Initialize PDF handler
    handler = PDFHandler()
    
    # Process the PDF
    result = handler.process_pdf(pdf_path, generate_embeddings=False)
    
    if not result.get("parsable"):
        return {"error": f"Error processing PDF: {result.get('error')}"}
    
    # Collect all paragraphs into a single string
    all_paragraphs = []
    for page in result["pages"]:
        all_paragraphs.extend(page["paragraphs"])
    
    # Clean and join paragraphs with double newlines
    full_text = "\n\n".join(handler.clean_text(para) for para in all_paragraphs if para.strip())
    
    # Initialize Azure OpenAI client
    try:
        client = AzureOpenAI(
            api_key=azure_openai_config["api_key"],
            api_version=azure_openai_config["api_version"],
            azure_endpoint=azure_openai_config["azure_endpoint"]
        )
        
        # Send content to LLM with a sample prompt
        prompt = (
            "You are a legal assistant. Please summarize the following contract text in 200 words or less:\n\n"
            f"{full_text}"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust based on desired output length
            temperature=0.7
        )
        
        return {
            "filename": result["filename"],
            "summary": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    
    except Exception as e:
        return {"error": f"Error calling Azure OpenAI: {str(e)}"}

if __name__ == "__main__":
    # Example usage
    pdf_path = "contracts/sample_contract.pdf"
    azure_openai_config = {
        "api_key": "your-azure-openai-api-key",
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2023-05-15"
    }
    
    if os.path.isfile(pdf_path):
        result = process_pdf_to_llm(pdf_path, azure_openai_config, model="gpt-4")
        if "error" in result:
            print(result["error"])
        else:
            print(f"Summary for {result['filename']}:\n{result['summary']}")
            print(f"Tokens used: {result['tokens_used']}")
    else:
        print(f"PDF file not found: {pdf_path}")
