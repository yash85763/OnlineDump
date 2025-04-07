import os
import pdfplumber
import anthropic
import argparse
import json
from typing import List, Dict, Optional, Tuple
import textwrap
import re

# Constants
MAX_TOKENS_PER_CHUNK = 100000  # Adjust based on your API's context window limit
OVERLAP_TOKENS = 1000  # Overlap between chunks to maintain context


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    print(f"Extracting text from {pdf_path}...")
    
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    extracted_text += text + "\n\n"
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise


def split_text_into_chunks(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """
    Split text into chunks that fit within the token limit
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Simple approximation: 1 token ≈ 4 characters for English text
    char_limit = max_tokens * 4
    overlap_chars = overlap * 4
    
    if len(text) <= char_limit:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + char_limit
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a natural break point (newline or period followed by space)
        natural_break = text.rfind("\n\n", start, end)
        if natural_break == -1:
            natural_break = text.rfind(". ", start, end)
        
        if natural_break != -1 and natural_break > start + char_limit // 2:
            # If we found a good break point, use it
            end = natural_break + 2  # Include the newline or period+space
        else:
            # Otherwise just break at the character limit
            end = start + char_limit
        
        chunks.append(text[start:end])
        # Start the next chunk with overlap to maintain context
        start = max(start, end - overlap_chars)
    
    return chunks


def analyze_contract_with_llm(text_chunk: str, api_key: str, model: str = "claude-3-haiku-20240307") -> Dict:
    """
    Analyze contract text using the Anthropic API
    
    Args:
        text_chunk: Contract text to analyze
        api_key: Anthropic API key
        model: Model to use
        
    Returns:
        Dictionary containing analysis results
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """
    You are a contract analysis assistant focused specifically on data usage terms.
    Your task is to:
    1. Identify whether the contract mentions data usage or data collection
    2. Determine if there are any limitations on data use
    3. Extract specific clauses or sections that discuss data usage
    
    Provide your analysis in JSON format with the following structure:
    {
        "data_usage_mentioned": true/false,
        "data_limitations_exist": true/false,
        "summary": "A single sentence summary of data usage terms",
        "relevant_clauses": [
            {
                "text": "The exact text of the clause",
                "type": "usage/limitation/both"
            }
        ]
    }
    """
    
    user_prompt = f"""
    Analyze the following contract text for data usage terms and limitations.
    Focus only on terms related to how data can be used, collected, shared, or restricted.
    
    CONTRACT TEXT:
    {text_chunk}
    
    Respond only with the JSON structure described in your instructions.
    """
    
    try:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract the JSON response
        response_text = response.content[0].text
        
        # If the response has markdown code blocks with JSON, extract just the JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        # Clean any non-JSON text
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        return json.loads(json_str)
        
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return {
            "data_usage_mentioned": False,
            "data_limitations_exist": False,
            "summary": f"Error analyzing contract: {str(e)}",
            "relevant_clauses": []
        }


def merge_analysis_results(results: List[Dict]) -> Dict:
    """
    Merge analysis results from multiple chunks
    
    Args:
        results: List of analysis results from different chunks
        
    Returns:
        Merged analysis results
    """
    merged = {
        "data_usage_mentioned": any(r.get("data_usage_mentioned", False) for r in results),
        "data_limitations_exist": any(r.get("data_limitations_exist", False) for r in results),
        "summary": "",
        "relevant_clauses": []
    }
    
    # Combine unique clauses
    unique_clauses = {}
    for result in results:
        for clause in result.get("relevant_clauses", []):
            clause_text = clause.get("text", "")
            # Use first 100 chars as an approximate deduplication key
            key = clause_text[:100].strip()
            if key and key not in unique_clauses:
                unique_clauses[key] = clause
    
    merged["relevant_clauses"] = list(unique_clauses.values())
    
    # Create a comprehensive summary
    if merged["data_usage_mentioned"] and merged["data_limitations_exist"]:
        merged["summary"] = "The contract expresses usage of data AND imposes limitations on data use."
    elif merged["data_usage_mentioned"]:
        merged["summary"] = "The contract expresses usage of data but does NOT specify limitations on data use."
    elif merged["data_limitations_exist"]:
        merged["summary"] = "The contract imposes limitations on data use but does not explicitly address data usage."
    else:
        merged["summary"] = "The contract does NOT express usage of data nor impose limitations on data use."
    
    return merged


def analyze_contract(pdf_path: str, api_key: str, output_file: Optional[str] = None, model: str = "claude-3-haiku-20240307") -> Dict:
    """
    Extract text from a PDF contract and analyze it for data usage terms
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Anthropic API key
        output_file: Path to save the analysis results (optional)
        model: Model to use for analysis
        
    Returns:
        Analysis results
    """
    # Extract text from PDF
    contract_text = extract_text_from_pdf(pdf_path)
    
    # Split into chunks if necessary
    chunks = split_text_into_chunks(contract_text)
    print(f"Split contract into {len(chunks)} chunks")
    
    # Analyze each chunk
    results = []
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1} of {len(chunks)}...")
        chunk_result = analyze_contract_with_llm(chunk, api_key, model)
        results.append(chunk_result)
    
    # Merge results from all chunks
    if len(results) > 1:
        final_result = merge_analysis_results(results)
    else:
        final_result = results[0]
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(final_result, f, indent=2)
        print(f"Analysis saved to {output_file}")
    
    return final_result


def print_analysis_results(results: Dict):
    """
    Print analysis results in a human-readable format
    
    Args:
        results: Analysis results
    """
    print("\n" + "="*80)
    print("CONTRACT ANALYSIS RESULTS".center(80))
    print("="*80 + "\n")
    
    print(f"DATA USAGE MENTIONED: {'YES' if results['data_usage_mentioned'] else 'NO'}")
    print(f"DATA LIMITATIONS EXIST: {'YES' if results['data_limitations_exist'] else 'NO'}")
    print("\nSUMMARY:")
    print(textwrap.fill(results['summary'], width=80))
    
    if results['relevant_clauses']:
        print("\nRELEVANT CLAUSES:")
        for i, clause in enumerate(results['relevant_clauses'], 1):
            print(f"\n{i}. Type: {clause['type'].upper()}")
            print("-" * 80)
            print(textwrap.fill(clause['text'], width=80))
    else:
        print("\nNo relevant clauses found.")
    
    print("\n" + "="*80)


def main():
    """
    Main function to run the contract analyzer from command line
    """
    parser = argparse.ArgumentParser(description='Analyze contracts for data usage terms')
    parser.add_argument('pdf_path', help='Path to the contract PDF file')
    parser.add_argument('--api-key', help='Anthropic API key')
    parser.add_argument('--output', help='Path to save analysis results JSON')
    parser.add_argument('--model', default="claude-3-haiku-20240307", 
                        help='Model to use (default: claude-3-haiku-20240307)')
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
    
    # Analyze contract
    results = analyze_contract(args.pdf_path, api_key, args.output, args.model)
    
    # Print results
    print_analysis_results(results)


if __name__ == "__main__":
    main()