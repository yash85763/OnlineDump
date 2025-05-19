import json
import os
from typing import List, Dict, Any
import spacy
from collections import Counter
import numpy as np
from pdfhandle import PDFHandler  # Assuming your provided code is saved as pdfhandle.py

class POSTaggingExperiment:
    """Class to handle POS tagging and comparison between PDF content and provided paragraph."""
    
    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize the experiment.
        
        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save results
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy English model
        self.pdf_handler = PDFHandler()  # Initialize your PDF handler
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_pdf_sentences(self) -> List[Dict[str, Any]]:
        """
        Extract sentences from PDF content using PDFHandler.
        
        Returns:
            List of dictionaries containing sentence data
        """
        # Process PDF
        pdf_result = self.pdf_handler.process_pdf(self.pdf_path, generate_embeddings=False)
        
        if not pdf_result.get("parsable", False):
            raise ValueError(f"PDF processing failed: {pdf_result.get('error', 'Unknown error')}")
        
        sentences = []
        
        # Extract sentences from each paragraph
        for page in pdf_result.get("pages", []):
            page_num = page["page_number"]
            for para_idx, paragraph in enumerate(page.get("paragraphs", [])):
                # Process paragraph with spaCy
                doc = self.nlp(paragraph)
                for sent_idx, sent in enumerate(doc.sents):
                    sentences.append({
                        "page_number": page_num,
                        "paragraph_index": para IDX,
                        "sentence_index": sent_idx,
                        "text": sent.text.strip(),
                        "pos_tags": [(token.text, token.pos_) for token in sent]
                    })
        
        return sentences
    
    def process_input_paragraph(self, paragraph: str) -> List[Dict[str, Any]]:
        """
        Process a provided paragraph and extract sentences with POS tags.
        
        Args:
            paragraph: Input paragraph text
            
        Returns:
            List of dictionaries containing sentence data
        """
        # Process paragraph with spaCy
        doc = self.nlp(paragraph)
        sentences = []
        
        for sent_idx, sent in enumerate(doc.sents):
            sentences.append({
                "source": "input_paragraph",
                "sentence_index": sent_idx,
                "text": sent.text.strip(),
                "pos_tags": [(token.text, token.pos_) for token in sent]
            })
        
        return sentences
    
    def compare_pos_distributions(self, pdf_sentences: List[Dict[str, Any]], 
                               input_sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare POS tag distributions between PDF sentences and input paragraph sentences.
        
        Args:
            pdf_sentences: List of sentence dictionaries from PDF
            input_sentences: List of sentence dictionaries from input paragraph
            
        Returns:
            Dictionary with comparison results
        """
        # Count POS tags
        pdf_pos_counts = Counter()
        input_pos_counts = Counter()
        
        # Collect POS tags from PDF sentences
        for sent in pdf_sentences:
            for _, pos in sent["pos_tags"]:
                pdf_pos_counts[pos] += 1
        
        # Collect POS tags from input paragraph sentences
        for sent in input_sentences:
            for _, pos in sent["pos_tags"]:
                input_pos_counts[pos] += 1
        
        # Normalize counts to get distributions
        pdf_total = sum(pdf_pos_counts.values())
        input_total = sum(input_pos_counts.values())
        
        pdf_dist = {pos: count/pdf_total for pos, count in pdf_pos_counts.items()}
        input_dist = {pos: count/input_total for pos, count in input_pos_counts.items()}
        
        # Combine all POS tags seen in either source
        all_pos_tags = set(pdf_dist.keys()) | set(input_dist.keys())
        
        # Calculate differences
        distribution_diff = {}
        for pos in all_pos_tags:
            pdf_freq = pdf_dist.get(pos, 0.0)
            input_freq = input_dist.get(pos, 0.0)
            distribution_diff[pos] = {
                "pdf_freq": pdf_freq,
                "input_freq": input_freq,
                "difference": pdf_freq - input_freq
            }
        
        # Calculate statistical similarity (cosine similarity)
        pdf_vector = [pdf_dist.get(pos, 0.0) for pos in all_pos_tags]
        input_vector = [input_dist.get(pos, 0.0) for pos in all_pos_tags]
        
        if pdf_vector and input_vector:
            cosine_sim = np.dot(pdf_vector, input_vector) / (
                np.linalg.norm(pdf_vector) * np.linalg.norm(input_vector)
            )
        else:
            cosine_sim = 0.0
        
        return {
            "pdf_pos_counts": dict(pdf_pos_counts),
            "input_pos_counts": dict(input_pos_counts),
            "distribution_difference": distribution_diff,
            "cosine_similarity": float(cosine_sim)
        }
    
    def run_experiment(self, input_paragraph: str) -> Dict[str, Any]:
        """
        Run the complete POS tagging comparison experiment.
        
        Args:
            input_paragraph: Paragraph to compare against PDF content
            
        Returns:
            Dictionary with complete results
        """
        try:
            # Extract sentences from PDF
            pdf_sentences = self.extract_pdf_sentences()
            
            # Process input paragraph
            input_sentences = self.process_input_paragraph(input_paragraph)
            
            # Compare POS distributions
            comparison_results = self.compare_pos_distributions(pdf_sentences, input_sentences)
            
            # Prepare result
            result = {
                "pdf_filename": os.path.basename(self.pdf_path),
                "pdf_sentences": pdf_sentences,
                "input_sentences": input_sentences,
                "comparison": comparison_results
            }
            
            # Save results
            output_path = os.path.join(self.output_dir, "pos_tagging_results.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            return {"error": f"Experiment failed: {str(e)}"}
    
    def print_summary(self, result: Dict[str, Any]):
        """
        Print a summary of the experiment results.
        
        Args:
            result: Experiment result dictionary
        """
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nPOS Tagging Experiment Summary for {result['pdf_filename']}")
        print(f"Number of PDF sentences: {len(result['pdf_sentences'])}")
        print(f"Number of input paragraph sentences: {len(result['input_sentences'])}")
        print("\nPOS Tag Distribution Comparison:")
        
        for pos, data in result["comparison"]["distribution_difference"].items():
            print(f"{pos}:")
            print(f"  PDF freq: {data['pdf_freq']:.4f}")
            print(f"  Input freq: {data['input_freq']:.4f}")
            print(f"  Difference: {data['difference']:.4f}")
        
        print(f"\nCosine Similarity: {result['comparison']['cosine_similarity']:.4f}")

def main():
    # Configuration
    pdf_path = "contracts/sample_contract.pdf"
    output_dir = "pos_tagging_results"
    input_paragraph = """This agreement is made between the parties. It outlines the terms clearly. 
    All disputes shall be resolved through arbitration."""
    
    # Initialize and run experiment
    experiment = POSTaggingExperiment(pdf_path, output_dir)
    result = experiment.run_experiment(input_paragraph)
    
    # Print summary
    experiment.print_summary(result)
    
    print(f"\nResults saved to: {os.path.join(output_dir, 'pos_tagging_results.json')}")

if __name__ == "__main__":
    main()
