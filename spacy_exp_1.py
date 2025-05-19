import json
import os
from typing import List, Dict, Any
import numpy as np
from pdfhandle import PDFHandler  # Assuming your provided code is saved as pdfhandle.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ParagraphSimilarityExperiment:
    """Class to compare PDF paragraphs with a provided paragraph using embeddings."""
    
    def __init__(self, pdf_path: str, output_dir: str, 
                 embedding_provider: str = "sentence_transformers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 azure_openai_config: Dict[str, str] = None):
        """
        Initialize the experiment.
        
        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save results
            embedding_provider: "sentence_transformers" or "azure_openai"
            embedding_model: Model name for embeddings
            azure_openai_config: Config for Azure OpenAI if used
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self.azure_openai_config = azure_openai_config or {}
        
        # Initialize PDF handler with embedding configuration
        self.pdf_handler = PDFHandler(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            azure_openai_config=azure_openai_config
        )
        
        # Initialize embedding model
        self.embedding_model = None
        self.azure_client = None
        
        if embedding_provider == "sentence_transformers":
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                raise ValueError(f"Failed to load Sentence Transformers model: {e}")
        
        elif embedding_provider == "azure_openai":
            try:
                from openai import AzureOpenAI
                required_params = ["api_key", "azure_endpoint", "api_version"]
                if not all(param in self.azure_openai_config for param in required_params):
                    raise ValueError(f"Missing Azure OpenAI config: {', '.join(required_params)}")
                self.azure_client = AzureOpenAI(
                    api_key=self.azure_openai_config["api_key"],
                    api_version=self.azure_openai_config["api_version"],
                    azure_endpoint=self.azure_openai_config["azure_endpoint"]
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Azure OpenAI client: {e}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_pdf_paragraphs(self) -> List[Dict[str, Any]]:
        """
        Extract paragraphs from PDF using PDFHandler.
        
        Returns:
            List of dictionaries containing paragraph data
        """
        # Process PDF with embeddings
        pdf_result = self.pdf_handler.process_pdf(self.pdf_path, generate_embeddings=True)
        
        if not pdf_result.get("parsable", False):
            raise ValueError(f"PDF processing failed: {pdf_result.get('error', 'Unknown error')}")
        
        paragraphs = []
        embeddings = pdf_result.get("embeddings", {})
        
        # Extract paragraphs and their embeddings
        for page_idx, page in enumerate(pdf_result.get("pages", [])):
            page_num = page["page_number"]
            for para_idx, paragraph in enumerate(page.get("paragraphs", [])):
                # Get embedding if available
                embedding = None
                if page_idx in embeddings and para_idx in embeddings[page_idx]:
                    embedding = embeddings[page_idx][para_idx]
                
                paragraphs.append({
                    "page_number": page_num,
                    "paragraph_index": para_idx,
                    "text": paragraph.strip(),
                    "embedding": embedding
                })
        
        return paragraphs
    
    def get_paragraph_embedding(self, paragraph: str) -> np.ndarray:
        """
        Generate embedding for a single paragraph.
        
        Args:
            paragraph: Input paragraph text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_provider == "sentence_transformers":
            return self.embedding_model.encode([paragraph])[0]
        
        elif self.embedding_provider == "azure_openai":
            response = self.azure_client.embeddings.create(
                input=[paragraph],
                model=self.embedding_model_name
            )
            return np.array(response.data[0].embedding)
        
        raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def compute_similarity_scores(self, pdf_paragraphs: List[Dict[str, Any]], 
                               input_paragraph: str) -> List[Dict[str, Any]]:
        """
        Compute cosine similarity between input paragraph and each PDF paragraph.
        
        Args:
            pdf_paragraphs: List of PDF paragraph dictionaries
            input_paragraph: Input paragraph text
            
        Returns:
            List of dictionaries with similarity scores
        """
        # Get embedding for input paragraph
        input_embedding = self.get_paragraph_embedding(input_paragraph)
        
        results = []
        
        for para in pdf_paragraphs:
            if para["embedding"] is None:
                continue  # Skip paragraphs without embeddings
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                [input_embedding],
                [para["embedding"]]
            )[0][0]
            
            results.append({
                "page_number": para["page_number"],
                "paragraph_index": para["paragraph_index"],
                "text": para["text"],
                "similarity_score": float(similarity)
            })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results
    
    def run_experiment(self, input_paragraph: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Run the paragraph similarity experiment.
        
        Args:
            input_paragraph: Paragraph to compare against PDF content
            top_n: Number of top similar paragraphs to highlight
            
        Returns:
            Dictionary with experiment results
        """
        try:
            # Extract paragraphs from PDF
            pdf_paragraphs = self.extract_pdf_paragraphs()
            
            # Compute similarity scores
            similarity_results = self.compute_similarity_scores(pdf_paragraphs, input_paragraph)
            
            # Get top N most similar paragraphs
            top_matches = similarity_results[:top_n]
            
            # Prepare result
            result = {
                "pdf_filename": os.path.basename(self.pdf_path),
                "input_paragraph": input_paragraph,
                "similarity_results": similarity_results,
                "top_matches": top_matches,
                "statistics": {
                    "total_paragraphs": len(similarity_results),
                    "max_similarity": max((r["similarity_score"] for r in similarity_results), default=0.0),
                    "avg_similarity": np.mean([r["similarity_score"] for r in similarity_results]) if similarity_results else 0.0
                }
            }
            
            # Save results
            output_path = os.path.join(self.output_dir, "paragraph_similarity_results.json")
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
        
        print(f"\nParagraph Similarity Experiment Summary for {result['pdf_filename']}")
        print(f"Input Paragraph: {result['input_paragraph']}")
        print(f"Total Paragraphs Compared: {result['statistics']['total_paragraphs']}")
        print(f"Maximum Similarity Score: {result['statistics']['max_similarity']:.4f}")
        print(f"Average Similarity Score: {result['statistics']['avg_similarity']:.4f}")
        
        print(f"\nTop {len(result['top_matches'])} Most Similar Paragraphs:")
        for match in result['top_matches']:
            print(f"\nPage {match['page_number']}, Paragraph {match['paragraph_index']}:")
            print(f"Similarity Score: {match['similarity_score']:.4f}")
            print(f"Text: {match['text'][:200]}{'...' if len(match['text']) > 200 else ''}")

def main():
    # Configuration
    pdf_path = "contracts/sample_contract.pdf"
    output_dir = "paragraph_similarity_results"
    input_paragraph = """This agreement is made between the parties. It outlines the terms clearly. 
    All disputes shall be resolved through arbitration."""
    
    # Embedding configuration (default to Sentence Transformers)
    embedding_provider = "sentence_transformers"
    embedding_model = "all-MiniLM-L6-v2"
    azure_openai_config = {
        "api_key": "your-azure-openai-api-key",
        "azure_endpoint": "https://your-resource-name.openai.azure.com/",
        "api_version": "2023-05-15"
    }
    
    # Initialize and run experiment
    experiment = ParagraphSimilarityExperiment(
        pdf_path=pdf_path,
        output_dir=output_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        azure_openai_config=azure_openai_config
    )
    result = experiment.run_experiment(input_paragraph, top_n=5)
    
    # Print summary
    experiment.print_summary(result)
    
    print(f"\nResults saved to: {os.path.join(output_dir, 'paragraph_similarity_results.json')}")

if __name__ == "__main__":
    main()
