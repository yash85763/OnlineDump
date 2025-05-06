"""
PDF Paragraph Similarity Comparison

This script implements and compares multiple methods for finding similar paragraphs within a PDF file:
1. Cosine Similarity with TF-IDF
2. BM25 Similarity
3. Embedding-based Semantic Similarity (using Sentence Transformers)
4. Word-to-word Similarity (using Jaccard similarity)

The script reads a PDF file, extracts paragraphs, and compares them to a query paragraph
using different similarity measures, then presents the results.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from collections import Counter
import math
import time

# For PDF handling
import PyPDF2

# For text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# For embeddings
from sentence_transformers import SentenceTransformer

# For BM25
from rank_bm25 import BM25Okapi

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PDFSimilarityTester:
    """
    A class to test various similarity methods for finding paragraphs in a PDF
    that are similar to a query paragraph.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the tester with a PDF file path.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.paragraphs = []
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract paragraphs from the PDF
        self.extract_paragraphs()
        
        # Create TF-IDF and Count vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.count_vectorizer = CountVectorizer(stop_words='english')
        
        # Create document matrices
        if self.paragraphs:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.paragraphs)
            self.count_matrix = self.count_vectorizer.fit_transform(self.paragraphs)
            
            # Create BM25 model
            tokenized_paragraphs = [self.preprocess_text(para) for para in self.paragraphs]
            self.bm25 = BM25Okapi(tokenized_paragraphs)
            
            # Create embedding matrix
            self.paragraph_embeddings = self.embedding_model.encode(self.paragraphs)
    
    def extract_paragraphs(self):
        """Extract paragraphs from the PDF file."""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Split text into paragraphs (using double newlines as separator)
                raw_paragraphs = re.split(r'\n\s*\n', text)
                
                # Clean paragraphs (remove extra whitespace, etc.)
                self.paragraphs = [
                    re.sub(r'\s+', ' ', para).strip() 
                    for para in raw_paragraphs 
                    if para.strip()
                ]
                
                print(f"Extracted {len(self.paragraphs)} paragraphs from PDF.")
        except Exception as e:
            print(f"Error extracting paragraphs from PDF: {e}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, remove stopwords, and stem.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return tokens
    
    def get_cosine_similarity(self, query: str) -> List[Tuple[int, float]]:
        """
        Calculate cosine similarity between query and paragraphs using TF-IDF.
        
        Args:
            query: Query paragraph
            
        Returns:
            List of (paragraph_index, similarity_score) tuples
        """
        # Transform query using fitted vectorizer
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all paragraphs
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Create list of (paragraph_index, similarity_score) tuples
        results = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_bm25_similarity(self, query: str) -> List[Tuple[int, float]]:
        """
        Calculate BM25 similarity between query and paragraphs.
        
        Args:
            query: Query paragraph
            
        Returns:
            List of (paragraph_index, similarity_score) tuples
        """
        # Preprocess query
        tokenized_query = self.preprocess_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create list of (paragraph_index, similarity_score) tuples
        results = [(i, score) for i, score in enumerate(scores)]
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_embedding_similarity(self, query: str) -> List[Tuple[int, float]]:
        """
        Calculate semantic similarity between query and paragraphs using embeddings.
        
        Args:
            query: Query paragraph
            
        Returns:
            List of (paragraph_index, similarity_score) tuples
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity between query embedding and paragraph embeddings
        similarities = cosine_similarity(
            [query_embedding], 
            self.paragraph_embeddings
        )[0]
        
        # Create list of (paragraph_index, similarity_score) tuples
        results = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_jaccard_similarity(self, query: str) -> List[Tuple[int, float]]:
        """
        Calculate word-to-word similarity using Jaccard similarity.
        
        Args:
            query: Query paragraph
            
        Returns:
            List of (paragraph_index, similarity_score) tuples
        """
        # Preprocess query
        query_tokens = set(self.preprocess_text(query))
        
        results = []
        
        # Calculate Jaccard similarity for each paragraph
        for i, para in enumerate(self.paragraphs):
            para_tokens = set(self.preprocess_text(para))
            
            # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = len(query_tokens.intersection(para_tokens))
            union = len(query_tokens.union(para_tokens))
            
            similarity = intersection / union if union > 0 else 0
            results.append((i, similarity))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def find_similar_paragraphs(self, query: str, top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find similar paragraphs to the query using multiple methods.
        
        Args:
            query: Query paragraph
            top_n: Number of top similar paragraphs to return
            
        Returns:
            Dictionary with results for each method
        """
        methods = {
            'Cosine Similarity (TF-IDF)': self.get_cosine_similarity,
            'BM25': self.get_bm25_similarity,
            'Embedding Similarity': self.get_embedding_similarity,
            'Jaccard Similarity': self.get_jaccard_similarity
        }
        
        results = {}
        performance = {}
        
        for method_name, method_func in methods.items():
            start_time = time.time()
            method_results = method_func(query)
            end_time = time.time()
            
            # Get top N results
            top_results = method_results[:top_n]
            
            # Format results
            formatted_results = [
                {
                    'paragraph_id': idx,
                    'similarity_score': score,
                    'paragraph_text': self.paragraphs[idx][:200] + "..." if len(self.paragraphs[idx]) > 200 else self.paragraphs[idx]
                }
                for idx, score in top_results
            ]
            
            results[method_name] = formatted_results
            performance[method_name] = end_time - start_time
        
        return results, performance
    
    def compare_methods(self, query: str, top_n: int = 5):
        """
        Compare different similarity methods and display results.
        
        Args:
            query: Query paragraph
            top_n: Number of top similar paragraphs to display
        """
        print(f"\n{'='*80}\nQuery: {query[:100]}...\n{'='*80}\n")
        
        # Get results from all methods
        results, performance = self.find_similar_paragraphs(query, top_n)
        
        # Display results for each method
        for method, method_results in results.items():
            print(f"\n{'-'*80}\nMethod: {method} (Execution time: {performance[method]:.4f} seconds)\n{'-'*80}")
            
            for i, result in enumerate(method_results):
                print(f"{i+1}. Paragraph {result['paragraph_id']} (Score: {result['similarity_score']:.4f})")
                print(f"   {result['paragraph_text']}\n")
        
        # Create and display a comparison chart
        self.plot_comparison(results, query)
    
    def plot_comparison(self, results: Dict[str, List[Dict[str, Any]]], query: str):
        """
        Plot a comparison of different methods.
        
        Args:
            results: Results from different methods
            query: Query paragraph
        """
        # Extract methods and top paragraph IDs
        methods = list(results.keys())
        top_paragraph_ids = []
        
        for method in methods:
            top_paragraph_ids.append([r['paragraph_id'] for r in results[method]])
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Plot similarity scores for top paragraphs for each method
        for i, method in enumerate(methods):
            scores = [r['similarity_score'] for r in results[method]]
            plt.plot(range(1, len(scores) + 1), scores, marker='o', label=method)
        
        plt.title('Comparison of Similarity Methods')
        plt.xlabel('Rank')
        plt.ylabel('Similarity Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot
        plt.tight_layout()
        plt.savefig('similarity_comparison.png')
        print("Comparison chart saved as 'similarity_comparison.png'")


def main():
    """Main function to run the similarity tests."""
    # Example usage
    pdf_path = input("Enter the path to the PDF file: ")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return
    
    # Create tester
    tester = PDFSimilarityTester(pdf_path)
    
    # Get query paragraph
    query = input("\nEnter the query paragraph: ")
    
    # Get number of top results to display
    try:
        top_n = int(input("\nEnter the number of top similar paragraphs to display: "))
    except ValueError:
        top_n = 5
        print(f"Using default value: {top_n}")
    
    # Compare methods
    tester.compare_methods(query, top_n)


if __name__ == "__main__":
    main()