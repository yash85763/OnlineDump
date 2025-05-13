Here's a function that finds sentences with POS tag patterns similar to a specified reference sentence, with a match threshold of 95%:

```python
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from typing import List, Dict, Any, Tuple

# Make sure to download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Import our PDFHandler class - assuming it's in a file called pdf_handler.py
from pdf_handler import PDFHandler


def calculate_pos_pattern_similarity(reference_tags: List[str], candidate_tags: List[str]) -> float:
    """
    Calculate the similarity between two POS tag sequences.
    
    Args:
        reference_tags: List of POS tags from the reference sentence
        candidate_tags: List of POS tags from the candidate sentence
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # If either list is empty, return 0
    if not reference_tags or not candidate_tags:
        return 0.0
    
    # If lengths differ dramatically, similarity will be low
    length_ratio = min(len(reference_tags), len(candidate_tags)) / max(len(reference_tags), len(candidate_tags))
    
    # If length ratio is too small, return it as the similarity
    if length_ratio < 0.8:  # Length differs by more than 20%
        return length_ratio
    
    # Count matching tags
    matches = 0
    min_length = min(len(reference_tags), len(candidate_tags))
    
    for i in range(min_length):
        if reference_tags[i] == candidate_tags[i]:
            matches += 1
    
    # Calculate similarity as proportion of matching tags
    similarity = matches / max(len(reference_tags), len(candidate_tags))
    
    return similarity


def find_similar_pos_pattern_sentences(all_sentences: List[str], 
                                      reference_sentence: str, 
                                      threshold: float = 0.95) -> List[Dict[str, Any]]:
    """
    Find sentences with POS tag patterns similar to the reference sentence.
    
    Args:
        all_sentences: List of all sentences to check
        reference_sentence: The sentence to use as a reference for the POS pattern
        threshold: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        List of dictionaries containing similar sentences with their similarity scores
    """
    # Tokenize and tag the reference sentence
    reference_words = nltk.word_tokenize(reference_sentence)
    reference_pos_tags = nltk.pos_tag(reference_words)
    reference_tags = [tag for _, tag in reference_pos_tags]
    
    similar_sentences = []
    
    # Check each sentence for similarity
    for i, sentence in enumerate(all_sentences):
        # Tokenize and tag the candidate sentence
        candidate_words = nltk.word_tokenize(sentence)
        candidate_pos_tags = nltk.pos_tag(candidate_words)
        candidate_tags = [tag for _, tag in candidate_pos_tags]
        
        # Calculate similarity
        similarity = calculate_pos_pattern_similarity(reference_tags, candidate_tags)
        
        # If similarity is above threshold, add to results
        if similarity >= threshold:
            similar_sentences.append({
                "sentence_id": i,
                "text": sentence,
                "similarity": similarity,
                "pos_tags": candidate_pos_tags
            })
    
    # Sort by similarity (highest first)
    similar_sentences.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similar_sentences


def process_pdf_to_sentences(pdf_path: str, output_path: str, reference_sentence: str = None) -> None:
    """
    Process a PDF file, extract text, split into sentences, and save as JSON.
    Also performs POS tagging on the first 10 sentences and finds sentences
    with similar POS patterns to the reference sentence.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path where to save the JSON output
        reference_sentence: Optional reference sentence to find similar patterns
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize the PDF handler
    handler = PDFHandler()
    
    # Process the PDF file
    result = handler.process_pdf(pdf_path, generate_embeddings=False)
    
    if not result["parsable"]:
        print(f"Error processing PDF: {result['error']}")
        return
    
    # Extract all paragraphs from all pages
    all_paragraphs = []
    for page in result["pages"]:
        all_paragraphs.extend(page["paragraphs"])
    
    print(f"Extracted {len(all_paragraphs)} paragraphs")
    
    # Split paragraphs into sentences
    all_sentences = []
    for paragraph in all_paragraphs:
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(paragraph)
        all_sentences.extend(sentences)
    
    print(f"Split into {len(all_sentences)} sentences")
    
    # Perform POS tagging on the first 10 sentences (or all if fewer)
    num_sentences_to_tag = min(10, len(all_sentences))
    tagged_sentences = []
    
    for i in range(num_sentences_to_tag):
        sentence = all_sentences[i]
        # Tokenize words and tag parts of speech
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        
        tagged_sentences.append({
            "sentence_id": i,
            "text": sentence,
            "pos_tags": pos_tags
        })
    
    print(f"Performed POS tagging on {num_sentences_to_tag} sentences")
    
    # Create output data structure
    output_data = {
        "document": {
            "filename": os.path.basename(pdf_path),
            "total_sentences": len(all_sentences),
            "sentences": [{"sentence_id": i, "text": s} for i, s in enumerate(all_sentences)]
        },
        "pos_tagged_sentences": tagged_sentences
    }
    
    # If a reference sentence is provided, find similar pattern sentences
    if reference_sentence:
        similar_sentences = find_similar_pos_pattern_sentences(
            all_sentences, reference_sentence, threshold=0.95)
        
        print(f"Found {len(similar_sentences)} sentences with similar POS patterns (≥95% match)")
        output_data["similar_pattern_sentences"] = similar_sentences
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print POS tagging results for the first 10 sentences
    print("\nPOS Tagging Results for First 10 Sentences:")
    for sentence in tagged_sentences:
        print(f"\nSentence {sentence['sentence_id'] + 1}: {sentence['text']}")
        print("POS Tags:")
        for word, tag in sentence['pos_tags']:
            print(f"  {word}: {tag}")
    
    # Print similar pattern sentences if found
    if reference_sentence and 'similar_pattern_sentences' in output_data:
        print("\nSentences with Similar POS Patterns:")
        print(f"Reference: \"{reference_sentence}\"")
        
        for i, sentence in enumerate(similar_sentences[:5]):  # Show top 5
            print(f"\nMatch {i+1} (Similarity: {sentence['similarity']:.2f}):")
            print(f"  \"{sentence['text']}\"")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PDF to JSON with sentence splitting and POS tagging")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path", default=None)
    parser.add_argument("--reference", "-r", help="Reference sentence to find similar POS patterns", default=None)
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        base_filename = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = f"{base_filename}_sentences.json"
    
    # Process the PDF
    process_pdf_to_sentences(args.pdf_path, args.output, args.reference)


if __name__ == "__main__":
    main()
```

### Explanation of the POS Pattern Similarity Function

The `find_similar_pos_pattern_sentences` function looks for sentences with POS tag patterns that match a reference sentence with at least 95% similarity. Here's how it works:

1. **POS Pattern Extraction**:
   - It extracts the POS tags from the reference sentence
   - It extracts the POS tags from each candidate sentence

2. **Similarity Calculation**:
   - The `calculate_pos_pattern_similarity` function calculates a similarity score between 0 and 1
   - It first checks for dramatic length differences
   - Then it counts matching tags at corresponding positions
   - The similarity is the proportion of matching tags to the length of the longer sequence

3. **Threshold Filtering**:
   - Only sentences with a similarity score ≥ 0.95 (95%) are included in the results
   - Results are sorted by similarity score (highest first)

### Usage Instructions

You can use this script with a reference sentence like this:

```bash
python pdf_to_sentences.py path/to/document.pdf --reference "This is the reference sentence that establishes the pattern."
```

### Output

The output JSON will include:

1. All sentences from the document
2. POS tags for the first 10 sentences
3. Sentences with similar POS patterns to the reference sentence (≥95% match)

For example:

```json
{
  "document": {
    "filename": "document.pdf",
    "total_sentences": 150,
    "sentences": [
      {"sentence_id": 0, "text": "This is the first sentence."},
      {"sentence_id": 1, "text": "This is the second sentence."},
      // ... all sentences
    ]
  },
  "pos_tagged_sentences": [
    // ... first 10 sentences with POS tags
  ],
  "similar_pattern_sentences": [
    {
      "sentence_id": 42,
      "text": "That was the initial attempt.",
      "similarity": 0.98,
      "pos_tags": [
        ["That", "DT"],
        ["was", "VBD"],
        ["the", "DT"],
        ["initial", "JJ"],
        ["attempt", "NN"],
        [".", "."]
      ]
    },
    // ... other similar sentences
  ]
}
```

### Why This Approach Works Well

1. **Position-Sensitive Matching**: It considers both the tags and their positions, which captures the syntactic structure
2. **Length Penalty**: Sentences with very different lengths automatically get lower similarity scores
3. **High Threshold**: The 95% threshold ensures only very close matches are included
4. **Flexible Usage**: You can easily adjust the threshold or use any reference sentence

This implementation is ideal for finding sentences with nearly identical grammatical structures while allowing for minor variations in the actual words used.​​​​​​​​​​​​​​​​