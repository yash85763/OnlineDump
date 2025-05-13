Here's a script that uses our enhanced PDFHandler class to extract content from a PDF, split it into sentences, and then perform POS tagging on the first 10 sentences using NLTK:

```python
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from typing import List, Dict, Any

# Make sure to download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Import our PDFHandler class - assuming it's in a file called pdf_handler.py
from pdf_handler import PDFHandler


def process_pdf_to_sentences(pdf_path: str, output_path: str) -> None:
    """
    Process a PDF file, extract text, split into sentences, and save as JSON.
    Also performs POS tagging on the first 10 sentences.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path where to save the JSON output
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
    
    # Create JSON structure with all sentences and tagged sentences
    output_data = {
        "document": {
            "filename": os.path.basename(pdf_path),
            "total_sentences": len(all_sentences),
            "sentences": [{"sentence_id": i, "text": s} for i, s in enumerate(all_sentences)]
        },
        "pos_tagged_sentences": tagged_sentences
    }
    
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PDF to JSON with sentence splitting and POS tagging")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output JSON file path", 
                        default=None)
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        base_filename = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = f"{base_filename}_sentences.json"
    
    # Process the PDF
    process_pdf_to_sentences(args.pdf_path, args.output)


if __name__ == "__main__":
    main()
```

And here's a step-by-step explanation of how to use this script:

### Usage Instructions

1. **Save the Script**: Save the above code as `pdf_to_sentences.py`

2. **Install Dependencies**:
   ```
   pip install nltk
   ```

3. **Run the Script**:
   ```
   python pdf_to_sentences.py path/to/your/document.pdf --output results.json
   ```

### What the Script Does

1. Processes the PDF using our enhanced PDFHandler
2. Extracts all paragraphs from the document
3. Splits paragraphs into sentences using NLTK's sentence tokenizer
4. Performs part-of-speech (POS) tagging on the first 10 sentences
5. Saves all sentences and POS-tagged sentences to a JSON file

### JSON Output Structure

The resulting JSON file has this structure:

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
    {
      "sentence_id": 0,
      "text": "This is the first sentence.",
      "pos_tags": [
        ["This", "DT"],
        ["is", "VBZ"],
        ["the", "DT"],
        ["first", "JJ"],
        ["sentence", "NN"],
        [".", "."]
      ]
    },
    // ... up to 10 sentences with POS tags
  ]
}
```

### Understanding NLTK POS Tags

NLTK uses the Penn Treebank tag set. Some common tags are:

- `NN`: Noun, singular
- `NNS`: Noun, plural
- `VB`: Verb, base form
- `VBD`: Verb, past tense
- `JJ`: Adjective
- `RB`: Adverb
- `IN`: Preposition
- `DT`: Determiner

For a full list of tags, you can visit the [Penn Treebank Tag Set documentation](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

### Extending the Script

This script can be easily extended to:

1. Tag all sentences instead of just the first 10
2. Apply other NLP analyses like named entity recognition
3. Filter sentences based on specific criteria
4. Perform more advanced linguistic analyses

Just modify the `process_pdf_to_sentences` function to include the additional processing you need.​​​​​​​​​​​​​​​​