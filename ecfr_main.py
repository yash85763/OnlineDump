"""
Enhanced PDF Handling Module for Contract Analysis with Language Detection

This module extends the PDFHandler class to support multiple language detection methods
for better parsability checks.
"""

import os
import json
import re
import io
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from sklearn.cluster import KMeans

# Import pdfminer.six components
try:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar, LTPage
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    raise ImportError("pdfminer.six is required. Install it with 'pip install pdfminer.six'")

# Check if spaCy is available
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Check if spaCy language detector is available
try:
    from spacy_langdetect import LanguageDetector
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Check if NLTK is available
try:
    import nltk
    from nltk.corpus import words as nltk_words
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class LanguageDetector:
    """Class for language detection with different backends (spaCy, NLTK, etc.)"""
    
    SPACY = "spacy"
    NLTK = "nltk"
    BASIC = "basic"
    
    def __init__(self, method="basic", min_english_ratio=0.5):
        """
        Initialize language detector with specified method.
        
        Args:
            method: Detection method to use ('spacy', 'nltk', or 'basic')
            min_english_ratio: Minimum ratio threshold for English content
        """
        self.method = method.lower()
        self.min_english_ratio = min_english_ratio
        self.nlp = None
        self.english_words = None
        
        # Initialize the chosen method
        if self.method == self.SPACY:
            self._init_spacy()
        elif self.method == self.NLTK:
            self._init_nltk()
        else:
            # Default to basic if specified method not available
            self.method = self.BASIC
    
    def _init_spacy(self):
        """Initialize spaCy language detection"""
        if not SPACY_AVAILABLE:
            print("Warning: spaCy is not installed. Falling back to basic detection.")
            self.method = self.BASIC
            return False
        
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add language detector component
            if LANGDETECT_AVAILABLE:
                if not Language.has_factory("language_detector"):
                    Language.factory("language_detector", func=lambda nlp, name: LanguageDetector())
                
                # Add language detector pipe if not already present
                if "language_detector" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("language_detector", last=True)
                
                return True
            else:
                print("Warning: spacy-langdetect is not installed. Falling back to NLTK.")
                self.method = self.NLTK
                return self._init_nltk()
                
        except OSError:
            print("Warning: en_core_web_sm model not found. Falling back to NLTK.")
            self.method = self.NLTK
            return self._init_nltk()
    
    def _init_nltk(self):
        """Initialize NLTK-based language detection"""
        if not NLTK_AVAILABLE:
            print("Warning: NLTK is not installed. Falling back to basic detection.")
            self.method = self.BASIC
            return False
        
        try:
            # Try to find words corpus
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                # Download if not found
                nltk.download('words', quiet=True)
            
            # Load English words set
            self.english_words = set(w.lower() for w in nltk_words.words())
            return len(self.english_words) > 0
        
        except Exception as e:
            print(f"Warning: Failed to initialize NLTK: {str(e)}. Falling back to basic detection.")
            self.method = self.BASIC
            return False
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """
        Detect language of the given text using the current method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language codes as keys and confidence scores as values
        """
        if not text:
            return {"en": 0.0}
        
        if self.method == self.SPACY:
            return self._detect_with_spacy(text)
        elif self.method == self.NLTK:
            return self._detect_with_nltk(text)
        else:
            return self._detect_basic(text)
    
    def _detect_with_spacy(self, text: str) -> Dict[str, float]:
        """Use spaCy with language detector to identify the language"""
        if not self.nlp:
            return {"en": 0.0}
        
        # Limit text length to improve performance
        sample_text = text[:10000]
        
        try:
            # Process text with spaCy
            doc = self.nlp(sample_text)
            
            # Get language scores
            if hasattr(doc._, "language_scores"):
                return doc._.language_scores
            else:
                # If language_detector pipe isn't working, fall back to NLTK
                return self._detect_with_nltk(text)
                
        except Exception as e:
            print(f"spaCy language detection error: {str(e)}")
            return self._detect_with_nltk(text)
    
    def _detect_with_nltk(self, text: str) -> Dict[str, float]:
        """Use NLTK's English word corpus to estimate language"""
        if not self.english_words:
            return {"en": 0.0}
        
        try:
            # Extract words (3+ letters to avoid most abbreviations)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            if not words:
                return {"en": 0.0}
            
            # Count English words
            english_count = sum(1 for word in words if word in self.english_words)
            english_ratio = english_count / len(words)
            
            return {"en": english_ratio}
            
        except Exception as e:
            print(f"NLTK language detection error: {str(e)}")
            return self._detect_basic(text)
    
    def _detect_basic(self, text: str) -> Dict[str, float]:
        """
        Basic heuristic for English text:
        - Count spaces (English has word separation)
        - Check for common English letters/patterns
        """
        if not text:
            return {"en": 0.0}
        
        try:
            # Simple heuristics for approximate detection
            # 1. Count frequency of common English letters (e, t, a, o, i, n)
            common_letters = "etaoin"
            letter_count = sum(text.lower().count(letter) for letter in common_letters)
            
            # 2. Count spaces (language with spaces between words)
            space_count = text.count(' ')
            
            # 3. Calculate rough ratio based on these factors
            total_chars = len(text)
            if total_chars == 0:
                return {"en": 0.0}
            
            letter_ratio = letter_count / total_chars
            space_ratio = space_count / total_chars
            
            # Combined score with higher weight for spaces
            combined_score = (letter_ratio * 0.4) + (space_ratio * 3.0)
            
            # Normalize to a 0-1 range (empirical thresholds)
            english_ratio = min(max(combined_score / 0.8, 0.0), 1.0)
            
            return {"en": english_ratio}
            
        except Exception as e:
            print(f"Basic language detection error: {str(e)}")
            return {"en": 0.5}  # Return neutral score
    
    def is_english(self, text: str) -> Tuple[bool, float]:
        """
        Check if the given text is mostly English.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (is_english, english_ratio)
        """
        language_scores = self.detect_language(text)
        english_ratio = language_scores.get("en", 0.0)
        
        return english_ratio >= self.min_english_ratio, english_ratio


class PDFHandler:
    """Main class for handling PDF extraction and processing with language detection."""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 paragraph_spacing_threshold: int = 10,
                 page_continuity_threshold: float = 0.1,
                 min_words_threshold: int = 5,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_provider: str = "sentence_transformers",
                 azure_openai_config: Dict[str, str] = None,
                 lang_detection_method: str = "basic",
                 min_english_ratio: float = 0.5):
        """
        Initialize the PDF handler with configurable thresholds.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric chars to total chars (default 0.5)
            paragraph_spacing_threshold: Max vertical spacing between text blocks to be considered
                                         part of the same paragraph, in points (default 10)
            page_continuity_threshold: Percentage of page height to check for paragraph 
                                       continuation across pages (default 0.1 or 10%)
            min_words_threshold: Minimum number of words for a paragraph to be considered standalone (default 5)
            embedding_model: Name of the embedding model to use
            embedding_provider: Which embedding provider to use, either "sentence_transformers" or "azure_openai"
            azure_openai_config: Configuration for Azure OpenAI, required if using "azure_openai" provider
            lang_detection_method: Method to use for language detection ('spacy', 'nltk', or 'basic')
            min_english_ratio: Minimum ratio of English content required for parsability
        """
        self.min_quality_ratio = min_quality_ratio
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.page_continuity_threshold = page_continuity_threshold
        self.min_words_threshold = min_words_threshold
        self.embedding_provider = embedding_provider
        self.azure_openai_config = azure_openai_config or {}
        self.lang_detection_method = lang_detection_method
        self.min_english_ratio = min_english_ratio
        
        # Set up pdfminer configuration
        self.laparams = LAParams(
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            detect_vertical=True,
            all_texts=True
        )
        
        # Initialize language detector with the specified method
        self.language_detector = LanguageDetector(
            method=lang_detection_method,
            min_english_ratio=min_english_ratio
        )
        
        # Initialize embedding model and other components
        self._init_embedding_model(embedding_model)
    
    def _init_embedding_model(self, embedding_model):
        """Initialize the embedding model based on the provider"""
        # Optional: Import libraries for embeddings
        try:
            # For sentence-transformers
            from sentence_transformers import SentenceTransformer
            import faiss
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False

        # For Azure OpenAI embeddings
        try:
            from openai import AzureOpenAI
            AZURE_OPENAI_AVAILABLE = True
        except ImportError:
            AZURE_OPENAI_AVAILABLE = False
            
        # Initialize the embedding model if available
        self.embedding_model = None
        self.azure_client = None
        
        if self.embedding_provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Initialized Sentence Transformers model: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load Sentence Transformers model: {e}")
        
        elif self.embedding_provider == "azure_openai" and AZURE_OPENAI_AVAILABLE:
            try:
                # Check if we have all the required config parameters
                required_params = ["api_key", "azure_endpoint", "api_version"]
                if not all(param in self.azure_openai_config for param in required_params):
                    missing = [p for p in required_params if p not in self.azure_openai_config]
                    raise ValueError(f"Missing required Azure OpenAI config parameters: {missing}")
                
                self.azure_client = AzureOpenAI(
                    api_key=self.azure_openai_config["api_key"],
                    api_version=self.azure_openai_config["api_version"],
                    azure_endpoint=self.azure_openai_config["azure_endpoint"]
                )
                self.embedding_model = embedding_model  # Store the deployment name
                print(f"Initialized Azure OpenAI client with deployment: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not initialize Azure OpenAI client: {e}")
    
    def extract_pdf_content(self, pdf_path: str, lang_detection_method: str = None) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Extract content from a PDF file using pdfminer.six and check parsability.
        Includes language detection for better quality assessment.
        
        Args:
            pdf_path: Path to the PDF file
            lang_detection_method: Optional override for language detection method
                                 (options: 'spacy', 'nltk', 'basic')
            
        Returns:
            Tuple of (pages_data, is_parsable, quality_info)
        """
        # If method override provided, create a new detector for this call only
        if lang_detection_method and lang_detection_method != self.lang_detection_method:
            language_detector = LanguageDetector(
                method=lang_detection_method,
                min_english_ratio=self.min_english_ratio
            )
        else:
            language_detector = self.language_detector
        
        # Initialize required pdfminer objects
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(resource_manager, laparams=self.laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        
        pages_data = []
        total_text = ""
        total_chars = 0
        alpha_chars = 0
        
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            
            # Check if document is empty or encrypted
            if not document.is_extractable:
                return [], False, "Document is encrypted or not extractable"
            
            # Extract content from each page
            for page_num, page in enumerate(PDFPage.create_pages(document)):
                interpreter.process_page(page)
                layout = device.get_result()
                
                # Extract textboxes
                text_boxes = []
                page_text = ""
                
                # Get page dimensions
                page_width = layout.width if hasattr(layout, 'width') else 0
                page_height = layout.height if hasattr(layout, 'height') else 0
                
                for element in layout:
                    if isinstance(element, LTTextBox):
                        box_text = element.get_text().strip()
                        if box_text:
                            # Create a text box entry with position and text
                            text_boxes.append({
                                'x0': element.x0,
                                'y0': element.y0,
                                'x1': element.x1,
                                'y1': element.y1,
                                'text': box_text
                            })
                            page_text += box_text + " "
                
                # Add page data to the results
                pages_data.append({
                    'page_num': page_num + 1,  # 1-based page numbering
                    'width': page_width,
                    'height': page_height,
                    'text_boxes': text_boxes,
                    'text': page_text.strip()
                })
                
                # Accumulate text for quality check
                total_text += page_text
                
        # If no text was extracted, the PDF might not be OCR'd or has issues
        if not total_text.strip():
            return pages_data, False, "No text extracted from PDF. The PDF might need OCR processing."
        
        # ---- Basic quality checks ----
        
        # 1. Count alphanumeric characters vs. total characters
        total_chars = len(total_text)
        alpha_chars = sum(1 for char in total_text if char.isalnum())
        
        # Calculate quality ratio
        if total_chars > 0:
            quality_ratio = alpha_chars / total_chars
        else:
            quality_ratio = 0
        
        # 2. Check if text length is reasonable for the number of pages
        if pages_data:
            avg_chars_per_page = total_chars / len(pages_data)
            if avg_chars_per_page < 100:  # Arbitrary threshold, adjust as needed
                return pages_data, False, f"Text extraction yielded too little content ({avg_chars_per_page:.1f} chars/page)"
        
        # 3. Check quality ratio against threshold
        if quality_ratio < self.min_quality_ratio:
            return pages_data, False, f"Low text quality (alphanumeric ratio: {quality_ratio:.2f})"
        
        # ---- Language detection quality check ----
        
        # Get a sample of the text for language detection (limit size for performance)
        sample_text = total_text[:10000]
        
        # Use the configured language detector to determine if content is English
        is_english, english_ratio = language_detector.is_english(sample_text)
        
        # Check if English content is sufficient
        if not is_english:
            return pages_data, False, f"Low English content detected ({english_ratio:.2f}). This may not be an English document."
        
        # Return success with quality metrics
        quality_info = f"PDF is parsable with quality ratio {quality_ratio:.2f} and English ratio {english_ratio:.2f}"
        return pages_data, True, quality_info
    
    # [Rest of the PDFHandler class methods remain unchanged]