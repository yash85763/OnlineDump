"""
Enhanced PDF Parsability Checker with Multiple Criteria

This module provides comprehensive parsability assessment using:
1. Special character ratio analysis
2. Spelling/orthographic error detection (hunspell)
3. Language coherence detection (langdetect/CLD3)
4. LLM-based quality assessment
"""

import os
import re
import string
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import json

# Language detection libraries
try:
    from langdetect import detect, detect_langs, LangDetectException
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Install with 'pip install langdetect'")

# Alternative: polyglot for language detection
try:
    from polyglot.detect import Detector
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    print("Warning: polyglot not available. Install with 'pip install polyglot'")

# Hunspell for spell checking
try:
    import hunspell
    HUNSPELL_AVAILABLE = True
except ImportError:
    HUNSPELL_AVAILABLE = False
    print("Warning: hunspell not available. Install with 'pip install cyhunspell'")

# OpenAI for LLM-based assessment
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Install with 'pip install openai'")


class EnhancedParsabilityChecker:
    """Advanced parsability checker with multiple assessment criteria"""
    
    def __init__(self, 
                 min_quality_ratio: float = 0.5,
                 max_special_char_ratio: float = 0.3,
                 max_typo_ratio: float = 0.15,
                 min_language_confidence: float = 0.7,
                 expected_language: str = "en",
                 openai_api_key: str = None,
                 hunspell_dict_path: str = None):
        """
        Initialize the enhanced parsability checker.
        
        Args:
            min_quality_ratio: Minimum ratio of alphanumeric chars to total chars
            max_special_char_ratio: Maximum allowed ratio of special characters
            max_typo_ratio: Maximum allowed ratio of spelling errors
            min_language_confidence: Minimum confidence for language detection
            expected_language: Expected language code (e.g., 'en', 'es', 'fr')
            openai_api_key: API key for OpenAI (optional)
            hunspell_dict_path: Path to hunspell dictionary files (optional)
        """
        self.min_quality_ratio = min_quality_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.max_typo_ratio = max_typo_ratio
        self.min_language_confidence = min_language_confidence
        self.expected_language = expected_language
        
        # Initialize spell checker
        self.spell_checker = None
        if HUNSPELL_AVAILABLE:
            try:
                if hunspell_dict_path:
                    self.spell_checker = hunspell.HunSpell(
                        f"{hunspell_dict_path}/{expected_language}.dic",
                        f"{hunspell_dict_path}/{expected_language}.aff"
                    )
                else:
                    # Try to use system dictionaries
                    self.spell_checker = hunspell.HunSpell(
                        f"/usr/share/hunspell/{expected_language}.dic",
                        f"/usr/share/hunspell/{expected_language}.aff"
                    )
            except Exception as e:
                print(f"Warning: Could not initialize hunspell: {e}")
                self.spell_checker = None
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE and openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.openai_client = openai
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
    
    def assess_parsability(self, text: str, sample_size: int = 2000) -> Dict[str, Any]:
        """
        Comprehensive parsability assessment using multiple criteria.
        
        Args:
            text: Extracted text from PDF
            sample_size: Size of text sample for analysis (for performance)
            
        Returns:
            Dictionary with detailed assessment results
        """
        if not text or not text.strip():
            return {
                "is_parsable": False,
                "confidence": 0.0,
                "reason": "No text extracted",
                "details": {}
            }
        
        # Use a sample for performance on large texts
        text_sample = text[:sample_size] if len(text) > sample_size else text
        
        # Run all assessment criteria
        char_ratio_result = self._assess_character_ratio(text_sample)
        typo_result = self._assess_spelling_errors(text_sample)
        language_result = self._assess_language_coherence(text_sample)
        llm_result = self._assess_with_llm(text_sample) if self.openai_client else None
        
        # Combine results for final decision
        final_assessment = self._combine_assessments(
            char_ratio_result, typo_result, language_result, llm_result
        )
        
        return final_assessment
    
    def _assess_character_ratio(self, text: str) -> Dict[str, Any]:
        """Assess text quality based on character ratios."""
        if not text:
            return {"passed": False, "ratio": 0.0, "details": "No text"}
        
        total_chars = len(text)
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        special_chars = sum(1 for c in text if c in string.punctuation)
        whitespace_chars = sum(1 for c in text if c.isspace())
        other_chars = total_chars - alphanumeric_chars - special_chars - whitespace_chars
        
        # Calculate ratios
        alphanumeric_ratio = alphanumeric_chars / total_chars
        special_char_ratio = special_chars / total_chars
        other_char_ratio = other_chars / total_chars
        
        # Check criteria
        quality_passed = alphanumeric_ratio >= self.min_quality_ratio
        special_passed = special_char_ratio <= self.max_special_char_ratio
        
        passed = quality_passed and special_passed
        
        return {
            "passed": passed,
            "alphanumeric_ratio": alphanumeric_ratio,
            "special_char_ratio": special_char_ratio,
            "other_char_ratio": other_char_ratio,
            "quality_threshold_met": quality_passed,
            "special_char_threshold_met": special_passed,
            "details": {
                "total_chars": total_chars,
                "alphanumeric_chars": alphanumeric_chars,
                "special_chars": special_chars,
                "whitespace_chars": whitespace_chars,
                "other_chars": other_chars
            }
        }
    
    def _assess_spelling_errors(self, text: str) -> Dict[str, Any]:
        """Assess text quality based on spelling errors."""
        if not text or not self.spell_checker:
            return {
                "passed": True,  # Pass by default if no spell checker
                "typo_ratio": 0.0,
                "details": "Spell checking not available"
            }
        
        # Extract words (remove punctuation and convert to lowercase)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return {
                "passed": False,
                "typo_ratio": 1.0,
                "details": "No valid words found"
            }
        
        # Check spelling for each word
        misspelled_words = []
        total_words = len(words)
        
        for word in words:
            if len(word) > 2 and not self.spell_checker.spell(word):
                misspelled_words.append(word)
        
        typo_ratio = len(misspelled_words) / total_words
        passed = typo_ratio <= self.max_typo_ratio
        
        # Get most common misspelled words for debugging
        common_errors = Counter(misspelled_words).most_common(10)
        
        return {
            "passed": passed,
            "typo_ratio": typo_ratio,
            "total_words": total_words,
            "misspelled_count": len(misspelled_words),
            "common_errors": common_errors,
            "details": f"Found {len(misspelled_words)} errors in {total_words} words"
        }
    
    def _assess_language_coherence(self, text: str) -> Dict[str, Any]:
        """Assess language coherence and consistency."""
        if not text:
            return {
                "passed": False,
                "confidence": 0.0,
                "detected_language": None,
                "details": "No text provided"
            }
        
        results = {}
        
        # Try langdetect first
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                lang_probs = detect_langs(text)
                
                # Find confidence for detected language
                confidence = 0.0
                for lang_prob in lang_probs:
                    if lang_prob.lang == detected_lang:
                        confidence = lang_prob.prob
                        break
                
                results["langdetect"] = {
                    "detected_language": detected_lang,
                    "confidence": confidence,
                    "all_probabilities": [(lp.lang, lp.prob) for lp in lang_probs]
                }
                
            except LangDetectException as e:
                results["langdetect"] = {
                    "error": str(e),
                    "detected_language": None,
                    "confidence": 0.0
                }
        
        # Try polyglot as backup
        if POLYGLOT_AVAILABLE:
            try:
                detector = Detector(text)
                results["polyglot"] = {
                    "detected_language": detector.language.code,
                    "confidence": detector.language.confidence,
                    "name": detector.language.name
                }
            except Exception as e:
                results["polyglot"] = {
                    "error": str(e),
                    "detected_language": None,
                    "confidence": 0.0
                }
        
        # Determine final result
        best_detection = None
        best_confidence = 0.0
        
        for method, result in results.items():
            if result.get("confidence", 0) > best_confidence:
                best_confidence = result["confidence"]
                best_detection = result.get("detected_language")
        
        # Check if detected language matches expected
        language_matches = best_detection == self.expected_language
        confidence_sufficient = best_confidence >= self.min_language_confidence
        
        passed = language_matches and confidence_sufficient
        
        return {
            "passed": passed,
            "detected_language": best_detection,
            "confidence": best_confidence,
            "expected_language": self.expected_language,
            "language_matches": language_matches,
            "confidence_sufficient": confidence_sufficient,
            "detection_results": results,
            "details": f"Detected {best_detection} with {best_confidence:.2f} confidence"
        }
    
    def _assess_with_llm(self, text: str, max_tokens: int = 1000) -> Optional[Dict[str, Any]]:
        """Use LLM to assess if PDF needs rescanning."""
        if not self.openai_client:
            return None
        
        # Truncate text for API efficiency
        sample_text = text[:max_tokens] if len(text) > max_tokens else text
        
        prompt = f"""
        Analyze the following text extracted from a PDF and determine if the PDF needs to be rescanned or re-OCR'd.

        Consider these factors:
        1. Are there many garbled characters or nonsensical character sequences?
        2. Is the text coherent and readable?
        3. Are there excessive formatting artifacts or OCR errors?
        4. Does the text appear to be properly extracted from a document?

        Text to analyze:
        "{sample_text}"

        Respond with a JSON object containing:
        {{
            "needs_rescan": boolean,
            "confidence": float (0-1),
            "reason": "brief explanation",
            "quality_score": float (0-1),
            "issues_found": ["list", "of", "specific", "issues"]
        }}
        """
        
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing OCR text quality."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                llm_assessment = json.loads(content)
                
                return {
                    "passed": not llm_assessment.get("needs_rescan", True),
                    "confidence": llm_assessment.get("confidence", 0.0),
                    "quality_score": llm_assessment.get("quality_score", 0.0),
                    "reason": llm_assessment.get("reason", ""),
                    "issues_found": llm_assessment.get("issues_found", []),
                    "raw_response": content
                }
                
            except json.JSONDecodeError:
                # Fallback: parse response manually
                needs_rescan = "true" in content.lower() and "needs_rescan" in content.lower()
                return {
                    "passed": not needs_rescan,
                    "confidence": 0.5,
                    "quality_score": 0.5,
                    "reason": "Could not parse structured response",
                    "issues_found": [],
                    "raw_response": content
                }
                
        except Exception as e:
            return {
                "passed": True,  # Default to passing if LLM fails
                "confidence": 0.0,
                "quality_score": 0.0,
                "reason": f"LLM assessment failed: {str(e)}",
                "issues_found": [],
                "error": str(e)
            }
    
    def _combine_assessments(self, char_ratio_result: Dict[str, Any], 
                           typo_result: Dict[str, Any],
                           language_result: Dict[str, Any],
                           llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine all assessment results into final decision."""
        
        # Collect individual results
        assessments = {
            "character_ratio": char_ratio_result,
            "spelling_errors": typo_result,
            "language_coherence": language_result
        }
        
        if llm_result:
            assessments["llm_assessment"] = llm_result
        
        # Calculate weighted score
        weights = {
            "character_ratio": 0.3,
            "spelling_errors": 0.3,
            "language_coherence": 0.2,
            "llm_assessment": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for assessment_name, result in assessments.items():
            if result and "passed" in result:
                weight = weights.get(assessment_name, 0.0)
                score = 1.0 if result["passed"] else 0.0
                
                # Use confidence if available
                if "confidence" in result and result["confidence"] > 0:
                    score *= result["confidence"]
                
                total_score += score * weight
                total_weight += weight
        
        # Calculate final confidence
        final_confidence = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if parsable (require majority of tests to pass)
        passed_tests = sum(1 for result in assessments.values() 
                          if result and result.get("passed", False))
        total_tests = len([r for r in assessments.values() if r is not None])
        
        is_parsable = passed_tests >= (total_tests / 2) and final_confidence > 0.5
        
        # Generate summary reason
        failed_tests = []
        for name, result in assessments.items():
            if result and not result.get("passed", True):
                failed_tests.append(name.replace("_", " "))
        
        if failed_tests:
            reason = f"Failed tests: {', '.join(failed_tests)}"
        else:
            reason = "All quality checks passed"
        
        return {
            "is_parsable": is_parsable,
            "confidence": final_confidence,
            "reason": reason,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "details": assessments,
            "recommendation": "Use as-is" if is_parsable else "Recommend rescanning/re-OCR"
        }


# Integration with existing PDF handler
def update_pdf_handler_parsability_check():
    """
    Updated extract_pdf_content_from_bytes method for EnhancedPDFHandler
    """
    def extract_pdf_content_from_bytes(self, pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Extract content from PDF bytes and check parsability using enhanced criteria.
        """
        # ... existing PDF extraction code ...
        
        # After extracting total_text, use enhanced parsability checker
        if not total_text.strip():
            return pages_data, False, "No text extracted from PDF. The PDF might need OCR processing."
        
        # Initialize enhanced parsability checker
        parsability_checker = EnhancedParsabilityChecker(
            min_quality_ratio=self.min_quality_ratio,
            expected_language="en",  # Configure as needed
            openai_api_key=os.getenv('OPENAI_API_KEY')  # Optional
        )
        
        # Run comprehensive assessment
        assessment = parsability_checker.assess_parsability(total_text)
        
        if not assessment["is_parsable"]:
            return pages_data, False, f"Quality assessment failed: {assessment['reason']}"
        
        quality_info = f"PDF is parsable (confidence: {assessment['confidence']:.2f}) - {assessment['reason']}"
        return pages_data, True, quality_info


# Example usage and testing
def test_parsability_checker():
    """Test the enhanced parsability checker with sample texts."""
    
    print("🧪 Testing Enhanced Parsability Checker...")
    print("=" * 60)
    
    # Initialize checker
    checker = EnhancedParsabilityChecker(
        expected_language="en",
        openai_api_key=os.getenv('OPENAI_API_KEY')  # Optional
    )
    
    # Test cases
    test_cases = [
        ("Good text", "This is a well-written document with proper spelling and grammar. It contains coherent sentences and follows standard formatting."),
        ("Poor OCR", "Th1s 1s @ p00r1y 0CR'd d0cum3nt w1th m@ny 3rr0rs @nd g@rbl3d ch@r@ct3rs th@t m@k3 1t h@rd t0 r3@d."),
        ("Too many special chars", "###@@@ !!!$$$ %%%^^^ &&&*** ())(( []]{} ||\\// <<<>>> ???~~~"),
        ("Foreign language", "Este es un documento en español que debería ser detectado correctamente por el sistema."),
        ("Mixed content", "This document has some good content but also has s0m3 0CR 3rr0rs scattered throughout the text.")
    ]
    
    for name, text in test_cases:
        print(f"\n📝 Testing: {name}")
        print("-" * 40)
        
        result = checker.assess_parsability(text)
        
        print(f"✅ Parsable: {result['is_parsable']}")
        print(f"✅ Confidence: {result['confidence']:.2f}")
        print(f"✅ Reason: {result['reason']}")
        print(f"✅ Recommendation: {result['recommendation']}")
        
        if result['details']:
            print("📊 Detailed Results:")
            for assessment_type, details in result['details'].items():
                if details:
                    passed = details.get('passed', 'N/A')
                    print(f"   - {assessment_type}: {passed}")


if __name__ == "__main__":
    test_parsability_checker()
