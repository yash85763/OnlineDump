# utils/text_processor.py

from typing import Tuple, Optional, Dict
import re

def get_reference_context(text: str, reference_match: re.Match) -> Tuple[str, str, str]:
    """
    Extract the context around a reference.
    
    Args:
        text (str): Full text containing the reference
        reference_match (re.Match): Match object for the reference
        
    Returns:
        Tuple[str, str, str]: (before_context, reference_text, after_context)
    """
    start = reference_match.start()
    end = reference_match.end()
    
    context_before = text[max(0, start - 200):start].strip()
    reference_text = text[start:end]
    context_after = text[end:min(len(text), end + 200)].strip()
    
    return context_before, reference_text, context_after

def parse_reference(reference_text: str) -> Optional[Dict[str, str]]:
    """
    Parse a reference string like "§ 217.12 of this chapter" into components.
    
    Args:
        reference_text (str): The reference text to parse
        
    Returns:
        Optional[Dict[str, str]]: Dictionary with part and section numbers, or None if parsing fails
    """
    pattern = r'§\s*(\d+)\.(\d+)'
    match = re.search(pattern, reference_text)
    
    if match:
        return {
            'part': match.group(1),
            'section': match.group(1) + '.' + match.group(2)
        }
    return None