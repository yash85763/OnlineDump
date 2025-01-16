# utils/file_handler.py

from typing import List, Dict

def write_regulation_to_file(data: List[Dict], filename: str) -> None:
    """
    Write the regulation data to a formatted text file.
    
    Args:
        data: List of dictionaries containing regulation data
        filename: Name of the output file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ELECTRONIC CODE OF FEDERAL REGULATIONS (eCFR) OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in data:
            f.write("\n" + "=" * 80 + "\n")
            
            if entry['subpart']:
                f.write(f"\nSUBPART {entry['subpart']}: {entry['subpart_title']}\n")
            
            f.write(f"\nSECTION {entry['section']}: {entry['section_title']}\n")
            
            if entry['source']:
                f.write(f"\nSOURCE:\n{entry['source']}\n")
            
            f.write("\nCONTENT:\n")
            f.write(entry['section_text'])
            f.write("\n")
            
            if entry.get('footnotes'):
                f.write("\nFOOTNOTES:\n")
                for i, footnote in enumerate(entry['footnotes'], 1):
                    f.write(f"[{i}] {footnote}\n")
            
            f.write("\n" + "=" * 80 + "\n")