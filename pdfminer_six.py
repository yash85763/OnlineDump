import os
import re
import json
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams


class ContractParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text()
        self.structure = {}

    def _extract_text(self):
        """Extract text from PDF using pdfminer.six"""
        output = StringIO()
        with open(self.pdf_path, 'rb') as pdf_file:
            extract_text_to_fp(pdf_file, output, laparams=LAParams())
        return output.getvalue()

    def parse(self):
        """Parse the contract text and create hierarchical structure"""
        # Split text into lines and clean
        lines = [line.strip() for line in self.text.split('\n') if line.strip()]
        
        # Initialize structure
        structure = {
            "title": "",
            "sections": []
        }
        
        # Try to identify document title (usually at the beginning)
        if lines and not re.match(r'^(article|section|\d+\.)', lines[0].lower()):
            structure["title"] = lines[0]
            lines = lines[1:]
        
        current_section = None
        current_subsection = None
        current_clause = None
        
        # Regular expressions for identifying different levels
        section_pattern = re.compile(r'^(?:ARTICLE|Section)\s+(\d+|[IVXLCDM]+)[\.:\s]+(.+)$', re.IGNORECASE)
        subsection_pattern = re.compile(r'^(\d+\.\d+)[\.:\s]+(.+)$')
        clause_pattern = re.compile(r'^(\([a-z]\))[\.:\s]*(.+)$')
        
        content_buffer = []
        
        for line in lines:
            # Check if line is a section header
            section_match = section_pattern.match(line)
            if section_match:
                # If we have content in buffer, add to the appropriate level
                if content_buffer and current_section is not None:
                    if current_clause is not None:
                        current_clause["content"] += " " + " ".join(content_buffer)
                    elif current_subsection is not None:
                        current_subsection["content"] += " " + " ".join(content_buffer)
                    else:
                        current_section["content"] += " " + " ".join(content_buffer)
                content_buffer = []
                
                # Create new section
                current_section = {
                    "id": section_match.group(1),
                    "title": section_match.group(2),
                    "content": "",
                    "subsections": []
                }
                structure["sections"].append(current_section)
                current_subsection = None
                current_clause = None
                continue
            
            # Check if line is a subsection header
            subsection_match = subsection_pattern.match(line)
            if subsection_match and current_section is not None:
                # If we have content in buffer, add to the appropriate level
                if content_buffer:
                    if current_clause is not None:
                        current_clause["content"] += " " + " ".join(content_buffer)
                    elif current_subsection is not None:
                        current_subsection["content"] += " " + " ".join(content_buffer)
                    else:
                        current_section["content"] += " " + " ".join(content_buffer)
                content_buffer = []
                
                # Create new subsection
                current_subsection = {
                    "id": subsection_match.group(1),
                    "title": subsection_match.group(2),
                    "content": "",
                    "clauses": []
                }
                current_section["subsections"].append(current_subsection)
                current_clause = None
                continue
            
            # Check if line is a clause
            clause_match = clause_pattern.match(line)
            if clause_match and current_subsection is not None:
                # If we have content in buffer, add to the appropriate level
                if content_buffer:
                    if current_clause is not None:
                        current_clause["content"] += " " + " ".join(content_buffer)
                    elif current_subsection is not None:
                        current_subsection["content"] += " " + " ".join(content_buffer)
                    else:
                        current_section["content"] += " " + " ".join(content_buffer)
                content_buffer = []
                
                # Create new clause
                current_clause = {
                    "id": clause_match.group(1),
                    "content": clause_match.group(2)
                }
                current_subsection["clauses"].append(current_clause)
                continue
            
            # If none of the above, treat as content
            content_buffer.append(line)
        
        # Add any remaining content
        if content_buffer:
            if current_clause is not None:
                current_clause["content"] += " " + " ".join(content_buffer)
            elif current_subsection is not None:
                current_subsection["content"] += " " + " ".join(content_buffer)
            elif current_section is not None:
                current_section["content"] += " " + " ".join(content_buffer)
            else:
                # If we couldn't identify any structure, add as main content
                structure["content"] = " ".join(content_buffer)
        
        self.structure = structure
        return structure

    def save_json(self, output_path):
        """Save the structure to a JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.structure, f, indent=2)
        print(f"JSON saved to {output_path}")


def process_directory(directory_path, output_directory):
    """Process all PDF files in a directory"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            json_path = os.path.join(output_directory, filename.replace('.pdf', '.json'))
            
            try:
                parser = ContractParser(pdf_path)
                parser.parse()
                parser.save_json(json_path)
                print(f"Successfully parsed {filename}")
            except Exception as e:
                print(f"Error parsing {filename}: {str(e)}")


def main(input_path, output_path):
    """
    Process a single PDF file or a directory of PDF files
    
    Args:
        input_path (str): Path to a PDF file or directory containing PDFs
        output_path (str): Path to output JSON file or directory
    """
    if os.path.isdir(input_path):
        process_directory(input_path, output_path)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        parser = ContractParser(input_path)
        parser.parse()
        parser.save_json(output_path)
        
    return "Processing completed successfully"


if __name__ == "__main__":
    # Example usage - modify these paths for your needs
    pdf_path = "path/to/contracts"
    json_output_path = "path/to/output"
    
    main(pdf_path, json_output_path)