from typing import Optional, List, Dict
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import re

from utils import LLMProcessor, get_reference_context, parse_reference, write_regulation_to_file

class ECFRAPIWrapper:
    def __init__(self, base_url: str = "https://www.ecfr.gov/api/versioner/v1/", openai_api_key: Optional[str] = None):
        """
        Initialize the eCFR API wrapper.
        
        Args:
            base_url (str): Base URL for the eCFR API
            openai_api_key (str, optional): OpenAI API key for LLM processing
        """
        self.base_url = base_url
        self.headers = {"accept": "application/xml"}
        self.llm_processor = LLMProcessor(openai_api_key) if openai_api_key else None

    def resolve_references(self, text: str, date: str, title: str, is_referenced_content: bool = False) -> str:
        """
        Find and resolve references in the text.
        
        Args:
            text (str): Original text content
            date (str): Date for the regulation
            title (str): Title number
            is_referenced_content (bool): Flag indicating if this is referenced content
            
        Returns:
            str: Text with resolved references
        """
        if is_referenced_content:
            return text

        reference_pattern = r'(?:as defined in |see )?(?:§\s*\d+\.\d+\s*(?:of this chapter)?)'
        references = list(re.finditer(reference_pattern, text))
        
        resolved_text = text
        for ref in reversed(references):
            reference_text = ref.group(0)
            parsed_ref = parse_reference(reference_text)
            
            if parsed_ref:
                context_before, ref_text, context_after = get_reference_context(text, ref)
                full_context = f"{context_before} {ref_text} {context_after}"
                
                referenced_content = self.get_referenced_content(
                    date=date,
                    title=title,
                    part=parsed_ref['part'],
                    section=parsed_ref['section']
                )
                
                if referenced_content:
                    if self.llm_processor:
                        relevant_content = self.llm_processor.process_reference(
                            original_context=full_context,
                            reference_text=reference_text,
                            referenced_content=referenced_content
                        )
                    else:
                        relevant_content = referenced_content
                    
                    replacement = f"{reference_text} [Relevant reference: {relevant_content}]"
                    start, end = ref.span()
                    resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
        
        return resolved_text

    def get_structured_regulation(self, **kwargs) -> List[Dict]:
        """
        Get regulation content in a structured format with resolved references.
        
        Args:
            **kwargs: Same arguments as get_regulation()
            
        Returns:
            List[Dict]: List of dictionaries containing structured regulation data
        """
        response = self.get_regulation(**kwargs)
        structured_data = self.parse_regulation_content(response['xml_content'])
        
        for entry in structured_data:
            entry['section_text'] = self.resolve_references(
                text=entry['section_text'],
                date=kwargs['date'],
                title=kwargs['title'],
                is_referenced_content=False
            )
        
        return structured_data
    
    
    def get_regulation(
        self,
        date: str,
        title: str,
        subtitle: Optional[str] = None,
        chapter: Optional[str] = None,
        subchapter: Optional[str] = None,
        part: Optional[str] = None,
        subpart: Optional[str] = None,
        section: Optional[str] = None,
        appendix: Optional[str] = None
    ) -> Dict:
        """
        Retrieve regulation content from the eCFR API.
        
        Args:
            date (str): Date in YYYY-MM-DD format (Required)
            title (str): Title number (Required)
            subtitle (str, optional): Uppercase letter: 'A', 'B', 'C'
            chapter (str, optional): Roman Numerals and digits 0-9, 'I', 'X', '1'
            subchapter (str, optional): Uppercase letters with optional underscore or dash
            part (str, optional): Uppercase letters with optional underscore or dash
            subpart (str, optional): Generally an uppercase letter
            section (str, optional): Generally a number followed by dot and number (e.g., '121.1')
            appendix (str, optional): Multiple formats: 'A', 'III', 'App. A'
            
        Returns:
            Dict: Response containing the regulation data and metadata
            
        Raises:
            ValueError: If the date format is invalid
            requests.exceptions.RequestException: If the API request fails
        """
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

        # Construct the endpoint URL
        endpoint = f"full/{date}/title-{title}.xml"
        
        # Add optional parameters
        params = {}
        if subtitle:
            params['subtitle'] = subtitle
        if chapter:
            params['chapter'] = chapter
        if subchapter:
            params['subchapter'] = subchapter
        if part:
            params['part'] = part
        if subpart:
            params['subpart'] = subpart
        if section:
            params['section'] = section
        if appendix:
            params['appendix'] = appendix

        try:
            # Make the API request
            url = urljoin(self.base_url, endpoint)
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            # Parse XML response
            xml_content = response.text
            root = ET.fromstring(xml_content)

            return {
                'status_code': response.status_code,
                'xml_content': xml_content,
                'parsed_xml': root,
                'headers': dict(response.headers)
            }

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"API request failed: {str(e)}")
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML response: {str(e)}")

    def parse_reference(self, reference_text: str) -> Optional[Dict[str, str]]:
        """
        Parse a reference string like "§ 217.12 of this chapter" into components.
        
        Args:
            reference_text (str): The reference text to parse
            
        Returns:
            Optional[Dict[str, str]]: Dictionary with part and section numbers, or None if parsing fails
        """
        import re
        
        # Match patterns like "§ 217.12" or "section 217.12"
        pattern = r'§\s*(\d+)\.(\d+)'
        match = re.search(pattern, reference_text)
        
        if match:
            return {
                'part': match.group(1),
                'section': match.group(1) + '.' + match.group(2)
            }
        return None

    def get_referenced_content(self, date: str, title: str, part: str, section: str) -> Optional[str]:
        """
        Fetch content for a referenced section.
        
        Args:
            date (str): Date for the regulation
            title (str): Title number
            part (str): Part number
            section (str): Section number
            
        Returns:
            Optional[str]: The referenced section's content, or None if not found
        """
        try:
            response = self.get_regulation(
                date=date,
                title=title,
                part=part
            )
            
            soup = BeautifulSoup(response['xml_content'], 'xml')
            referenced_section = soup.find('DIV8', attrs={'N': section})
            
            if referenced_section:
                # Get the section title if available
                section_title = referenced_section.find('HEAD').text.strip() if referenced_section.find('HEAD') else ''
                
                # Process paragraphs
                paragraphs = []
                for p in referenced_section.find_all(['P', 'FP']):
                    text = p.text.strip()
                    if text:
                        paragraphs.append(text)
                
                # Format the content with the title if available
                content = ' '.join(paragraphs)
                if section_title:
                    content = f"{section_title}: {content}"
                
                return content
            
            return None
            
        except Exception as e:
            print(f"Error fetching referenced content: {str(e)}")
            return None

    def parse_regulation_content(self, xml_content: str) -> List[Dict]:
        """
        Parse the XML content and return structured data, handling any level of the hierarchy.
        
        Args:
            xml_content (str): The XML content to parse
                
        Returns:
            List[Dict]: List of dictionaries containing structured regulation data
        """
        soup = BeautifulSoup(xml_content, 'xml')
        print("Starting XML parsing...")
        structured_data = []
        
        def extract_metadata(element) -> Dict:
            """Extract common metadata from an element."""
            metadata = {
                'number': element.attrs.get('N', ''),
                'title': element.find('HEAD').text.strip() if element.find('HEAD') else '',
            }
            
            # Extract AUTH/SOURCE info if present
            auth = element.find('AUTH')
            source = element.find('SOURCE')
            metadata['authority'] = auth.text.strip() if auth else ''
            metadata['source'] = source.text.strip() if source else ''
            
            return metadata

        def process_section(section_element, parent_metadata={}) -> Dict:
            """Process a section (DIV8) element and its contents."""
            metadata = extract_metadata(section_element)
            
            # Build standardized output structure
            section_data = {
                'subpart': parent_metadata.get('subpart_number', ''),
                'subpart_title': parent_metadata.get('subpart_title', ''),
                'section': metadata['number'],
                'section_title': metadata['title'],
                'source': metadata['source'],
            }
            
            # Process paragraphs
            paragraphs = []
            for p in section_element.find_all(['P', 'FP']):
                text = p.text.strip()
                if text:
                    paragraphs.append(text)
            section_data['section_text'] = '\n'.join(paragraphs)
            
            # Process footnotes
            footnotes = []
            for ftnt in section_element.find_all('FTNT'):
                ftnt_text = ftnt.text.strip()
                if ftnt_text:
                    footnotes.append(ftnt_text)
            section_data['footnotes'] = footnotes
            
            return section_data

        def process_subpart(subpart_element, parent_metadata={}) -> List[Dict]:
            """Process a subpart (DIV6) element and its sections."""
            subpart_data = []
            metadata = extract_metadata(subpart_element)
            
            subpart_metadata = {
                **parent_metadata,
                'subpart_number': metadata['number'],
                'subpart_title': metadata['title']
            }
            
            # Process sections within subpart
            for section in subpart_element.find_all('DIV8', recursive=False):
                section_data = process_section(section, subpart_metadata)
                subpart_data.append(section_data)
                
            return subpart_data

        def process_part(part_element) -> List[Dict]:
            """Process a part (DIV5) element and its children."""
            part_data = []
            metadata = extract_metadata(part_element)
            part_metadata = {'part_number': metadata['number']}
            
            # Process direct sections (if any)
            for section in part_element.find_all('DIV8', recursive=False):
                section_data = process_section(section, part_metadata)
                part_data.append(section_data)
            
            # Process subparts
            for subpart in part_element.find_all('DIV6', recursive=False):
                subpart_data = process_subpart(subpart, part_metadata)
                part_data.extend(subpart_data)
                
            return part_data

        def process_chapter(chapter_element) -> List[Dict]:
            """Process a chapter (DIV4) element and its parts."""
            chapter_data = []
            metadata = extract_metadata(chapter_element)
            chapter_metadata = {'chapter_number': metadata['number']}
            
            # Process parts
            for part in chapter_element.find_all('DIV5', recursive=False):
                part_data = process_part(part)
                chapter_data.extend(part_data)
                
            return chapter_data

        # Get the root element and determine its type
        root = soup.find()
        if not root:
            print("No content found")
            return []

        print(f"Processing root element: {root.name}")
        
        # Process based on root element type
        if root.name == 'DIV8':  # Section
            structured_data.append(process_section(root))
        elif root.name == 'DIV6':  # Subpart
            structured_data.extend(process_subpart(root))
        elif root.name == 'DIV5':  # Part
            structured_data.extend(process_part(root))
        elif root.name == 'DIV4':  # Chapter
            structured_data.extend(process_chapter(root))
        elif root.name == 'TITLE':  # Full title
            for chapter in root.find_all('DIV4', recursive=False):
                structured_data.extend(process_chapter(chapter))
        else:
            print(f"Unexpected root element type: {root.name}")
        
        print(f"Successfully parsed {len(structured_data)} content items")
        return structured_data

    def get_regulation_metadata(self, xml_content: str) -> Dict:
        """
        Extract metadata from the regulation XML.
        
        Args:
            xml_content (str): The XML content to parse
                
        Returns:
            Dict: Dictionary containing metadata about the regulation
        """
        soup = BeautifulSoup(xml_content, 'xml')
        metadata = {
            'structure_level': None,
            'title_number': None,
            'chapter_number': None,
            'part_number': None,
            'subpart_number': None,
            'section_number': None
        }
        
        # Determine the structure level and gather relevant metadata
        root = soup.find()
        if root:
            metadata['structure_level'] = root.name
            
            # Extract title number if available
            title_element = soup.find('TITLE')
            if title_element:
                metadata['title_number'] = title_element.get('N')
                
            # Extract other identifiers based on the root level
            if root.name == 'DIV4':  # Chapter
                metadata['chapter_number'] = root.get('N')
            elif root.name == 'DIV5':  # Part
                metadata['part_number'] = root.get('N')
            elif root.name == 'DIV6':  # Subpart
                metadata['subpart_number'] = root.get('N')
            elif root.name == 'DIV8':  # Section
                metadata['section_number'] = root.get('N')
        
        return metadata

    def get_regulation_text(self, **kwargs) -> str:
        """
        Get only the text content of the regulation.
        
        Args:
            **kwargs: Same arguments as get_regulation()
            
        Returns:
            str: Plain text content of the regulation
        """
        response = self.get_regulation(**kwargs)
        return response['xml_content']
