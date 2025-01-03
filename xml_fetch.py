import requests
import xml.etree.ElementTree as ET
from io import BytesIO


# https://drive.google.com/file/d/16SGjp8l9epI3-5pySaz7s8-ESaFIAKx2/view?usp=sharing
def fetch_ecfr_data(base_url, date, title, **kwargs):
    """
    Fetches the eCFR XML data for a specific title, date, and optional query parameters.
    """
    endpoint = f"{base_url}/api/versioner/v1/full/{date}/title-{title}.xml"
    params = {key: value for key, value in kwargs.items() if value is not None}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/xml"
    }

    try:
        print(f"Fetching data from: {endpoint} with parameters {params}")
        response = requests.get(endpoint, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        print("Successfully fetched XML data.")
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the data: {e}")
        return None

def write_text_file(file_path, content):
    """
    Writes the parsed content into a text file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Successfully wrote content to {file_path}")

def extract_text_with_nested_tags(element):
    """
    Extracts text from an element, including text from child tags.
    """
    return ''.join(element.itertext()).strip()

def parse_and_indent_content(root, indent=0):
    """
    Recursively parses the XML content and returns a formatted string with indentation.
    """
    output = []
    spaces = '    ' * indent  # Each level increases indentation by 4 spaces
    
    # Check for HEAD (titles or headings)
    head = root.find("HEAD")
    if head is not None and head.text:
        output.append(f"{spaces}{head.text.strip()}\n")
    
    # Process paragraphs (<P>) inside the current tag
    for paragraph in root.findall("P"):
        paragraph_text = extract_text_with_nested_tags(paragraph)
        output.append(f"{spaces}  - {paragraph_text}\n")
    
    # Recursively process child DIV tags
    for child in root.findall("./DIV6") + root.findall("./DIV8"):
        output.extend(parse_and_indent_content(child, indent=indent+1))
    
    return output

def process_ecfr_to_text(xml_content, file_path):
    """
    Processes the XML content and saves it as a formatted text file.
    """
    try:
        tree = ET.parse(xml_content)
        root = tree.getroot()

        # Initialize output list
        output = ["eCFR Document:\n\n"]
        
        # Check root level (DIV5) and parse
        if root.tag == "DIV5":
            output.extend(parse_and_indent_content(root, indent=0))
        else:
            print("Root element is not DIV5. Cannot parse content.")
            return
        
        # Write to text file
        write_text_file(file_path, "".join(output))
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")

if __name__ == "__main__":
    BASE_URL = "https://www.ecfr.gov"

    # Mandatory parameters
    date = "2024-12-13"
    title = "12"

    # Optional parameters
    optional_params = {
        "chapter": "2",
        "subchapter": "A",
        "part": "211"
    }

    # Fetch eCFR XML data
    xml_content = fetch_ecfr_data(BASE_URL, date, title, **optional_params)
    if xml_content:
        # Process and save to text file
        output_file_path = "ecfr_output.txt"
        process_ecfr_to_text(xml_content, output_file_path)
