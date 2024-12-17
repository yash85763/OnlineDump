import requests
import xml.etree.ElementTree as ET
from io import BytesIO

def fetch_ecfr_data(base_url, date, title, **kwargs):
    """
    Fetches the eCFR XML data for a specific title, date, and optional query parameters.
    
    :param base_url: Base URL for the API.
    :param date: The date in 'YYYY-MM-DD' format (mandatory).
    :param title: The title number (mandatory).
    :param kwargs: Optional query parameters like chapter, subchapter, part, etc.
    :return: XML content as a file-like object.
    """
    endpoint = f"{base_url}/api/versioner/v1/full/{date}/title-{title}.xml"
    params = {key: value for key, value in kwargs.items() if value is not None}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/xml"
    }

    try:
        print(f"Fetching data from: {endpoint} with parameters {params}")
        response = requests.get(endpoint, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        print("Successfully fetched XML data.")
        return BytesIO(response.content)  # Convert to file-like object for parsing
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the data: {e}")
        return None

def parse_ecfr_xml(xml_content):
    """
    Parses the eCFR XML content.
    :param xml_content: XML content as a file-like object.
    """
    try:
        tree = ET.parse(xml_content)
        root = tree.getroot()
        print("XML content parsed successfully.")
        return root
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None

def display_ecfr_content(root):
    """
    Display content from the parsed XML tree.
    :param root: Root of the parsed XML tree.
    """
    print("\nDisplaying eCFR Content...\n")
    for part in root.findall('.//PART'):
        part_number = part.findtext('PARTNO', default="N/A")
        part_name = part.findtext('PARTNAME', default="No Title")
        print(f"Part {part_number}: {part_name}")
        
        for section in part.findall('.//SECTION'):
            section_number = section.findtext('SECTNO', default="N/A")
            section_title = section.findtext('SUBJECT', default="No Title")
            print(f"  Section {section_number}: {section_title}")

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

    # Fetch and parse XML data
    xml_content = fetch_ecfr_data(BASE_URL, date, title, **optional_params)
    if xml_content:
        root = parse_ecfr_xml(xml_content)
        if root:
            display_ecfr_content(root)
