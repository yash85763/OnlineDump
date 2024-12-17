import xml.etree.ElementTree as ET
import requests
from io import StringIO

def fetch_and_parse_inline_xml(url):
    """
    Fetches and parses inline XML content from a URL.
    :param url: URL where the XML content is displayed.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }

    try:
        print(f"Fetching XML content from: {url}")
        response = requests.get(url, headers=headers, timeout=60)
        
        # Check for successful response
        if response.status_code == 200:
            print("XML content fetched successfully.\n")
            xml_content = response.text  # Read the raw XML content as text
            
            # Parse XML content using StringIO
            tree = ET.parse(StringIO(xml_content))
            root = tree.getroot()
            return root
        else:
            print(f"Failed to fetch XML. HTTP Status Code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the XML: {e}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing XML content: {e}")
        return None

def parse_ecfr_inline_xml(url):
    """
    Fetches inline XML content from a URL and parses it.
    """
    root = fetch_and_parse_inline_xml(url)
    if root is None:
        return

    print("Parsing ECFR XML content...\n")
    for part in root.findall(".//PART"):
        part_number = part.findtext("PARTNO", default="N/A")
        part_heading = part.findtext("PARTNAME", default="No Title")
        print(f"Part {part_number}: {part_heading}")

        for section in part.findall(".//SECTION"):
            section_number = section.findtext("SECTNO", default="N/A")
            section_title = section.findtext("SUBJECT", default="No Title")
            print(f"  Section {section_number}: {section_title}")

            for paragraph in section.findall(".//P"):
                print(f"    - {paragraph.text.strip()}")

if __name__ == "__main__":
    # Example URL where XML content is displayed inline
    xml_url = "https://www.govinfo.gov/bulkdata/ECFR/title-1/2024-01-01/ECFR-title1.xml"
    parse_ecfr_inline_xml(xml_url)
