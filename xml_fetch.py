import xml.etree.ElementTree as ET
import requests
from io import BytesIO

def fetch_ecfr_xml(url):
    """
    Fetches the ECFR XML file from a given URL.
    :param url: URL of the ECFR XML file.
    :return: ElementTree root of the fetched XML.
    """
    try:
        print(f"Fetching XML file from: {url}")
        response = requests.get(url)
        
        # Check for successful request
        if response.status_code != 200:
            print(f"Failed to fetch XML file. HTTP Status Code: {response.status_code}")
            return None
        
        print("XML file fetched successfully.")
        
        # Parse the XML content
        xml_content = BytesIO(response.content)  # Create a file-like object from content
        tree = ET.parse(xml_content)
        return tree.getroot()
    
    except Exception as e:
        print(f"An error occurred while fetching the XML file: {e}")
        return None

def parse_ecfr_xml_from_url(url):
    """
    Fetches and parses an ECFR XML file from a URL.
    :param url: URL of the ECFR XML file.
    """
    # Fetch the root element from the XML file
    root = fetch_ecfr_xml(url)
    if root is None:
        return

    # Check for root element
    if root.tag != "ECFR":
        print("Not a valid ECFR XML file.")
        return

    print("\nParsing ECFR XML file...\n")

    # Loop through PARTs and SECTIONS
    for part in root.findall(".//PART"):
        part_number = part.findtext("PARTNO")
        part_heading = part.findtext("PARTNAME")
        print(f"Part {part_number}: {part_heading}")

        for section in part.findall(".//SECTION"):
            section_number = section.findtext("SECTNO", default="N/A")
            section_title = section.findtext("SUBJECT", default="No Title")

            print(f"\n  Section {section_number}: {section_title}")

            # Extract paragraphs (<P>) under this section
            for paragraph in section.findall(".//P"):
                print(f"    - {paragraph.text.strip()}")

if __name__ == "__main__":
    # URL to the ECFR XML file
    xml_url = "https://www.govinfo.gov/bulkdata/ECFR/title-1/2024-01-01/ECFR-title1.xml"  # Example URL
    parse_ecfr_xml_from_url(xml_url)
