import requests
import xml.etree.ElementTree as ET
from io import BytesIO

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
        response = requests.get(endpoint, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        print("Successfully fetched XML data.")
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the data: {e}")
        return None

def parse_and_display_ecfr_content(xml_content):
    """
    Parses and displays the eCFR XML content dynamically.
    """
    try:
        tree = ET.parse(xml_content)
        root = tree.getroot()

        # Parse PARTS (DIV5)
        for part in root.findall(".//DIV5[@TYPE='PART']"):
            part_number = part.attrib.get("N", "N/A")
            part_title = part.findtext("HEAD", default="No Title")
            print(f"\nPart {part_number}: {part_title}")

            # Parse SUBPARTS (DIV6)
            for subpart in part.findall(".//DIV6[@TYPE='SUBPART']"):
                subpart_title = subpart.findtext("HEAD", default="No Subpart Title")
                print(f"  Subpart: {subpart_title}")

                # Parse SECTIONS (DIV8)
                for section in subpart.findall(".//DIV8[@TYPE='SECTION']"):
                    section_number = section.attrib.get("N", "N/A")
                    section_title = section.findtext("HEAD", default="No Section Title")
                    print(f"    Section {section_number}: {section_title}")

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

    # Fetch the XML
    xml_content = fetch_ecfr_data(BASE_URL, date, title, **optional_params)
    if xml_content:
        parse_and_display_ecfr_content(xml_content)
