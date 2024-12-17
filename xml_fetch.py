import xml.etree.ElementTree as ET
import requests
from xml.etree.ElementTree import iterparse

def fetch_and_stream_parse_ecfr_xml(url, chunk_size=1024):
    """
    Fetches a large ECFR XML file in chunks using streaming and parses it.
    :param url: URL of the ECFR XML file.
    :param chunk_size: Size of chunks to download in bytes.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }

    try:
        print(f"Fetching XML file in streaming mode from: {url}")
        
        # Start streaming the file
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses
            
            # Use iterparse to parse incrementally
            context = iterparse(response.raw, events=("start", "end"))
            context = iter(context)

            # Track PARTS and SECTIONS
            current_part_number = ""
            current_part_heading = ""

            for event, elem in context:
                if event == "start" and elem.tag == "PART":
                    # Part tag starts
                    current_part_number = elem.findtext("PARTNO")
                    current_part_heading = elem.findtext("PARTNAME")
                    print(f"\nPart {current_part_number}: {current_part_heading}")
                
                if event == "start" and elem.tag == "SECTION":
                    # Section starts
                    section_number = elem.findtext("SECTNO", default="N/A")
                    section_title = elem.findtext("SUBJECT", default="No Title")
                    print(f"  Section {section_number}: {section_title}")

                    # Print paragraphs under this section
                    for paragraph in elem.findall(".//P"):
                        print(f"    - {paragraph.text.strip()}")

                # Clear the element from memory to prevent memory buildup
                elem.clear()
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the XML file: {e}")
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")

if __name__ == "__main__":
    # Example ECFR XML URL
    xml_url = "https://www.govinfo.gov/bulkdata/ECFR/title-1/2024-01-01/ECFR-title1.xml"
    fetch_and_stream_parse_ecfr_xml(xml_url)
