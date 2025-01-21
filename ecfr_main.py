from ecfr_api_wrapper import ECFRAPIWrapper
from utils import write_regulation_to_file
import requests


def main():
    # Initialize with OpenAI API key
    openai_api_key = "your_api_key"  # Replace with actual API key
    api = ECFRAPIWrapper(openai_api_key=openai_api_key)
    
    try:
        # Define parameters
        date = "2024-12-12"
        title = "12"
        subtitle = "A"
        chapter = "II"
        subchapter = "A"
        part = "211"
        subpart = "A"
        section = "211.2"

        # Create filename
        filename = f"LLM_POWERED_ecfr_output_{date}_{title}_{subtitle}_{chapter}_{subchapter}_{part}_{subpart}_{section}.txt"
        
        # Get regulation data
        result = api.get_structured_regulation(
            date=date,
            title=title,
            subtitle=subtitle,
            chapter=chapter,
            subchapter=subchapter,
            part=part,
            subpart=subpart,
            section=section
        )
        
        # Write to file using utility function
        write_regulation_to_file(result, filename)
        print(f"Output has been written to {filename}")
        
    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()