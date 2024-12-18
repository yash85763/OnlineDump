from bs4 import BeautifulSoup
import json
import os
import re

def clean_filename(div_id):
    """Convert div ID to filename (e.g., 'p-211.1(a)' -> '211_1_a.json')"""
    clean_id = div_id.replace('p-', '')
    clean_id = clean_id.replace('.', '_').replace('(', '_').replace(')', '')
    return clean_id

def get_combined_content(div):
    """Get combined content from a div and all its nested divs"""
    return div.get_text(strip=True)

def process_html_file(html_file_path, output_dir = 'sectionss_data'):
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated directory: {output_dir}")
    
    print(f"Reading HTML file: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    
    created_files = []
    
    sections = soup.find_all('div', class_='section')
    
    for section in sections:
        # get section title and metadata from h4
        h4_tag = section.find('h4')
        section_title = h4_tag.get_text().strip() if h4_tag else ""
        section_metadata = h4_tag.get('data-hierarchy-metadata') if h4_tag else ""
        
        # get footnotes
        footnotes_div = section.find('div', class_='box-published')
        footnotes_text = ""
        if footnotes_div:
            footnotes_text = footnotes_div.get_text(strip=True)
            # remove the "Footnotes -" prefix if it exists
            footnotes_text = re.sub(r'^Footnotes\s*-\s*\d+\.\d+\s*', '', footnotes_text)
        
        # find all second level divs
        second_level_divs = section.find_all('div', id=lambda x: x and x.startswith('p-') and len(x.split('(')) == 2)
        
        for div in second_level_divs:
            div_id = div.get('id', '')
            if not div_id:
                continue
                
            # get div content
            content = get_combined_content(div)
            
            # append footnotes if they exist
            if footnotes_text:
                content = f"{content}\nFootnote: {footnotes_text}"
            
            # find all nested divs under this second level div
            nested_divs = div.find_all('div', recursive=False)
            
            # prepare the JSON data
            div_data = {
                'id': div_id,
                'section_title': section_title,
                'section_metadata': section_metadata,
                'content': content,
                'html_content': str(div)
            }
            
            # if there are nested divs, include them
            if nested_divs:
                div_data['nested_content'] = []
                for nested_div in nested_divs:
                    nested_content = nested_div.get_text(strip=True)
                    # also append footnotes to nested content
                    if footnotes_text:
                        nested_content = f"{nested_content}\nFootnote: {footnotes_text}"
                    
                    nested_data = {
                        'id': nested_div.get('id', ''),
                        'content': nested_content,
                        'html_content': str(nested_div)
                    }
                    div_data['nested_content'].append(nested_data)
            
            # create filename from div_id
            filename = f"{clean_filename(div_id)}.json"
            filepath = os.path.join(output_dir, filename)
            
            # save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(div_data, f, indent=4, ensure_ascii=False)
            
            created_files.append(filepath)
            print(f"Created: {filepath}")
    
    print("\nAll files created successfully!")
    print("\nCreated files:")
    for file in created_files:
        print(f"- {file}")

# run the script
if __name__ == "__main__":
    try:
        html_path = "/content/title-12.htm"
        process_html_file(html_path)
    except FileNotFoundError:
        print(f"\nError: Could not find the file {html_path}")
        print("Make sure you entered the correct path and the file exists.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nMake sure you have the required packages installed:")
