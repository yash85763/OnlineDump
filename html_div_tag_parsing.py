# from bs4 import BeautifulSoup
# import json
# import os
# import re

# def clean_filename(div_id):
#     """Convert div ID to filename (e.g., 'p-211.1(a)' -> '211_1_a.json')"""
#     clean_id = div_id.replace('p-', '')
#     clean_id = clean_id.replace('.', '_').replace('(', '_').replace(')', '')
#     return clean_id

# def get_combined_content(div):
#     """Get combined content from a div and all its nested divs"""
#     return div.get_text(strip=True)

# def process_html_file(html_file_path, output_dir = 'sectionss_data'):
    
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\nCreated directory: {output_dir}")
    
#     print(f"Reading HTML file: {html_file_path}")
#     with open(html_file_path, 'r', encoding='utf-8') as file:
#         soup = BeautifulSoup(file, 'lxml')
    
#     created_files = []
    
#     sections = soup.find_all('div', class_='section')
    
#     for section in sections:
#         # get section title and metadata from h4
#         h4_tag = section.find('h4')
#         section_title = h4_tag.get_text().strip() if h4_tag else ""
#         section_metadata = h4_tag.get('data-hierarchy-metadata') if h4_tag else ""
        
#         # get footnotes
#         footnotes_div = section.find('div', class_='box-published')
#         footnotes_text = ""
#         if footnotes_div:
#             footnotes_text = footnotes_div.get_text(strip=True)
#             # remove the "Footnotes -" prefix if it exists
#             footnotes_text = re.sub(r'^Footnotes\s*-\s*\d+\.\d+\s*', '', footnotes_text)
        
#         # find all second level divs
#         second_level_divs = section.find_all('div', id=lambda x: x and x.startswith('p-') and len(x.split('(')) == 2)
        
#         for div in second_level_divs:
#             div_id = div.get('id', '')
#             if not div_id:
#                 continue
                
#             # get div content
#             content = get_combined_content(div)
            
#             # append footnotes if they exist
#             if footnotes_text:
#                 content = f"{content}\nFootnote: {footnotes_text}"
            
#             # find all nested divs under this second level div
#             nested_divs = div.find_all('div', recursive=False)
            
#             # prepare the JSON data
#             div_data = {
#                 'id': div_id,
#                 'section_title': section_title,
#                 'section_metadata': section_metadata,
#                 'content': content,
#                 'html_content': str(div)
#             }
            
#             # if there are nested divs, include them
#             if nested_divs:
#                 div_data['nested_content'] = []
#                 for nested_div in nested_divs:
#                     nested_content = nested_div.get_text(strip=True)
#                     # also append footnotes to nested content
#                     if footnotes_text:
#                         nested_content = f"{nested_content}\nFootnote: {footnotes_text}"
                    
#                     nested_data = {
#                         'id': nested_div.get('id', ''),
#                         'content': nested_content,
#                         'html_content': str(nested_div)
#                     }
#                     div_data['nested_content'].append(nested_data)
            
#             # create filename from div_id
#             filename = f"{clean_filename(div_id)}.json"
#             filepath = os.path.join(output_dir, filename)
            
#             # save to JSON file
#             with open(filepath, 'w', encoding='utf-8') as f:
#                 json.dump(div_data, f, indent=4, ensure_ascii=False)
            
#             created_files.append(filepath)
#             print(f"Created: {filepath}")
    
#     print("\nAll files created successfully!")
#     print("\nCreated files:")
#     for file in created_files:
#         print(f"- {file}")

# # run the script
# if __name__ == "__main__":
#     try:
#         html_path = "/content/title-12.htm"
#         process_html_file(html_path)
#     except FileNotFoundError:
#         print(f"\nError: Could not find the file {html_path}")
#         print("Make sure you entered the correct path and the file exists.")
#     except Exception as e:
#         print(f"\nAn error occurred: {str(e)}")
#         print("\nMake sure you have the required packages installed:")




from bs4 import BeautifulSoup
import json
import os
import re

def clean_text(text):
    """Clean text content by removing extra spaces and newlines"""
    return ' '.join(text.split())

def create_node(id="", text="", children=None):
    """Create a standard node structure"""
    node = {
        "id": id,
        "text": text,
        "children": children if children is not None else []
    }
    return node

def combine_nested_content(div, footnotes_text=""):
    """Combine content of a div with its nested divs"""
    # Get main text from the div itself
    main_text = div.find('p', recursive=False).get_text(strip=True) if div.find('p', recursive=False) else ""
    
    # Get text from nested divs
    nested_divs = div.find_all('div', recursive=False)
    nested_texts = [nested_div.get_text(strip=True) for nested_div in nested_divs]
    
    # Combine all texts
    all_text = [main_text] + nested_texts
    combined_text = ' '.join(text for text in all_text if text)
    
    # Add footnotes if they exist
    if footnotes_text:
        combined_text = f"{combined_text}\nFootnote: {footnotes_text}"
        
    return clean_text(combined_text)

def process_html_file(html_file_path):
    print(f"Reading HTML file: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    
    # Create root node
    root_node = create_node(id="regulation", text="Regulation")
    
    # Process each section
    sections = soup.find_all('div', class_='section')
    
    for section in sections:
        section_id = section.get('id', '')
        
        # Get section title from h4
        h4_tag = section.find('h4')
        section_text = h4_tag.get_text(strip=True) if h4_tag else ""
        
        # Create section node
        section_node = create_node(
            id=section_id,
            text=section_text
        )
        
        # Get footnotes
        footnotes_div = section.find('div', class_='box-published')
        footnotes_text = ""
        if footnotes_div:
            footnotes_text = footnotes_div.get_text(strip=True)
            footnotes_text = re.sub(r'^Footnotes\s*-\s*\d+\.\d+\s*', '', footnotes_text)
        
        # Process second level divs (subsections)
        second_level_divs = section.find_all('div', id=lambda x: x and x.startswith('p-') and len(x.split('(')) == 2)
        
        for div in second_level_divs:
            div_id = div.get('id', '')
            
            # Combine main content with nested content
            combined_text = combine_nested_content(div, footnotes_text)
            
            subsection_node = create_node(
                id=div_id,
                text=combined_text
            )
            
            section_node["children"].append(subsection_node)
        
        root_node["children"].append(section_node)
    
    # Save the tree structure
    output_dir = 'section_update_data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'regulation_tree.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(root_node, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated nested JSON structure: {output_file}")
    
    # Print tree structure for verification
    def print_tree_dfs(node, level=0):
        print("  " * level + f"ID: {node['id']}")
        print("  " * level + f"Text preview: {node['text'][:100]}...")
        for child in node["children"]:
            print_tree_dfs(child, level + 1)
    
    print("\nTree structure preview:")
    print_tree_dfs(root_node)

# Run the script
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
        print("pip install beautifulsoup4")
        print("pip install lxml")

