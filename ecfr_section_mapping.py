def build_ecfr_section_mapping(data):
    """
    Builds a mapping of section identifiers to their descriptions from eCFR JSON data
    
    Args:
        data (dict): The eCFR JSON data structure
        
    Returns:
        dict: Mapping of section identifiers to their descriptions
    """
    mapping = {}
    
    def traverse(node):
        # Check if this node is a section
        if node.get('type') == 'section':
            mapping[node['identifier']] = node['label_description']
        
        # If this node has children, recursively process them
        children = node.get('children', [])
        if children and isinstance(children, list):
            for child in children:
                traverse(child)
    
    # Start traversal from the root
    traverse(data)
    
    return mapping

# Example usage:
if __name__ == "__main__":
    # Sample eCFR data
    ecfr_data = {
        "type": "title",
        "label": "Title 3 - The President",
        "label_level": "Title 3",
        "label_description": "The President",
        "identifier": "3",
        "children": [
            {
                "type": "chapter",
                "label": "Chapter I - Executive Office of the President",
                "label_level": "Chapter I",
                "label_description": "Executive Office of the President",
                "identifier": "I",
                "reserved": False,
                "children": [
                    {
                        "type": "part",
                        "label": "Part 100 - Standards of Conduct",
                        "label_level": "Part 100",
                        "label_description": "Standards of Conduct",
                        "identifier": "100",
                        "reserved": False,
                        "children": [
                            {
                                "type": "section",
                                "label": "§ 100.1 Ethical conduct standards and financial disclosure regulations.",
                                "label_level": "§ 100.1",
                                "label_description": "Ethical conduct standards and financial disclosure regulations.",
                                "identifier": "100.1",
                                "reserved": False
                            }
                        ],
                        "section_range": "§ 100.1"
                    }
                ]
            }
        ]
    }
    
    # Build the mapping
    section_mapping = build_ecfr_section_mapping(ecfr_data)
    
    # Print the result
    print(section_mapping)
    # Output: {'100.1': 'Ethical conduct standards and financial disclosure regulations.'}
