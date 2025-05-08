You've identified an important edge case - PDFs that contain mixed layouts with single-column sections and double-column sections on the same page. Let me provide the necessary code updates to handle this situation.

Here are the key code changes needed:

### 1. Update the Layout Detection at Page Section Level

```python
def analyze_page_layout(self, text_boxes, page_width, page_height):
    """
    Analyze the layout structure within a single page,
    detecting sections with different column layouts.
    
    Args:
        text_boxes: List of text boxes with position information
        page_width: Width of the page
        page_height: Height of the page
        
    Returns:
        List of section dictionaries with layout information and y-boundaries
    """
    if not text_boxes or len(text_boxes) < 5:
        return [{"layout": "single_column", "y_start": 0, "y_end": page_height}]
    
    # Sort text boxes by y-position (top to bottom)
    text_boxes_sorted = sorted(text_boxes, key=lambda b: b['y0'])
    
    # Use a sliding window approach to analyze sections of the page
    sections = []
    window_size = min(8, max(3, len(text_boxes) // 4))  # Adaptive window size
    min_section_height = page_height * 0.1  # Minimum height for a section
    
    current_y = 0
    i = 0
    
    while i < len(text_boxes_sorted):
        # Get a window of text boxes
        end_idx = min(i + window_size, len(text_boxes_sorted))
        window_boxes = text_boxes_sorted[i:end_idx]
        
        # Find y-range of current window
        section_y_start = window_boxes[0]['y0']
        section_y_end = window_boxes[-1]['y1']
        
        # Skip very small sections
        if section_y_end - section_y_start < min_section_height and i + window_size < len(text_boxes_sorted):
            i += 1
            continue
        
        # Analyze layout for this window of boxes
        x_positions = []
        for box in window_boxes:
            x_center = (box['x0'] + box['x1']) / 2
            x_positions.append(x_center)
        
        # Apply k-means to detect columns
        if len(x_positions) >= 3:
            X = np.array(x_positions).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            centers = sorted(kmeans.cluster_centers_.flatten())
            counts = np.bincount(kmeans.labels_)
            
            # Check if centers are far apart with sufficient points in each cluster
            if (len(centers) > 1 and 
                abs(centers[0] - centers[1]) > page_width * 0.3 and 
                min(counts) >= max(2, len(x_positions) * 0.15)):
                layout = "double_column"
                midpoint = (centers[0] + centers[1]) / 2
            else:
                layout = "single_column"
                midpoint = page_width / 2
        else:
            layout = "single_column"
            midpoint = page_width / 2
        
        # Add section
        sections.append({
            "layout": layout,
            "y_start": section_y_start,
            "y_end": section_y_end,
            "midpoint": midpoint if layout == "double_column" else None
        })
        
        # Advance to next section
        i = end_idx
    
    # Merge adjacent sections with the same layout
    merged_sections = []
    for section in sections:
        if (merged_sections and 
            merged_sections[-1]["layout"] == section["layout"] and
            abs(merged_sections[-1]["y_end"] - section["y_start"]) < min_section_height):
            # Merge with previous section
            merged_sections[-1]["y_end"] = section["y_end"]
        else:
            merged_sections.append(section)
    
    return merged_sections
```

### 2. Update the Page Processing Logic

```python
def parse_paragraphs(self, pages_data):
    """
    Parse PDF content into paragraphs, handling mixed layouts within pages.
    """
    pages_content = []
    last_paragraph_info = None
    
    for page in pages_data:
        page_num = page['page_num']
        page_height = page.get('height', 792)
        page_width = page.get('width', 612)
        
        # Convert text boxes to blocks format
        text_boxes = page.get('text_boxes', [])
        blocks = self.convert_to_blocks(text_boxes)
        
        if not blocks:
            pages_content.append({
                "page_number": page_num,
                "paragraphs": [],
                "layout": "single_column"
            })
            continue
        
        # Analyze the page to identify sections with different layouts
        sections = self.analyze_page_layout(text_boxes, page_width, page_height)
        
        # Define header and footer boundaries
        header_boundary = page_height * 0.1
        footer_boundary = page_height * 0.9
        
        # Process each section separately
        all_paragraphs = []
        primary_layout = "single_column"  # Default, will be updated based on largest section
        
        # Track section sizes to determine primary layout
        section_sizes = {}
        
        for section in sections:
            section_layout = section["layout"]
            y_start = section["y_start"]
            y_end = section["y_end"]
            section_size = y_end - y_start
            
            # Update section size tracking
            section_sizes[section_layout] = section_sizes.get(section_layout, 0) + section_size
            
            # Filter blocks that fall within this section
            section_blocks = []
            for block in blocks:
                # Only include blocks where the majority falls within this section
                block_mid_y = (block[1] + block[3]) / 2
                if y_start <= block_mid_y <= y_end:
                    section_blocks.append(block)
            
            if not section_blocks:
                continue
            
            # Process blocks based on section layout
            section_paragraphs = []
            
            if section_layout == "double_column" and section_blocks:
                midpoint = section.get("midpoint", page_width / 2)
                
                # Separate into left and right columns
                left_column = []
                right_column = []
                
                for block in section_blocks:
                    block_center_x = (block[0] + block[2]) / 2
                    
                    if block_center_x < midpoint:
                        left_column.append(block)
                    else:
                        right_column.append(block)
                
                # Sort each column by y-coordinate
                left_column.sort(key=lambda b: b[1])
                right_column.sort(key=lambda b: b[1])
                
                # Process each column into paragraphs
                left_paragraphs = self._process_blocks_into_paragraphs(left_column)
                right_paragraphs = self._process_blocks_into_paragraphs(right_column)
                
                # Determine reading order (usually left column first, then right)
                section_paragraphs = self.process_sequential_paragraphs(left_paragraphs)
                section_paragraphs.extend(self.process_sequential_paragraphs(right_paragraphs))
                
            else:  # Single column
                # Sort blocks by y-coordinate
                section_blocks.sort(key=lambda b: b[1])
                raw_paragraphs = self._process_blocks_into_paragraphs(section_blocks)
                section_paragraphs = self.process_sequential_paragraphs(raw_paragraphs)
            
            # Add section paragraphs to all paragraphs
            all_paragraphs.extend(section_paragraphs)
        
        # Determine the primary layout based on section sizes
        if section_sizes:
            primary_layout = max(section_sizes, key=section_sizes.get)
        
        # Handle cross-page paragraph continuity
        if last_paragraph_info and all_paragraphs:
            prev_text, ends_with_punctuation, word_count = last_paragraph_info
            
            if not ends_with_punctuation or word_count < self.min_words_threshold:
                if all_paragraphs:
                    first_content_para = all_paragraphs[0]
                    joined_paragraph = prev_text + " " + first_content_para
                    all_paragraphs[0] = joined_paragraph
            else:
                all_paragraphs.insert(0, prev_text)
            
            last_paragraph_info = None
        
        # Check last paragraph for potential continuation to next page
        if all_paragraphs:
            last_para = all_paragraphs[-1]
            ends_with_punctuation = bool(re.search(r'[.!?:;]$', last_para.strip()))
            word_count = len(last_para.split())
            
            if not ends_with_punctuation or word_count < self.min_words_threshold:
                last_paragraph_info = (last_para, ends_with_punctuation, word_count)
                all_paragraphs.pop()
        
        # Add processed paragraphs to result
        pages_content.append({
            "page_number": page_num,
            "paragraphs": all_paragraphs,
            "layout": primary_layout,
            "sections": [{"layout": s["layout"], "y_range": [s["y_start"], s["y_end"]]} for s in sections]
        })
    
    # If there's still a paragraph carried over at the end of the document, add it
    if last_paragraph_info:
        last_page = pages_content[-1]
        last_page["paragraphs"].append(last_paragraph_info[0])
    
    return pages_content
```

### 3. Update the Layout Determination Method

```python
def determine_layout(self, pages_data):
    """
    Determine the predominant layout type across the document.
    Now supports identifying documents with mixed layouts.
    
    Args:
        pages_data: List of dictionaries containing page data
        
    Returns:
        String indicating layout type: "single_column", "double_column", or "mixed_layout"
    """
    # Analyze each page for section layouts
    layout_counts = {"single_column": 0, "double_column": 0}
    total_area = 0
    
    for page in pages_data:
        page_width = page.get('width', 612)
        page_height = page.get('height', 792)
        text_boxes = page.get('text_boxes', [])
        
        # Skip pages with very few text boxes
        if len(text_boxes) < 3:
            continue
            
        # Analyze the page sections
        sections = self.analyze_page_layout(text_boxes, page_width, page_height)
        
        # Accumulate layout area
        for section in sections:
            section_height = section["y_end"] - section["y_start"]
            layout_counts[section["layout"]] += section_height
            total_area += section_height
    
    # Determine the predominant layout
    if total_area == 0:
        return "single_column"  # Default if no sections were analyzed
        
    # Calculate percentages
    single_pct = layout_counts["single_column"] / total_area
    double_pct = layout_counts["double_column"] / total_area
    
    # If both layout types are present in significant amounts, it's mixed
    if single_pct > 0.25 and double_pct > 0.25:
        return "mixed_layout"
    elif double_pct > single_pct:
        return "double_column"
    else:
        return "single_column"
```

These code updates will:

1. Analyze layouts at the section level within each page using a sliding window approach
2. Detect and handle multiple column layouts within the same page
3. Process each section according to its specific layout type
4. Track and report mixed layouts in the document metadata
5. Maintain proper reading order within and across sections
6. Preserve paragraph continuity between different layout sections

This approach provides a much more robust handling of complex PDF layouts where a single page may contain both single-column and double-column sections, such as academic papers, magazines, or technical documents with mixed formatting.
