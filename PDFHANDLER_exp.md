The issue is that we're using `eval()` in the `ICONS` dictionary and returning HTML strings that aren't being properly rendered. Here's the corrected approach:

## **1. Fix the Icon Loading Functions:**

**Replace the icon functions with:**

```python
import base64
import os

def load_image_as_base64(image_path):
    """Load image and convert to base64 for HTML display"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    return None

def get_icon_html(icon_name, size=16, fallback_emoji="", alt_text=""):
    """Get HTML for icon with fallback to emoji"""
    icon_path = f"pics/{icon_name}.png"
    base64_image = load_image_as_base64(icon_path)
    
    if base64_image:
        return f'<img src="data:image/png;base64,{base64_image}" width="{size}" height="{size}" style="vertical-align: middle; margin-right: 5px; display: inline-block;" alt="{alt_text}">'
    else:
        return f'<span style="margin-right: 5px; display: inline-block;">{fallback_emoji}</span>'

# Remove the ICONS dictionary - we'll call functions directly
```

## **2. Update CSS for Better Icon Display:**

**Add this to your CSS:**

```css
.icon-container {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    vertical-align: middle;
}

.status-icon {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.processing-message {
    color: #0068c9;
    font-size: 0.9rem;
    padding: 0.3rem 0;
    border-left: 3px solid #0068c9;
    padding-left: 0.8rem;
    margin: 0.2rem 0;
    background-color: #f8f9ff;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
```

## **3. Fix Processing Messages:**

**Replace the processing message updates with:**

```python
# In process_pdf_enhanced function
st.session_state.processing_messages[pdf_name].append("Starting PDF processing with obfuscation...")
st.session_state.processing_messages[pdf_name].append("PDF processed and stored in database")
st.session_state.processing_messages[pdf_name].append(f"Database ID: {result.get('pdf_id')}")

# When displaying messages, use this format:
for msg in st.session_state.processing_messages[pdf_name]:
    icon_html = get_icon_html('processing', 16, '🔄')
    st.markdown(
        f'<div class="processing-message">{icon_html}{msg}</div>', 
        unsafe_allow_html=True
    )
```

## **4. Fix Service Status Display:**

**Replace with:**

```python
# Service status checks
services = st.session_state.get('services_status', {})

for service_name, service_info in services.items():
    status = service_info['status']
    message = service_info['message']
    
    if status:
        icon_html = get_icon_html('success', 16, '✅', 'Success')
        st.markdown(
            f'<div class="icon-container">{icon_html}<strong>{service_name.replace("_", " ").title()}:</strong> {message}</div>', 
            unsafe_allow_html=True
        )
    else:
        icon_html = get_icon_html('error', 16, '❌', 'Error') 
        st.markdown(
            f'<div class="icon-container">{icon_html}<strong>{service_name.replace("_", " ").title()}:</strong> {message}</div>', 
            unsafe_allow_html=True
        )
```

## **5. Fix Contract Status Display:**

**Replace with:**

```python
# Contract Status - Enhanced UI with icons
st.markdown("### Contract Status")

for i, (key, label) in enumerate(status_items):
    target_col = col_status1 if i % 2 == 0 else col_status2
    with target_col:
        status = json_data.get(key, None)
        status_str = str(status).lower() if status is not None else 'unknown'
        
        # Determine icon and style
        if status_str in ['true', 'yes']:
            icon_html = get_icon_html('success', 16, '✅', 'Success')
            button_class = 'status-button-true'
        elif status_str in ['false', 'no']:
            icon_html = get_icon_html('error', 16, '❌', 'Error')
            button_class = 'status-button-false'
        else:
            icon_html = get_icon_html('warning', 16, '❓', 'Unknown')
            button_class = 'status-button-missing'
        
        st.markdown(f"""
        <div class='{button_class}'>
            <div class='status-icon'>
                {icon_html}
                <div>
                    <strong>{label}</strong><br>
                    <small>{status}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
```

## **6. Fix Button Displays with st.columns for Layout:**

**Replace the search buttons with:**

```python
# Search and navigation buttons
col_search1, col_search2 = st.columns(2)

with col_search1:
    search_icon = get_icon_html('search', 16, '🔍', 'Search')
    # Use markdown for icon, but regular button for functionality
    st.markdown(f'<div class="icon-container">{search_icon}Search in PDF</div>', unsafe_allow_html=True)
    if st.button("Search Text", key=f"search_clause_{i}"):
        st.session_state.search_text = clause['text'][:100]
        st.success("Searching for clause...")
        st.rerun()

with col_search2:
    if clause_page:
        goto_icon = get_icon_html('goto', 16, '📄', 'Go to page')
        st.markdown(f'<div class="icon-container">{goto_icon}Navigate to Page {clause_page}</div>', unsafe_allow_html=True)
        if st.button(f"Go to Page {clause_page}", key=f"goto_page_{i}"):
            st.session_state.current_page_number = clause_page
            st.session_state.search_text = clause['text'][:50]
            st.success(f"Navigating to page {clause_page}...")
            st.rerun()
```

## **7. Alternative: Use st.columns for Icon+Text Layout:**

**For better icon integration, use this pattern:**

```python
# For buttons with icons
def render_icon_button(icon_name, text, key, fallback_emoji="", on_click=None):
    """Render button with icon"""
    col_icon, col_text = st.columns([1, 4])
    
    with col_icon:
        icon_html = get_icon_html(icon_name, 20, fallback_emoji)
        st.markdown(icon_html, unsafe_allow_html=True)
    
    with col_text:
        if st.button(text, key=key):
            if on_click:
                on_click()
            return True
    return False

# Usage example:
if render_icon_button('search', 'Search in PDF', f'search_{i}', '🔍'):
    st.session_state.search_text = clause['text'][:100]
    st.rerun()
```

## **8. Debug Icon Loading:**

**Add this debug function to check if images are loading:**

```python
def debug_icons():
    """Debug icon loading"""
    st.write("### Icon Debug")
    
    test_icons = ['success', 'error', 'warning', 'processing']
    
    for icon_name in test_icons:
        icon_path = f"pics/{icon_name}.png"
        exists = os.path.exists(icon_path)
        
        st.write(f"**{icon_name}.png**: {'Found' if exists else 'Missing'}")
        
        if exists:
            # Test the HTML output
            icon_html = get_icon_html(icon_name, 24, '❓')
            st.markdown(f"Preview: {icon_html} {icon_name}", unsafe_allow_html=True)
        else:
            st.write(f"Path checked: {os.path.abspath(icon_path)}")

# Add to sidebar temporarily
if st.sidebar.button("Debug Icons"):
    debug_icons()
```

## **9. Make Sure Your PNG Files Are:**

- **Small size**: 16x16 or 24x24 pixels
- **Transparent background**: PNG with alpha channel
- **Named correctly**: exactly matching the icon names
- **In the right location**: `pics/` folder in your project root

## **10. Test with a Simple Icon First:**

**Add this test at the top of your main function:**

```python
# Test icon loading
st.write("Icon test:")
test_icon = get_icon_html('success', 24, '✅', 'Test')
st.markdown(f"Test icon: {test_icon} This should show an icon", unsafe_allow_html=True)
```

The key fixes:
- ✅ **Removed eval()** from icon loading
- ✅ **Fixed HTML structure** for proper rendering  
- ✅ **Added proper CSS** for icon alignment
- ✅ **Separated icon display from button functionality**
- ✅ **Added debugging** to check file loading
- ✅ **Used inline-block** display for proper positioning

Try the debug function first to see if your PNG files are being found and loaded correctly!​​​​​​​​​​​​​​​​