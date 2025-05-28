Here's how to display PNG images instead of emojis in your Streamlit app:

## **1. Create Image Loading Function:**

**Add this function after the imports:**

```python
import base64
from pathlib import Path

def load_image_as_base64(image_path):
    """Load image and convert to base64 for HTML display"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def get_icon_html(icon_name, size=20, fallback_emoji=""):
    """Get HTML for icon with fallback to emoji"""
    icon_path = f"pics/{icon_name}.png"
    base64_image = load_image_as_base64(icon_path)
    
    if base64_image:
        return f'<img src="data:image/png;base64,{base64_image}" width="{size}" height="{size}" style="vertical-align: middle; margin-right: 5px;">'
    else:
        return fallback_emoji  # Fallback to emoji if image not found

# Icon mapping dictionary
ICONS = {
    'success': 'get_icon_html("success", 16, "✅")',
    'error': 'get_icon_html("error", 16, "❌")', 
    'warning': 'get_icon_html("warning", 16, "⚠️")',
    'info': 'get_icon_html("info", 16, "ℹ️")',
    'processing': 'get_icon_html("processing", 16, "🔄")',
    'document': 'get_icon_html("document", 16, "📄")',
    'database': 'get_icon_html("database", 16, "💾")',
    'privacy': 'get_icon_html("privacy", 16, "🔒")',
    'analysis': 'get_icon_html("analysis", 16, "🔍")',
    'batch': 'get_icon_html("batch", 16, "🚀")',
    'feedback': 'get_icon_html("feedback", 16, "📝")',
    'clause': 'get_icon_html("clause", 16, "📑")',
    'page': 'get_icon_html("page", 16, "📖")',
    'search': 'get_icon_html("search", 16, "🔍")',
    'goto': 'get_icon_html("goto", 16, "📄")',
    'download': 'get_icon_html("download", 16, "📥")',
    'upload': 'get_icon_html("upload", 16, "📤")',
    'settings': 'get_icon_html("settings", 16, "🔧")',
    'star': 'get_icon_html("star", 16, "⭐")'
}
```

## **2. Update Status Button CSS:**

**Replace the status button CSS with:**

```css
.status-button-true {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.3rem 0;
    display: inline-block;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(40,167,69,0.3);
}

.status-button-false {
    background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.3rem 0;
    display: inline-block;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(220,53,69,0.3);
}

.status-button-missing {
    background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
    color: #212529;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.3rem 0;
    display: inline-block;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(255,193,7,0.3);
}

.icon-text {
    display: flex;
    align-items: center;
    gap: 8px;
}
```

## **3. Replace Emoji Usage Throughout:**

**Replace processing messages with:**

```python
# In process_pdf_enhanced function
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('processing', 16, '🔄')} Starting PDF processing with obfuscation...")
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('success', 16, '✅')} PDF processed and stored in database")
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('database', 16, '💾')} Database ID: {result.get('pdf_id')}")
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('privacy', 16, '🔒')} Privacy protection applied: {pages_removed} pages removed")
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('analysis', 16, '🔍')} Starting contract analysis...")
st.session_state.processing_messages[pdf_name].append(f"{get_icon_html('success', 16, '✅')} Contract analysis completed successfully")
```

## **4. Update Service Status Display:**

**Replace the service status section with:**

```python
# Service status checks
services = st.session_state.get('services_status', {})

for service_name, service_info in services.items():
    status = service_info['status']
    message = service_info['message']
    
    if status:
        st.markdown(f"<div class='icon-text'>{get_icon_html('success', 16, '✅')} <strong>{service_name.replace('_', ' ').title()}:</strong> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='icon-text'>{get_icon_html('error', 16, '❌')} <strong>{service_name.replace('_', ' ').title()}:</strong> {message}</div>", unsafe_allow_html=True)
```

## **5. Update Contract Status Display:**

**Replace the contract status section with:**

```python
# Contract Status - Enhanced UI with icons
st.markdown("### Contract Status")
status_fields = {
    'data_usage_mentioned': 'Data Usage Mentioned',
    'data_limitations_exists': 'Data Limitations Exists', 
    'pi_clause': 'Presence of PI Clause',
    'ci_clause': 'Presence of CI Clause'
}

col_status1, col_status2 = st.columns(2)
status_items = list(status_fields.items())

for i, (key, label) in enumerate(status_items):
    target_col = col_status1 if i % 2 == 0 else col_status2
    with target_col:
        status = json_data.get(key, None)
        status_str = str(status).lower() if status is not None else 'unknown'
        
        # Determine icon and style
        if status_str in ['true', 'yes']:
            icon_html = get_icon_html('success', 16, '✅')
            button_class = 'status-button-true'
        elif status_str in ['false', 'no']:
            icon_html = get_icon_html('error', 16, '❌')
            button_class = 'status-button-false'
        else:
            icon_html = get_icon_html('warning', 16, '❓')
            button_class = 'status-button-missing'
        
        st.markdown(f"""
        <div class='{button_class}'>
            <div class='icon-text'>
                {icon_html} <strong>{label}</strong><br>
                <small>{status}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
```

## **6. Update Button Displays:**

**Replace buttons with icon versions:**

```python
# Search and navigation buttons
col_search1, col_search2 = st.columns(2)
with col_search1:
    if st.button(f"{get_icon_html('search', 16, '🔍')} Search Text", key=f"search_clause_{i}"):
        st.session_state.search_text = clause['text'][:100]
        st.success(f"Searching for clause {i+1}...")
        st.rerun()

with col_search2:
    if clause_page and st.button(f"{get_icon_html('goto', 16, '📄')} Go to Page {clause_page}", key=f"goto_page_{i}"):
        st.session_state.current_page_number = clause_page
        st.session_state.search_text = clause['text'][:50]
        st.success(f"Navigating to page {clause_page}...")
        st.rerun()
```

## **7. Update Batch Processing Section:**

**Replace batch processing buttons with:**

```python
if st.button(f"{get_icon_html('batch', 16, '🚀')} Start Batch Processing", 
            disabled=batch_button_disabled,
            help="Process all selected documents"):
    # ... existing code

if st.session_state.batch_job_active:
    if st.button(f"{get_icon_html('error', 16, '⏹️')} Cancel Batch"):
        st.session_state.batch_job_active = False
        st.rerun()
```

## **8. Update Header and Titles:**

**Replace the main header with:**

```python
st.markdown(f"""
<div class='main-header'>
    <h1>{get_icon_html('document', 24, '📄')} Enhanced Contract Analysis Platform</h1>
    <p>AI-powered contract analysis with privacy protection and intelligent feedback</p>
</div>
""", unsafe_allow_html=True)
```

## **9. Add Fallback Icon Checker:**

**Add this function to check which icons are missing:**

```python
def check_available_icons():
    """Check which icons are available and show missing ones"""
    icon_names = ['success', 'error', 'warning', 'info', 'processing', 'document', 
                  'database', 'privacy', 'analysis', 'batch', 'feedback', 'clause',
                  'page', 'search', 'goto', 'download', 'upload', 'settings', 'star']
    
    missing_icons = []
    for icon_name in icon_names:
        icon_path = f"pics/{icon_name}.png"
        if not os.path.exists(icon_path):
            missing_icons.append(icon_name)
    
    return missing_icons

# Add to sidebar for debugging
if st.sidebar.button("Check Icons"):
    missing = check_available_icons()
    if missing:
        st.sidebar.write(f"Missing icons: {', '.join(missing)}")
    else:
        st.sidebar.write("All icons found!")
```

## **Required PNG Files:**

Create these PNG files in your `pics/` folder (16x16 or 24x24 pixels recommended):

- `success.png` (green checkmark)
- `error.png` (red X)
- `warning.png` (yellow triangle)
- `info.png` (blue i)
- `processing.png` (spinning wheel)
- `document.png` (document icon)
- `database.png` (database icon)
- `privacy.png` (lock icon)
- `analysis.png` (magnifying glass)
- `batch.png` (rocket icon)
- `feedback.png` (pencil/note icon)
- `clause.png` (document with lines)
- `page.png` (page icon)
- `search.png` (magnifying glass)
- `goto.png` (arrow or page icon)
- `download.png` (down arrow)
- `upload.png` (up arrow)
- `settings.png` (gear icon)
- `star.png` (star icon)

This approach will:
- ✅ **Display custom PNG icons** instead of emojis
- ✅ **Fallback to emojis** if PNG files are missing
- ✅ **Maintain consistent sizing** across the app
- ✅ **Load efficiently** using base64 encoding
- ✅ **Check for missing icons** with debugging function​​​​​​​​​​​​​​​​