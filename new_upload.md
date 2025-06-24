Great! Step 1 is working (PDF bytes are loading). Now let's implement Step 2 properly to load the analysis data.

## Update your loading function to include analysis

Replace your `load_pdf_from_database_basic` function with this enhanced version:

```python
def load_pdf_from_database_with_analysis(pdf_id, pdf_name):
    try: 
        with st.spinner(f"Loading {pdf_name} with analysis..."):
            # Load complete data including analysis
            complete_data = load_complete_pdf_data(pdf_id)

            if not complete_data:
                st.error("Could not load PDF data")
                return False
                
            if not complete_data.get('pdf_bytes'):
                st.error("No PDF bytes found")
                return False
                
            # Load PDF bytes (Step 1 functionality)
            pdf_bytes = bytes(complete_data['pdf_bytes'])
            st.session_state.pdf_files[pdf_name] = pdf_bytes
            st.session_state.loaded_pdfs.add(pdf_name)
            st.session_state.current_pdf = pdf_name
            st.session_state.pdf_database_ids[pdf_name] = complete_data["id"]
            
            # NEW: Load analysis data (Step 2)
            analyses = complete_data.get('analyses', [])
            if analyses and len(analyses) > 0:
                # Get latest analysis
                latest_analysis = analyses[0]
                analysis_data = latest_analysis.get('raw_json', {})
                
                # Handle JSON parsing
                if isinstance(analysis_data, str):
                    try:
                        analysis_data = json.loads(analysis_data)
                    except json.JSONDecodeError:
                        st.warning("Could not parse analysis data")
                        analysis_data = {}
                
                # Store analysis for right pane display
                file_stem = Path(pdf_name).stem
                st.session_state.json_data[file_stem] = analysis_data
                st.session_state.analysis_status[pdf_name] = "Processed"
                
                st.success(f"✅ Loaded {pdf_name} with analysis (Version {latest_analysis.get('version', 1)})")
            else:
                st.session_state.analysis_status[pdf_name] = "Parsed Only"
                st.warning(f"⚠️ {pdf_name} loaded but no analysis found")
            
            return True
            
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return False
```

## Add the missing imports

Add these imports at the top of your `main.py` if not already there:

```python
import json
from pathlib import Path
```

## Make sure you have this function in your `database.py`:

```python
def load_complete_pdf_data(pdf_id: int) -> Dict[str, Any]:
    """Load complete PDF data including bytes, parsing data, and analyses"""
    
    sql = """
        SELECT 
            p.*,
            -- Get all analyses for this PDF
            COALESCE(
                json_agg(
                    json_build_object(
                        'id', a.id,
                        'version', a.version,
                        'analysis_date', a.analysis_date,
                        'form_number', a.form_number,
                        'pi_clause', a.pi_clause,
                        'ci_clause', a.ci_clause,
                        'data_usage_mentioned', a.data_usage_mentioned,
                        'data_limitations_exists', a.data_limitations_exists,
                        'summary', a.summary,
                        'raw_json', a.raw_json,
                        'processed_by', a.processed_by,
                        'processing_time', a.processing_time
                    ) ORDER BY a.version DESC
                ) FILTER (WHERE a.id IS NOT NULL),
                '[]'::json
            ) as analyses
        FROM pdfs p
        LEFT JOIN analyses a ON p.id = a.pdf_id
        WHERE p.id = %s
        GROUP BY p.id
    """
    
    with db.get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (pdf_id,))
            result = cur.fetchone()
            return dict(result) if result else None
```

## Update your row selection handler

Change the function call in your row selection handler:

```python
# In your row selection handler, change this line:
if load_pdf_from_database_basic(pdf_id, selected_pdf_name):

# To this:
if load_pdf_from_database_with_analysis(pdf_id, selected_pdf_name):
```

## Test Step 2:

After making these changes:

1. **Click on a database PDF** that should have analysis data
2. **Check the success/warning message** - does it say "with analysis" or "no analysis found"?
3. **Look at the right pane** - does analysis data appear?
4. **Check the analysis status** in the grid - does it update to "Processed"?


Looking at your code, I can see the issue. The page navigation isn't working because we're not loading the **original parsing data** that's needed for page number matching. 

Currently, your `load_pdf_from_database_with_analysis` function only loads:
1. PDF bytes (for display)
2. Analysis data (for right pane)

But it's missing:
3. **PDF parsing data** (for page number matching)

## Here's what needs to be added to your loading function:

Add this section to your `load_pdf_from_database_with_analysis` function, right after loading the analysis data:

```python
def load_pdf_from_database_with_analysis(pdf_id, pdf_name):
    try: 
        with st.spinner(f"Loading {pdf_name} with analysis..."):
            # ... your existing code for PDF bytes and analysis ...
            
            # NEW: Load PDF parsing data for page number matching
            pdf_parsing_data = complete_data.get('pdf_parsing_data', {})
            if pdf_parsing_data:
                # Handle JSON parsing if it's stored as string
                if isinstance(pdf_parsing_data, str):
                    try:
                        pdf_parsing_data = json.loads(pdf_parsing_data)
                    except json.JSONDecodeError:
                        st.warning("Could not parse PDF parsing data")
                        pdf_parsing_data = {}
                
                # Extract original pages for page matching
                original_pages = pdf_parsing_data.get('original_pages', [])
                if original_pages:
                    st.session_state.raw_pdf_data[pdf_name] = {
                        'pages': original_pages
                    }
                    st.write(f"🔍 DEBUG: Loaded {len(original_pages)} pages for page matching")
                else:
                    st.warning("No original pages data found for page matching")
                    st.session_state.raw_pdf_data[pdf_name] = {'pages': []}
            else:
                st.warning("No PDF parsing data found")
                st.session_state.raw_pdf_data[pdf_name] = {'pages': []}
            
            return True
            
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return False
```

## Debug to verify the data structure:

Add this debug section temporarily to see what's in your `pdf_parsing_data`:

```python
# Add this right after loading pdf_parsing_data:
if pdf_parsing_data:
    st.write("🔍 DEBUG: PDF parsing data keys:", list(pdf_parsing_data.keys()))
    
    original_pages = pdf_parsing_data.get('original_pages', [])
    st.write(f"🔍 DEBUG: Original pages count: {len(original_pages)}")
    
    if original_pages:
        first_page = original_pages[0]
        st.write(f"🔍 DEBUG: First page keys: {list(first_page.keys())}")
        
        paragraphs = first_page.get('paragraphs', [])
        st.write(f"🔍 DEBUG: First page paragraphs count: {len(paragraphs)}")
        
        if paragraphs:
            st.write(f"🔍 DEBUG: First paragraph preview: {paragraphs[0][:100]}...")
```

## Test the page matching:

After making these changes:

1. **Click on a database PDF** with analysis
2. **Check the debug messages** - do you see the parsing data being loaded?
3. **Go to the right pane** and expand a clause dropdown
4. **Look for page numbers** - they should now appear

The key insight is that your `find_clause_page_number` function needs `st.session_state.raw_pdf_data[pdf_name]` to be populated with the original pages data, which we're now loading from the `pdf_parsing_data` field in the database.

Let me know what the debug output shows and if the page numbers start appearing!
