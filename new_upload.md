Great! Now let's implement Step 2 to load the analysis data. Here are the code updates needed:

## 1. Update the database function in `database.py`

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

## 2. Update the loading function in `main.py`

Replace your current `load_pdf_from_database()` function with this enhanced version:

```python
def load_pdf_from_database_with_analysis(pdf_id: int, pdf_name: str):
    """Load complete PDF data including analysis from database into session state"""
    try:
        from config.database import load_complete_pdf_data
        import json
        from pathlib import Path
        
        with st.spinner(f"Loading {pdf_name} with analysis data..."):
            # Get complete data from database
            complete_data = load_complete_pdf_data(pdf_id)
            
            if not complete_data:
                st.error(f"Could not load PDF data for ID {pdf_id}")
                return False
            
            # Load PDF bytes (from Step 1)
            if complete_data.get('pdf_bytes'):
                st.session_state.pdf_files[pdf_name] = bytes(complete_data['pdf_bytes'])
                st.session_state.loaded_pdfs.add(pdf_name)
            else:
                st.error(f"No PDF bytes found for {pdf_name}")
                return False
            
            # NEW: Load analysis data
            analyses = complete_data.get('analyses', [])
            if analyses and len(analyses) > 0:
                # Get the latest analysis (first in list due to ORDER BY version DESC)
                latest_analysis = analyses[0]
                
                # Extract analysis data
                analysis_data = latest_analysis.get('raw_json', {})
                
                # Handle both string and dict formats
                if isinstance(analysis_data, str):
                    try:
                        analysis_data = json.loads(analysis_data)
                    except json.JSONDecodeError:
                        st.warning(f"Could not parse analysis data for {pdf_name}")
                        analysis_data = {}
                
                # Store analysis in session state
                file_stem = Path(pdf_name).stem
                st.session_state.json_data[file_stem] = analysis_data
                st.session_state.analysis_status[pdf_name] = "Processed"
                
                # Store metadata for feedback and other uses
                st.session_state.pdf_database_ids[pdf_name] = pdf_id
                
                # Store analysis metadata if needed
                if 'analysis_metadata' not in st.session_state:
                    st.session_state.analysis_metadata = {}
                
                st.session_state.analysis_metadata[pdf_name] = {
                    'latest_version': latest_analysis.get('version', 1),
                    'analysis_date': latest_analysis.get('analysis_date'),
                    'total_analyses': len(analyses),
                    'all_analyses': analyses  # Store all analyses for potential version switching
                }
                
                st.success(f"✅ Loaded {pdf_name} with analysis (Version {latest_analysis.get('version', 1)})")
                return True
                
            else:
                # No analysis data found
                st.session_state.analysis_status[pdf_name] = "Parsed Only"
                st.session_state.pdf_database_ids[pdf_name] = pdf_id
                st.warning(f"⚠️ {pdf_name} loaded but no analysis data found")
                return True
                
    except Exception as e:
        st.error(f"Error loading PDF with analysis: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return False
```

## 3. Update the row selection handler in the left pane

Replace the row selection handling section in your left pane with this:

```python
# Handle row selection
selected_rows = grid_response.get('selected_rows', pd.DataFrame())
if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_pdf_name = selected_rows.iloc[0]['PDF Name']
    source = selected_rows.iloc[0]['Source']
    
    if source == 'Database':
        # Get the PDF ID for database loading
        db_id = selected_rows.iloc[0]['DB ID']
        
        if db_id != "New":  # Make sure it's a valid database ID
            try:
                pdf_id = int(db_id)
                
                # Check if already loaded
                if selected_pdf_name not in st.session_state.loaded_pdfs:
                    # Load from database with analysis
                    if load_pdf_from_database_with_analysis(pdf_id, selected_pdf_name):
                        set_current_pdf(selected_pdf_name)
                        st.rerun()  # Refresh to show loaded data
                else:
                    # Already loaded, just set as current
                    if selected_pdf_name != st.session_state.get('current_pdf'):
                        set_current_pdf(selected_pdf_name)
                        st.rerun()
                        
            except ValueError:
                st.error(f"Invalid database ID: {db_id}")
        else:
            st.error("Cannot load PDF - invalid database ID")
    else:
        # Session PDF - just set as current (existing functionality)
        if selected_pdf_name != st.session_state.get('current_pdf'):
            set_current_pdf(selected_pdf_name)
```

## 4. Add analysis metadata display (optional)

You can add this to the left pane to show analysis information:

```python
# Show analysis metadata if available (add this after the grid)
if st.session_state.get('current_pdf') and st.session_state.current_pdf in st.session_state.get('analysis_metadata', {}):
    metadata = st.session_state.analysis_metadata[st.session_state.current_pdf]
    
    with st.expander(f"📊 Analysis Info for {st.session_state.current_pdf}", expanded=False):
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.write(f"**Version:** {metadata['latest_version']}")
            st.write(f"**Total Analyses:** {metadata['total_analyses']}")
        with col_meta2:
            if metadata['analysis_date']:
                date_str = str(metadata['analysis_date']).split('T')[0]
                st.write(f"**Date:** {date_str}")
            
        if metadata['total_analyses'] > 1:
            st.info(f"This PDF has {metadata['total_analyses']} analysis versions available")
```

## 5. Update session state initialization

Add this to your `initialize_session_state()` function:

```python
# Add this to the session_vars dictionary:
'analysis_metadata': {},  # Store analysis version info and metadata
```

## Testing Step 2:

After implementing these changes, test by:

1. **Click on a database PDF** that has analysis data
2. **Verify the PDF loads** in the middle pane (from Step 1)
3. **Check the right pane** - you should see analysis results appear
4. **Check the analysis status** - should show "Processed" 
5. **Look for the success message** - should mention the analysis version
6. **Try a PDF without analysis** - should show "Parsed Only" warning

The right pane should now populate with analysis data when you click on analyzed PDFs from the database!
