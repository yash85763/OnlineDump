Here are the two functionalities you requested:

## 1. Display Previous Feedbacks Below Feedback Form

**Create a function to retrieve previous feedbacks:**

```python
def get_previous_feedbacks(pdf_id):
    """Retrieve all previous feedbacks for a given PDF from database"""
    try:
        from config.database import get_database_connection
        
        conn = get_database_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT feedback_date, form_number_feedback, general_feedback, rating, user_session_id
        FROM feedback_table 
        WHERE pdf_id = %s 
        ORDER BY feedback_date DESC
        """
        
        cursor.execute(query, (pdf_id,))
        feedbacks = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return feedbacks
        
    except Exception as e:
        print(f"Error retrieving feedbacks: {e}")
        return []

def render_previous_feedbacks(pdf_name):
    """Display previous feedbacks for the current PDF"""
    pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
    
    if not pdf_id:
        return
    
    previous_feedbacks = get_previous_feedbacks(pdf_id)
    
    if not previous_feedbacks:
        st.info("📝 No previous feedback found for this document.")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Previous Feedback History")
    st.markdown(f"**{len(previous_feedbacks)} feedback(s) submitted for this document:**")
    
    for i, feedback in enumerate(previous_feedbacks):
        feedback_date, form_number, general_feedback, rating, session_id = feedback
        
        # Convert rating back to text
        rating_text = {5: "Correct", 3: "Partially Correct", 1: "Incorrect"}.get(rating, f"Rating: {rating}")
        
        with st.expander(f"📅 Feedback #{i+1} - {feedback_date.strftime('%Y-%m-%d %H:%M')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📋 Form Number:** {form_number}")
                st.markdown(f"**📝 Feedback:** {general_feedback}")
                
                # Show session info (truncated for privacy)
                session_display = f"{session_id[:8]}..." if session_id else "Unknown"
                st.caption(f"Session: {session_display}")
            
            with col2:
                # Show rating with appropriate emoji
                rating_emoji = {"Correct": "✅", "Partially Correct": "⚠️", "Incorrect": "❌"}.get(rating_text, "📊")
                st.markdown(f"**{rating_emoji} {rating_text}**")
                st.caption(f"📅 {feedback_date.strftime('%b %d, %Y')}")

def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF"""
    feedback_key = f"feedback_{file_stem}"
    
    # Check if feedback was already submitted
    if st.session_state.feedback_submitted.get(feedback_key, False):
        st.success("✅ Thank you! Your feedback has been submitted for this document.")
        if st.button("Submit New Feedback", key=f"new_feedback_{file_stem}"):
            st.session_state.feedback_submitted[feedback_key] = False
            st.rerun()
        
        # Show previous feedbacks after submission
        render_previous_feedbacks(pdf_name)
        return
    
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    st.subheader("📝 Your Feedback Matters")
    st.write("Help us improve our analysis by providing feedback on the results:")
    
    with st.form(f"feedback_form_{file_stem}"):
        # Form number selection (1-10)
        form_number = st.selectbox(
            "Select Form Number",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=0,  # Default to form 1
            help="Select the form number that best matches this document",
            key=f"form_number_{file_stem}"
        )
        
        # Rating selection
        rating = st.radio(
            "Is this analysis correct?",
            ["Correct", "Partially Correct", "Incorrect"],
            help="Rate the accuracy of the analysis",
            key=f"rating_{file_stem}"
        )
        
        # Convert rating to integer for database storage
        rating_map = {
            "Correct": 5,
            "Partially Correct": 3,
            "Incorrect": 1
        }
        rating_value = rating_map[rating]
        
        # Single feedback text box
        feedback_text = st.text_area(
            "Your feedback/comments",
            placeholder="Please provide your feedback or comments about this analysis...",
            height=120,
            help="Provide detailed feedback about the analysis accuracy, suggestions for improvement, etc.",
            key=f"feedback_text_{file_stem}"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            # Validation - require feedback text
            if not feedback_text.strip():
                st.error("Please provide some feedback before submitting.")
                return
            
            # Get PDF ID from session state
            pdf_id = st.session_state.pdf_database_ids.get(pdf_name)
            
            if pdf_id:
                # Prepare feedback data according to database schema
                feedback_data = {
                    'pdf_id': pdf_id,
                    'feedback_date': datetime.now(),
                    'form_number_feedback': form_number,  # INTEGER form number
                    'general_feedback': feedback_text.strip(),  # Single feedback text
                    'rating': rating_value,  # INTEGER rating (1-5)
                    'user_session_id': get_session_id()
                }
                
                try:
                    # Store feedback using the database function
                    feedback_id = store_feedback_data(feedback_data)
                    
                    if feedback_id:
                        st.success("🎉 Thank you for your valuable feedback! It helps us improve our analysis.")
                        st.session_state.feedback_submitted[feedback_key] = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. No feedback ID returned.")
                        
                except Exception as e:
                    st.error(f"❌ Failed to save feedback: {str(e)}")
                    print(f"Feedback submission error: {str(e)}")  # For debugging
            else:
                st.error("❌ Cannot submit feedback - PDF not found in database")
    
    # Show previous feedbacks below the form (always visible)
    render_previous_feedbacks(pdf_name)
    
    st.markdown("</div>", unsafe_allow_html=True)
```

## 2. Load Existing PDFs from Database on App Start

**Create function to load existing PDFs from database:**

```python
def load_existing_pdfs_from_database():
    """Load all existing PDFs from database into session state"""
    try:
        from config.database import get_database_connection
        
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Query to get all processed PDFs with their latest analysis
        query = """
        SELECT DISTINCT p.id, p.filename, p.file_size, p.upload_date, p.final_content,
               a.form_number, a.summary, a.analysis_date,
               a.pi_clause, a.ci_clause, a.data_usage_mentioned, a.data_limitations_exists
        FROM pdf_table p
        LEFT JOIN analysis_table a ON p.id = a.pdf_id
        WHERE a.id = (
            SELECT MAX(id) FROM analysis_table a2 WHERE a2.pdf_id = p.id
        )
        ORDER BY p.upload_date DESC
        """
        
        cursor.execute(query)
        existing_pdfs = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Load PDFs into session state
        for pdf_data in existing_pdfs:
            (pdf_id, filename, file_size, upload_date, final_content,
             form_number, summary, analysis_date,
             pi_clause, ci_clause, data_usage_mentioned, data_limitations_exists) = pdf_data
            
            # Store PDF metadata
            st.session_state.pdf_database_ids[filename] = pdf_id
            st.session_state.analysis_status[filename] = "Processed"
            
            # Create analysis data structure
            file_stem = Path(filename).stem
            st.session_state.json_data[file_stem] = {
                'form_number': form_number or 'Not available',
                'summary': summary or 'No summary available',
                'pi_clause': bool(pi_clause),
                'ci_clause': bool(ci_clause),
                'data_usage_mentioned': bool(data_usage_mentioned),
                'data_limitations_exists': bool(data_limitations_exists),
                'relevant_clauses': [],  # Will be loaded separately if needed
                'analysis_date': analysis_date
            }
            
            # Convert final_content to pages structure for page matching
            if final_content:
                st.session_state.raw_pdf_data[filename] = {
                    'pages': convert_string_to_pages_structure(final_content, filename)
                }
            
            # Create mock processing messages for UI consistency
            st.session_state.processing_messages[filename] = [
                f"📄 Loaded from database (ID: {pdf_id})",
                f"📅 Originally processed: {analysis_date.strftime('%Y-%m-%d %H:%M') if analysis_date else 'Unknown'}",
                f"📊 File size: {file_size} bytes" if file_size else "📊 File size: Unknown",
                "✅ Ready for viewing"
            ]
        
        return len(existing_pdfs)
        
    except Exception as e:
        print(f"Error loading existing PDFs: {e}")
        return 0

def load_pdf_clauses(pdf_id, file_stem):
    """Load relevant clauses for a specific PDF"""
    try:
        from config.database import get_database_connection
        
        conn = get_database_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT c.clause_type, c.clause_text, c.clause_order
        FROM clause_table c
        JOIN analysis_table a ON c.analysis_id = a.id
        WHERE a.pdf_id = %s
        ORDER BY c.clause_order
        """
        
        cursor.execute(query, (pdf_id,))
        clauses = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Convert to the expected format
        relevant_clauses = []
        for clause_type, clause_text, clause_order in clauses:
            relevant_clauses.append({
                'type': clause_type,
                'text': clause_text,
                'order': clause_order
            })
        
        # Update the session data
        if file_stem in st.session_state.json_data:
            st.session_state.json_data[file_stem]['relevant_clauses'] = relevant_clauses
        
        return relevant_clauses
        
    except Exception as e:
        print(f"Error loading clauses: {e}")
        return []
```

**Update the session state initialization:**

```python
def initialize_session_state():
    """Initialize all session state variables"""
    # Database initialization
    if 'database_initialized' not in st.session_state:
        try:
            if check_database_connection():
                initialize_database()
                st.session_state.database_initialized = True
                st.session_state.database_status = "Connected"
            else:
                st.session_state.database_initialized = False
                st.session_state.database_status = "Failed to connect"
        except Exception as e:
            st.session_state.database_initialized = False
            st.session_state.database_status = f"Error: {str(e)}"
    
    # Session state variables
    session_vars = {
        'pdf_files': {},
        'json_data': {},
        'raw_pdf_data': {},
        'current_pdf': None,
        'analysis_status': {},
        'processing_messages': {},
        'pdf_database_ids': {},
        'search_text': None,
        'feedback_submitted': {},
        'obfuscation_summaries': {},
        'session_id': get_session_id(),
        'batch_processing_status': None,
        'batch_processed_count': 0,
        'processing_mode': None,
        'processing_in_progress': False,
        'existing_pdfs_loaded': False  # New flag to track if we've loaded existing PDFs
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    # Load existing PDFs from database on first run
    if (st.session_state.database_initialized and 
        not st.session_state.existing_pdfs_loaded):
        
        try:
            loaded_count = load_existing_pdfs_from_database()
            st.session_state.existing_pdfs_loaded = True
            
            if loaded_count > 0:
                print(f"Loaded {loaded_count} existing PDFs from database")
        except Exception as e:
            print(f"Failed to load existing PDFs: {e}")
```

**Update the document list section to always show the AgGrid:**

```python
        # Document list and selection - ALWAYS SHOW (even without uploaded files)
        st.subheader("📋 Available Documents")
        
        # Combine uploaded PDFs and existing PDFs from database
        all_pdfs = {}
        
        # Add existing PDFs from database
        for pdf_name in st.session_state.pdf_database_ids.keys():
            if pdf_name not in st.session_state.pdf_files:  # Don't duplicate uploaded files
                all_pdfs[pdf_name] = "from_database"
        
        # Add currently uploaded PDFs
        for pdf_name in st.session_state.pdf_files.keys():
            all_pdfs[pdf_name] = "uploaded"
        
        if all_pdfs:
            pdf_data = []
            for pdf_name, source in all_pdfs.items():
                status = st.session_state.analysis_status.get(pdf_name, "Ready")
                db_id = st.session_state.pdf_database_ids.get(pdf_name, "N/A")
                
                # Get file size
                if source == "uploaded":
                    file_size = len(st.session_state.pdf_files[pdf_name]) / 1024
                else:
                    file_size = "N/A"  # Size not available for database PDFs
                
                status_emoji = "✅" if status == "Processed" else "⏳" if "processing" in status.lower() else "📄"
                source_emoji = "📤" if source == "uploaded" else "💾"
                
                pdf_data.append({
                    'Status': status_emoji,
                    'Source': source_emoji,
                    'PDF Name': pdf_name,
                    'Size (KB)': f"{file_size:.1f}" if isinstance(file_size, (int, float)) else str(file_size),
                    'DB ID': str(db_id)
                })
            
            pdf_df = pd.DataFrame(pdf_data)
            gb = GridOptionsBuilder.from_dataframe(pdf_df)
            gb.configure_selection(selection_mode='single', use_checkbox=False)
            gb.configure_grid_options(domLayout='normal')
            gb.configure_default_column(cellStyle={'fontSize': '14px'})
            gb.configure_column("PDF Name", cellStyle={'fontWeight': 'bold'})
            gridOptions = gb.build()

            grid_response = AgGrid(
                pdf_df,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                fit_columns_on_grid_load=True,
                theme='streamlit'
            )

            selected_rows = grid_response.get('selected_rows', pd.DataFrame())
            if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
                selected_pdf = selected_rows.iloc[0]['PDF Name']
                
                # Handle on-demand processing (only for uploaded PDFs)
                if (st.session_state.processing_mode == "on_demand" and 
                    selected_pdf != st.session_state.get('current_pdf') and
                    st.session_state.analysis_status.get(selected_pdf) != "Processed" and
                    selected_pdf in st.session_state.pdf_files):  # Only process uploaded files
                    
                    # Process the selected PDF immediately
                    st.session_state.processing_messages[selected_pdf] = []
                    with st.spinner(f"🔄 Processing {selected_pdf} on-demand..."):
                        success, result = process_pdf_enhanced(
                            st.session_state.pdf_files[selected_pdf], 
                            selected_pdf, 
                            st.empty(),
                            logger
                        )
                        
                        if success:
                            st.session_state.analysis_status[selected_pdf] = "Processed"
                            st.success(f"✅ On-demand analysis complete for {selected_pdf}")
                        else:
                            st.session_state.analysis_status[selected_pdf] = f"❌ Failed: {result}"
                            st.error(f"❌ Failed to process {selected_pdf}: {result}")
                
                # Set current PDF for viewing (works for both uploaded and database PDFs)
                if selected_pdf != st.session_state.get('current_pdf'):
                    set_current_pdf(selected_pdf)
                    
                    # Load clauses for database PDFs if not already loaded
                    file_stem = Path(selected_pdf).stem
                    if (selected_pdf in st.session_state.pdf_database_ids and 
                        file_stem in st.session_state.json_data and
                        not st.session_state.json_data[file_stem].get('relevant_clauses')):
                        
                        pdf_id = st.session_state.pdf_database_ids[selected_pdf]
                        load_pdf_clauses(pdf_id, file_stem)
        else:
            st.info("📄 No documents available. Upload PDFs or check database connection.")
            
        # Show legend for source icons
        st.caption("📤 = Uploaded this session | 💾 = From database")
```

## Key Features Added:

### 1. Previous Feedbacks Display:
- Shows all previous feedback for the current PDF
- Displays feedback date, comments, ratings, and session info
- Expandable format for easy viewing
- Automatically loads when viewing any PDF

### 2. Existing PDFs Loading:
- Loads all previously analyzed PDFs from database on app start
- Shows them in the AgGrid with a source indicator (📤 for uploaded, 💾 for database)
- Preserves analysis results and allows viewing without reprocessing
- Lazy loads clauses when a database PDF is selected
- Maintains consistent UI for both uploaded and database PDFs

### 3. Enhanced AgGrid:
- Always visible (even without uploads)
- Shows source of each PDF
- Distinguishes between uploaded and database PDFs
- Maintains all existing functionality

The system now provides a complete view of all available documents and their feedback history, making it much more user-friendly for ongoing work with contract analysis.
