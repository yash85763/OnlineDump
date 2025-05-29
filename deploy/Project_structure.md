Here are the code changes needed:

## 1. Update Processing Strategy Section (Remove Spinner Between Buttons and Grid)

**Replace the single PDF processing button code in the Processing Strategy Section:**

```python
elif pdf_count == 1:
    # Single PDF - simple processing button
    pdf_name = list(st.session_state.pdf_files.keys())[0]
    if st.session_state.analysis_status.get(pdf_name) != "Processed":
        if st.button("🔄 Process PDF", use_container_width=True):
            st.session_state.processing_messages[pdf_name] = []
            # Remove the spinner and processing logic from here
            # Just set a flag to indicate processing should start
            st.session_state.analysis_status[pdf_name] = "Processing"
            st.rerun()
    else:
        st.success("✅ PDF already processed")
```

## 2. Move Processing Logic to Processing Status Section

**In the Processing Status Display section, add this processing logic:**

```python
# Processing Status Display (Single Column Below PDF List)
st.subheader("📊 Processing Status")

if st.session_state.pdf_files:
    pdf_count = len(st.session_state.pdf_files)
    processed_count = len([s for s in st.session_state.analysis_status.values() if s == "Processed"])
    
    # Handle actual processing for single PDF mode
    if pdf_count == 1:
        pdf_name = list(st.session_state.pdf_files.keys())[0]
        if st.session_state.analysis_status.get(pdf_name) == "Processing":
            with st.spinner(f"🔄 Processing {pdf_name}..."):
                success, result = process_pdf_enhanced(
                    st.session_state.pdf_files[pdf_name], 
                    pdf_name, 
                    st.empty(),  # placeholder for messages
                    logger
                )
                
                if success:
                    st.session_state.analysis_status[pdf_name] = "Processed"
                    st.success(f"✅ Analysis complete for {pdf_name}")
                else:
                    st.session_state.analysis_status[pdf_name] = f"❌ Failed: {result}"
                    st.error(f"❌ Failed to process {pdf_name}: {result}")
                st.rerun()
    
    # Show current processing mode status
    if st.session_state.processing_mode == "on_demand":
        unprocessed = [name for name, status in st.session_state.analysis_status.items() 
                     if status != "Processed"]
        if unprocessed:
            st.info(f"🎯 **On-Demand Mode**: {len(unprocessed)} PDFs ready. Click any PDF above to analyze instantly.")
        else:
            st.success(f"✅ **All PDFs Analyzed**: {processed_count}/{pdf_count} completed in on-demand mode.")
            
    elif st.session_state.processing_mode == "batch":
        if st.session_state.batch_processing_status == "Running":
            st.info(f"📚 **Batch Processing**: {processed_count}/{pdf_count} completed...")
        elif processed_count == pdf_count:
            st.success(f"✅ **Batch Complete**: All {pdf_count} PDFs processed successfully!")
        else:
            st.info(f"📚 **Batch Status**: {processed_count}/{pdf_count} PDFs processed")
            
    elif pdf_count == 1:
        pdf_name = list(st.session_state.pdf_files.keys())[0]
        if st.session_state.analysis_status.get(pdf_name) == "Processed":
            st.success("✅ **PDF Analysis Complete**")
        elif st.session_state.analysis_status.get(pdf_name) == "Processing":
            st.info("🔄 **Processing in progress**...")
        else:
            st.info("📄 **Single PDF Ready**: Click 'Process PDF' button above to analyze")
    else:
        st.info(f"📋 **{pdf_count} PDFs Uploaded**: Choose a processing strategy above to begin")
    
    # Show detailed processing messages in dropdown (expandable)
    current_pdf = st.session_state.current_pdf
    if current_pdf and current_pdf in st.session_state.processing_messages:
        if st.session_state.processing_messages[current_pdf]:
            with st.expander(f"📋 Processing Details: {current_pdf}", expanded=False):
                # Single column format for messages
                for msg in st.session_state.processing_messages[current_pdf]:
                    st.markdown(f"<div class='processing-message'>{msg}</div>", unsafe_allow_html=True)
```

## 3. Update Batch Processing Function (Remove Inline Spinners)

**Update the `process_batch_pdfs` function to remove inline spinners:**

```python
def process_batch_pdfs(logger):
    """Process all uploaded PDFs one by one with progress updates"""
    pdf_files = list(st.session_state.pdf_files.keys())
    total_pdfs = len(pdf_files)
    
    if total_pdfs == 0:
        st.warning("⚠️ No PDFs uploaded for batch processing.")
        return
    
    st.session_state.batch_processing_status = "Running"
    st.session_state.batch_processed_count = 0
    
    # Remove progress_bar and status_text from here
    # Processing will be shown in the Processing Status section
    
    for i, pdf_name in enumerate(pdf_files):
        if st.session_state.analysis_status.get(pdf_name) == "Processed":
            continue  # Skip already processed PDFs
        
        # Remove status display from here
        success, result = process_pdf_enhanced(
            st.session_state.pdf_files[pdf_name], 
            pdf_name, 
            st.empty(),  # placeholder for messages
            logger
        )
        
        if success:
            st.session_state.analysis_status[pdf_name] = "Processed"
            st.session_state.batch_processed_count += 1
        else:
            st.session_state.analysis_status[pdf_name] = f"❌ Failed: {result}"
    
    st.session_state.batch_processing_status = "Completed"
```

## 4. Horizontalize Feedback Form (Move Below 3-Pane Grid)

**Move the feedback form call from inside the right pane to after the main 3-column layout:**

**Remove this line from the right pane (col3):**
```python
st.markdown("---")
render_feedback_form(st.session_state.current_pdf, file_stem, json_data)
```

**Add this section after the 3-column layout ends (after `col3` closes):**

```python
# Horizontal Feedback Form (Below 3-Pane Grid)
if (st.session_state.current_pdf and 
    Path(st.session_state.current_pdf).stem in st.session_state.json_data):
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                padding: 2rem; border-radius: 15px; margin: 2rem 0;'>
        <h2 style='text-align: center; color: #856404; margin-bottom: 1.5rem;'>
            📝 Your Feedback Matters - Help Us Improve
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    file_stem = Path(st.session_state.current_pdf).stem
    json_data = st.session_state.json_data[file_stem]
    render_feedback_form(st.session_state.current_pdf, file_stem, json_data)
```

## 5. Update Feedback Form for Horizontal Layout

**Update the `render_feedback_form` function to use horizontal layout:**

```python
def render_feedback_form(pdf_name, file_stem, json_data):
    """Render feedback form for a specific PDF in horizontal layout"""
    feedback_key = f"feedback_{file_stem}"
    if st.session_state.feedback_submitted.get(feedback_key, False):
        st.success("✅ Thank you! Your feedback has been submitted for this document.")
        if st.button("Submit New Feedback", key=f"new_feedback_{file_stem}"):
            st.session_state.feedback_submitted[feedback_key] = False
            st.rerun()
        return
    
    st.write("Help us improve our analysis by providing feedback on the results:")
    
    with st.form(f"feedback_form_{file_stem}"):
        # Horizontal layout with 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Form Number Analysis**")
            form_number_correct = st.selectbox(
                "Is the form number correct?",
                ["Select...", "Yes, correct", "No, incorrect", "Not applicable"],
                key=f"form_correct_{file_stem}"
            )
            form_number_feedback = st.text_area(
                "Comments",
                placeholder="Details...",
                height=80,
                key=f"form_feedback_{file_stem}"
            )
        
        with col2:
            st.write("**PI Clause Detection**")
            pi_clause_correct = st.selectbox(
                "Is PI clause detection accurate?",
                ["Select...", "Yes, accurate", "No, missed clauses", "No, false positives", "Not applicable"],
                key=f"pi_correct_{file_stem}"
            )
            pi_clause_feedback = st.text_area(
                "Comments",
                placeholder="Details...",
                height=80,
                key=f"pi_feedback_{file_stem}"
            )
        
        with col3:
            st.write("**CI Clause Detection**")
            ci_clause_correct = st.selectbox(
                "Is CI clause detection accurate?",
                ["Select...", "Yes, accurate", "No, missed clauses", "No, false positives", "Not applicable"],
                key=f"ci_correct_{file_stem}"
            )
            ci_clause_feedback = st.text_area(
                "Comments",
                placeholder="Details...",
                height=80,
                key=f"ci_feedback_{file_stem}"
            )
        
        with col4:
            st.write("**Summary Quality**")
            summary_quality = st.selectbox(
                "Rate summary quality",
                ["Select...", "Excellent", "Good", "Fair", "Poor"],
                key=f"summary_quality_{file_stem}"
            )
            summary_feedback = st.text_area(
                "Comments",
                placeholder="Details...",
                height=80,
                key=f"summary_feedback_{file_stem}"
            )
        
        # Bottom section for general feedback and rating
        st.write("**Overall Assessment**")
        col5, col6 = st.columns([3, 1])
        with col5:
            general_feedback = st.text_area(
                "General Comments",
                placeholder="Any other comments, suggestions, or issues you noticed?",
                height=60,
                key=f"general_feedback_{file_stem}"
            )
        with col6:
            rating = st.slider(
                "Overall Rating", 
                1, 5, 3, 
                help="1=Poor, 5=Excellent",
                key=f"rating_{file_stem}"
            )
            st.write(f"Rating: {'⭐' * rating}")
        
        # Submit button - full width
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        # Rest of the submission logic remains the same...
```

## Summary of Changes:

1. **Removed spinners from between buttons and grid** - moved to Processing Status section
2. **Single column processing display** - all processing info in one column below PDF list
3. **Dropdown processing details** - using `st.expander` for collapsible details
4. **Horizontal feedback form** - moved below 3-pane grid with 4-column layout
5. **Cleaner separation** - processing UI separate from selection UI​​​​​​​​​​​​​​​​