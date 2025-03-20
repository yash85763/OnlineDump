You're right, the Streamlit chat message components are much cleaner and purpose-built for chat interfaces. Let's use those instead:

```python
# Function to handle question click
def handle_question_click(idx):
    if idx < len(st.session_state['previous_questions']):
        question = st.session_state['previous_questions'][idx]
        if question in st.session_state['previous_qa_pairs']:
            qa_pair = st.session_state['previous_qa_pairs'][question]
            st.session_state['current_qa'] = {
                "question": question,
                "answer": qa_pair['answer'],
                "section": qa_pair['section'],
                "is_previous": True  # Mark this as a previous answer
            }
            
            # Immediately update the display in the right column
            with col2:
                right_col = st.empty()
                with right_col.container():
                    # Display the question as a user message
                    with st.chat_message("user"):
                        st.write(question)
                    
                    # Display the section info if available
                    if qa_pair['section'] and qa_pair['section'] != "All Sections":
                        section = qa_pair['section']
                        section_description = section_mapping.get(section, "")
                        st.info(f"Section: {section} {section_description}")
                    
                    # Display the answer as an assistant message
                    with st.chat_message("assistant"):
                        st.write(qa_pair['answer'])
                        if st.session_state['current_qa']['is_previous']:
                            st.caption("Previous answer")
```

For the main display:

```python
# Right column for displaying the current Q&A
with col2:
    # Use st.empty() to completely replace previous content
    right_col = st.empty()
    
    # Reset all content in the right column
    with right_col.container():
        if st.session_state['current_qa']['question']:
            # Display the question as a user message
            with st.chat_message("user"):
                st.write(st.session_state['current_qa']['question'])
            
            # Display the section info if available
            if st.session_state['current_qa'].get('section') and st.session_state['current_qa']['section'] != "All Sections":
                section = st.session_state['current_qa']['section']
                section_description = section_mapping.get(section, "")
                st.info(f"Section: {section} {section_description}")
            
            # Display the answer as an assistant message
            with st.chat_message("assistant"):
                st.write(st.session_state['current_qa'].get('answer', ''))
                if st.session_state['current_qa'].get('is_previous', False):
                    st.caption("Previous answer")
        else:
            # Initial message
            with st.chat_message("assistant"):
                st.write("Select a question and click Submit, or click on a previous question from the list.")
```

And for the submission process:

```python
# First, update the right column to show the question and a loading indicator
with col2:
    right_col = st.empty()
    with right_col.container():
        # Display the question as a user message
        with st.chat_message("user"):
            st.write(question)
        
        # Display the section info if available
        if selected_section and selected_section != "All Sections":
            section_description = section_mapping.get(selected_section, "")
            st.info(f"Section: {selected_section} {section_description}")
        
        # Show a loading message as an assistant message
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                # Process the question and get the answer
                # (The actual processing code remains here)
                pass

# After processing, update the display with the final answer
with right_col.container():
    # Display the question as a user message
    with st.chat_message("user"):
        st.write(question)
    
    # Display the section info if available
    if selected_section and selected_section != "All Sections":
        section_description = section_mapping.get(selected_section, "")
        st.info(f"Section: {selected_section} {section_description}")
    
    # Display the final answer as an assistant message
    with st.chat_message("assistant"):
        st.write(final_answer)
```

This approach:

1. Uses the native Streamlit `st.chat_message()` function for proper chat formatting
2. Keeps the user and assistant messages visually distinct
3. Adds a "Previous answer" caption to indicate previous answers
4. Uses `st.info()` for section information which gives it a subtle highlight
5. Handles loading state using Streamlit's built-in spinner

The Streamlit chat message component should automatically handle:
- Proper spacing between messages
- Avatar/icon display
- Message styling 
- Markdown rendering within the messages

This should give you a much more reliable display of chat messages with proper formatting of markdown content.​​​​​​​​​​​​​​​​