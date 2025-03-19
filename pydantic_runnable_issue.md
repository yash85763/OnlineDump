You're right, let's simplify and use Streamlit's built-in markdown rendering instead of trying to convert markdown to HTML manually. Here's a much simpler solution:

```python
# Function to handle question click - ensure this triggers an immediate display update
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
                    # Question with user icon (using HTML for icon only)
                    st.markdown(f'<div class="user-icon">👤</div>', unsafe_allow_html=True)
                    st.markdown(f"**Question:** {question}")
                    
                    # Display the section info if available
                    if qa_pair['section'] and qa_pair['section'] != "All Sections":
                        section = qa_pair['section']
                        section_description = section_mapping.get(section, "")
                        st.markdown(f"*Section: {section} {section_description}*")
                    
                    # Bot icon (using HTML for icon only)
                    st.markdown(f'<div class="bot-icon">🤖</div>', unsafe_allow_html=True)
                    
                    # Display the answer directly as markdown
                    st.markdown(f"**Answer:**")
                    st.markdown(qa_pair['answer'])
```

And similarly, update the right column display:

```python
# Right column for displaying the current Q&A
with col2:
    # Use st.empty() to completely replace previous content
    right_col = st.empty()
    
    # Reset all content in the right column
    with right_col.container():
        if st.session_state['current_qa']['question']:
            # Question with user icon (using HTML for icon only)
            st.markdown(f'<div class="user-icon">👤</div>', unsafe_allow_html=True)
            st.markdown(f"**Question:** {st.session_state['current_qa']['question']}")
            
            # Display the section info if available
            if st.session_state['current_qa'].get('section') and st.session_state['current_qa']['section'] != "All Sections":
                section = st.session_state['current_qa']['section']
                section_description = section_mapping.get(section, "")
                st.markdown(f"*Section: {section} {section_description}*")
            
            # Bot icon (using HTML for icon only)
            st.markdown(f'<div class="bot-icon">🤖</div>', unsafe_allow_html=True)
            
            # Answer as direct markdown
            st.markdown(f"**Answer:**")
            
            # If it's a previous answer, use a container with custom styling
            is_previous = st.session_state['current_qa'].get('is_previous', False)
            if is_previous:
                with st.container():
                    st.markdown(
                        """<style>
                        .previous-answer {opacity: 0.7;}
                        </style>""",
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<div class="previous-answer">{st.session_state["current_qa"].get("answer", "")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(st.session_state['current_qa'].get('answer', ''))
        else:
            # Initial message
            st.markdown(f'<div class="bot-icon">🤖</div>', unsafe_allow_html=True)
            st.markdown("Select a question and click Submit, or click on a previous question from the list.")
```

And the update for the processing part:

```python
# Update the answer display with the final answer
with right_col.container():
    # Question with user icon (using HTML for icon only)
    st.markdown(f'<div class="user-icon">👤</div>', unsafe_allow_html=True)
    st.markdown(f"**Question:** {question}")
    
    # Display the section info if available
    if selected_section and selected_section != "All Sections":
        section_description = section_mapping.get(selected_section, "")
        st.markdown(f"*Section: {selected_section} {section_description}*")
    
    # Bot icon (using HTML for icon only)
    st.markdown(f'<div class="bot-icon">🤖</div>', unsafe_allow_html=True)
    
    # Display the final answer as markdown
    st.markdown(f"**Answer:**")
    st.markdown(final_answer)
```

This approach:

1. Uses Streamlit's native markdown rendering for the question and answer content
2. Only uses HTML for the icons and minimal styling
3. Preserves all formatting, links, and markdown structure in the original content
4. Creates a translucent effect for previous answers using CSS opacity

This should render your markdown content correctly including links, numbered lists, and other formatting.​​​​​​​​​​​​​​​​