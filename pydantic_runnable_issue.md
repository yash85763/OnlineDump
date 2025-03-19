Let's fix the layout so the icons appear next to the question and answer text properly:

```python
# Add this CSS to your styles
st.markdown("""
<style>
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 20px;
    }
    .icon {
        background-color: #6c757d;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .bot-icon {
        background-color: #0d6efd;
    }
    .message-content {
        flex-grow: 1;
    }
    .previous-answer {
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

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
                    # User question with icon
                    st.markdown("""
                    <div class="message-container">
                        <div class="icon">👤</div>
                        <div class="message-content">
                            <strong>Question:</strong> {0}
                        </div>
                    </div>
                    """.format(question), unsafe_allow_html=True)
                    
                    # Display the section info if available
                    if qa_pair['section'] and qa_pair['section'] != "All Sections":
                        section = qa_pair['section']
                        section_description = section_mapping.get(section, "")
                        st.markdown(f"<div style='margin-left: 42px; font-style: italic;'>Section: {section} {section_description}</div>", unsafe_allow_html=True)
                    
                    # Display answer with bot icon
                    st.markdown("""
                    <div class="message-container">
                        <div class="icon bot-icon">🤖</div>
                        <div class="message-content previous-answer">
                            <strong>Answer:</strong><br/>
                            {0}
                        </div>
                    </div>
                    """.format(qa_pair['answer'].replace('\n', '<br/>')), unsafe_allow_html=True)
```

And for the main display:

```python
# Right column for displaying the current Q&A
with col2:
    # Use st.empty() to completely replace previous content
    right_col = st.empty()
    
    # Reset all content in the right column
    with right_col.container():
        if st.session_state['current_qa']['question']:
            # Display user question with icon
            st.markdown("""
            <div class="message-container">
                <div class="icon">👤</div>
                <div class="message-content">
                    <strong>Question:</strong> {0}
                </div>
            </div>
            """.format(st.session_state['current_qa']['question']), unsafe_allow_html=True)
            
            # Display the section info if available
            if st.session_state['current_qa'].get('section') and st.session_state['current_qa']['section'] != "All Sections":
                section = st.session_state['current_qa']['section']
                section_description = section_mapping.get(section, "")
                st.markdown(f"<div style='margin-left: 42px; font-style: italic;'>Section: {section} {section_description}</div>", unsafe_allow_html=True)
            
            # Display answer with bot icon
            is_previous = st.session_state['current_qa'].get('is_previous', False)
            prev_class = " previous-answer" if is_previous else ""
            
            st.markdown("""
            <div class="message-container">
                <div class="icon bot-icon">🤖</div>
                <div class="message-content{1}">
                    <strong>Answer:</strong><br/>
                    {0}
                </div>
            </div>
            """.format(st.session_state['current_qa'].get('answer', '').replace('\n', '<br/>'), prev_class), unsafe_allow_html=True)
        else:
            # Initial message
            st.markdown("""
            <div class="message-container">
                <div class="icon bot-icon">🤖</div>
                <div class="message-content">
                    Select a question and click Submit, or click on a previous question from the list.
                </div>
            </div>
            """, unsafe_allow_html=True)
```

And for the submit process:

```python
# First, update the right column to show the question and a loading indicator
with col2:
    right_col = st.empty()
    with right_col.container():
        # Display user question with icon
        st.markdown("""
        <div class="message-container">
            <div class="icon">👤</div>
            <div class="message-content">
                <strong>Question:</strong> {0}
            </div>
        </div>
        """.format(question), unsafe_allow_html=True)
        
        # Display the section info if available
        if selected_section and selected_section != "All Sections":
            section_description = section_mapping.get(selected_section, "")
            st.markdown(f"<div style='margin-left: 42px; font-style: italic;'>Section: {selected_section} {section_description}</div>", unsafe_allow_html=True)
        
        # Show loading indicator with bot icon
        st.markdown("""
        <div class="message-container">
            <div class="icon bot-icon">🤖</div>
            <div class="message-content">
                <strong>Answer:</strong><br/>
                <div class="loading">Generating answer...</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# After processing
with right_col.container():
    # Display user question with icon
    st.markdown("""
    <div class="message-container">
        <div class="icon">👤</div>
        <div class="message-content">
            <strong>Question:</strong> {0}
        </div>
    </div>
    """.format(question), unsafe_allow_html=True)
    
    # Display the section info if available
    if selected_section and selected_section != "All Sections":
        section_description = section_mapping.get(selected_section, "")
        st.markdown(f"<div style='margin-left: 42px; font-style: italic;'>Section: {selected_section} {section_description}</div>", unsafe_allow_html=True)
    
    # Display final answer with bot icon
    st.markdown("""
    <div class="message-container">
        <div class="icon bot-icon">🤖</div>
        <div class="message-content">
            <strong>Answer:</strong><br/>
            {0}
        </div>
    </div>
    """.format(final_answer.replace('\n', '<br/>')), unsafe_allow_html=True)
```

This approach:

1. Uses CSS flexbox to align the icon and content horizontally
2. Keeps the icon and text together in the same container
3. Handles line breaks in the text 
4. Provides consistent styling for all messages
5. Still maintains the translucent effect for previous answers

The icons should now appear properly next to their respective text rather than above it.​​​​​​​​​​​​​​​​