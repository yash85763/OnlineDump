import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional

# File paths
DATA_PATH = "path/to/data"  # Adjust as needed
context_json_file_path = os.path.join(DATA_PATH, "context")
consolidated_ans_json_file_path = os.path.join(DATA_PATH, "consolidated")

# Load section mapping (assuming this function is defined elsewhere)
# section_mapping = build_ecfr_section_mapping(ecfr_data)

# Load context data
with open(os.path.join(context_json_file_path, 'context.json'), 'r') as f:
    context = json.load(f)

# Load consolidated data with error handling
consolidated_data_file = os.path.join(consolidated_ans_json_file_path, 'consolidated_data.json')
if not os.path.exists(consolidated_data_file):
    with open(consolidated_data_file, 'w') as f:
        json.dump([], f)
    consolidated_data = []
else:
    try:
        with open(consolidated_data_file, 'r') as f:
            consolidated_data = json.load(f)
            if not isinstance(consolidated_data, list):
                raise ValueError("Invalid JSON format: Root element must be a list")
    except (json.JsonDecodeError, ValueError) as e:
        st.error(f"Error loading consolidated data: {e}")
        consolidated_data = []

# Load QA data similarly
qa_data_file = os.path.join(DATA_PATH, 'qa_new_data.json')
if not os.path.exists(qa_data_file):
    with open(qa_data_file, 'w') as f:
        json.dump([], f)
    qa_data = []
else:
    try:
        with open(qa_data_file, 'r') as f:
            qa_data = json.load(f)
            if not isinstance(qa_data, list):
                raise ValueError("Invalid JSON format: Root element must be a list")
    except (json.JsonDecodeError, ValueError) as e:
        st.error(f"Error loading QA data: {e}")
        qa_data = []

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'custom_question' not in st.session_state:
    st.session_state['custom_question'] = ''
if 'submit_button_disabled' not in st.session_state:
    st.session_state['submit_button_disabled'] = True
if 'previous_questions' not in st.session_state:
    st.session_state['previous_questions'] = []
if 'previous_qa_pairs' not in st.session_state:
    st.session_state['previous_qa_pairs'] = {}  # Dictionary to store question-answer pairs
if 'last_selected_question' not in st.session_state:
    st.session_state['last_selected_question'] = None

# Add default questions to the list of available questions
default_question_options = [
    'What does this regulation require our bank to do?', 
    'What does this regulation prohibit our bank from doing?', 
    'What does this regulation permit our bank to do? (drawing a distinction here between something that is permitted vs. other...)', 
    'Other...'
]

# Function to find best match (placeholder)
def find_best_match(question, data):
    # Implement your matching logic here
    return None

# Main UI
st.title("Regulation As of 2025-03-06")

# Create two columns
col1, col2 = st.columns([1, 2])

# Left column for selection and inputs
with col1:
    selected_regulation = st.selectbox("Select a regulation", ["Regulation k"], index=0)

    sections = list(context.keys())
    selected_section = st.selectbox("Select query scope", ["All Sections"] + sections, index=0)
    
    # Combine default questions with previous questions from this session
    # Make sure we don't have duplicates
    all_questions = default_question_options.copy()
    for prev_q in st.session_state['previous_questions']:
        if prev_q not in all_questions and prev_q != "Other...":
            all_questions.insert(-1, prev_q)  # Insert before "Other..."
    
    # Question selection with callback
    def on_question_select():
        # Update the last selected question to detect changes
        st.session_state['last_selected_question'] = selected_question
        
    selected_question = st.selectbox(
        "Select a question", 
        all_questions, 
        index=0,
        on_change=on_question_select,
        key="question_selector"
    )
    
    # Check if user selected a previously asked question
    is_previous_question = selected_question in st.session_state['previous_qa_pairs']
    
    if selected_question == "Other...":
        st.session_state['custom_question'] = st.text_input("Enter your question", value=st.session_state['custom_question'])
        question = st.session_state['custom_question'] if st.session_state['custom_question'] else None
    else:
        question = selected_question
    
    submit_button_disabled = False if question else True
    
    # Submit button
    submit_clicked = st.button("Submit", key="submit_button", disabled=submit_button_disabled)
    
    # Show additional info for previous questions
    if is_previous_question and not submit_clicked:
        st.info("This question has been answered before. You can view the previous answer in the chat history or submit again for a fresh response.")

# Right column for chat history and answers
with col2:
    # Create two containers: one for current Q&A and one for history
    current_qa_container = st.container()
    
    st.markdown("### Previous Conversations")
    # Create a container with fixed height and scrolling for chat history
    chat_history_container = st.container()
    
    # Apply custom CSS for the scrollable container
    st.markdown("""
        <style>
        .history-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #f0f0f0;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Wrap the history container in a div with the custom class
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    
    # Display chat history in reverse order (most recent last)
    # We'll skip the most recent Q&A pair if it's not marked as previous
    history_to_display = []
    i = 0
    while i < len(st.session_state['chat_history']):
        if i < len(st.session_state['chat_history']) - 2:  # Check if we're not at the last pair
            history_to_display.append(st.session_state['chat_history'][i])
            history_to_display.append(st.session_state['chat_history'][i+1])
        i += 2  # Move to the next Q&A pair
    
    for msg in history_to_display:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    
    # Close the container div
    st.markdown('</div>', unsafe_allow_html=True)

# Check if a previously answered question was selected
if selected_question in st.session_state['previous_qa_pairs'] and selected_question != st.session_state.get('last_selected_question'):
    # Retrieve previous answer
    previous_pair = st.session_state['previous_qa_pairs'][selected_question]
    question = selected_question
    
    with col2:
        with current_qa_container:
            st.markdown("### Current Question & Answer")
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                st.write(previous_pair['answer'])
                st.caption("Previous answer - Select the question and press Submit to regenerate")
    
    # Add to chat history without duplicating in QA data
    st.session_state['chat_history'].append({"role": "user", "content": question})
    st.session_state['chat_history'].append({
        "role": "assistant", 
        "content": previous_pair['answer'],
        "is_previous": True
    })
    
    # Update last selected question to prevent retriggering
    st.session_state['last_selected_question'] = selected_question

# Process new submission
elif submit_clicked:
    # Add question to the previous questions list if it's not already there and not "Other..."
    if question and question != "Other..." and question not in st.session_state['previous_questions']:
        st.session_state['previous_questions'].append(question)
    
    with col2:
        with current_qa_container:
            st.markdown("### Current Question & Answer")
            with st.chat_message("user"):
                st.write(question)
            
            # Show a spinner while generating the answer
            with st.spinner(f"Interpreting section: {selected_section} {section_mapping.get(selected_section, '') if selected_section != 'All Sections' else ''} of eCFR"):
                if selected_section == "All Sections":
                    answer = find_best_match(question, consolidated_data)
                    if answer:
                        final_answer = answer
                    else:
                        section_answers = []
                        for section, section_content in context.items():
                            # Get section description
                            section_description = section_mapping.get(section, "")
                            with st.spinner(f"Interpreting section: {section} {section_description} of eCFR"):
                                answers_for_section = llm_processor.answer_from_sections(section_content, [question])
                                if answers_for_section:
                                    section_answers.append(answers_for_section[0])
                        if section_answers:
                            final_answer = llm_processor.consolidator(question, section_answers)
                        else:
                            final_answer = "No answers found from any section."
                else:
                    final_answer = llm_processor.answer_from_sections(context[selected_section], [question])[0]
            
            with st.chat_message("assistant"):
                st.write(final_answer)
    
    # Update session state
    st.session_state['chat_history'].append({"role": "user", "content": question})
    st.session_state['chat_history'].append({"role": "assistant", "content": final_answer})
    
    # Store in previous QA pairs dictionary
    st.session_state['previous_qa_pairs'][question] = {
        "section": selected_section,
        "answer": final_answer
    }
    
    # Save to QA data
    qa_data.append({
        "question": question,
        "section": selected_section,
        "answer": final_answer
    })
    with open(qa_data_file, 'w') as f:
        json.dump(qa_data, f, indent=4)
    
    # Clear custom question if it was used
    if selected_question == "Other...":
        st.session_state['custom_question'] = ''
    
    # Update last selected question
    st.session_state['last_selected_question'] = selected_question
    
    # No need to force rerun as we're handling the display directly