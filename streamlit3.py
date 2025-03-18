import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# File paths
DATA_PATH = "path/to/data"  # Adjust as needed
context_json_file_path = os.path.join(DATA_PATH, "context")
consolidated_ans_json_file_path = os.path.join(DATA_PATH, "consolidated")

# Function to get latest regulation data (placeholder - implement your actual function)
def get_latest_regulation_data():
    # Your implementation here
    return datetime.now().strftime("%Y-%m-%d")

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
if 'current_qa' not in st.session_state:
    st.session_state['current_qa'] = {"question": "", "answer": "", "section": ""}
if 'regulation_date' not in st.session_state:
    st.session_state['regulation_date'] = "2025-03-06"  # Default date

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

# Custom CSS for the layout
st.markdown("""
<style>
    .main-container {
        display: flex;
        flex-direction: row;
    }
    .stButton {
        margin-bottom: 5px;
    }
    .stButton button {
        width: 100%;
        text-align: left;
        padding: 8px;
        border-radius: 4px;
        background-color: #f8f9fa;
        border: 1px solid #eee;
    }
    .stButton button:hover {
        background-color: #f0f0f0;
    }
    .stButton button[data-active="true"] {
        background-color: #e6f3ff;
        border-left: 3px solid #2e74b5;
    }
    .scrollable-column {
        height: 80vh;
        overflow-y: auto;
    }
    .date-display {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    .user-message {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .section-info {
        font-style: italic;
        margin: 5px 0;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("Regulation K Interpreter")

# Create two columns with custom widths
col1, col2 = st.columns([3, 7])

# Add scrolling to col2
st.markdown("""
<style>
    [data-testid="column"]:nth-of-type(2) {
        height: 80vh;
        overflow-y: auto;
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
                "section": qa_pair['section']
            }

# Left column for selections and history
with col1:
    selected_regulation = st.selectbox("Select a regulation", ["Regulation K"], index=0)
    
    # Button to get latest regulation data
    if st.button("Get Latest Data"):
        st.session_state['regulation_date'] = get_latest_regulation_data()
    
    # Display regulation date
    st.markdown(f'<div class="date-display">As of: {st.session_state["regulation_date"]}</div>', unsafe_allow_html=True)
    
    sections = list(context.keys())
    selected_section = st.selectbox("Select query scope", ["All Sections"] + sections, index=0)
    
    # Combine default questions with previous questions from this session
    all_questions = default_question_options.copy()
    
    selected_question = st.selectbox("Select a question", all_questions, index=0)
    
    if selected_question == "Other...":
        st.session_state['custom_question'] = st.text_input("Enter your question", value=st.session_state['custom_question'])
        question = st.session_state['custom_question'] if st.session_state['custom_question'] else None
    else:
        question = selected_question
    
    submit_button_disabled = False if question else True
    
    # Submit button
    submit_clicked = st.button("Submit", key="submit_button", disabled=submit_button_disabled)
    
    # History section
    st.markdown("### Previous Questions")
    
    # Display previous questions as clickable items
    if st.session_state['previous_questions']:
        st.markdown("### Previous Questions")
        
        for i, prev_q in enumerate(st.session_state['previous_questions']):
            # Check if this is the current question being displayed
            is_active = st.session_state['current_qa'].get('question') == prev_q
            
            # Create a unique key for each question button using index
            question_key = f"question_btn_{i}"
            
            # Use a button with the question text
            if st.button(
                prev_q, 
                key=question_key,
                help="Click to view this previous question and answer",
                use_container_width=True
            ):
                handle_question_click(i)

# Right column for displaying the current Q&A
with col2:
    # Clear any previous content
    right_col_container = st.empty()
    
    # Display only the current Q&A
    with right_col_container.container():
        if st.session_state['current_qa']['question']:
            # Display the question
            st.markdown(f'<div class="user-message"><strong>Question:</strong> {st.session_state["current_qa"]["question"]}</div>', unsafe_allow_html=True)
            
            # Display the section info if available
            if st.session_state['current_qa']['section'] and st.session_state['current_qa']['section'] != "All Sections":
                section = st.session_state['current_qa']['section']
                section_description = section_mapping.get(section, "")
                st.markdown(f'<div class="section-info">Section: {section} {section_description}</div>', unsafe_allow_html=True)
            
            # Display the answer
            st.markdown(f'<div class="assistant-message"><strong>Answer:</strong> {st.session_state["current_qa"]["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="assistant-message">Select a question and click Submit, or click on a previous question from the list.</div>', unsafe_allow_html=True)

# Process new submission
if submit_clicked:
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
    
    # Add question to the previous questions list if it's not already there and not "Other..."
    if question and question != "Other..." and question not in st.session_state['previous_questions']:
        st.session_state['previous_questions'].append(question)
    
    # Store in previous QA pairs dictionary
    st.session_state['previous_qa_pairs'][question] = {
        "section": selected_section,
        "answer": final_answer
    }
    
    # Update current QA
    st.session_state['current_qa'] = {
        "question": question,
        "answer": final_answer,
        "section": selected_section
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
    
    # Display the answer in col2 immediately after submission
    with col2:
        # Clear any previous content and create a fresh container
        right_col_container = st.empty()
        
        with right_col_container.container():
            # Display the question
            st.markdown(f'<div class="user-message"><strong>Question:</strong> {question}</div>', unsafe_allow_html=True)
            
            # Display the section info if available
            if selected_section and selected_section != "All Sections":
                section_description = section_mapping.get(selected_section, "")
                st.markdown(f'<div class="section-info">Section: {selected_section} {section_description}</div>', unsafe_allow_html=True)
            
            # Display the answer
            st.markdown(f'<div class="assistant-message"><strong>Answer:</strong> {final_answer}</div>', unsafe_allow_html=True)
        
    # Use rerun only if we need to update the UI elements outside of col2
    # st.rerun()

# Remove JavaScript which is not needed with the new button approach
# st.markdown("""
# <script>
#     document.addEventListener('DOMContentLoaded', function() {
#         const historyItems = document.querySelectorAll('.history-item');
#         historyItems.forEach(item => {
#             item.addEventListener('click', function() {
#                 const questionId = this.id;
#                 // Use Streamlit's postMessage to communicate with Python
#                 window.parent.postMessage({
#                     type: 'streamlit:setComponentValue',
#                     value: questionId
#                 }, '*');
#             });
#         });
#     });
# </script>
# """, unsafe_allow_html=True)