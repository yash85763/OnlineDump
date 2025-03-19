import streamlit as st
import json
import os
import concurrent.futures
from functools import partial
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# File paths
DATA_PATH = "path/to/data"  # Adjust as needed
context_json_file_path = os.path.join(DATA_PATH, "context")
consolidated_ans_json_file_path = os.path.join(DATA_PATH, "consolidated")

# Function to get latest regulation data (placeholder - implement your actual function)
def get_latest_regulation_data():
    # Your implementation here
    return datetime.now().strftime("%Y-%m-%d")

# Helper function to safely render markdown text
def safe_markdown(text):
    # Replace any characters that might break HTML rendering
    if text is None:
        return ""
    
    # Convert markdown to HTML safely
    import re
    
    # Escape HTML special characters first
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Process code blocks with syntax highlighting
    text = re.sub(r'```(\w*)\n(.*?)\n```', r'<pre><code class="language-\1">\2</code></pre>', text, flags=re.DOTALL)
    
    # Process inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Process bold text
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Process italic text
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Process bullet lists
    text = re.sub(r'^\s*\*\s(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = text.replace('<li>', '<ul><li>').replace('</li>', '</li></ul>')
    
    # Process numbered lists
    text = re.sub(r'^\s*(\d+)\.\s(.+)$', r'<li>\2</li>', text, flags=re.MULTILINE)
    text = text.replace('<li>', '<ol><li>').replace('</li>', '</li></ol>')
    
    # Process headers
    text = re.sub(r'^###\s(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Process paragraphs - split by double newlines and wrap in <p> tags
    paragraphs = []
    for p in text.split('\n\n'):
        if p.strip() and not (p.startswith('<h') or p.startswith('<ul>') or p.startswith('<ol>')):
            paragraphs.append(f'<p>{p}</p>')
        else:
            paragraphs.append(p)
    
    text = '\n'.join(paragraphs)
    
    # Replace newlines with <br> tags inside paragraphs
    paragraphs = []
    for p in text.split('</p>'):
        if p.startswith('<p>'):
            p = p.replace('\n', '<br>')
        paragraphs.append(p)
    
    text = '</p>'.join(paragraphs)
    
    return text

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
    st.session_state['current_qa'] = {"question": "", "answer": "", "section": "", "is_previous": False}
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
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .user-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .assistant-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 20px;
    }
    .assistant-message-previous {
        display: flex;
        align-items: flex-start;
        margin-bottom: 20px;
        opacity: 0.7;
    }
    .user-icon {
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
    .message-content {
        padding: 10px;
        border-radius: 5px;
        max-width: calc(100% - 50px);
        overflow-wrap: break-word;
    }
    .user-message .message-content {
        background-color: #f1f1f1;
    }
    .assistant-message .message-content, .assistant-message-previous .message-content {
        background-color: #e6f7ff;
    }
    .section-info {
        font-style: italic;
        margin: 5px 0 15px 42px;
        color: #555;
    }
    .loading {
        display: flex;
        align-items: center;
    }
    .loading:after {
        content: "...";
        width: 24px;
        text-align: left;
        animation: dots 1.5s steps(5, end) infinite;
    }
    @keyframes dots {
        0%, 20% { content: ""; }
        40% { content: "."; }
        60% { content: ".."; }
        80%, 100% { content: "..."; }
    }
    
    /* Markdown formatting within the message content */
    .message-content p {
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }
    .message-content h1, .message-content h2, .message-content h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .message-content h1 {
        font-size: 1.5rem;
    }
    .message-content h2 {
        font-size: 1.25rem;
    }
    .message-content h3 {
        font-size: 1.1rem;
    }
    .message-content ul, .message-content ol {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
    }
    .message-content li {
        margin-bottom: 0.25rem;
    }
    .message-content code {
        font-family: monospace;
        background-color: rgba(0,0,0,0.05);
        padding: 0.1rem 0.2rem;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .message-content pre {
        background-color: rgba(0,0,0,0.05);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        overflow-x: auto;
    }
    .message-content pre code {
        background-color: transparent;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("Regulation K Interpreter")

# Create two columns with custom widths
col1, col2 = st.columns([3, 7])

# Add scrolling to col2 and ensure content doesn't compound
st.markdown("""
<style>
    [data-testid="column"]:nth-of-type(2) {
        height: 80vh;
        overflow-y: auto;
    }
    /* Clear all contents in the right column on rerun */
    [data-testid="column"]:nth-of-type(2) > div {
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

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
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    
                    # Question with user icon
                    st.markdown(f'''
                    <div class="user-message">
                        <div class="user-icon">👤</div>
                        <div class="message-content">
                            {question}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Display the section info if available
                    if qa_pair['section'] and qa_pair['section'] != "All Sections":
                        section = qa_pair['section']
                        section_description = section_mapping.get(section, "")
                        st.markdown(f'<div class="section-info">Section: {section} {section_description}</div>', unsafe_allow_html=True)
                    
                    # Answer with bot icon (previous style) - use safe_markdown for the answer
                    st.markdown(f'''
                    <div class="assistant-message-previous">
                        <div class="bot-icon">🤖</div>
                        <div class="message-content">
                            {safe_markdown(qa_pair['answer'])}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

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
    # Use st.empty() to completely replace previous content
    right_col = st.empty()
    
    # Reset all content in the right column
    with right_col.container():
        if st.session_state['current_qa']['question']:
            # Display only the current Q&A in chat format with icons
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Question with user icon
            st.markdown(f'''
            <div class="user-message">
                <div class="user-icon">👤</div>
                <div class="message-content">
                    {st.session_state["current_qa"]["question"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display the section info if available
            if st.session_state['current_qa'].get('section') and st.session_state['current_qa']['section'] != "All Sections":
                section = st.session_state['current_qa']['section']
                section_description = section_mapping.get(section, "")
                st.markdown(f'<div class="section-info">Section: {section} {section_description}</div>', unsafe_allow_html=True)
            
            # Answer with bot icon
            is_previous = st.session_state['current_qa'].get('is_previous', False)
            message_class = "assistant-message-previous" if is_previous else "assistant-message"
            
            st.markdown(f'''
            <div class="{message_class}">
                <div class="bot-icon">🤖</div>
                <div class="message-content">
                    {safe_markdown(st.session_state["current_qa"].get("answer", ""))}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Initial message
            st.markdown('''
            <div class="chat-container">
                <div class="assistant-message">
                    <div class="bot-icon">🤖</div>
                    <div class="message-content">
                        Select a question and click Submit, or click on a previous question from the list.
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

# Process new submission
if submit_clicked:
    # First, update the right column to show the question and a loading indicator
    with col2:
        right_col = st.empty()
        with right_col.container():
            # Display the question
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="user-message">
                <div class="user-icon">👤</div>
                <div class="message-content">
                    {question}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display the section info if available
            if selected_section and selected_section != "All Sections":
                section_description = section_mapping.get(selected_section, "")
                st.markdown(f'<div class="section-info">Section: {selected_section} {section_description}</div>', unsafe_allow_html=True)
            
            # Show a spinner in the answer spot
            answer_placeholder = st.empty()
            with answer_placeholder:
                st.markdown('''
                <div class="assistant-message">
                    <div class="bot-icon">🤖</div>
                    <div class="message-content">
                        <div class="loading">
                            Generating answer...
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the question
    if selected_section == "All Sections":
        answer = find_best_match(question, consolidated_data)
        if answer:
            final_answer = answer
        else:
            section_answers = []
            for section, section_content in context.items():
                try:
                    # Process each section sequentially
                    answers = llm_processor.answer_from_sections(section_content, [question])
                    if answers and len(answers) > 0:
                        section_answers.append(answers[0])
                except Exception as e:
                    st.error(f"Error processing section {section}: {e}")
            
            if section_answers:
                final_answer = llm_processor.consolidator(question, section_answers)
            else:
                final_answer = "No answers found from any section."
    else:
        # Process a single section
        try:
            answers = llm_processor.answer_from_sections(context[selected_section], [question])
            if answers and len(answers) > 0:
                final_answer = answers[0]
            else:
                final_answer = "No answer was generated for this question. Please try rephrasing it."
        except Exception as e:
            st.error(f"Error processing question: {e}")
            final_answer = "An error occurred while processing your question. Please try again."
    
    # Update the answer display with the final answer
    with right_col.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="user-message">
            <div class="user-icon">👤</div>
            <div class="message-content">
                {question}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display the section info if available
        if selected_section and selected_section != "All Sections":
            section_description = section_mapping.get(selected_section, "")
            st.markdown(f'<div class="section-info">Section: {selected_section} {section_description}</div>', unsafe_allow_html=True)
        
        # Display the final answer - use safe_markdown for proper formatting
        st.markdown(f'''
        <div class="assistant-message">
            <div class="bot-icon">🤖</div>
            <div class="message-content">
                {safe_markdown(final_answer)}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
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
        "section": selected_section,
        "is_previous": False  # This is a new answer
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