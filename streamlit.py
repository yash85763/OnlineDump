import streamlit as st
import json
import os
import re
from utils.llm_processor import LLMProcessor
from ecfr_api_wrapper import ECFRAPIWrapper
from utils.configuration import (
    DATA_PATH,
    client_id,
    client_secret,
    rai_base_url,
    base_url,
    PROXIES,
    context_json_file_path,
    consolidated_ans_json_file_path,
    results_path_ext_cont
)

# Initialize LLMProcessor and ECFRAPIWrapper
llm_processor = LLMProcessor(
    rai_base_url,
    client_secret,
    client_id
)

api = ECFRAPIWrapper(
    base_url=base_url,
    client_secret=client_secret,
    client_id=client_id,
    rai_base_url=rai_base_url,
    proxies=PROXIES
)

# Load the context data
with open(os.path.join(context_json_file_path, 'context.json'), 'r') as f:
    context = json.load(f)

# Load the Consolidated data
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
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error loading consolidated data: {e}")
        consolidated_data = []

# Load the QA data
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
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error loading QA data: {e}")
        qa_data = []

# Function to parse citations and convert them to eCFR links
def format_answer_with_links(answer, title="12", part="211"):
    """
    Parse citations like [211.1] or [211.2(a)] in the answer and replace them with eCFR links.
    
    Args:
        answer (str): The answer string containing citations.
        title (str): The eCFR title (default: "12").
        part (str): The eCFR part (default: "211").
    
    Returns:
        str: The answer with citations replaced by Markdown links.
    """
    # Regular expression to match citations like [211.1] or [211.2(a)]
    citation_pattern = r'\[(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:,\s*\d+\.\d+(?:\([a-zA-Z0-9]+\))?)*)\]'
    
    def create_link(match):
        # Extract the citation text (e.g., "211.1" or "211.2(a)")
        citation_group = match.group(1)
        citations = citation_group.split(", ")  # Handle multiple citations like [211.1, 211.2]
        
        links = []
        for citation in citations:
            # Split into section and subsection (if any)
            if "(" in citation:
                section, subsection = citation.split("(")
                subsection = subsection.rstrip(")")
                section_url = f"{section}#{section}({subsection})"
            else:
                section = citation
                section_url = section
            
            # Construct the eCFR URL
            ecfr_url = f"https://www.ecfr.gov/current/title-{title}/part-{part}/section-{section_url}"
            # Create a Markdown link
            link = f"[{citation}]({ecfr_url})"
            links.append(link)
        
        # Join multiple links with commas
        return ", ".join(links)
    
    # Replace all citations with links
    formatted_answer = re.sub(citation_pattern, create_link, answer)
    return formatted_answer

# Create a Streamlit app
st.title("Regulation Assistant: As of 2025-03-06")

# Create a session state to store the chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a dropdown menu to select the regulation
selected_regulation = st.selectbox("Select a regulation", ["Regulation k"], index=0)

# Create a dropdown menu to select the section
sections = list(context.keys())
selected_section = st.selectbox("Select query scope", ["ALL Sections"] + sections, index=0)

# Initialize session state for the user question
if 'custom_question' not in st.session_state:
    st.session_state.custom_question = ''

# Initialize session state for the submit button
if 'submit_button_disabled' not in st.session_state:
    st.session_state.submit_button_disabled = True

# Predefined questions
question_options = [
    'What does this regulation require our bank to do?',
    'What does this regulation prohibit our bank from doing?',
    'What does this regulation permit our bank to do? (drawing a distinction here between something that is permitted vs. other...)',
    'Other...'
]

selected_question = st.selectbox(
    'Select a question',
    question_options,
    index=0  # Default to first option
)

if selected_question == 'Other...':
    st.session_state.custom_question = st.text_input(
        'Enter your question',
        value=st.session_state.custom_question
    )

    if not st.session_state.custom_question:
        st.warning('Please enter your question or select a predefined one')
        st.session_state.submit_button_disabled = True
        st.stop()
    else:
        question = st.session_state.custom_question
        st.session_state.submit_button_disabled = False
else:
    question = selected_question
    st.session_state.submit_button_disabled = False

# Function to find the best-matching question in consolidated_data
def find_best_match(user_question, consolidated_data):
    for qa in consolidated_data:
        if qa["question"].strip().lower() == user_question.strip().lower():
            return qa["answer"]
    return None  # Return None if no exact match is found

# Create a button to submit the question
if st.button("Submit", key="submit_button", disabled=st.session_state.submit_button_disabled):
    with st.spinner("Processing your question..."):
        final_answer = None
        if selected_section == "ALL Sections":
            # Check if the question exists in consolidated_data
            answer = find_best_match(question, consolidated_data)
            if answer:
                final_answer = answer
            else:
                # Process each section individually to avoid context length issues
                section_answers = []
                for section, section_content in context.items():
                    answers_for_section = llm_processor.answer_from_sections(section_content, [question])
                    if answers_for_section:
                        section_answers.append(answers_for_section[0])
                
                # Consolidate the answers from all sections
                if section_answers:
                    final_answer = llm_processor.consolidator(question, section_answers)
                else:
                    final_answer = "No answers found from any section."

                # Update consolidated_data with the new question-answer pair
                consolidated_data.append({
                    "question": question,
                    "answer": final_answer
                })
                with open(consolidated_data_file, 'w') as f:
                    json.dump(consolidated_data, f, indent=4)
        else:
            # Use answer_from_sections for the specific section
            final_answer = llm_processor.answer_from_sections(context[selected_section], [question])[0]

        # Format the answer with clickable eCFR links
        formatted_answer = format_answer_with_links(final_answer, title="12", part="211")

        # Display the answer using st.markdown to render links
        st.markdown(f"**Answer:** {formatted_answer}", unsafe_allow_html=True)

        # Update chat history (store the formatted answer)
        chat_history = st.session_state.chat_history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": formatted_answer})
        st.session_state.chat_history = chat_history

        # Save the QA data (store the original answer without Markdown links)
        qa_data.append({
            "question": question,
            "section": selected_section,
            "answer": final_answer  # Store the raw answer, not the formatted one
        })
        with open(qa_data_file, 'w') as f:
            json.dump(qa_data, f, indent=4)

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.write(message["content"])