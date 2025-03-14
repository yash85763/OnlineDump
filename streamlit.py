import streamlit as st
import json
from utils.llm_processor import LLMProcessor
from ecfr_api_wrapper import ECFRAPIWrapper
import os

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
else:
    with open(consolidated_data_file, 'r') as f:
        consolidated_data = json.load(f)

# Create a Streamlit app
st.title(f"Regulation Assistant: As of 2025-03-06")  # This is the last eCFR updated version, currently not using the API.

# Create a session state to store the chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a dropdown menu to select the section
sections = list(context.keys())
section_option = st.selectbox("Select a section", sections + ["All Sections"])

# Create a text input to ask questions
question = st.text_input("Ask a question")

# Function to find the best-matching question in consolidated_data
def find_best_match(user_question, consolidated_data):
    for qa in consolidated_data:
        if qa["question"].lower() == user_question.lower():
            return qa["answer"]
    return None  # Return None if no exact match is found

# Create a button to submit the question
if st.button("Submit", key="submit_button"):
    with st.spinner("Processing your question..."):
        
        # If "All Sections" is selected, use the consolidated data
        if section_option == "All Sections":
            answer = find_best_match(question, consolidated_data)
            if answer:
                st.write(f"Answer: {answer}")
                st.write("=" * 80)

                chat_history = st.session_state.chat_history
                chat_history.append({
                    'role': 'user',
                    'content': question
                })
                chat_history.append({
                    'role': 'assistant',
                    'content': answer
                })
                st.session_state.chat_history = chat_history

                # Save the QA data
                qa_data.append({
                    'question': question,
                    'section': section_option,
                    'answer': answer
                })
            else:
                st.write("No matching answer found in the consolidated data.")

        else:
            # If a specific section is selected, use the existing approach
            answer = llm_processor.answer_from_sections(context[section_option], [question])[0]
            st.write(f"Section: {section_option}")
            st.write(f"Answer: {answer}")

            chat_history = st.session_state.chat_history
            chat_history.append({
                'role': 'user',
                'content': question
            })
            chat_history.append({
                'role': 'assistant',
                'content': answer
            })
            st.session_state.chat_history = chat_history

            # Save the QA data
            qa_data.append({
                'question': question,
                'section': section_option,
                'answer': answer
            })

        # Save the QA data to file
        with open(consolidated_data_file, 'w') as f:
            json.dump(qa_data, f)

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.write(message["content"])
        else:
            st.write(message["content"])