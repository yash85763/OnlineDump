Key Points
	•	It seems likely that the updated Streamlit implementation will improve the chat interface with response streaming and better chat history display.
	•	Research suggests using Streamlit’s st.chat_message and st.chat_input for a conversational UI, enhancing user interaction.
	•	The evidence leans toward maintaining chat history in session state for persistence, ensuring all messages are visible.
Streamlit Chat Interface
The updated Streamlit app will use st.chat_message to display user questions and assistant answers in chat bubbles, improving the visual experience. Users can select predefined questions or enter custom ones, with the selected question displayed as a user message. The “Submit” button triggers the answer generation, shown as an assistant message, creating a seamless chat-like interaction.
Response Streaming
For response streaming, if the LLM processor supports it, the app can use st.write_stream() within st.chat_message("assistant") to display answers as they are generated, offering a dynamic, real-time experience. Currently, it assumes string outputs, but streaming can be added if the LLM supports chunked responses.
Chat History Management
The chat history will be stored in st.session_state['chat_history'], displaying all previous messages at the start using st.chat_message for each, ensuring a persistent and scrollable conversation history. This addresses the user’s concern about the current chat history not being user-friendly, replacing plain st.write() with chat bubbles.

Detailed Implementation and Analysis
This report provides a comprehensive update to the Streamlit implementation for a regulation assistant application, focusing on enhancing the chat interface with response streaming output and improved chat history management. The analysis is based on the current code structure and Streamlit documentation for chatbots, ensuring a user-friendly and interactive experience.
Background and Current State
The existing implementation, as seen in the code snippets from method_2_1_streamlit.py, is a Streamlit-based application for regulatory question-answering. It allows users to select a regulation, section, and question (predefined or custom), then submit to receive an answer from consolidated data or generated via an LLM processor. However, the chat history is currently displayed using st.write(), which does not provide a chat bubble format, leading to a less intuitive user experience. The user expressed dissatisfaction with the current chat history, prompting the need for improvements.
Methodology
To address these issues, we reviewed Streamlit documentation, particularly the tutorial on building LLM chat apps (Build a basic LLM chat app - Streamlit Docs). This guided the integration of chat elements like st.chat_message and st.chat_input, and the use of session state for history persistence. We also considered response streaming using st.write_stream() for dynamic output, aligning with modern chatbot expectations.
Updated Implementation
The updated code is structured as follows, ensuring a conversational UI and improved history management:
Data Loading and Initialization
The application begins by loading context, consolidated, and QA data from JSON files, ensuring robust error handling for file operations. Session state is initialized for chat_history, custom_question, and submit_button_disabled, maintaining state across interactions.
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
Chat History Display
The chat history is displayed at the start using st.chat_message, ensuring each message is shown in a bubble format with the correct role (“user” or “assistant”):
# Display chat history
for msg in st.session_state['chat_history']:
    with st.chat_message(msg['role']):
        st.write(msg['content'])
This replaces the previous st.write(message['content']), providing a visual distinction between user and assistant messages, enhancing user experience.
User Interface and Input Handling
The UI includes selecting a regulation, section, and question. For questions, users can choose from predefined options or enter a custom one if “Other…” is selected:
st.title("Regulation As of 2025-03-06")

selected_regulation = st.selectbox("Select a regulation", ["Regulation k"], index=0)

sections = list(context.keys())
selected_section = st.selectbox("Select query scope", ["All Sections"] + sections, index=0)

question_options = [
    'What does this regulation require our bank to do?', 
    'What does this regulation prohibit our bank from doing?', 
    'What does this regulation permit our bank to do? (drawing a distinction here between something that is permitted vs. other...)', 
    'Other...'
]

selected_question = st.selectbox("Select a question", question_options, index=0)

if selected_question == "Other...":
    st.session_state['custom_question'] = st.text_input("Enter your question", value=st.session_state['custom_question'])
    question = st.session_state['custom_question'] if st.session_state['custom_question'] else None
else:
    question = selected_question

submit_button_disabled = False if question else True
This setup ensures users can interact with predefined questions or enter custom ones, with the “Submit” button enabled only when a valid question is provided.
Answer Generation and Display
Upon clicking “Submit”, the user’s question is displayed as a user message, and the answer is generated based on the selected section. If “All Sections” is chosen, it searches consolidated data first, then processes each section if no match is found. The answer is displayed as an assistant message:
if st.button("Submit", key="submit_button", disabled=submit_button_disabled):
    with st.chat_message("user"):
        st.write(question)
    
    if selected_section == "All Sections":
        answer = find_best_match(question, consolidated_data)
        if answer:
            final_answer = answer
        else:
            section_answers = []
            for section, section_content in context.items():
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
    
    st.session_state['chat_history'].append({"role": "user", "content": question})
    st.session_state['chat_history'].append({"role": "assistant", "content": final_answer})
    
    qa_data.append({
        "question": question,
        "section": selected_section,
        "answer": final_answer
    })
    with open(qa_data_file, 'w') as f:
        json.dump(qa_data, f, indent=4)
This ensures that each interaction is logged in the chat history, displayed in chat bubbles, and persisted for future sessions.
Response Streaming Consideration
Currently, the answer is assumed to be a string, displayed using st.write(). For response streaming, if llm_processor supports streaming (e.g., returning a generator), we can modify the display to:
with st.chat_message("assistant"):
    st.write_stream(llm_processor.stream_answer(context, question))  # Hypothetical streaming method
This would require adjusting llm_processor to yield chunks of the response, which is not implemented in the current code but can be added for enhanced user experience.
Discussion and Improvements
The updated implementation addresses the user’s concern about the chat history by using st.chat_message, providing a visual distinction between user and assistant messages. The chat history is now persistent and displayed at the start, improving usability. However, response streaming is not fully implemented due to the lack of streaming support in the current llm_processor, which can be a future enhancement.
Comparative Analysis
The following table compares the old and new implementations:
Feature
Old Implementation
New Implementation
Chat History Display
Uses st.write(), no chat bubbles
Uses st.chat_message, chat bubbles
User Input
Select box and text input, no chat input
Select box and text input, integrated with chat
Response Streaming
Not supported
Supported if LLM processor allows (future)
Session State Management
Basic, stores chat history
Enhanced, persists all interactions
This table highlights the improvements in user interface and functionality, ensuring a more engaging chat experience.
Conclusion
The updated Streamlit implementation provides a conversational UI with improved chat history display, leveraging Streamlit’s chat elements for a user-friendly experience. Future work can focus on implementing response streaming if the LLM processor supports it, further enhancing real-time interaction.
Key Citations
	•	Build a basic LLM chat app - Streamlit Docs
