Let’s address your request by first examining the relevant Streamlit documentation for building a chatbot UI, then updating your code to incorporate modern features like response streaming, improved chat history management, and a better user experience for the chat history. I’ll also ensure the citations are handled with the [text](url) format you prefer, and include the fallback to a default eCFR link if needed.
Step 1: Review Streamlit Documentation for Chatbot UI
The Streamlit documentation provides guidance on building conversational apps using chat elements like st.chat_message and st.chat_input. Key features include:
	•	Chat Elements:
	◦	st.chat_message: Displays messages in a chat-like container, supporting roles like “user” and “assistant”. It can contain text, Markdown, and other Streamlit elements.
	◦	st.chat_input: Provides an input widget for users to type messages, which can trigger responses.
	•	Session State for Chat History:
	◦	Streamlit’s st.session_state is used to store chat history, allowing messages to persist across interactions. The documentation emphasizes initializing and appending to the chat history to maintain conversation context.
	•	Streaming Responses:
	◦	Streamlit supports streaming responses for LLMs, where the assistant’s response can be displayed incrementally as it’s generated. This is often achieved using st.write_stream() or by iterating over a streaming response object from an LLM.
	•	Recent Upgrades:
	◦	Streamlit has introduced features like st.write_stream() for streaming responses, which enhances the user experience by showing the assistant’s response in real-time.
	◦	Improved chat history management can be implemented by ensuring the history is scrollable, visually distinct, and possibly clickable (e.g., to revisit or continue past conversations).
Given your feedback that the current chat history implementation is “not good,” I’ll improve it by:
	•	Making the chat history scrollable and visually distinct.
	•	Adding timestamps to messages for better context.
	•	Optionally, allowing users to clear the chat history with a button.
Step 2: Update the Code
Below is the updated Streamlit implementation, incorporating response streaming, improved chat history, and the citation link formatting with a default eCFR fallback. I’ll assume your LLMProcessor class supports streaming responses (if it doesn’t, you’ll need to adjust the streaming logic based on your LLM’s API).

'''python 

  import streamlit as st
  import json
  import os
  import re
  from datetime import datetime
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
  
  # Function to use LLM to format citations as links
  def llm_format_answer_with_links(answer, title="12", part="211"):
      """
      Use the LLM to identify citations in the answer and convert them to eCFR Markdown links.
      Citations can be in the format [Section 211.1] or [Content 211.4(a)].
      
      Args:
          answer (str): The answer string containing citations.
          title (str): The eCFR title (default: "12").
          part (str): The eCFR part (default: "211").
      
      Returns:
          str: The answer with citations replaced by Markdown links in the format [text](url).
      """
      prompt = f"""
      You are a helpful assistant tasked with formatting citations in a text to include clickable links to the eCFR website. The text contains citations in square brackets, such as [Section 211.1], [Content 211.4(a)], or [Section 211.1, Content 211.2(a)]. Your task is to:
  
      1. Identify all citations in the text within square brackets.
      2. For each citation, ignore any text before the first digit (e.g., 'Section', 'Content') and extract the section number (e.g., 211.1, 211.4(a)) for the URL.
      3. Create a Markdown link for each section number using the format [full citation text](url), where the URL points to the eCFR page.
      4. Replace the entire citation with the Markdown link, preserving the full citation text (e.g., [Section 211.1] becomes [Section 211.1](https://www.ecfr.gov/current/title-{title}/part-{part}/section-211.1)).
      5. Return the modified text with all citations converted to links.
  
      ### Citation Format Rules:
      - A citation like [Section 211.1] should link to https://www.ecfr.gov/current/title-{title}/part-{part}/section-211.1
      - A citation like [Content 211.4(a)] should link to https://www.ecfr.gov/current/title-{title}/part-{part}/section-211.4#p-211.4(a)
      - Multiple citations like [Section 211.1, Content 211.2(a)] should be split and each converted to a link, e.g., [Section 211.1](...), [Content 211.2(a)](...)
      - Preserve the exact citation text (including the prefix) in the link text.
  
      ### Input Text:
      {answer}
  
      ### Output:
      Provide the text with citations replaced by Markdown links in the format [text](url).
      """
      
      try:
          formatted_answer = llm_processor.answer_from_sections(prompt, [prompt])[0]
          return formatted_answer
      except Exception as e:
          st.error(f"Error using LLM to format links: {e}")
          return answer
  
  # Fallback regex-based function to parse citations
  def regex_format_answer_with_links(answer, title="12", part="211"):
      """
      Parse citations like [Section 211.1] or [Content 211.4(a)] in the answer and replace them with eCFR links.
      
      Args:
          answer (str): The answer string containing citations.
          title (str): The eCFR title (default: "12").
          part (str): The eCFR part (default: "211").
      
      Returns:
          str: The answer with citations replaced by Markdown links in the format [text](url).
      """
      citation_pattern = r'\[([A-Za-z\s]+)?(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:,\s*(?:[A-Za-z\s]+)?\d+\.\d+(?:\([a-zA-Z0-9]+\))?)*)\]'
      
      def create_link(match):
          full_citation = match.group(0)
          citation_group = match.group(2)
          citations = citation_group.split(", ")
          full_citations = full_citation[1:-1].split(", ")
          
          links = []
          for i, citation in enumerate(citations):
              full_citation_text = full_citations[i].strip()
              section_match = re.search(r'\d+\.\d+(?:\([a-zA-Z0-9]+\))?', citation)
              if not section_match:
                  continue
              section_number = section_match.group(0)
              
              if "(" in section_number:
                  section, subsection = section_number.split("(")
                  subsection = subsection.rstrip(")")
                  section_url = f"{section}#{section}({subsection})"
              else:
                  section = section_number
                  section_url = section
              
              ecfr_url = f"https://www.ecfr.gov/current/title-{title}/part-{part}/section-{section_url}"
              link = f"[{full_citation_text}]({ecfr_url})"
              links.append(link)
          
          return ", ".join(links)
      
      citations_found = re.findall(citation_pattern, answer)
      if citations_found:
          st.write(f"Debug: Citations found in answer: {citations_found}")
      else:
          st.write("Debug: No citations found in answer with regex.")
      
      formatted_answer = re.sub(citation_pattern, create_link, answer)
      return formatted_answer
  
  # Create a Streamlit app
  st.title("Regulation Assistant: As of 2025-03-06")
  
  # Test manual link rendering
  st.markdown("Test [eCFR Link](https://www.ecfr.gov)", unsafe_allow_html=True)
  
  # Initialize chat history with timestamps
  if 'chat_history' not in st.session_state:
      st.session_state.chat_history = []
  
  # Sidebar for chat history management
  with st.sidebar:
      st.header("Chat History")
      if st.button("Clear Chat History"):
          st.session_state.chat_history = []
          st.success("Chat history cleared!")
  
      # Display chat history as a list of past questions
      if st.session_state.chat_history:
          for i, message in enumerate(st.session_state.chat_history):
              if message["role"] == "user":
                  timestamp = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                  if st.button(f"{timestamp}: {message['content'][:30]}...", key=f"history_{i}"):
                      # Optionally, you can implement logic to jump to this conversation
                      st.session_state.current_conversation = i
      else:
          st.write("No chat history yet.")
  
  # Create a dropdown menu to select the regulation
  selected_regulation = st.selectbox("Select a regulation", ["Regulation K"], index=0)
  
  # Create a dropdown menu to select the section
  sections = list(context.keys())
  selected_section = st.selectbox("Select query scope", ["ALL Sections"] + sections, index=0)
  
  # Initialize session state for the user question
  if 'custom_question' not in st.session_state:
      st.session_state.custom_question = ''
  
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
      index=0
  )
  
  if selected_question == 'Other...':
      st.session_state.custom_question = st.text_input(
          'Enter your question',
          value=st.session_state.custom_question
      )
  
      if not st.session_state.custom_question:
          st.warning('Please enter your question or select a predefined one')
          st.stop()
      else:
          question = st.session_state.custom_question
  else:
      question = selected_question
  
  # Display chat messages in a scrollable container
  st.header("Conversation")
  chat_container = st.container(height=400)  # Scrollable container
  
  with chat_container:
      for message in st.session_state.chat_history:
          with st.chat_message(message["role"]):
              timestamp = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
              st.markdown(f"**[{timestamp}]** {message['content']}", unsafe_allow_html=True)
  
  # Function to find the best-matching question in consolidated_data
  def find_best_match(user_question, consolidated_data):
      for qa in consolidated_data:
          if qa["question"].strip().lower() == user_question.strip().lower():
              return qa["answer"]
      return None
  
  # Accept user input via chat_input
  if prompt := st.chat_input("Type your question here..."):
      # Add user message to chat history with timestamp
      timestamp = datetime.now()
      st.session_state.chat_history.append({
          "role": "user",
          "content": prompt,
          "timestamp": timestamp
      })
  
      # Display user message immediately
      with chat_container:
          with st.chat_message("user"):
              st.markdown(f"**[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}]** {prompt}", unsafe_allow_html=True)
  
      with st.spinner("Processing your question..."):
          final_answer = None
          if selected_section == "ALL Sections":
              answer = find_best_match(prompt, consolidated_data)
              if answer:
                  final_answer = answer
              else:
                  section_answers = []
                  for section, section_content in context.items():
                      # Assuming answer_from_sections supports streaming
                      answers_for_section = llm_processor.answer_from_sections(section_content, [prompt], stream=True)
                      section_answers.append(answers_for_section)
                  
                  # Consolidate the answers from all sections
                  if section_answers:
                      # Assuming consolidator supports streaming
                      final_answer_stream = llm_processor.consolidator(prompt, section_answers, stream=True)
                      final_answer = ""
                      with chat_container:
                          with st.chat_message("assistant"):
                              placeholder = st.empty()
                              for chunk in final_answer_stream:
                                  final_answer += chunk
                                  placeholder.markdown(f"**[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]** {final_answer}", unsafe_allow_html=True)
                  else:
                      final_answer = "No answers found from any section."
  
                  consolidated_data.append({
                      "question": prompt,
                      "answer": final_answer
                  })
                  with open(consolidated_data_file, 'w') as f:
                      json.dump(consolidated_data, f, indent=4)
          else:
              # Stream the response for a specific section
              answer_stream = llm_processor.answer_from_sections(context[selected_section], [prompt], stream=True)
              final_answer = ""
              with chat_container:
                  with st.chat_message("assistant"):
                      placeholder = st.empty()
                      for chunk in answer_stream:
                          final_answer += chunk
                          placeholder.markdown(f"**[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]** {final_answer}", unsafe_allow_html=True)
  
          # Debug: Print the raw answer
          st.write(f"Debug: Raw answer: {final_answer}")
  
          # Format the answer with clickable eCFR links using LLM
          llm_formatted_answer = llm_format_answer_with_links(final_answer, title="12", part="211")
  
          # Fallback to regex
          regex_formatted_answer = regex_format_answer_with_links(final_answer, title="12", part="211")
          st.write(f"Debug: Regex formatted answer: {regex_formatted_answer}")
  
          # Check if LLM formatted answer contains links, fallback to regex or default link
          link_pattern = r'\[.*?\]\(https?://[^\s]+?\)'
          if not re.search(link_pattern, llm_formatted_answer) and not re.findall(r'\[([A-Za-z\s]+)?(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:,\s*(?:[A-Za-z\s]+)?\d+\.\d+(?:\([a-zA-Z0-9]+\))?)*)\]', final_answer):
              default_url = "https://www.ecfr.gov"
              formatted_answer = f"{final_answer} [Default eCFR Link]({default_url})"
          elif not re.search(link_pattern, llm_formatted_answer):
              formatted_answer = regex_formatted_answer
          else:
              formatted_answer = llm_formatted_answer
  
          # Update the last assistant message with the formatted answer
          with chat_container:
              with st.chat_message("assistant"):
                  timestamp = datetime.now()
                  st.markdown(f"**[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}]** {formatted_answer}", unsafe_allow_html=True)
  
          # Add the formatted assistant response to chat history
          st.session_state.chat_history.append({
              "role": "assistant",
              "content": formatted_answer,
              "timestamp": timestamp
          })
  
          # Save the QA data (store the original answer without Markdown links)
          qa_data.append({
              "question": prompt,
              "section": selected_section,
              "answer": final_answer
          })
          with open(qa_data_file, 'w') as f:
              json.dump(qa_data, f, indent=4)
'''

Key Changes and Improvements
	1	Response Streaming:
	◦	Added streaming support by assuming llm_processor.answer_from_sections() and llm_processor.consolidator() can return a streaming response (via a stream=True parameter). If your LLMProcessor doesn’t support streaming, you’ll need to modify this to use the actual streaming API of your LLM (e.g., OpenAI’s client.chat.completions.create with stream=True).
	◦	Used a placeholder (st.empty()) to update the assistant’s response in real-time as chunks are received, improving the user experience.
	2	Improved Chat History:
	◦	Added timestamps to each message using datetime.now() for better context.
	◦	Moved the chat history display into a scrollable container (st.container(height=400)) to prevent the page from becoming cluttered.
	◦	Added a sidebar to list past user questions with timestamps, making it easier to navigate. You can click a question to revisit it (though the “jump to conversation” logic is left as optional for now).
	◦	Added a “Clear Chat History” button in the sidebar to reset the conversation.
	3	Citation Link Formatting:
	◦	Retained the [text](url) format for links, as per your preference.
	◦	Kept the fallback logic to use a default eCFR link (https://www.ecfr.gov) if neither the LLM nor regex finds/formats citations.
	4	Debugging:
	◦	Included debug prints to help diagnose issues with citation detection and link formatting.
Assumptions and Notes
	•	Streaming Support: The code assumes llm_processor supports streaming responses. If it doesn’t, you’ll need to modify the streaming logic. For example, if using OpenAI’s API, you might do: response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], stream=True)
	•	for chunk in response:
	•	    final_answer += chunk.choices[0].delta.get("content", "")
	•	 Replace the streaming logic in the answer_from_sections and consolidator calls accordingly.
	•	Chat History Navigation: The sidebar lists past questions, but the “jump to conversation” functionality is not fully implemented. You can extend this by storing conversations in separate lists within st.session_state and allowing users to switch between them.
	•	Timestamps: Timestamps are added to each message to improve readability and context.
Debugging Steps
If links or streaming don’t work as expected:
	1	Check the Debug: Raw answer to ensure citations are present in the format [Section 211.1].
	2	Check the Debug: Regex formatted answer to verify the regex is generating links correctly.
	3	Add st.write(f"Debug: LLM formatted answer: {llm_formatted_answer}") to see the LLM’s output.
	4	Verify the test link (Test [eCFR Link](https://www.ecfr.gov)) renders as clickable.
Request for Further Debugging
If issues persist, please share:
	•	The debug outputs (Debug: Raw answer, Debug: Regex formatted answer, and optionally Debug: LLM formatted answer).
	•	Whether the test link renders as clickable.
	•	Details about your LLMProcessor’s streaming capabilities, if streaming doesn’t work.
This updated implementation should provide a more modern and user-friendly chatbot UI with streaming responses and an improved chat history experience.
