I understand your request: you're currently outputting the answer from the LLM (`llm_format_answer_with_links()`), but if the LLM-generated links don't match the expected format or if the regex formatter (`regex_format_answer_with_links()`) fails to find citations, you want to fall back to a default eCFR link (e.g., the homepage or a generic section). This will ensure that the app always provides some form of clickable link even if the citation parsing fails.

Let’s modify the code to:
1. Compare the LLM-formatted answer with the regex-formatted answer.
2. If the LLM output doesn’t contain the expected `[text](url)` links (or if the regex finds no citations), use a default eCFR link (e.g., `https://www.ecfr.gov`).
3. Ensure the final output is rendered correctly in Streamlit using your preferred `[text](url)` format.

### Updated Streamlit Code
Below is the revised code with the fallback logic for a default eCFR link:

```python
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
        # Assuming llm_processor has a method to send a prompt and get a response
        formatted_answer = llm_processor.answer_from_sections(prompt, [prompt])[0]
        return formatted_answer
    except Exception as e:
        st.error(f"Error using LLM to format links: {e}")
        return answer  # Fallback to original answer if LLM fails

# Fallback regex-based function to parse citations (for debugging)
def regex_format_answer_with_links(answer, title="12", part="211"):
    """
    Parse citations like [Section 211.1] or [Content 211.4(a)] in the answer and replace them with eCFR links.
    Ignores any text before the first digit in the citation.
    
    Args:
        answer (str): The answer string containing citations.
        title (str): The eCFR title (default: "12").
        part (str): The eCFR part (default: "211").
    
    Returns:
        str: The answer with citations replaced by Markdown links in the format [text](url).
    """
    # Regex to match citations like [Section 211.1] or [Content 211.4(a)]
    citation_pattern = r'\[([A-Za-z\s]+)?(\d+\.\d+(?:\([a-zA-Z0-9]+\))?(?:,\s*(?:[A-Za-z\s]+)?\d+\.\d+(?:\([a-zA-Z0-9]+\))?)*)\]'
    
    def create_link(match):
        full_citation = match.group(0)  # The entire citation, e.g., [Section 211.1]
        citation_group = match.group(2)  # The part after the prefix, e.g., 211.1 or 211.1, 211.2(a)
        citations = citation_group.split(", ")  # Split multiple citations
        
        # Find the corresponding full citations (including prefixes)
        full_citations = full_citation[1:-1].split(", ")  # Remove brackets and split
        
        links = []
        for i, citation in enumerate(citations):
            full_citation_text = full_citations[i].strip()  # e.g., "Section 211.1"
            
            # Extract the section number by finding the first digit and everything after
            section_match = re.search(r'\d+\.\d+(?:\([a-zA-Z0-9]+\))?', citation)
            if not section_match:
                continue
            section_number = section_match.group(0)  # e.g., 211.1 or 211.4(a)
            
            # Construct the URL based on whether there's a subsection
            if "(" in section_number:
                section, subsection = section_number.split("(")
                subsection = subsection.rstrip(")")
                section_url = f"{section}#{section}({subsection})"
            else:
                section = section_number
                section_url = section
            
            ecfr_url = f"https://www.ecfr.gov/current/title-{title}/part-{part}/section-{section_url}"
            link = f"[{full_citation_text}]({ecfr_url})"  # Match your preferred [text](url) format
            links.append(link)
        
        return ", ".join(links)
    
    # Debug: Print the citations found
    citations_found = re.findall(citation_pattern, answer)
    if citations_found:
        st.write(f"Debug: Citations found in answer: {citations_found}")
    else:
        st.write("Debug: No citations found in answer with regex.")
    
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

        # Debug: Print the raw answer to check for citations
        st.write(f"Debug: Raw answer: {final_answer}")

        # Format the answer with clickable eCFR links using LLM
        llm_formatted_answer = llm_format_answer_with_links(final_answer, title="12", part="211")

        # Fallback to regex
        regex_formatted_answer = regex_format_answer_with_links(final_answer, title="12", part="211")
        st.write(f"Debug: Regex formatted answer: {regex_formatted_answer}")

        # Check if LLM formatted answer contains links, fallback to regex or default link
        import re
        link_pattern = r'\[.*?\]\(https?://[^\s]+?\)'  # Matches [text](url) format
        if not re.search(link_pattern, llm_formatted_answer) and not re.findall(citation_pattern, final_answer):
            # If LLM didn't generate links and no citations were found, use a default link
            default_url = "https://www.ecfr.gov"
            formatted_answer = f"{final_answer} [Default eCFR Link]({default_url})"
        elif not re.search(link_pattern, llm_formatted_answer):
            # If LLM failed but citations were found, use regex formatted answer
            formatted_answer = regex_formatted_answer
        else:
            # Use LLM formatted answer if it contains links
            formatted_answer = llm_formatted_answer

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
            "answer": final_answer  # Store the raw answer
        })
        with open(qa_data_file, 'w') as f:
            json.dump(qa_data, f, indent=4)

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"], unsafe_allow_html=True)  # Render links for assistant responses
        else:
            st.write(message["content"])  # Plain text for user messages
```

### Key Changes
1. **Fallback Logic**:
   - Added a check using a `link_pattern` (`r'\[.*?\]\(https?://[^\s]+?\)`) to detect if the `llm_formatted_answer` contains Markdown links in the `[text](url)` format.
   - Conditions for selecting the final `formatted_answer`:
     - **If no links in LLM output and no citations found in the raw answer**: Uses a default eCFR link (`https://www.ecfr.gov`) appended to the original answer (e.g., "No answers found from any section. [Default eCFR Link](https://www.ecfr.gov)").
     - **If no links in LLM output but citations are found**: Falls back to the `regex_formatted_answer`.
     - **If LLM output contains links**: Uses the `llm_formatted_answer`.

2. **Debugging**:
   - Retained the debug prints to help identify the issue:
     - `Debug: Raw answer` to check the input.
     - `Debug: Regex formatted answer` to verify the regex output.

3. **Rendering**:
   - Continues to use `st.markdown()` with `unsafe_allow_html=True` to render the links, matching your example syntax (`[link](%s)` % url).

### Example
- **Raw Answer**: `"The regulation requires compliance as per [Section 211.1] and [Content 211.4(a)]."`
- **LLM Formatted Answer**: If the LLM correctly formats it:
  ```
  "The regulation requires compliance as per [Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1) and [Content 211.4(a)](https://www.ecfr.gov/current/title-12/part-211/section-211.4#p-211.4(a))."
  ```
- **Regex Formatted Answer**: If LLM fails:
  ```
  "The regulation requires compliance as per [Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1) and [Content 211.4(a)](https://www.ecfr.gov/current/title-12/part-211/section-211.4#p-211.4(a))."
  ```
- **If No Citations or LLM Fails**: 
  ```
  "No answers found from any section. [Default eCFR Link](https://www.ecfr.gov)"
  ```

### Testing
1. **Run the App**:
   - Submit a question (e.g., "What are the reporting requirements?").
   - Check the debug outputs:
     - `Debug: Raw answer`: Does it contain `[Section 211.1]` or similar?
     - `Debug: Regex formatted answer`: Does it show the expected `[text](url)` links?

2. **Verify Links**:
   - Ensure the final answer under `**Answer:**` has clickable links if citations are present, or the default link if not.
   - Check the chat history for the same.

3. **Fallback Behavior**:
   - If you intentionally break the LLM (e.g., by passing an invalid prompt), verify that the regex fallback or default link appears.

### Debugging Why Links Aren’t Appearing
If links still don’t appear:
- **Check Citation Format**: Share the `Debug: Raw answer` output to confirm the citation format (e.g., `[Section 211.1]`).
- **Check LLM Output**: Add `st.write(f"Debug: LLM formatted answer: {llm_formatted_answer}")` after the LLM call to see what the LLM returns.
- **Test Manual Link**: Add `st.markdown("Test [link](https://www.ecfr.gov)", unsafe_allow_html=True)` after `st.title()` to verify Streamlit can render links. If this doesn’t work, there might be a Streamlit issue.

### Request for Debugging Output
Please provide:
1. The `Debug: Raw answer`, `Debug: Regex formatted answer`, and (optionally) `Debug: LLM formatted answer` outputs after submitting a question.
2. Confirmation if the test manual link renders as clickable.
3. An example of the `final_answer` string if the debug prints don’t suffice.

This will help me ensure the links are correctly generated and displayed!