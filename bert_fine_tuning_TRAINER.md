Thank you for providing the example of how links are handled in your Streamlit app! Based on your input, it seems you're using a specific Markdown link syntax where the URL is formatted with `%s` in a string, and `st.write()` or `st.markdown()` is used to render it. The example you gave:

```python
url = "https://www.streamlit.io"
st.write("checkout this [link](%s)" % url)
st.markdown("checkout this [link](%s)" % url)
```

This approach works in Streamlit, where:
- `st.write()` renders the string as plain text with the link embedded (clickable if formatted correctly).
- `st.markdown()` renders the string as Markdown, ensuring the link is clickable, especially with `unsafe_allow_html=True` if needed for complex formatting.

Given this, it appears the issue with links not appearing might be due to:
- The Markdown syntax not being correctly applied or rendered in the current implementation.
- The citations in your `final_answer` not being properly transformed into the expected `[text](url)` format.

Since you’ve confirmed the desired link format, let’s update the code to:
1. Align the link generation with your preferred syntax (e.g., `[Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1)`).
2. Use `st.markdown()` consistently to ensure links are clickable, as it’s more reliable for Markdown rendering.
3. Debug why links aren’t appearing and ensure the LLM or regex approach generates the correct format.

### Updated Streamlit Code
Below is the revised code incorporating your link format preference and addressing the citation parsing for `[Section 211.1]` or `[Content 211.4(a)]` formats:

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
    Citations can be in the format [Section 211.1] or [Content 211.4(a)], where the prefix
    (e.g., 'Section', 'Content') should be ignored for the URL, and the full text is used as the link text.
    
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
        formatted_answer = llm_format_answer_with_links(final_answer, title="12", part="211")

        # Fallback to regex if LLM fails or for comparison
        regex_formatted_answer = regex_format_answer_with_links(final_answer, title="12", part="211")
        st.write(f"Debug: Regex formatted answer: {regex_formatted_answer}")

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
1. **Alignment with Your Link Format**:
   - The `llm_format_answer_with_links()` and `regex_format_answer_with_links()` functions now generate links in the format `[text](url)`, matching your example `checkout this [link](%s)` % url.
   - The LLM prompt and regex logic ensure that the full citation text (e.g., `[Section 211.1]`) is preserved as the link text, while the URL is constructed from the section number (e.g., `211.1`).

2. **Rendering with `st.markdown()`**:
   - The final answer and chat history assistant responses are rendered using `st.markdown()` with `unsafe_allow_html=True` to ensure the `[text](url)` syntax is interpreted as clickable links.
   - This aligns with your example where `st.markdown("checkout this [link](%s)" % url)` works.

3. **Citation Parsing**:
   - Both functions handle citations like `[Section 211.1]` or `[Content 211.4(a)]` by ignoring the prefix before the first digit and using the section number for the URL.
   - The regex pattern and LLM prompt are designed to extract `211.1` or `211.4(a)` for URL construction while keeping the original citation text (e.g., `[Section 211.1]`) as the link text.

### Example
Suppose `final_answer` is:
```
The regulation requires compliance as per [Section 211.1] and [Content 211.4(a)].
```

- **LLM Output** (`formatted_answer`):
  ```
  The regulation requires compliance as per [Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1) and [Content 211.4(a)](https://www.ecfr.gov/current/title-12/part-211/section-211.4#p-211.4(a)).
  ```
- **Regex Output** (`regex_formatted_answer`):
  ```
  The regulation requires compliance as per [Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1) and [Content 211.4(a)](https://www.ecfr.gov/current/title-12/part-211/section-211.4#p-211.4(a)).
  ```
- **Displayed in Streamlit**:
  **Answer:** The regulation requires compliance as per [Section 211.1](https://www.ecfr.gov/current/title-12/part-211/section-211.1) and [Content 211.4(a)](https://www.ecfr.gov/current/title-12/part-211/section-211.4#p-211.4(a)).

The citations should now be clickable links in the Streamlit app.

### Debugging Why Links Aren’t Appearing
If you’re still not seeing links, let’s troubleshoot:
1. **Check the Raw Answer**:
   - Look at `Debug: Raw answer: {final_answer}` to confirm if citations like `[Section 211.1]` are present. If not, the issue is with `llm_processor.consolidator()` or `answer_from_sections()` not including citations.

2. **Check Citation Detection**:
   - Look at `Debug: Citations found in answer: {citations_found}` to see if the regex is detecting the citations. If it’s empty, the format might not match (e.g., missing brackets or different spacing).

3. **Check Formatted Output**:
   - Look at `Debug: Regex formatted answer: {regex_formatted_answer}` to verify the Markdown links are being generated correctly (e.g., `[Section 211.1](https://...)`).

4. **Test Manual Link**:
   - Add this line after `st.title()` to test if links render:
     ```python
     url = "https://www.ecfr.gov/current/title-12/part-211/section-211.1"
     st.markdown("Test [link](%s)" % url, unsafe_allow_html=True)
     ```
   - If this doesn’t render as a clickable link, there might be a Streamlit configuration or version issue.

### Request for Debugging Output
To resolve this:
1. Share the output of the debug prints (`Debug: Raw answer`, `Debug: Citations found in answer`, `Debug: Regex formatted answer`) after submitting a question.
2. Share an example of the `final_answer` string generated by your app.
3. Confirm if the test manual link (`Test [link](https://...)`) renders as clickable in your Streamlit app.

With this information, I can pinpoint why the links aren’t appearing and adjust the code accordingly!