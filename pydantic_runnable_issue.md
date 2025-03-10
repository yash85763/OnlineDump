I understand your situation: you're hitting a "runnable" issue with `prompt | llm` because your LLM isn’t a standard LangChain-compatible model like `ChatOpenAI`. Instead, it’s hosted on a custom server, requiring an ID and key to get an access token, which you then use to query the LLM. The `|` (pipe) operator in LangChain expects a `Runnable` object, but your LLM setup doesn’t naturally fit that interface yet. Let’s fix this by either adapting your custom LLM to work with LangChain chains or writing a workaround to achieve the same result.

### The Problem
- **LangChain Chains**: The `prompt | llm` syntax relies on `llm` being a `Runnable` (e.g., a LangChain chat model like `ChatOpenAI`). This interface expects `invoke()` to handle inputs directly.
- **Your Setup**: Your LLM is on a server, accessed via an API with an ID/key for authentication, returning responses in a custom way. This doesn’t plug into LangChain’s chain system out of the box.

### Solutions
I’ll offer two approaches:
1. **Integrate with LangChain**: Create a custom `Runnable` LLM wrapper that handles your authentication and API calls, making it compatible with `prompt | llm`.
2. **Manual Chain Workaround**: Skip LangChain’s chain syntax and write a function to manually combine the prompt and LLM call.

Since you’re using a custom server, I’ll assume it’s an HTTP API (e.g., REST) that requires an access token. Adjust the details (endpoint URLs, token format) based on your server’s specifics.

---

### Option 1: Integrate with LangChain (Custom Runnable LLM)
This approach makes your LLM work seamlessly with `prompt | llm` by wrapping it in a LangChain-compatible class.

#### Steps
1. **Fetch Access Token**: Write a function to get the token using your ID and key.
2. **Create a Custom LLM**: Subclass LangChain’s `BaseChatModel` or `SimpleChatModel` to handle API calls.
3. **Use in Chain**: Plug it into the `prompt | llm` chain.

#### Code
```python
import json
import requests
from typing import Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable

# Define Pydantic schemas (unchanged)
class Citation(BaseModel):
    source: str = Field(description="Section or source of the answer")
    text: str = Field(description="The original answer from that section")

class QuestionResponse(BaseModel):
    question: str = Field(description="The question being answered")
    combined_answer: str = Field(description="The consolidated answer for this question")
    citations: list[Citation] = Field(description="List of citations for this question")

# Function to get access token
def get_access_token(api_id: str, api_key: str) -> str:
    token_url = "https://your-server.com/auth/token"  # Replace with your auth endpoint
    payload = {"id": api_id, "key": api_key}
    response = requests.post(token_url, json=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get token: {response.text}")

# Custom LLM class for your server
class CustomServerLLM(Runnable):
    def __init__(self, api_id: str, api_key: str, api_url: str):
        self.api_id = api_id
        self.api_key = api_key
        self.api_url = api_url  # e.g., "https://your-server.com/v1/chat"
        self.token = get_access_token(api_id, api_key)

    def invoke(self, input: Any, config: Optional[dict] = None) -> AIMessage:
        # Input is the rendered prompt from the chain
        if isinstance(input, str):
            prompt = input
        else:
            raise ValueError("Expected string input")

        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"prompt": prompt}  # Adjust based on your API's expected format
        response = requests.post(self.api_url, json=payload, headers=headers)

        if response.status_code == 200:
            # Assuming your API returns JSON like {"response": "..."}
            json_response = response.json()
            return AIMessage(content=json.dumps(json_response["response"]))
        else:
            raise Exception(f"API call failed: {response.text}")

    def _get_input_schema(self, config=None):
        return None  # Simplification for this example

    def _get_output_schema(self, config=None):
        return None  # Simplification for this example

# Prompt template
combine_prompt = PromptTemplate(
    input_variables=["question", "answers"],
    template="""
    Consolidate the following answers for the question "{question}" into a single coherent response and include citations in valid JSON format:

    {answers}

    Return your response as a JSON object with this structure:
    {
        "question": "{question}",
        "combined_answer": "your consolidated answer here",
        "citations": [
            {"source": "section_name", "text": "original answer text"}
        ]
    }
    Ensure the JSON is valid and contains no extra text outside the JSON object.
    """
)

# Configuration
api_id = "your-api-id-here"
api_key = "your-api-key-here"
api_url = "https://your-server.com/v1/chat"  # Replace with your LLM endpoint

# Instantiate custom LLM
llm = CustomServerLLM(api_id=api_id, api_key=api_key, api_url=api_url)

# Create the chain
llm_chain = combine_prompt | llm

# Function to combine answers (unchanged logic, adapted to chain)
def combine_answers_for_question(question: str, answer_list: list, max_retries: int = 3) -> dict:
    answers_str = "\n".join([f"Section: {section}, Answer: {answer}" for section, answer in answer_list])
    
    for attempt in range(max_retries):
        try:
            raw_response = llm_chain.invoke({"question": question, "answers": answers_str})
            json_data = json.loads(raw_response.content)
            validated_response = QuestionResponse(**json_data)
            return validated_response.dict()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed for '{question}': {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Failed to combine answers for '{question}'",
                    "citations": []
                }
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1} for '{question}': {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Unexpected error: {str(e)}",
                    "citations": []
                }

# Test it
question = "What is the weather like?"
answer_list = [
    ("Section1", "It’s sunny today."),
    ("Section2", "The weather is clear and warm.")
]
response = combine_answers_for_question(question, answer_list)
print(json.dumps(response, indent=2))
```

#### How It Works
1. **Token Fetching**: `get_access_token` requests a token from your server using the ID and key.
2. **Custom LLM**: `CustomServerLLM` implements the `invoke` method, making it a `Runnable`. It:
   - Takes the rendered prompt from the chain.
   - Sends it to your server with the token.
   - Returns an `AIMessage` with the response.
3. **Chain Compatibility**: `combine_prompt | llm` now works because `llm` is a `Runnable`.

#### Adjustments
- **API Details**: Update `token_url`, `api_url`, and payload/response formats to match your server’s API.
- **Error Handling**: Add token refresh logic if it expires (e.g., re-call `get_access_token` on 401 errors).
- **Schema**: I simplified `_get_input_schema` and `_get_output_schema`. For full compatibility, define these if your chain needs structured input/output validation.

---

### Option 2: Manual Chain Workaround
If integrating with LangChain feels overkill, you can skip the `|` syntax and manually handle the prompt and LLM call.

#### Code
```python
import json
import requests
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate

# Pydantic schemas (unchanged)
class Citation(BaseModel):
    source: str = Field(description="Section or source of the answer")
    text: str = Field(description="The original answer from that section")

class QuestionResponse(BaseModel):
    question: str = Field(description="The question being answered")
    combined_answer: str = Field(description="The consolidated answer for this question")
    citations: list[Citation] = Field(description="List of citations for this question")

# Function to get access token
def get_access_token(api_id: str, api_key: str) -> str:
    token_url = "https://your-server.com/auth/token"
    payload = {"id": api_id, "key": api_key}
    response = requests.post(token_url, json=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get token: {response.text}")

# Function to query the LLM
def query_llm(prompt: str, api_url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"prompt": prompt}  # Adjust based on your API
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]  # Adjust based on your API's response
    else:
        raise Exception(f"API call failed: {response.text}")

# Prompt template
combine_prompt = PromptTemplate(
    input_variables=["question", "answers"],
    template="""
    Consolidate the following answers for the question "{question}" into a single coherent response and include citations in valid JSON format:

    {answers}

    Return your response as a JSON object with this structure:
    {
        "question": "{question}",
        "combined_answer": "your consolidated answer here",
        "citations": [
            {"source": "section_name", "text": "original answer text"}
        ]
    }
    Ensure the JSON is valid and contains no extra text outside the JSON object.
    """
)

# Configuration
api_id = "your-api-id-here"
api_key = "your-api-key-here"
api_url = "https://your-server.com/v1/chat"
token = get_access_token(api_id, api_key)

# Function to combine answers
def combine_answers_for_question(question: str, answer_list: list, max_retries: int = 3) -> dict:
    answers_str = "\n".join([f"Section: {section}, Answer: {answer}" for section, answer in answer_list])
    prompt = combine_prompt.format(question=question, answers=answers_str)
    
    for attempt in range(max_retries):
        try:
            raw_response = query_llm(prompt, api_url, token)
            json_data = json.loads(raw_response)
            validated_response = QuestionResponse(**json_data)
            return validated_response.dict()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed for '{question}': {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Failed to combine answers for '{question}'",
                    "citations": []
                }
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1} for '{question}': {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Unexpected error: {str(e)}",
                    "citations": []
                }

# Test it
question = "What is the weather like?"
answer_list = [
    ("Section1", "It’s sunny today."),
    ("Section2", "The weather is clear and warm.")
]
response = combine_answers_for_question(question, answer_list)
print(json.dumps(response, indent=2))
```

#### How It Works
1. **Manual Prompt Rendering**: `combine_prompt.format()` generates the prompt string.
2. **Direct API Call**: `query_llm` sends the prompt to your server and gets the response.
3. **No LangChain Dependency**: You bypass the `|` syntax, handling everything yourself.

#### Adjustments
- **API Details**: Update `token_url`, `api_url`, and payload/response handling.
- **Token Management**: Add logic to refresh the token if it expires.

---

### Which Option to Choose?
- **Option 1 (LangChain Integration)**:
  - **Pros**: Reusable, integrates with LangChain’s ecosystem (e.g., memory, tools), cleaner for complex chains.
  - **Cons**: More setup, requires understanding LangChain’s `Runnable` interface.
  - **Use If**: You plan to expand with more LangChain features later.

- **Option 2 (Manual Workaround)**:
  - **Pros**: Simpler, no LangChain dependency beyond `PromptTemplate`, full control.
  - **Cons**: Less modular, manual effort for chaining additional steps.
  - **Use If**: You just need this one task done quickly.

---

### Recommendation
Since you’re already using LangChain’s `PromptTemplate`, I’d lean toward **Option 1**. It keeps your code future-proof and aligns with your existing setup. However, if you’re not planning to use more LangChain features, **Option 2** is perfectly fine and simpler to start with.

Let me know your server’s API details (e.g., endpoints, expected payload) if you need help tailoring either option further!




## Inline citations like cross references:

If the final answer (`combined_answer`) from your LLM includes citations as cross-references (e.g., "The weather is sunny [1] and warm [2]"), and you also want a separate `citations` list in the JSON linking back to those references (e.g., `[1]` maps to a specific section), we need to ensure the LLM generates both consistently and that your code can handle them. This introduces a new layer of complexity, but it’s manageable with the right approach.

Let’s break this down and figure out how to handle inline citations/cross-references effectively.

---

### What’s Happening
- **Current Setup**: The LLM combines answers into a `combined_answer` and provides a `citations` list, but there’s no explicit connection between the text in `combined_answer` and the `citations` entries.
- **New Requirement**: You want inline citations (e.g., "[1]", "[2]") in the `combined_answer` that correspond to specific entries in the `citations` list, acting as cross-references to the original sections.

For example, you might want:
```json
{
  "question": "What is the weather like?",
  "combined_answer": "The weather is sunny [1] and warm [2] today.",
  "citations": [
    {"source": "Section1", "text": "It’s sunny today."},
    {"source": "Section2", "text": "The weather is clear and warm."}
  ]
}
```
Here, `[1]` refers to `Section1`, and `[2]` refers to `Section2`.

---

### Challenges
1. **Consistency**: The LLM must number the citations in `combined_answer` and match them to the `citations` list order.
2. **Parsing**: Your code needs to trust or validate that the cross-references align with the `citations` list.
3. **Flexibility**: Should the LLM decide the citation numbers, or should you predefine them based on input?

---

### Approaches to Handle Inline Citations
I’ll propose two strategies:
1. **Ask for Inline Citations in the Prompt**: Instruct the LLM to include numbered cross-references in `combined_answer` and align them with the `citations` list.
2. **Predefine Citations with Numbers**: Pass the answers with pre-assigned numbers to the LLM, ensuring it uses those in the response.

#### Option 1: Ask for Inline Citations in the Prompt
This relies on the LLM to assign numbers and maintain consistency.

##### Updated Prompt
Modify the `combine_prompt` to explicitly request inline citations:
```python
combine_prompt = PromptTemplate(
    input_variables=["question", "answers"],
    template="""
    Consolidate the following answers for the question "{question}" into a single coherent response. Include inline citations as numbered cross-references (e.g., [1], [2]) in the combined answer, corresponding to the order of citations listed below:

    {answers}

    Return your response as a JSON object with this structure:
    {
        "question": "{question}",
        "combined_answer": "your consolidated answer with [1], [2], etc.",
        "citations": [
            {"source": "section_name", "text": "original answer text"},
            {"source": "section_name", "text": "original answer text"}
        ]
    }
    Ensure the JSON is valid, the inline citations match the citations list order (e.g., [1] refers to the first citation), and there’s no extra text outside the JSON object.
    """
)
```

##### Code (Using Option 1 from Previous Answer)
Here’s how it fits with the custom LangChain `Runnable` approach:
```python
import json
import requests
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

# Pydantic schemas (unchanged)
class Citation(BaseModel):
    source: str = Field(description="Section or source of the answer")
    text: str = Field(description="The original answer from that section")

class QuestionResponse(BaseModel):
    question: str = Field(description="The question being answered")
    combined_answer: str = Field(description="The consolidated answer with inline citations")
    citations: list[Citation] = Field(description="List of citations for this question")

# Get access token
def get_access_token(api_id: str, api_key: str) -> str:
    token_url = "https://your-server.com/auth/token"
    payload = {"id": api_id, "key": api_key}
    response = requests.post(token_url, json=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    raise Exception(f"Failed to get token: {response.text}")

# Custom LLM
class CustomServerLLM:
    def __init__(self, api_id: str, api_key: str, api_url: str):
        self.api_id = api_id
        self.api_key = api_key
        self.api_url = api_url
        self.token = get_access_token(api_id, api_key)

    def invoke(self, input: str, config=None) -> AIMessage:
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"prompt": input}
        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            json_response = response.json()
            return AIMessage(content=json.dumps(json_response["response"]))
        raise Exception(f"API call failed: {response.text}")

# Prompt
combine_prompt = PromptTemplate(
    input_variables=["question", "answers"],
    template="""
    Consolidate the following answers for the question "{question}" into a single coherent response. Include inline citations as numbered cross-references (e.g., [1], [2]) in the combined answer, corresponding to the order of citations listed below:

    {answers}

    Return your response as a JSON object with this structure:
    {
        "question": "{question}",
        "combined_answer": "your consolidated answer with [1], [2], etc.",
        "citations": [
            {"source": "section_name", "text": "original answer text"}
        ]
    }
    Ensure the JSON is valid, the inline citations match the citations list order (e.g., [1] refers to the first citation), and there’s no extra text outside the JSON object.
    """
)

# Configuration
api_id = "your-api-id-here"
api_key = "your-api-key-here"
api_url = "https://your-server.com/v1/chat"
llm = CustomServerLLM(api_id=api_id, api_key=api_key, api_url=api_url)
llm_chain = combine_prompt | llm

# Combine answers
def combine_answers_for_question(question: str, answer_list: list, max_retries: int = 3) -> dict:
    answers_str = "\n".join([f"Section: {section}, Answer: {answer}" for section, answer in answer_list])
    
    for attempt in range(max_retries):
        try:
            raw_response = llm_chain.invoke({"question": question, "answers": answers_str})
            json_data = json.loads(raw_response.content)
            validated_response = QuestionResponse(**json_data)
            return validated_response.dict()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Failed to combine answers for '{question}'",
                    "citations": []
                }

# Test
question = "What is the weather like?"
answer_list = [
    ("Section1", "It’s sunny today."),
    ("Section2", "The weather is clear and warm.")
]
response = combine_answers_for_question(question, answer_list)
print(json.dumps(response, indent=2))
```

##### Pros
- Simple for the LLM: It decides the numbering and order.
- Flexible: Works with any number of answers.

##### Cons
- Relies on LLM accuracy: If it messes up the numbering (e.g., skips [2] or mismatches), you’ll need to validate or retry.
- No predefined control: You can’t dictate which section gets which number.

##### Validation (Optional)
To ensure the `combined_answer` citations match the `citations` list:
- Parse `combined_answer` for `[n]` references using regex (e.g., `re.findall(r'\[\d+\]', combined_answer)`).
- Check that the numbers align with the `citations` list length and order.

---

#### Option 2: Predefine Citations with Numbers
This approach assigns numbers to the answers before sending them to the LLM, ensuring exact control over cross-references.

##### Updated Prompt
```python
combine_prompt = PromptTemplate(
    input_variables=["question", "answers"],
    template="""
    Consolidate the following answers for the question "{question}" into a single coherent response. Use the provided citation numbers (e.g., [1], [2]) as inline cross-references in the combined answer, matching the order below:

    {answers}

    Return your response as a JSON object with this structure:
    {
        "question": "{question}",
        "combined_answer": "your consolidated answer with [1], [2], etc.",
        "citations": [
            {"source": "section_name", "text": "original answer text"}
        ]
    }
    Ensure the JSON is valid, the inline citations use the numbers provided, and there’s no extra text outside the JSON object.
    """
)
```

##### Modified Function
```python
def combine_answers_for_question(question: str, answer_list: list, max_retries: int = 3) -> dict:
    # Assign numbers to answers
    numbered_answers = [(i + 1, section, answer) for i, (section, answer) in enumerate(answer_list)]
    answers_str = "\n".join([f"Citation [{num}], Section: {section}, Answer: {answer}" for num, section, answer in numbered_answers])
    
    for attempt in range(max_retries):
        try:
            raw_response = llm_chain.invoke({"question": question, "answers": answers_str})
            json_data = json.loads(raw_response.content)
            validated_response = QuestionResponse(**json_data)
            return validated_response.dict()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {
                    "question": question,
                    "combined_answer": f"Failed to combine answers for '{question}'",
                    "citations": []
                }
```

##### Example Output
Input `answer_list`:
```python
[("Section1", "It’s sunny today."), ("Section2", "The weather is clear and warm.")]
```

`answers_str` becomes:
```
Citation [1], Section: Section1, Answer: It’s sunny today.
Citation [2], Section: Section2, Answer: The weather is clear and warm.
```

Output:
```json
{
  "question": "What is the weather like?",
  "combined_answer": "The weather is sunny [1] and warm [2] today.",
  "citations": [
    {"source": "Section1", "text": "It’s sunny today."},
    {"source": "Section2", "text": "The weather is clear and warm."}
  ]
}
```

##### Pros
- Exact Control: You dictate the numbers, reducing LLM errors.
- Easier Validation: Numbers are predefined, so you can check if `[1]`, `[2]`, etc., appear correctly.

##### Cons
- More Prep Work: You need to number the answers before sending them.
- Less LLM Flexibility: It must follow your numbering, which might limit its creativity.

---

### Which Should You Use?
- **Option 1 (Inline Citations)**:
  - Use if your LLM is reliable at numbering and you want it to handle the citation logic.
  - Add validation if you’re worried about mismatches.
- **Option 2 (Predefined Numbers)**:
  - Use if you want guaranteed alignment between `combined_answer` and `citations`, or if your LLM struggles with consistent numbering.
  - Preferred for precision and control.

---

### Handling Cross-References in Practice
- **Displaying**: In a UI, `[1]` could link to the first citation in the list (e.g., via HTML `<sup>[1]</sup>` with a tooltip).
- **Validation**: If critical, add a check:
  ```python
  import re
  def validate_citations(response: dict) -> bool:
      refs = re.findall(r'\[\d+\]', response["combined_answer"])
      max_ref = max([int(r.strip("[]")) for r in refs], default=0)
      return max_ref <= len(response["citations"]) and all(r in response["combined_answer"] for r in [f"[{i+1}]" for i in range(len(response["citations"]))])
  ```
  Call this after `validated_response.dict()` to ensure all citations are referenced.

---

### Recommendation
Start with **Option 2** (predefined numbers) because:
- It aligns your input (`answer_list`) directly with the output, minimizing LLM mistakes.
- It’s easier to debug and validate.
- You can switch to Option 1 later if the LLM proves reliable.

Try the Option 2 code with your server, and let me know how it goes or if you need tweaks (e.g., API specifics, citation format)!