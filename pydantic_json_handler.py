import json
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI  # Replace with your LLM provider
from langchain.prompts import PromptTemplate

# Define Pydantic schemas
class Citation(BaseModel):
    source: str = Field(description="Section or source of the answer")
    text: str = Field(description="The original answer from that section")

class QuestionResponse(BaseModel):
    question: str = Field(description="The question being answered")
    combined_answer: str = Field(description="The consolidated answer for this question")
    citations: list[Citation] = Field(description="List of citations for this question")

# Mock function to simulate answering questions (replace with your actual implementation)
def answer_questions(content: str, questions: list) -> list:
    # Placeholder—replace with your LLM call or logic
    return [
        f"{q} Based on '{content}'" for q in questions
    ]

# LLM configuration
api_key = "your-api-key-here"
api_id = "your-api-id-here"
base_url = "https://api.your-llm-provider.com/v1"

llm = ChatOpenAI(
    api_key=api_key,
    model="your-model-name",
    base_url=base_url,
    model_kwargs={"user_id": api_id}
)

# Prompt to combine answers for a single question
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

llm_chain = combine_prompt | llm

# Function to combine answers for a single question
def combine_answers_for_question(question: str, answer_list: list, max_retries: int = 3) -> dict:
    # Format answers for the prompt
    answers_str = "\n".join([f"Section: {section}, Answer: {answer}" for section, answer in answer_list])
    
    for attempt in range(max_retries):
        try:
            raw_response = llm_chain.invoke({"question": question, "answers": answers_str})
            if hasattr(raw_response, "content"):
                raw_response = raw_response.content
            
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

# Main function to process all questions
def get_all_combined_responses(answers_dict: dict) -> list:
    combined_responses = []
    for question, answer_list in answers_dict.items():
        if answer_list:  # Only process if there are answers
            response = combine_answers_for_question(question, answer_list)
            combined_responses.append(response)
    return combined_responses

# Main execution
questions = ["What is the weather like?", "What is the temperature?", "Is it sunny?"]
answers = {question: [] for question in questions}

context = {
    "Section1": "The weather is sunny with a temperature of 72°F.",
    "Section2": "It’s clear and warm today."
}

# Populate the answers dictionary
for section, content in context.items():
    section_answers = answer_questions(content, questions)
    for i, answer in enumerate(section_answers):
        answers[questions[i]].append((section, answer))

# Get combined responses
combined_responses = get_all_combined_responses(answers)
print(json.dumps(combined_responses, indent=2))