import openai
from typing import List
from .logger import logger

class QAProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key

    def read_regulation_file(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = content.split("CONTENT:\n")
        processed_content = []
        for section in sections[1:]:
            end = section.find("\nFOOTNOTES:") if "\nFOOTNOTES:" in section else section.find("\n=")
            if end == -1:
                processed_content.append(section.strip())
            else:
                processed_content.append(section[:end].strip())
        return "\n\n".join(processed_content)

    def answer_questions(self, context: str, questions: List[str]) -> List[str]:
        try:
            prompt = f"""Based on this regulation content:

{context}

Please answer these questions directly and concisely:

{chr(10).join(f'Q: {q}' for q in questions)}"""

            openai_client = openai.OpenAI(api_key="your_api_key")
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide direct, concise answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            return [response.choices[0].message.content.strip()]

        except Exception as e:
            logger.error(f"Error in Q&A processing: {str(e)}")
            return [f"Error processing questions: {str(e)}"]
