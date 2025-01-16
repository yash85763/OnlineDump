# utils/llm_processor.py

import openai
from typing import Optional

class LLMProcessor:
    def __init__(self, api_key: str):
        """
        Initialize the LLM processor.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        openai.api_key = api_key

    def process_reference(
        self,
        original_context: str,
        reference_text: str,
        referenced_content: str
    ) -> str:
        """
        Process a reference using LLM to extract relevant content.
        
        Args:
            original_context (str): The original text containing the reference
            reference_text (str): The actual reference text
            referenced_content (str): The full content of the referenced section
            
        Returns:
            str: Relevant extract from the referenced content
        """
        try:
            prompt = f"""Given the following context and reference:

ORIGINAL TEXT CONTEXT:
{original_context}

REFERENCE MARKER:
{reference_text}

FULL REFERENCED CONTENT:
{referenced_content}

Task: Analyze the original context where the reference appears and the full referenced content. 
Extract only the specific parts of the referenced content that are directly relevant to how 
the reference is being used in the original context. Consider:

1. If it's referencing a definition, extract only the relevant definition
2. If it's referencing a procedure, extract only the relevant procedural steps
3. If it's referencing requirements, extract only the pertinent requirements
4. Maintain legal accuracy while being concise
5. IMPORTANT: Ignore any references within the referenced content itself

Provide only the relevant extract, no explanations or additional text."""

            openai_client = openai.OpenAI(api_key="your_api_key")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal expert assistant that helps extract relevant information from legal references while maintaining accuracy and context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            return referenced_content
