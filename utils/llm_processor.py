import openai
from typing import Optional, List
import tiktoken
from .logger import logger

class LLMProcessor:
    def __init__(self, api_key: str):
        """
        Initialize the LLM processor.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 16000  # Leave some buffer for the prompt and response

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit while maintaining coherence."""
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Log the truncation
        logger.info(f"Truncating content from {current_tokens} to {max_tokens} tokens")
        
        # Truncate tokens and decode
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        # Try to end at a sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > 0:
            truncated_text = truncated_text[:last_period + 1]
            
        return truncated_text

    def process_reference(
        self,
        original_context: str,
        reference_text: str,
        referenced_content: str
    ) -> str:
        """
        Process a reference using LLM to extract relevant content.
        Handles large content by limiting context and splitting if necessary.
        """
        try:
            logger.debug(f"\n{'='*80}\nPROCESSING REFERENCE\n{'='*80}")
            logger.debug(f"Reference marker: {reference_text}")
            logger.debug(f"\nOriginal context snippet:\n{original_context[:200]}...")
            
            # Count initial tokens
            initial_tokens = self.count_tokens(referenced_content)
            logger.debug(f"\nInitial content size: {initial_tokens} tokens")
            
            # Calculate available tokens for content
            base_prompt = """Task: Analyze the context and reference to extract relevant content.
                        Maintain legal accuracy while being concise."""
            system_message = "You are a legal expert assistant that helps extract relevant information from legal references."
            
            base_tokens = self.count_tokens(base_prompt + system_message + original_context + reference_text)
            available_tokens = self.max_tokens - base_tokens - 500  # Reserve 500 tokens for response

            # Truncate if necessary
            if initial_tokens > available_tokens:
                referenced_content = self.truncate_to_token_limit(referenced_content, available_tokens)
                final_tokens = self.count_tokens(referenced_content)
                logger.info(f"Content truncated from {initial_tokens} to {final_tokens} tokens")
            
            logger.debug(f"\nProcessing content:\n{referenced_content}")

            prompt = f"""Given the following context and reference:

ORIGINAL TEXT CONTEXT:
{original_context}

REFERENCE MARKER:
{reference_text}

REFERENCED CONTENT:
{referenced_content}

Task: Analyze the original context where the reference appears and extract only the most relevant 
parts of the referenced content. Focus on:

1. If it's referencing a definition, extract only that specific definition
2. If it's referencing a procedure, extract only the relevant procedural steps
3. If it's referencing requirements, extract only the pertinent requirements
4. Keep only what's directly relevant to how the reference is used in the original context
5. Maintain legal accuracy while being concise
6. Ignore any nested references within the referenced content

Provide only the relevant extract, no explanations or additional text."""

            openai_client = openai.OpenAI(api_key="your_api_key")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            llm_output = response.choices[0].message.content.strip()
            
            # Log the LLM processing results
            logger.debug(f"\nLLM PROCESSING RESULTS\n{'-'*50}")
            logger.debug(f"Original content length: {len(referenced_content)} characters")
            logger.debug(f"Processed content length: {len(llm_output)} characters")
            logger.debug("\nProcessed content:")
            logger.debug(f"{llm_output}")
            logger.debug(f"\n{'='*80}\n")
            
            return llm_output

        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")
            # Return a truncated version of the original content as fallback
            return self.truncate_to_token_limit(referenced_content, 1000)  # Return first ~1000 tokens

    def split_and_process_large_content(
        self,
        original_context: str,
        reference_text: str,
        referenced_content: str
    ) -> str:
        """
        Handle very large content by splitting it into chunks and processing each separately.
        """
        try:
            logger.info("Starting split processing of large content")
            # Split content into ~10000 token chunks with some overlap
            chunk_size = 10000
            tokens = self.encoding.encode(referenced_content)
            chunks = []
            
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            logger.info(f"Split content into {len(chunks)} chunks")
            
            # Process each chunk
            relevant_extracts = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i} of {len(chunks)}")
                extract = self.process_reference(
                    original_context=original_context,
                    reference_text=reference_text,
                    referenced_content=chunk
                )
                if extract:
                    relevant_extracts.append(extract)
            
            # Combine extracts if multiple chunks were processed
            if len(relevant_extracts) > 1:
                logger.info("Combining and refining multiple extracts")
                # Send combined extracts for final refinement
                combined = " ".join(relevant_extracts)
                return self.process_reference(
                    original_context=original_context,
                    reference_text=reference_text,
                    referenced_content=combined
                )
            elif relevant_extracts:
                return relevant_extracts[0]
            else:
                return "Unable to process reference content."
                
        except Exception as e:
            logger.error(f"Error in split processing: {str(e)}")
            return self.truncate_to_token_limit(referenced_content, 1000)
