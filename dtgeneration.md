Your idea of generating a batch of 10 questions from a given context and then validating them fits well with your database router use case. Since you’re building a BERT-based classifier to route queries to specific databases, I’ll adapt the combo approach to generate batches of 10 queries at a time and propose a validation strategy tailored to your needs. I’ll assume your LLM function still takes a single `template` string (`call_llm(template)`), and I’ll update the codebase accordingly.

### Updated Context and Intents
- **Context**: "A data retrieval system with multiple databases."
- **Intents (Databases)**: 
  - `CustomerDB`: Customer-related queries.
  - `ProductDB`: Product-related queries.
  - `OrderDB`: Order-related queries.
  - `AnalyticsDB`: Analytical queries.

### Batch Generation Strategy
- Generate 10 queries per batch, each tied to a specific database intent.
- Use the combo approach (initial generation + paraphrasing + noise) to scale up after validation.
- Validate each batch to ensure quality before adding to the dataset.

### Validation Strategy
Since the three-body solution might be overkill for your needs (and requires three LLM calls per batch), I’ll propose a simpler yet effective validation method using a single LLM call per batch:
- **Validation LLM**: 
  - Checks two things for each query:
    1. **Context Fit**: Does the query align with the data retrieval system context?
    2. **Intent Accuracy**: Does the query match the assigned database intent?
  - Provides reasoning for approval or rejection.
- **Process**:
  - Generate a batch of 10 queries for a specific intent.
  - Validate the batch, keeping only approved queries.
  - Augment and inject noise on validated queries to scale the dataset.

Here’s the updated codebase:

---

### Complete Codebase with Batch Generation and Validation
```python

import json
import random
import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Simulated LLM call function (replace with your actual function)
# def call_llm(template: str) -> str:
#     # Example: return "Generated response from LLM"
#     pass

# JSON parsing helper functions
def parse_json_from_llm_response(response: str) -> Optional[Any]:
    """
    Extract and parse JSON from an LLM response, handling various edge cases.
    
    Args:
        response: Raw text response from LLM
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    if not response:
        logging.error("Empty response received from LLM")
        return None
        
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # Log original response for debugging (truncated if too long)
        log_response = response[:500] + "..." if len(response) > 500 else response
        logging.debug(f"JSON parsing failed. Response: {log_response}")
        
        # Look for JSON within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                json_content = json_match.group(1)
                return json.loads(json_content)
            except json.JSONDecodeError:
                logging.debug("Failed to parse JSON from code block")
        
        # Look for JSON without code blocks
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Fall back to regular expression parsing for list format
        queries = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
        if queries:
            queries = [re.sub(r"^\d+\.\s", "", q) for q in queries]
            return {"items": queries}  # Return as JSON-compatible dict
        
        logging.warning("Failed to parse JSON or list format from LLM response")
        return None

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with a default fallback."""
    return data.get(key, default)

# Template creation function with improved JSON instructions
def create_template(context: str, question: str, json_output: bool = False) -> str:
    """Create a single template string with instructions, context, and question."""
    format_instructions = (
        "- Return results ONLY in the exact JSON format requested in the question.\n"
        "- Do not wrap your JSON in markdown code blocks, provide explanatory text, or any content outside the JSON.\n"
    ) if json_output else (
        "- Return results as a numbered list (e.g., 1. text, 2. text) unless specified otherwise.\n"
    )
    
    template = (
        "You are an AI tasked with generating or validating synthetic data for a database router classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate or analyze responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified database intents, if applicable.\n"
        f"{format_instructions}"
        f"Context: {context}\n"
        f"Question: {question}"
    )
    return template

# Step 1: Generate a batch of queries using JSON output
def generate_batch(context: str, intent: str, batch_size: int = 10) -> List[str]:
    """Generate a batch of queries for a given database intent."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What's the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What's the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n\n"
        f"Generate {batch_size} unique user queries with the intent '{intent}'. "
        f"These should reflect realistic questions a user might ask in a data retrieval system.\n\n"
        f"IMPORTANT: Your response must be ONLY a valid JSON object with exactly this structure:\n"
        f"{{\"queries\": [\n"
        f"  \"first query text\",\n"
        f"  \"second query text\",\n"
        f"  ... and so on\n"
        f"]}}\n\n"
        f"Do not include any explanations, markdown formatting, or surrounding text."
    )
    template = create_template(context, question, json_output=True)
    response = call_llm(template)
    
    json_data = parse_json_from_llm_response(response)
    if not json_data or not isinstance(json_data, dict):
        logging.error("Failed to parse queries from LLM response")
        # Fall back to regex parsing if JSON parsing fails
        queries = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
        queries = [re.sub(r"^\d+\.\s", "", q) for q in queries]
        return queries[:batch_size]
    
    # Try both "queries" and "items" keys (from fallback parsing)
    queries = safe_get(json_data, "queries", []) or safe_get(json_data, "items", [])
    if not queries or not isinstance(queries, list):
        logging.error("Invalid queries format in LLM response")
        return []
    
    # Ensure we only return strings
    valid_queries = [q for q in queries if isinstance(q, str) and q.strip()]
    return valid_queries[:batch_size]

# Validation: Validate the batch using JSON output
def validate_batch(context: str, intent: str, queries: List[str]) -> List[Dict[str, str]]:
    """Validate a batch of queries for context fit and intent accuracy."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What's the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What's the average sales trend?')."
    )
    
    # Prepare the queries for the prompt
    queries_for_prompt = [{"id": i+1, "query": q} for i, q in enumerate(queries)]
    
    question = (
        f"{intent_definitions}\n\n"
        f"For each query, determine:\n"
        f"1. Does the query fit the context of a data retrieval system? (true/false)\n"
        f"2. Does the query match the intent '{intent}'? (true/false)\n"
        f"3. Provide reasoning for your decisions.\n\n"
        f"IMPORTANT: Your response must be ONLY a valid JSON object with exactly this structure:\n"
        f"{{\"validations\": [\n"
        f"  {{\"query\": \"query text\", \"context_fit\": true/false, \"intent_match\": true/false, \"reasoning\": \"explanation\"}},\n"
        f"  ... and so on for each query\n"
        f"]}}\n\n"
        f"Do not include any explanations, markdown formatting, or surrounding text.\n\n"
        f"Queries to validate: {json.dumps(queries_for_prompt, indent=2)}"
    )
    
    template = create_template(context, question, json_output=True)
    response = call_llm(template)
    
    json_data = parse_json_from_llm_response(response)
    if not json_data or not isinstance(json_data, dict):
        logging.error("Failed to parse validations from LLM response")
        # Fall back to regex parsing
        validated_data = []
        for line, query in zip(response.split("\n"), queries):
            if re.match(r"^\d+\.\s\[.*\]\s-\sReasoning:", line):
                match = re.match(r"^\d+\.\s\[(Yes|No),\s(Yes|No)\]\s-\sReasoning:\s(.*)$", line.strip())
                if match:
                    context_ok, intent_ok, reasoning = match.groups()
                    if context_ok == "Yes" and intent_ok == "Yes":
                        validated_data.append({
                            "text": query,
                            "intent": intent,
                            "reasoning": reasoning
                        })
        return validated_data
    
    # Handle both potential JSON formats
    validations = None
    if isinstance(json_data, dict):
        # Try several possible keys
        validations = (safe_get(json_data, "validations") or 
                       safe_get(json_data, "validation") or 
                       safe_get(json_data, "results"))
    
    if not validations or not isinstance(validations, list):
        logging.error("Invalid validations format in LLM response")
        return []
    
    validated_data = []
    for item in validations:
        if not isinstance(item, dict):
            continue
        
        query = safe_get(item, "query", "")
        context_fit = safe_get(item, "context_fit", False)
        intent_match = safe_get(item, "intent_match", False)
        reasoning = safe_get(item, "reasoning", "")
        
        # Convert string boolean representations if needed
        if isinstance(context_fit, str):
            context_fit = context_fit.lower() in ("true", "yes", "1")
        if isinstance(intent_match, str):
            intent_match = intent_match.lower() in ("true", "yes", "1")
            
        if query and isinstance(query, str):
            if context_fit and intent_match:
                validated_data.append({
                    "text": query,
                    "intent": intent,
                    "reasoning": reasoning
                })
    
    return validated_data

# Step 2: Augmentation with Paraphrasing using JSON
def augment_with_paraphrasing(context: str, examples: List[Dict[str, str]], num_paraphrases: int) -> List[Dict[str, str]]:
    """Augment validated queries with paraphrases."""
    augmented_data = examples.copy()
    
    for example in examples:
        query_text = example['text']
        intent = example['intent']
        
        question = (
            f"Generate {num_paraphrases} paraphrased versions of this query: '{query_text}'. "
            f"Keep the same database intent '{intent}'.\n\n"
            f"IMPORTANT: Your response must be ONLY a valid JSON object with exactly this structure:\n"
            f"{{\"paraphrases\": [\n"
            f"  \"first paraphrased version\",\n"
            f"  \"second paraphrased version\",\n"
            f"  ... and so on\n"
            f"]}}\n\n"
            f"Do not include any explanations, markdown formatting, or surrounding text."
        )
        
        template = create_template(context, question, json_output=True)
        response = call_llm(template)
        
        json_data = parse_json_from_llm_response(response)
        if not json_data or not isinstance(json_data, dict):
            logging.error(f"Failed to parse paraphrases for query: {query_text}")
            # Fall back to regex parsing
            paraphrases = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
            paraphrases = [re.sub(r"^\d+\.\s", "", para) for para in paraphrases]
        else:
            # Try multiple possible keys
            paraphrases = (safe_get(json_data, "paraphrases", []) or 
                           safe_get(json_data, "results", []) or
                           safe_get(json_data, "items", []))
            
            if not paraphrases or not isinstance(paraphrases, list):
                logging.error(f"Invalid paraphrases format for query: {query_text}")
                continue
        
        # Only add valid string paraphrases
        valid_paraphrases = [p for p in paraphrases if isinstance(p, str) and p.strip()]
        augmented_data.extend([{"text": para, "intent": intent} for para in valid_paraphrases[:num_paraphrases]])
    
    return augmented_data

# Step 3: Noise Injection for Robustness using JSON
def inject_noise(context: str, examples: List[Dict[str, str]], num_noisy: int) -> List[Dict[str, str]]:
    """Add noise to some queries for robustness."""
    noisy_data = examples.copy()
    
    if num_noisy <= 0 or len(examples) == 0:
        return noisy_data
    
    sampled_examples = random.sample(examples, min(num_noisy, len(examples)))
    
    for example in sampled_examples:
        query_text = example['text']
        intent = example['intent']
        
        question = (
            f"Rewrite this query with typos, slang, or casual phrasing: '{query_text}'. "
            f"Keep the same database intent '{intent}'.\n\n"
            f"IMPORTANT: Your response must be ONLY a valid JSON object with exactly this structure:\n"
            f"{{\"noisy_query\": \"the rewritten query with noise\"}}\n\n"
            f"Do not include any explanations, markdown formatting, or surrounding text."
        )
        
        template = create_template(context, question, json_output=True)
        response = call_llm(template)
        
        json_data = parse_json_from_llm_response(response)
        if not json_data or not isinstance(json_data, dict):
            logging.error(f"Failed to parse noisy query for: {query_text}")
            # Use the raw response as fallback if it's not empty
            if response and response.strip():
                noisy_data.append({"text": response.strip(), "intent": intent})
        else:
            noisy_query = safe_get(json_data, "noisy_query", "")
            if noisy_query and isinstance(noisy_query, str):
                noisy_data.append({"text": noisy_query, "intent": intent})
    
    return noisy_data

# Retry logic for LLM calls
def retry_llm_call(template: str, max_retries: int = 3) -> str:
    """Retry LLM call with backoff in case of failures."""
    import time
    
    for attempt in range(max_retries):
        try:
            response = call_llm(template)
            if response:
                return response
        except Exception as e:
            logging.warning(f"LLM call attempt {attempt+1} failed: {str(e)}")
            
        # Exponential backoff
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    logging.error("All LLM call attempts failed")
    return ""

# Main function to generate the dataset with batches
def generate_synthetic_dataset(
    context: str,
    intents: List[str],
    batches_per_intent: int = 5,
    batch_size: int = 10,
    paraphrases_per_query: int = 3,
    noisy_per_batch: int = 2
) -> List[Dict[str, str]]:
    """Generate synthetic dataset using batches with validation."""
    dataset = []
    intent_counts = {intent: 0 for intent in intents}

    for intent in intents:
        logging.info(f"Generating data for database intent: {intent}")
        intent_data = []

        # Generate and validate batches
        for batch_num in range(batches_per_intent):
            logging.info(f"Processing batch {batch_num + 1}/{batches_per_intent}")
            
            # Generate batch
            batch_queries = generate_batch(context, intent, batch_size)
            logging.info(f"Generated {len(batch_queries)} queries in batch.")
            
            if not batch_queries:
                logging.warning(f"Failed to generate queries for batch {batch_num + 1}. Skipping...")
                continue

            # Validate batch
            validated_queries = validate_batch(context, intent, batch_queries)
            logging.info(f"Validated {len(validated_queries)}/{len(batch_queries)} queries.")
            intent_data.extend(validated_queries)

        logging.info(f"Total validated queries for {intent}: {len(intent_data)}")
        
        if not intent_data:
            logging.warning(f"No valid queries generated for {intent}. Skipping augmentation...")
            continue

        # Step 2: Augment validated queries
        augmented_data = augment_with_paraphrasing(context, intent_data, paraphrases_per_query)
        logging.info(f"After paraphrasing: {len(augmented_data)} queries.")

        # Step 3: Inject noise
        final_data = inject_noise(context, augmented_data, noisy_per_batch * batches_per_intent)
        logging.info(f"After noise injection: {len(final_data)} queries.")

        # Add to dataset
        dataset.extend(final_data)
        intent_counts[intent] = len(final_data)

    # Log distribution statistics
    total = sum(intent_counts.values())
    if total > 0:
        distribution = {intent: f"{count} ({count/total*100:.1f}%)" for intent, count in intent_counts.items()}
        logging.info(f"Intent distribution: {distribution}")
    
    return dataset

# Check dataset quality
def check_dataset_quality(dataset: List[Dict[str, str]], intents: List[str]) -> Dict[str, Any]:
    """Check dataset quality and provide statistics."""
    if not dataset:
        return {"status": "error", "message": "Empty dataset"}
    
    total_examples = len(dataset)
    intent_counts = {intent: 0 for intent in intents}
    query_lengths = []
    query_word_counts = []
    
    # Set for duplicate detection
    unique_texts = set()
    duplicates = []
    
    for entry in dataset:
        if entry["intent"] in intent_counts:
            intent_counts[entry["intent"]] += 1
        
        query_text = entry["text"]
        query_lengths.append(len(query_text))
        query_word_counts.append(len(query_text.split()))
        
        # Check for duplicates
        if query_text in unique_texts:
            duplicates.append(query_text)
        else:
            unique_texts.add(query_text)
    
    # Calculate distribution balance
    if total_examples > 0:
        expected_per_intent = total_examples / len(intents)
        max_deviation = max(abs(count - expected_per_intent) for count in intent_counts.values())
        balance_score = 1 - (max_deviation / total_examples)
    else:
        balance_score = 0
    
    return {
        "status": "success",
        "total_examples": total_examples,
        "intent_counts": intent_counts,
        "balance_score": balance_score,  # 1.0 is perfectly balanced
        "duplicate_count": len(duplicates),
        "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        "avg_word_count": sum(query_word_counts) / len(query_word_counts) if query_word_counts else 0,
        "min_query_length": min(query_lengths) if query_lengths else 0,
        "max_query_length": max(query_lengths) if query_lengths else 0,
    }

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "database_router_batch_dataset.json"):
    """Save dataset to JSON file with error handling."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"data": dataset, "metadata": {"generated_at": datetime.now().isoformat()}}, f, indent=2)
        logging.info(f"Dataset saved to {filename} with {len(dataset)} examples.")
    except Exception as e:
        logging.error(f"Failed to save dataset: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Define context and database intents
        context = "A data retrieval system with multiple databases"
        intents = ["CustomerDB", "ProductDB", "OrderDB", "AnalyticsDB"]

        # Override call_llm for testing
        def call_llm(template: str) -> str:
            # This is a placeholder - replace with your actual LLM call
            import time
            time.sleep(0.1)  # Simulate API call latency
            return '{"queries": ["Who are my top customers?", "What\'s the status of product X?"]}'

        # Generate dataset with smaller values for testing
        dataset = generate_synthetic_dataset(
            context=context,
            intents=intents,
            batches_per_intent=2,       # 2 batches per intent (for testing)
            batch_size=5,               # 5 queries per batch (for testing)
            paraphrases_per_query=2,    # 2 paraphrases per validated query
            noisy_per_batch=1           # 1 noisy version per batch
        )

        # Check quality
        quality_report = check_dataset_quality(dataset, intents)
        logging.info(f"Dataset quality report: {quality_report}")

        # Save to file
        save_dataset(dataset)

        # Print sample
        print("\nSample of generated data:")
        for entry in dataset[:5]:
            print(f"Text: {entry['text']}, Intent: {entry['intent']}")
            
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {str(e)}", exc_info=True)

```

---

### How It Works
1. **Batch Generation (`generate_batch`)**:
   - Generates 10 queries per batch for a specific intent (e.g., "Who are my top customers?" for `CustomerDB`).

2. **Validation (`validate_batch`)**:
   - For each query in the batch:
     - Checks if it fits the context ("A data retrieval system").
     - Confirms it matches the intended database intent.
     - Provides reasoning (e.g., "Yes, Yes - Reasoning: Query seeks customer data, matches CustomerDB").
   - Only queries with `[Yes, Yes]` are kept.

3. **Augmentation (`augment_with_paraphrasing`)**:
   - Takes validated queries and generates 3 paraphrases each (e.g., "Who are my top customers?" → "Which customers spent the most?").

4. **Noise Injection (`inject_noise`)**:
   - Adds 2 noisy versions per batch (e.g., "When did order 123 ship?" → "Wen did ord 123 go out?").

5. **Main Function (`generate_synthetic_dataset`)**:
   - Loops through intents, generating 5 batches of 10 queries each (50 initial queries per intent).
   - Validates, augments, and injects noise to build the final dataset.

### Dataset Size (Per Intent)
- Initial: 5 batches * 10 queries = 50 queries.
- After validation: ~40–50 queries (assuming 80–100% pass rate).
- After paraphrasing: 40–50 * (1 + 3) = 160–200 queries.
- After noise: 160–200 + (5 * 2) = 170–210 queries.
- Total for 4 intents: ~680–840 queries.

### Validation Details
- **Why This Approach**: A single LLM validation step is efficient (one call per batch) and ensures quality without the complexity of the three-body system.
- **Output Format**: 
  ```
  1. [Yes, Yes] - Reasoning: Query seeks customer data, matches CustomerDB intent.
  2. [No, Yes] - Reasoning: Query is about weather, not data retrieval system.
  ```
- **Filtering**: Only `[Yes, Yes]` queries proceed to augmentation.

### Customization
- **Your Databases**: Update `intents` and `intent_definitions` with your actual database names and purposes.
- **Batch Size**: Change `batch_size` or `batches_per_intent` to adjust the initial query count.
- **Validation Strictness**: Modify the validation prompt if you want stricter or looser criteria.

### Next Steps
1. Replace `call_llm` with your actual function.
2. Run and inspect `database_router_batch_dataset.json`.
3. Adjust `batches_per_intent` or `batch_size` if you need a larger dataset (e.g., 1000+ queries total).

What are your specific database names and their purposes? I can tailor the intent definitions further. Also, do you want a different batch size or validation tweak?
