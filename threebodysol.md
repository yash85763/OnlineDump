Thanks for the additional context! That’s a fascinating use case—building a BERT-based router to classify which database to use based on user queries. Since your intents represent database choices, the synthetic data needs to reflect realistic queries that imply a specific database, and the classifier will learn to route them accordingly. Let’s refine the previous "three-body solution" to fit this scenario.

### Updated Context and Intents
- **Context**: "A data retrieval system with multiple databases."
- **Intents (Databases)**: Let’s assume you have four databases as your intents (you can adjust these):
  - `CustomerDB`: For customer-related data (e.g., "Who bought the most last month?").
  - `ProductDB`: For product-related data (e.g., "What’s the stock level of item X?").
  - `OrderDB`: For order-related data (e.g., "When did order 123 ship?").
  - `AnalyticsDB`: For analytical queries (e.g., "What’s the average sales trend?").

### Goal
Generate synthetic queries that a user might ask, labeled with the appropriate database intent, to train your BERT classifier to route queries to the correct database.

### Adapted Three-Body Solution
- **Curiosity LLM**: Generates queries related to the data retrieval system.
- **Intent LLM**: Classifies which database each query should route to.
- **Supervisor LLM**: Ensures the query fits the context and the database classification makes sense.

Below is the updated codebase tailored to your database router use case.

---

### Complete Codebase
```python

import json
import random
import re
import logging
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # Look for JSON within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                json_content = json_match.group(1)
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Look for JSON without code blocks
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logging.warning("Failed to parse JSON from LLM response")
        return None

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with a default fallback."""
    return data.get(key, default)

def validate_json_format(data: Any, expected_format: Dict[str, type]) -> bool:
    """
    Validate if parsed JSON matches expected format.
    
    Args:
        data: Parsed JSON object
        expected_format: Dictionary mapping keys to expected types
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    for key, expected_type in expected_format.items():
        if key not in data:
            return False
        
        if expected_type == list:
            if not isinstance(data[key], list):
                return False
        elif not isinstance(data[key], expected_type):
            return False
    
    return True

# Template creation function
def create_template(context: str, question: str) -> str:
    """Create a single template string with instructions, context, and question."""
    template = (
        "You are an AI tasked with generating or analyzing synthetic data for a database router classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate or analyze responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified database intents, if applicable.\n"
        "- Return results ONLY in the exact JSON format requested in the question.\n"
        "- Do not wrap your JSON in markdown code blocks, provide explanatory text, or any content outside the JSON.\n"
        f"Context: {context}\n"
        f"Question: {question}"
    )
    return template

# Curiosity LLM: Generate queries
def curiosity_llm(context: str, num_queries: int) -> List[str]:
    """Generate queries based on the context of a data retrieval system."""
    question = (
        f"Generate {num_queries} unique user queries related to the context. "
        f"These should reflect realistic questions a user might ask about customers, products, orders, or analytics. "
        f"Examples might include 'Who are my top customers?' or 'What's the stock status of product X?'. "
        f"Return as a JSON array of strings with the format: "
        f"{{\"queries\": [\"query1\", \"query2\", ...]}}"
    )
    template = create_template(context, question)
    response = call_llm(template)
    
    json_data = parse_json_from_llm_response(response)
    if not json_data or not isinstance(json_data, dict):
        logging.error("Failed to parse queries from LLM response")
        return []
    
    queries = safe_get(json_data, "queries", [])
    if not queries or not isinstance(queries, list):
        logging.error("Invalid queries format in LLM response")
        return []
    
    # Ensure we only return strings
    valid_queries = [q for q in queries if isinstance(q, str) and q.strip()]
    return valid_queries[:num_queries]

# Intent LLM: Classify database intents
def intent_llm(context: str, queries: List[str]) -> List[Tuple[str, str]]:
    """Classify which database each query should route to."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What's the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What's the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n\n"
        f"Classify which database each of the following queries should route to.\n\n"
        f"IMPORTANT: Your response must be ONLY a valid JSON object (not an array) with exactly this structure:\n"
        f"{{\"classifications\": [\n"
        f"  {{\"query\": \"first query text\", \"intent\": \"DatabaseName\"}},\n"
        f"  {{\"query\": \"second query text\", \"intent\": \"DatabaseName\"}},\n"
        f"  ... and so on for each query\n"
        f"]}}\n\n"
        f"Do not include any explanations, markdown formatting, or surrounding text. "
        f"Return only the JSON object.\n\n"
        f"Queries to classify:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    )
    
    template = create_template(context, question)
    response = call_llm(template)
    
    json_data = parse_json_from_llm_response(response)
    if not json_data or not isinstance(json_data, dict):
        logging.error("Failed to parse classifications from LLM response")
        return []
    
    classifications = safe_get(json_data, "classifications", [])
    if not classifications or not isinstance(classifications, list):
        logging.error("Invalid classifications format in LLM response")
        return []
    
    pairs = []
    for item in classifications:
        if not isinstance(item, dict):
            continue
        
        query = safe_get(item, "query", "")
        intent = safe_get(item, "intent", "")
        
        if query and intent and isinstance(query, str) and isinstance(intent, str):
            pairs.append((query, intent))
    
    return pairs[:len(queries)]

# Supervisor LLM: Validate queries and database intents
def supervisor_llm(context: str, query_intent_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Validate context alignment and database classification with reasoning."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What's the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What's the average sales trend?')."
    )
    
    # Prepare pairs for the prompt
    pairs_for_prompt = [{"query": q, "intent": i} for q, i in query_intent_pairs]
    
    question = (
        f"{intent_definitions}\n\n"
        f"For each query-database pair, determine:\n"
        f"1. Does the query fit the context of a data retrieval system? (true/false)\n"
        f"2. Is the database classification correct? (true/false)\n"
        f"3. Provide reasoning for your decisions.\n\n"
        f"IMPORTANT: Your response must be ONLY a valid JSON object (not an array) with exactly this structure:\n"
        f"{{\"validations\": [\n"
        f"  {{\"query\": \"query text\", \"intent\": \"DatabaseName\", \"context_fit\": true/false, \"intent_correct\": true/false, \"reasoning\": \"explanation\"}},\n"
        f"  ... and so on for each pair\n"
        f"]}}\n\n"
        f"Do not include any explanations, markdown formatting, or surrounding text. "
        f"Return only the JSON object.\n\n"
        f"Pairs to validate: {json.dumps(pairs_for_prompt, indent=2)}"
    )
    template = create_template(context, question)
    response = call_llm(template)
    
    json_data = parse_json_from_llm_response(response)
    if not json_data or not isinstance(json_data, dict):
        logging.error("Failed to parse validations from LLM response")
        return []
    
    validations = safe_get(json_data, "validations", [])
    if not validations or not isinstance(validations, list):
        logging.error("Invalid validations format in LLM response")
        return []
    
    validated_data = []
    for item in validations:
        if not isinstance(item, dict):
            continue
        
        query = safe_get(item, "query", "")
        intent = safe_get(item, "intent", "")
        context_fit = safe_get(item, "context_fit", False)
        intent_correct = safe_get(item, "intent_correct", False)
        reasoning = safe_get(item, "reasoning", "")
        
        if (query and intent and isinstance(query, str) and isinstance(intent, str) 
            and isinstance(context_fit, bool) and isinstance(intent_correct, bool)):
            if context_fit and intent_correct:
                validated_data.append({
                    "text": query,
                    "intent": intent,
                    "reasoning": reasoning
                })
    
    return validated_data

# Main function to generate dataset for database router
def generate_database_router_dataset(
    context: str,
    intents: List[str],
    num_queries: int = 200
) -> List[Dict[str, str]]:
    """Generate synthetic dataset for a database router using the three-body system."""
    dataset = []

    # Step 1: Curiosity LLM generates queries
    logging.info("Generating queries...")
    queries = curiosity_llm(context, num_queries)
    logging.info(f"Generated {len(queries)} queries.")
    
    if not queries:
        logging.error("Failed to generate any valid queries.")
        return []

    # Step 2: Intent LLM classifies database intents
    logging.info("Classifying database intents...")
    query_intent_pairs = intent_llm(context, queries)
    logging.info(f"Classified {len(query_intent_pairs)} pairs.")
    
    if not query_intent_pairs:
        logging.error("Failed to classify any queries.")
        return []

    # Step 3: Supervisor LLM validates
    logging.info("Validating with supervisor...")
    validated_data = supervisor_llm(context, query_intent_pairs)
    logging.info(f"Validated {len(validated_data)} pairs.")

    # Collect validated data with intent filtering
    intent_counts = {intent: 0 for intent in intents}
    for entry in validated_data:
        if entry["intent"] in intents:
            intent_counts[entry["intent"]] += 1
            dataset.append({"text": entry["text"], "intent": entry["intent"]})

    logging.info("Database intent distribution: %s", intent_counts)
    return dataset

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "database_router_dataset.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"data": dataset}, f, indent=2)
        logging.info(f"Dataset saved to {filename} with {len(dataset)} examples.")
    except Exception as e:
        logging.error(f"Failed to save dataset: {str(e)}")

# Additional helper function for retry logic
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

# Function to check dataset quality
def check_dataset_quality(dataset: List[Dict[str, str]], intents: List[str]) -> Dict[str, Any]:
    """Check dataset quality and provide statistics."""
    if not dataset:
        return {"status": "error", "message": "Empty dataset"}
    
    total_examples = len(dataset)
    intent_counts = {intent: 0 for intent in intents}
    query_lengths = []
    
    for entry in dataset:
        if entry["intent"] in intent_counts:
            intent_counts[entry["intent"]] += 1
        query_lengths.append(len(entry["text"]))
    
    # Calculate distribution imbalance
    distribution = {intent: count/total_examples for intent, count in intent_counts.items()}
    
    # Check for duplicate queries
    texts = [entry["text"] for entry in dataset]
    unique_texts = set(texts)
    duplicate_count = len(texts) - len(unique_texts)
    
    return {
        "status": "success",
        "total_examples": total_examples,
        "intent_distribution": distribution,
        "duplicate_count": duplicate_count,
        "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        "min_query_length": min(query_lengths) if query_lengths else 0,
        "max_query_length": max(query_lengths) if query_lengths else 0,
    }

# Example usage
if __name__ == "__main__":
    # Define context and database intents
    context = "A data retrieval system with multiple databases"
    intents = ["CustomerDB", "ProductDB", "OrderDB", "AnalyticsDB"]

    # Generate dataset
    dataset = generate_database_router_dataset(
        context=context,
        intents=intents,
        num_queries=200  # Total queries to generate initially
    )

    # Check dataset quality
    quality_report = check_dataset_quality(dataset, intents)
    logging.info("Dataset quality report: %s", quality_report)

    # Save to file
    save_dataset(dataset)

    # Print sample
    print("\nSample of generated data:")
    for entry in dataset[:5]:
        print(f"Text: {entry['text']}, Intent: {entry['intent']}")

```

---

### Key Updates
1. **Context and Intents**:
   - Changed to reflect a data retrieval system with database-specific intents.
   - Intent definitions now describe which type of data each database handles.

2. **Curiosity LLM**:
   - Generates queries like "What’s the stock level of product X?" or "Who are my top customers?" to match the database use case.

3. **Intent LLM**:
   - Classifies queries into `CustomerDB`, `ProductDB`, `OrderDB`, or `AnalyticsDB` based on their content.

4. **Supervisor LLM**:
   - Validates that queries fit the data retrieval system context and that the database intent aligns with the query’s purpose.

### Example Workflow
- **Curiosity LLM Output**:
  ```
  1. Who are my top customers this year?
  2. What’s the stock level of item XYZ?
  ```
- **Intent LLM Output**:
  ```
  1. [CustomerDB] - Who are my top customers this year?
  2. [ProductDB] - What’s the stock level of item XYZ?
  ```
- **Supervisor LLM Output**:
  ```
  1. [Yes, Yes] - Reasoning: Query seeks customer data, correctly routed to CustomerDB.
  2. [Yes, Yes] - Reasoning: Query asks about product stock, correctly routed to ProductDB.
  ```
- **Dataset Entry**:
  ```json
  [
    {"text": "Who are my top customers this year?", "intent": "CustomerDB"},
    {"text": "What’s the stock level of item XYZ?", "intent": "ProductDB"}
  ]
  ```

### Dataset Size
- Starting with `num_queries = 200`, you’ll get a subset after validation (e.g., 150–180 entries), depending on Supervisor approvals.
- For a balanced dataset (~200 per intent, 800 total), increase `num_queries` to 1000 and add balancing logic if needed.

### Customization
- **Your Databases**: Replace `CustomerDB`, `ProductDB`, etc., with your actual database names and update their definitions.
- **Balancing**: Add a step to ensure roughly equal distribution across intents (e.g., cap at 200 per intent or regenerate for underrepresented ones).
- **Query Variety**: Tweak the Curiosity LLM prompt to include specific keywords or patterns your users might use.

### Next Steps
1. Replace `call_llm` with your actual function.
2. Run the code and inspect `database_router_dataset.json`.
3. Use the dataset with BERT (e.g., via Hugging Face) by tokenizing `text` and mapping `intent` to numeric labels.

What are your actual database names and their purposes? I can fine-tune the intent definitions and prompts further. Also, do you want to add balancing or refine the query generation process?
