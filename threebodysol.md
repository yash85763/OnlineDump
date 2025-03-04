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
from typing import List, Dict, Tuple

# Simulated LLM call function (replace with your actual function)
# def call_llm(template: str) -> str:
#     # Example: return "Generated response from LLM"
#     pass

# Template creation function
def create_template(context: str, question: str) -> str:
    """Create a single template string with instructions, context, and question."""
    template = (
        "You are an AI tasked with generating or analyzing synthetic data for a database router classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate or analyze responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified database intents, if applicable.\n"
        "- Return results in the requested format.\n"
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
        f"Examples might include 'Who are my top customers?' or 'What’s the stock status of product X?'. "
        f"Return as a numbered list (e.g., 1. text, 2. text)."
    )
    template = create_template(context, question)
    response = call_llm(template)
    queries = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
    queries = [re.sub(r"^\d+\.\s", "", q) for q in queries]
    return queries[:num_queries]

# Intent LLM: Classify database intents
def intent_llm(context: str, queries: List[str]) -> List[Tuple[str, str]]:
    """Classify which database each query should route to."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What’s the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What’s the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n"
        f"Classify which database each of the following queries should route to. "
        f"Return as a numbered list with the format: '1. [Database] - text'.\n"
        f"Queries:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    )
    template = create_template(context, question)
    response = call_llm(template)
    pairs = []
    for line in response.split("\n"):
        if re.match(r"^\d+\.\s\[.*\]\s-\s", line):
            match = re.match(r"^\d+\.\s\[(.*?)\]\s-\s(.*)$", line.strip())
            if match:
                intent, text = match.groups()
                pairs.append((text, intent))
    return pairs[:len(queries)]

# Supervisor LLM: Validate queries and database intents
def supervisor_llm(context: str, query_intent_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Validate context alignment and database classification with reasoning."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What’s the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What’s the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n"
        f"For each query-database pair, determine:\n"
        f"1. Does the query fit the context of a data retrieval system? (Yes/No)\n"
        f"2. Is the database classification correct? (Yes/No)\n"
        f"3. Provide reasoning for your decisions.\n"
        f"Return as a numbered list with format: "
        f"'1. [Yes/No, Yes/No] - Reasoning: <reason>'.\n"
        f"Pairs:\n" + "\n".join([f"{i+1}. '{q}' - {intent}" for i, (q, intent) in enumerate(query_intent_pairs)])
    )
    template = create_template(context, question)
    response = call_llm(template)
    validated_data = []
    for line, (query, intent) in zip(response.split("\n"), query_intent_pairs):
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

# Main function to generate dataset for database router
def generate_database_router_dataset(
    context: str,
    intents: List[str],
    num_queries: int = 200
) -> List[Dict[str, str]]:
    """Generate synthetic dataset for a database router using the three-body system."""
    dataset = []

    # Step 1: Curiosity LLM generates queries
    print("Generating queries...")
    queries = curiosity_llm(context, num_queries)
    print(f"Generated {len(queries)} queries.")

    # Step 2: Intent LLM classifies database intents
    print("Classifying database intents...")
    query_intent_pairs = intent_llm(context, queries)
    print(f"Classified {len(query_intent_pairs)} pairs.")

    # Step 3: Supervisor LLM validates
    print("Validating with supervisor...")
    validated_data = supervisor_llm(context, query_intent_pairs)
    print(f"Validated {len(validated_data)} pairs.")

    # Collect validated data with intent filtering
    intent_counts = {intent: 0 for intent in intents}
    for entry in validated_data:
        if entry["intent"] in intents:
            intent_counts[entry["intent"]] += 1
            dataset.append({"text": entry["text"], "intent": entry["intent"]})

    print("Database intent distribution:", intent_counts)
    return dataset

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "database_router_dataset.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename} with {len(dataset)} examples.")

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