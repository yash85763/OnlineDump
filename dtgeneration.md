Since your intent is now clarified as routing queries to specific databases for a BERT-based classifier, I’ll update the "combo approach" solution (Context-Driven Freeform Generation + Augmentation with Paraphrasing + Noise Injection) to align with this use case. The updated codebase will generate synthetic queries for a data retrieval system, labeled with database intents (`CustomerDB`, `ProductDB`, `OrderDB`, `AnalyticsDB`), using the single-template LLM call function you specified.

### Updated Context and Intents
- **Context**: "A data retrieval system with multiple databases."
- **Intents (Databases)**:
  - `CustomerDB`: Customer-related queries.
  - `ProductDB`: Product-related queries.
  - `OrderDB`: Order-related queries.
  - `AnalyticsDB`: Analytical queries.

### Updated Combo Approach
1. **Context-Driven Freeform Generation**: Generate initial queries for each database intent.
2. **Augmentation with Paraphrasing**: Expand the dataset with paraphrased versions.
3. **Noise Injection**: Add realistic variations (typos, slang) for robustness.

Here’s the updated codebase:

---

### Complete Codebase for Combo Approach
```python
import json
import random
import re
from typing import List, Dict

# Simulated LLM call function (replace with your actual function)
# def call_llm(template: str) -> str:
#     # Example: return "Generated response from LLM"
#     pass

# Template creation function
def create_template(context: str, question: str) -> str:
    """Create a single template string with instructions, context, and question."""
    template = (
        "You are an AI tasked with generating synthetic data for a database router classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified database intents, if applicable.\n"
        "- Return results as a numbered list (e.g., 1. text, 2. text) unless specified otherwise.\n"
        f"Context: {context}\n"
        f"Question: {question}"
    )
    return template

# Step 1: Context-Driven Freeform Generation
def generate_initial_data(context: str, intent: str, num_examples: int) -> List[str]:
    """Generate initial synthetic queries for a given database intent."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What’s the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What’s the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n"
        f"Generate {num_examples} unique user queries with the intent '{intent}'. "
        f"These should reflect realistic questions a user might ask in a data retrieval system. "
        f"Return as a numbered list."
    )
    template = create_template(context, question)
    response = call_llm(template)
    examples = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
    examples = [re.sub(r"^\d+\.\s", "", ex) for ex in examples]
    return examples[:num_examples]

# Step 2: Augmentation with Paraphrasing
def augment_with_paraphrasing(context: str, examples: List[str], num_paraphrases: int) -> List[str]:
    """Augment data by generating paraphrases for each query."""
    augmented_data = examples.copy()
    for example in examples:
        question = (
            f"Generate {num_paraphrases} paraphrased versions of this query: '{example}'. "
            f"Keep the same database intent. Return as a numbered list."
        )
        template = create_template(context, question)
        response = call_llm(template)
        paraphrases = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
        paraphrases = [re.sub(r"^\d+\.\s", "", para) for para in paraphrases]
        augmented_data.extend(paraphrases[:num_paraphrases])
    return augmented_data

# Step 3: Noise Injection for Robustness
def inject_noise(context: str, examples: List[str], num_noisy: int) -> List[str]:
    """Add noise (typos, slang) to some queries for robustness."""
    noisy_data = examples.copy()
    sampled_examples = random.sample(examples, min(num_noisy, len(examples)))
    for example in sampled_examples:
        question = (
            f"Rewrite this query with typos, slang, or casual phrasing: '{example}'. "
            f"Keep the same database intent. Return one version."
        )
        template = create_template(context, question)
        response = call_llm(template)
        noisy_data.append(response.strip())
    return noisy_data

# Main function to generate the full dataset
def generate_synthetic_dataset(
    context: str,
    intents: List[str],
    initial_per_intent: int = 50,
    paraphrases_per_example: int = 3,
    noisy_per_intent: int = 20
) -> List[Dict[str, str]]:
    """Generate a complete synthetic dataset using the combo strategy."""
    dataset = []

    for intent in intents:
        print(f"Generating data for database intent: {intent}")
        
        # Step 1: Generate initial data
        initial_data = generate_initial_data(context, intent, initial_per_intent)
        print(f"Initial queries generated: {len(initial_data)}")

        # Step 2: Augment with paraphrasing
        augmented_data = augment_with_paraphrasing(context, initial_data, paraphrases_per_example)
        print(f"After paraphrasing: {len(augmented_data)}")

        # Step 3: Inject noise
        final_data = inject_noise(context, augmented_data, noisy_per_intent)
        print(f"After noise injection: {len(final_data)}")

        # Add to dataset with labels
        dataset.extend([{"text": text, "intent": intent} for text in final_data])

    return dataset

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "database_router_combo_dataset.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename} with {len(dataset)} examples.")

# Example usage
if __name__ == "__main__":
    # Define context and database intents
    context = "A data retrieval system with multiple databases"
    intents = ["CustomerDB", "ProductDB", "OrderDB", "AnalyticsDB"]

    # Generate dataset
    dataset = generate_synthetic_dataset(
        context=context,
        intents=intents,
        initial_per_intent=50,      # 50 initial queries per intent
        paraphrases_per_example=3,  # 3 paraphrases per initial query
        noisy_per_intent=20         # 20 noisy versions per intent
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
   - Updated to reflect a data retrieval system with database-specific intents.
   - Intent definitions now guide the LLM to generate queries aligned with `CustomerDB`, `ProductDB`, `OrderDB`, or `AnalyticsDB`.

2. **Generate Initial Data**:
   - Prompts the LLM to create queries specific to each database intent (e.g., "Who are my top customers?" for `CustomerDB`).

3. **Paraphrasing**:
   - Ensures paraphrased queries retain the same database intent (e.g., "What’s the stock level of item X?" → "How much of product X is in stock?").

4. **Noise Injection**:
   - Adds realistic variations (e.g., "When did order 123 ship?" → "Wen did ord 123 go out?") while preserving the intent.

### Example Workflow
- **Initial Data (CustomerDB)**:
  ```
  1. Who are my top customers this year?
  2. What’s the email of customer ID 456?
  ```
- **After Paraphrasing**:
  ```
  1. Who are my top customers this year?
  2. Which customers spent the most in 2023?
  3. Who’s leading in purchases this year?
  4. What’s the email of customer ID 456?
  5. Can you get me the email for customer 456?
  ```
- **After Noise Injection**:
  ```
  1. Who r my top custmers this yr?
  ```

### Dataset Size
- Per intent:
  - Initial: 50 queries.
  - After paraphrasing: 50 + (50 * 3) = 200 queries.
  - After noise: 200 + 20 = 220 queries.
- Total for 4 intents: ~880 queries.

### Customization
- **Your Databases**: Replace `CustomerDB`, etc., with your actual database names and update their definitions in `intent_definitions`.
- **Size**: Adjust `initial_per_intent`, `paraphrases_per_example`, and `noisy_per_intent` to scale the dataset (e.g., 1000+ per intent).
- **Query Style**: Modify the prompts in `generate_initial_data` to match your users’ query patterns.

### Next Steps
1. Replace `call_llm` with your actual function.
2. Run and check `database_router_combo_dataset.json`.
3. Tokenize the `text` field and train your BERT classifier (e.g., using Hugging Face).

What are your specific database names and their purposes? I can refine the intent definitions and prompts further. Also, do you want to adjust the dataset size or add any specific query patterns?