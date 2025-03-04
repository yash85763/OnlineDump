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
from typing import List, Dict, Tuple

# Simulated LLM call function (replace with your actual function)
# def call_llm(template: str) -> str:
#     # Example: return "Generated response from LLM"
#     pass

# Template creation function
def create_template(context: str, question: str) -> str:
    """Create a single template string with instructions, context, and question."""
    template = (
        "You are an AI tasked with generating or validating synthetic data for a database router classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate or analyze responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified database intents, if applicable.\n"
        "- Return results as a numbered list (e.g., 1. text, 2. text) unless specified otherwise.\n"
        f"Context: {context}\n"
        f"Question: {question}"
    )
    return template

# Step 1: Generate a batch of 10 queries
def generate_batch(context: str, intent: str, batch_size: int = 10) -> List[str]:
    """Generate a batch of queries for a given database intent."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What’s the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What’s the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n"
        f"Generate {batch_size} unique user queries with the intent '{intent}'. "
        f"These should reflect realistic questions a user might ask in a data retrieval system. "
        f"Return as a numbered list."
    )
    template = create_template(context, question)
    response = call_llm(template)
    queries = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
    queries = [re.sub(r"^\d+\.\s", "", q) for q in queries]
    return queries[:batch_size]

# Validation: Validate the batch
def validate_batch(context: str, intent: str, queries: List[str]) -> List[Dict[str, str]]:
    """Validate a batch of queries for context fit and intent accuracy."""
    intent_definitions = (
        "Database intents:\n"
        "- CustomerDB: Queries about customer data (e.g., 'Who bought the most last month?').\n"
        "- ProductDB: Queries about product data (e.g., 'What’s the stock level of item X?').\n"
        "- OrderDB: Queries about order data (e.g., 'When did order 123 ship?').\n"
        "- AnalyticsDB: Queries about analytical data (e.g., 'What’s the average sales trend?')."
    )
    question = (
        f"{intent_definitions}\n"
        f"For each query, determine:\n"
        f"1. Does the query fit the context of a data retrieval system? (Yes/No)\n"
        f"2. Does the query match the intent '{intent}'? (Yes/No)\n"
        f"3. Provide reasoning for your decisions.\n"
        f"Return as a numbered list with format: "
        f"'1. [Yes/No, Yes/No] - Reasoning: <reason>'.\n"
        f"Queries:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    )
    template = create_template(context, question)
    response = call_llm(template)
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

# Step 2: Augmentation with Paraphrasing
def augment_with_paraphrasing(context: str, examples: List[Dict[str, str]], num_paraphrases: int) -> List[Dict[str, str]]:
    """Augment validated queries with paraphrases."""
    augmented_data = examples.copy()
    for example in examples:
        question = (
            f"Generate {num_paraphrases} paraphrased versions of this query: '{example['text']}'. "
            f"Keep the same database intent '{example['intent']}'. Return as a numbered list."
        )
        template = create_template(context, question)
        response = call_llm(template)
        paraphrases = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
        paraphrases = [re.sub(r"^\d+\.\s", "", para) for para in paraphrases]
        augmented_data.extend([{"text": para, "intent": example["intent"]} for para in paraphrases[:num_paraphrases]])
    return augmented_data

# Step 3: Noise Injection for Robustness
def inject_noise(context: str, examples: List[Dict[str, str]], num_noisy: int) -> List[Dict[str, str]]:
    """Add noise to some queries for robustness."""
    noisy_data = examples.copy()
    sampled_examples = random.sample(examples, min(num_noisy, len(examples)))
    for example in sampled_examples:
        question = (
            f"Rewrite this query with typos, slang, or casual phrasing: '{example['text']}'. "
            f"Keep the same database intent '{example['intent']}'. Return one version."
        )
        template = create_template(context, question)
        response = call_llm(template)
        noisy_data.append({"text": response.strip(), "intent": example["intent"]})
    return noisy_data

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

    for intent in intents:
        print(f"Generating data for database intent: {intent}")
        intent_data = []

        # Generate and validate batches
        for batch_num in range(batches_per_intent):
            print(f"Batch {batch_num + 1}/{batches_per_intent}")
            # Generate batch
            batch_queries = generate_batch(context, intent, batch_size)
            print(f"Generated {len(batch_queries)} queries in batch.")

            # Validate batch
            validated_queries = validate_batch(context, intent, batch_queries)
            print(f"Validated {len(validated_queries)} queries.")
            intent_data.extend(validated_queries)

        # Step 2: Augment validated queries
        augmented_data = augment_with_paraphrasing(context, intent_data, paraphrases_per_query)
        print(f"After paraphrasing: {len(augmented_data)} queries.")

        # Step 3: Inject noise
        final_data = inject_noise(context, augmented_data, noisy_per_batch * batches_per_intent)
        print(f"After noise injection: {len(final_data)} queries.")

        # Add to dataset
        dataset.extend(final_data)

    return dataset

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "database_router_batch_dataset.json"):
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
        batches_per_intent=5,       # 5 batches per intent
        batch_size=10,              # 10 queries per batch
        paraphrases_per_query=3,    # 3 paraphrases per validated query
        noisy_per_batch=2           # 2 noisy versions per batch
    )

    # Save to file
    save_dataset(dataset)

    # Print sample
    print("\nSample of generated data:")
    for entry in dataset[:5]:
        print(f"Text: {entry['text']}, Intent: {entry['intent']}")
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