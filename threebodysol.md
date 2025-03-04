

### Three-Body Solution Overview
1. **Curiosity LLM**:
   - Role: Generates questions or statements based on the provided context.
   - Output: A list of questions/statements (e.g., "When will my order arrive?").
2. **Intent LLM**:
   - Role: Analyzes each question and assigns one of the predefined intents (e.g., Inquiry, Request).
   - Output: Intent label for each question (e.g., "Inquiry").
3. **Supervisor LLM**:
   - Role: Validates (1) if the question fits the context and (2) if the intent classification is correct, providing reasoning.
   - Output: Approved question-intent pairs with reasoning, or rejection with feedback.

### Process Flow
- **Step 1**: Curiosity LLM generates a batch of questions.
- **Step 2**: Intent LLM classifies the intent of each question.
- **Step 3**: Supervisor LLM reviews each question-intent pair, approves or rejects it, and provides reasoning.
- **Step 4**: Collect approved pairs into the dataset; optionally recycle rejected ones for refinement.

### Assumptions
- Context: "customer support for an online shopping platform."
- Intents: `Positive Feedback`, `Negative Feedback`, `Inquiry`, `Request`.
- Your `call_llm(template: str) -> str` function returns a string response.

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
        "You are an AI tasked with generating or analyzing synthetic data for an intent classifier.\n"
        "Follow these instructions carefully:\n"
        "- Generate or analyze responses based on the provided context and question.\n"
        "- Ensure outputs align with the specified intent definitions, if applicable.\n"
        "- Return results in the requested format.\n"
        f"Context: {context}\n"
        f"Question: {question}"
    )
    return template

# Curiosity LLM: Generate questions
def curiosity_llm(context: str, num_questions: int) -> List[str]:
    """Generate questions/statements based on the context."""
    question = (
        f"Generate {num_questions} unique user questions or statements related to the context. "
        f"These should reflect realistic interactions a customer might have. "
        f"Return as a numbered list (e.g., 1. text, 2. text)."
    )
    template = create_template(context, question)
    response = call_llm(template)
    questions = [line.strip() for line in response.split("\n") if re.match(r"^\d+\.\s", line)]
    questions = [re.sub(r"^\d+\.\s", "", q) for q in questions]
    return questions[:num_questions]

# Intent LLM: Classify intents
def intent_llm(context: str, questions: List[str]) -> List[Tuple[str, str]]:
    """Classify the intent of each question."""
    intent_definitions = (
        "Intent definitions:\n"
        "- Positive Feedback: User expresses satisfaction.\n"
        "- Negative Feedback: User expresses dissatisfaction.\n"
        "- Inquiry: User asks a question.\n"
        "- Request: User asks for an action."
    )
    question = (
        f"{intent_definitions}\n"
        f"Classify the intent of each of the following questions/statements. "
        f"Return as a numbered list with the format: '1. [Intent] - text'.\n"
        f"Questions:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
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
    return pairs[:len(questions)]

# Supervisor LLM: Validate questions and intents
def supervisor_llm(context: str, question_intent_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Validate context alignment and intent classification with reasoning."""
    intent_definitions = (
        "Intent definitions:\n"
        "- Positive Feedback: User expresses satisfaction.\n"
        "- Negative Feedback: User expresses dissatisfaction.\n"
        "- Inquiry: User asks a question.\n"
        "- Request: User asks for an action."
    )
    question = (
        f"{intent_definitions}\n"
        f"For each question-intent pair, determine:\n"
        f"1. Does the question fit the context? (Yes/No)\n"
        f"2. Is the intent classification correct? (Yes/No)\n"
        f"3. Provide reasoning for your decisions.\n"
        f"Return as a numbered list with format: "
        f"'1. [Yes/No, Yes/No] - Reasoning: <reason>'.\n"
        f"Pairs:\n" + "\n".join([f"{i+1}. '{q}' - {intent}" for i, (q, intent) in enumerate(question_intent_pairs)])
    )
    template = create_template(context, question)
    response = call_llm(template)
    validated_data = []
    for line, (question, intent) in zip(response.split("\n"), question_intent_pairs):
        if re.match(r"^\d+\.\s\[.*\]\s-\sReasoning:", line):
            match = re.match(r"^\d+\.\s\[(Yes|No),\s(Yes|No)\]\s-\sReasoning:\s(.*)$", line.strip())
            if match:
                context_ok, intent_ok, reasoning = match.groups()
                if context_ok == "Yes" and intent_ok == "Yes":
                    validated_data.append({
                        "text": question,
                        "intent": intent,
                        "reasoning": reasoning
                    })
    return validated_data

# Main function to generate dataset using three-body system
def generate_three_body_dataset(
    context: str,
    intents: List[str],
    num_questions: int = 200
) -> List[Dict[str, str]]:
    """Generate synthetic dataset using the three-body LLM system."""
    dataset = []

    # Step 1: Curiosity LLM generates questions
    print("Generating questions...")
    questions = curiosity_llm(context, num_questions)
    print(f"Generated {len(questions)} questions.")

    # Step 2: Intent LLM classifies intents
    print("Classifying intents...")
    question_intent_pairs = intent_llm(context, questions)
    print(f"Classified {len(question_intent_pairs)} pairs.")

    # Step 3: Supervisor LLM validates
    print("Validating with supervisor...")
    validated_data = supervisor_llm(context, question_intent_pairs)
    print(f"Validated {len(validated_data)} pairs.")

    # Filter to ensure balanced intents (optional)
    intent_counts = {intent: 0 for intent in intents}
    for entry in validated_data:
        if entry["intent"] in intents:
            intent_counts[entry["intent"]] += 1
            dataset.append({"text": entry["text"], "intent": entry["intent"]})

    print("Intent distribution:", intent_counts)
    return dataset

# Save dataset to JSON file
def save_dataset(dataset: List[Dict[str, str]], filename: str = "three_body_dataset.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename} with {len(dataset)} examples.")

# Example usage
if __name__ == "__main__":
    # Define context and intents
    context = "customer support for an online shopping platform"
    intents = ["Positive Feedback", "Negative Feedback", "Inquiry", "Request"]

    # Generate dataset
    dataset = generate_three_body_dataset(
        context=context,
        intents=intents,
        num_questions=200  # Total questions to generate initially
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
1. **Curiosity LLM (`curiosity_llm`)**:
   - Generates `num_questions` questions/statements based on the context.
   - Example output: 
     ```
     1. When will my order arrive?
     2. I love the quick delivery!
     ```

2. **Intent LLM (`intent_llm`)**:
   - Takes the questions and assigns an intent to each.
   - Example output:
     ```
     1. [Inquiry] - When will my order arrive?
     2. [Positive Feedback] - I love the quick delivery!
     ```

3. **Supervisor LLM (`supervisor_llm`)**:
   - Validates each question-intent pair for context fit and intent accuracy.
   - Example output:
     ```
     1. [Yes, Yes] - Reasoning: Question is relevant to shipping in an online shopping context; intent matches a query.
     2. [Yes, Yes] - Reasoning: Statement reflects satisfaction with delivery, correctly classified as positive feedback.
     ```
   - Only pairs with `[Yes, Yes]` are added to the dataset.

4. **Main Function (`generate_three_body_dataset`)**:
   - Chains the three LLMs together.
   - Ensures the final dataset only includes validated entries.

### Dataset Size
- Starting with `num_questions = 200`, the final size depends on how many pairs the Supervisor LLM approves.
- Typically, you’d get fewer than 200 validated entries due to rejections, so adjust `num_questions` higher if you need a specific size (e.g., 800 total, ~200 per intent).

### Customization
- **Intent Balancing**: The current code doesn’t enforce balance across intents. You could add a post-processing step to cap or boost specific intents.
- **Rejection Handling**: Add logic to recycle rejected questions (e.g., prompt Curiosity LLM to refine them based on Supervisor feedback).
- **Output Format**: Adjust parsing in each function if your LLM returns JSON or a different structure.

### Pros and Cons
- **Pros**:
  - High-quality data with validated context and intent.
  - Reasoning provides transparency and debugging insights.
- **Cons**:
  - Slower due to three LLM calls per batch.
  - Dataset size may be unpredictable without additional balancing.

### Next Steps
1. Replace the placeholder `call_llm` with your actual function.
2. Run and check the output in `three_body_dataset.json`.
3. If you want more control over intent distribution or handling of rejected pairs, let me know—I can extend the code!

What do you think of this approach? Want to tweak the context, intents, or add features like intent balancing?