To fine-tune the `distilbert-base-uncased` model instead of FinBERT or `bert-base-uncased` in your existing codebase, we’ll need to make several adjustments to leverage DistilBERT’s lighter architecture (6 layers, 768 hidden size, 12 attention heads, ~66M parameters compared to BERT’s 110M). DistilBERT is a distilled version of BERT, designed for efficiency while retaining much of its performance, making it a great choice for your dataset (~150–400 samples per class). Below, I’ll outline the changes required across your files (`config.py`, `src/model.py`, `src/data_processing.py`, `src/train.py`, and `main.py`) to switch to DistilBERT, ensuring compatibility with your 4-class classification task.

---

### Key Considerations for DistilBERT

- **Architecture**: DistilBERT has fewer layers (6 vs. 12) and no token-type embeddings, but it uses the same tokenizer and vocabulary as `bert-base-uncased` (30522 tokens).
- **Pre-trained Weights**: Available from Hugging Face, loaded via `transformers`.
- **Classification Head**: Needs to be adjusted for your 4 classes, similar to FinBERT.
- **Compatibility**: Your existing tokenization and training pipeline (e.g., `QADataset`, `DataLoader`) should work with minor tweaks, as DistilBERT supports the same input format (`input_ids`, `attention_mask`).

---

### Changes to Your Codebase

#### 1. Update `config.py`
Modify `config.py` to point to DistilBERT and adjust hyperparameters if needed:

```python
# config.py
# Stores all configurable parameters and paths

# Paths
DATA_PATH = "data/raw_data.csv"
MODEL_DIR = "models/distilbert_finetuned"  # Directory to save fine-tuned model
PROCESSED_DIR = "data/processed"

# Model settings
MODEL_NAME = "distilbert-base-uncased"  # Switch to DistilBERT
NUM_LABELS = 4          # Number of classes in your task
MAX_LENGTH = 128        # Max token length for questions

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5    # May need adjustment for DistilBERT (e.g., 3e-5 or 5e-5)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100      # Keep for stability
PATIENCE = 3            # For early stopping
```

**Explanation**:
- `MODEL_NAME`: Changed to `"distilbert-base-uncased"` to load the pre-trained DistilBERT model.
- `LEARNING_RATE`: DistilBERT might converge faster due to its smaller size; you can experiment with 3e-5 or 5e-5 if 2e-5 underperforms.
- `MODEL_DIR`: Updated to a new directory for saving the fine-tuned DistilBERT model.

#### 2. Update `src/model.py`
Adjust `load_model()` to load DistilBERT and configure it for your 4 classes:

```python
# src/model.py
# Defines and initializes the DistilBERT model

from transformers import DistilBertForSequenceClassification, DistilBertConfig
import torch
from pathlib import Path

def load_model(model_dir="models/distilbert_finetuned", num_labels=4):
    """
    Load the pre-trained DistilBERT model with custom classification head for 4 classes.

    Args:
        model_dir: Directory to save/load the fine-tuned model.
        num_labels: Number of classes in your task (4 for your classes).

    Returns:
        Configured DistilBERT model ready for fine-tuning.
    """
    # Load the config for DistilBERT
    config = DistilBertConfig.from_pretrained(MODEL_NAME)  # Use MODEL_NAME from config.py
    config.num_labels = num_labels
    config.id2label = {0: "Daily Flow Monitor", 1: "Turbulence Indices Daily Report", 
                       2: "Fixed Income and Currency Insight", 3: "Presentations and Background Materials"}
    config.label2id = {"Daily Flow Monitor": 0, "Turbulence Indices Daily Report": 1, 
                       "Fixed Income and Currency Insight": 2, "Presentations and Background Materials": 3}

    # Load the pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )

    return model

# Note: Ensure 'MODEL_NAME' is imported from config.py or passed as an argument
```

**Explanation**:
- **Config**: Use `DistilBertConfig` instead of `BertConfig` to match DistilBERT’s architecture.
- **Model Loading**: Replace `BertForSequenceClassification` with `DistilBertForSequenceClassification` and load from `"distilbert-base-uncased"`.
- **No Safetensors**: Unlike FinBERT, we’re using Hugging Face’s pre-trained weights directly, so no manual `.safetensors` loading is needed.
- **Head Adjustment**: The pre-trained head (for binary classification) is replaced with a new head for 4 classes, initialized randomly.

#### 3. Update `src/data_processing.py`
No major changes are needed here, as DistilBERT uses the same tokenizer as BERT. Ensure the tokenizer is loaded correctly:

```python
# src/data_processing.py (snippet)

from transformers import DistilBertTokenizer  # Update tokenizer import

def prepare_datasets():
    """Main function to process and return datasets."""
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)  # Use DistilBERT tokenizer
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_to_id, id_to_label = load_and_split_data()
    
    # ... existing tokenization and dataset creation ...
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    
    # ... existing dataset creation ...
    return train_dataset, val_dataset, test_dataset, train_labels, label_to_id, id_to_label
```

**Explanation**:
- **Tokenizer**: Switch to `DistilBertTokenizer` to match the model. It’s compatible with `bert-base-uncased`’s vocabulary, so your existing tokenization logic works.
- **No Other Changes**: The dataset structure (`QADataset`) and label handling remain unchanged.

#### 4. Update `src/train.py`
The training loop is largely compatible, but ensure it uses DistilBERT’s inputs (no `token_type_ids`, which DistilBERT lacks). The existing gradient clipping and early stopping are still applicable:

```python
# src/train.py (snippet)

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

def train_model(model, train_dataset, val_dataset, train_labels, device, patience=3, clip_value=1.0):
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    class_weights = compute_class_weights(train_labels, num_classes=model.config.num_labels).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # No token_type_ids
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss_fn(outputs.logits, labels).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement, patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored model to best validation loss: {best_val_loss:.4f}")

    return model
```

**Explanation**:
- **Input Args**: Remove `token_type_ids` from the `model()` call, as DistilBERT doesn’t use it.
- **Compatibility**: The rest of the loop (gradient clipping, early stopping, class weights) works seamlessly with DistilBERT.

#### 5. Update `main.py`
Ensure the pipeline loads and saves DistilBERT correctly:

```python
# main.py
# Entry point to run the fine-tuning pipeline

import torch
from src.data_processing import prepare_datasets
from src.model import load_model
from src.train import train_model
from src.evaluate import evaluate_model
from config import MODEL_DIR
import pickle

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_dataset, val_dataset, test_dataset, train_labels, label_to_id, id_to_label = prepare_datasets()
    
    # Load model
    model = load_model(MODEL_DIR, num_labels=4)
    
    # Train
    trained_model = train_model(model, train_dataset, val_dataset, train_labels, device)
    
    # Evaluate
    metrics = evaluate_model(trained_model, test_dataset, device, id_to_label, train_dataset, val_dataset)
    
    # Print creative metrics
    print("\nCreative Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric.upper()}: {value:.4f}")
    
    # Save model, tokenizer, and label mappings
    trained_model.save_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)  # Use DistilBERT tokenizer
    tokenizer.save_pretrained(MODEL_DIR)
    with open(f"{MODEL_DIR}/label_mappings.pkl", "wb") as f:
        pickle.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
    print(f"Model and label mappings saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
```

**Explanation**:
- **Tokenizer**: Switch to `DistilBertTokenizer` to match the model.
- **Evaluation**: Includes the creative metrics from your previous request.
- **Saving**: Saves the fine-tuned DistilBERT model and tokenizer.

---

### Additional Notes

- **Performance Expectation**: DistilBERT should perform similarly to `bert-base-uncased` but faster due to its smaller size. With your ~150–400 samples per class, it may generalize better with fewer parameters, reducing overfitting.
- **Hyperparameter Tuning**: Start with your current settings (e.g., `LEARNING_RATE=2e-5`, `BATCH_SIZE=16`). If convergence is slow, increase the learning rate to 3e-5 or 5e-5, or reduce batch size to 8 for more gradient updates.
- **Domain Fit**: Unlike FinBERT, DistilBERT is not pre-trained on financial text. If your data is financial (e.g., reports, insights), FinBERT might still outperform, but DistilBERT’s efficiency could be advantageous with your small dataset.
- **Validation**: Monitor the creative metrics (LSI, GGR, FES, LER, RTN) to assess DistilBERT’s learning behavior compared to FinBERT or BERT.

---

### Running the Code
1. Update `MODEL_DIR` and `MODEL_NAME` in `config.py` to match your setup.
2. Ensure `raw_data.csv` labels align with your 4 classes (e.g., "Daily Flow Monitor" or 0–3).
3. Run `python main.py` to fine-tune DistilBERT.

This setup integrates `distilbert-base-uncased` into your pipeline, leveraging its efficiency while maintaining your task-specific requirements. Let me know if you need further adjustments or help with hyperparameter tuning!