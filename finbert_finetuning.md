To use FinBERT (a domain-specific BERT model fine-tuned for financial text) with the model weights stored in model.safetensors and the provided config.json, we’ll need to update your existing code to load and fine-tune this model for your multi-class classification task (with 4 classes: “Daily Flow Monitor”, “Turbulence Indices Daily Report”, “Fixed Income and Currency Insight”, “Presentations and Background Materials”). I’ll guide you through the process step-by-step, explaining how to integrate FinBERT, handle the .safetensors weights, and adapt your code to work with its configuration and your dataset.

Overview of FinBERT and Your Setup
	•	FinBERT: A BERT model pre-trained or fine-tuned on financial text, likely optimized for sentiment analysis (e.g., positive, negative, neutral, as seen in id2label and label2id in your config.json). It uses the same architecture as bert-base-uncased (12 layers, 768 hidden size, 12 attention heads) but is specialized for financial domains.
	•	Your Task: Multi-class classification with 4 classes, not 3 (as in FinBERT’s default sentiment setup). You’ll need to modify the classification head to match your 4 classes.
	•	Model Weights: Stored in model.safetensors, a format for efficient storage and loading of PyTorch/TensorFlow weights, commonly used by Hugging Face.
	•	Config.json: Provides the model architecture and metadata, including label2id and id2label for 3 sentiment classes, which we’ll override for your 4 classes.

Steps to Use FinBERT in Your Code
1. Install Required Dependencies
Ensure you have the necessary libraries installed, including support for .safetensors:
pip install transformers torch safetensors
2. Update `src/model.py` to Load FinBERT
Modify src/model.py to load FinBERT with your .safetensors weights and adjust the classification head for your 4 classes. Here’s the updated code:
# src/model.py
# Defines and initializes the FinBERT model

from transformers import BertForSequenceClassification, BertConfig
import torch
from pathlib import Path

def load_model(model_dir="path/to/finbert/weights", num_labels=4):
    """
    Load the pre-trained FinBERT model with custom classification head for 4 classes.

    Args:
        model_dir: Directory containing config.json and model.safetensors.
        num_labels: Number of classes in your task (4 for your classes).

    Returns:
        Configured FinBERT model ready for fine-tuning.
    """
    # Load the config from config.json
    config = BertConfig.from_json_file(Path(model_dir) / "config.json")
    
    # Update config for your task (4 classes instead of 3)
    config.num_labels = num_labels
    config.id2label = {0: "Daily Flow Monitor", 1: "Turbulence Indices Daily Report", 
                       2: "Fixed Income and Currency Insight", 3: "Presentations and Background Materials"}
    config.label2id = {"Daily Flow Monitor": 0, "Turbulence Indices Daily Report": 1, 
                       "Fixed Income and Currency Insight": 2, "Presentations and Background Materials": 3}
    
    # Load the model with the updated config
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        state_dict=None,  # We'll load safetensors manually
        ignore_mismatched_sizes=True  # Allow mismatches if the head changes
    )
    
    # Load weights from safetensors
    state_dict = torch.load(Path(model_dir) / "model.safetensors", map_location=device)
    model.load_state_dict(state_dict, strict=False)  # Allow mismatches for the classification head
    
    return model

# Note: Ensure 'device' is defined (e.g., torch.device("cuda" if torch.cuda.is_available() else "cpu"))
Explanation:
	•	Config Loading: We load config.json to get FinBERT’s architecture but override num_labels, id2label, and label2id for your 4 classes.
	•	Safetensors: We manually load model.safetensors using torch.load and apply it to the model. strict=False allows ignoring mismatches (e.g., the classification head size changing from 3 to 4 outputs).
	•	Head Adjustment: The pre-trained FinBERT has a head for 3 classes (positive, negative, neutral). We redefine it for 4 classes, initializing new weights randomly for the head.
3. Update `config.py`
Update config.py to reflect the new model path and your task-specific settings:
# config.py
# Stores all configurable parameters and paths

# Paths
DATA_PATH = "data/raw_data.csv"
MODEL_DIR = "path/to/finbert/weights"  # Update to your FinBERT weights directory
PROCESSED_DIR = "data/processed"

# Model settings
MODEL_NAME = "finbert"  # Custom identifier for FinBERT
NUM_LABELS = 4          # Number of classes in your task
MAX_LENGTH = 128        # Max token length for questions

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100      # Add warmup for better stability with FinBERT
PATIENCE = 3            # For early stopping
Explanation:
	•	MODEL_DIR: Points to where config.json and model.safetensors are stored.
	•	WARMUP_STEPS: Added 100 steps (10% of total steps, ~225) to stabilize FinBERT’s fine-tuning, given its domain-specific pre-training.
4. Update `src/data_processing.py`
Ensure your labels match the new label2id and id2label from FinBERT. Update load_and_split_data() to use these mappings explicitly:
# src/data_processing.py (snippet)

def load_and_split_data():
    """Load CSV and split into train/val/test sets."""
    df = pd.read_csv(DATA_PATH)
    questions = df['question'].tolist()
    labels = df['label'].tolist()

    # Define mappings based on your 4 classes (override FinBERT's default)
    label_to_id = {"Daily Flow Monitor": 0, "Turbulence Indices Daily Report": 1, 
                   "Fixed Income and Currency Insight": 2, "Presentations and Background Materials": 3}
    id_to_label = {0: "Daily Flow Monitor", 1: "Turbulence Indices Daily Report", 
                   2: "Fixed Income and Currency Insight", 3: "Presentations and Background Materials"}

    # Convert labels to IDs if they're not already numeric
    if isinstance(labels[0], str):
        label_ids = [label_to_id[label] for label in labels]
    else:
        # Assume labels are already 0-3 if numeric, but verify alignment
        label_ids = labels

    # Stratified split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        questions, label_ids, test_size=0.2, stratify=label_ids, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_to_id, id_to_label
Explanation:
	•	Override FinBERT’s default sentiment labels with your 4 classes.
	•	Ensure raw_data.csv uses these class names or numeric IDs (0–3) for consistency.
5. Update `src/train.py`
No major changes are needed here, but ensure gradient clipping (added earlier) and early stopping are included:
# src/train.py (snippet with gradient clipping and early stopping)

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, PATIENCE

def compute_class_weights(labels, num_classes):
    class_weights = compute_class_weight('balanced', classes=range(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

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

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
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
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
Explanation:
	•	Gradient clipping is already included (as requested earlier) with clip_value=1.0.
	•	Early stopping is preserved for overfitting prevention.
6. Update `main.py`
Ensure all components use FinBERT and pass id_to_label:
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
    accuracy, f1, preds, true_labels = evaluate_model(trained_model, test_dataset, device, id_to_label)
    
    # Save model, tokenizer, and label mappings
    trained_model.save_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use FinBERT tokenizer if available
    tokenizer.save_pretrained(MODEL_DIR)
    with open(f"{MODEL_DIR}/label_mappings.pkl", "wb") as f:
        pickle.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
    print(f"Model and label mappings saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
Explanation:
	•	Loads FinBERT with your weights and 4 classes.
	•	Uses BertTokenizer.from_pretrained('bert-base-uncased') as a placeholder; ideally, use FinBERT’s tokenizer if available (check if FinBERT provides a specific tokenizer).
7. Tokenization with FinBERT
If FinBERT has a custom tokenizer, replace BertTokenizer.from_pretrained('bert-base-uncased') with BertTokenizer.from_pretrained(MODEL_DIR) or the specific FinBERT tokenizer path. If not, bert-base-uncased is a reasonable fallback, as FinBERT likely uses a similar vocabulary.

Key Considerations
	•	Domain Adaptation: FinBERT is pre-trained on financial text, which may align well with your dataset (e.g., financial reports, insights). However, ensure your questions match this domain; if not, performance might degrade.
	•	Classification Head: FinBERT’s original head is for 3 classes (sentiment). We’ve redefined it for 4 classes, initializing new head weights randomly. Fine-tuning will adjust these, but expect slower convergence for the head.
	•	Safetensors: Ensures efficient loading of weights, compatible with Hugging Face’s from_pretrained.
	•	Imbalance: Use compute_class_weight('balanced') or manual weights (as discussed earlier) to handle your 150–400 samples per class imbalance.

Running the Code
	1	Replace path/to/finbert/weights with the actual path to your config.json and model.safetensors.
	2	Ensure raw_data.csv has labels as strings (“Daily Flow Monitor”, etc.) or numeric IDs (0–3).
	3	Run python main.py to fine-tune FinBERT on your dataset.
This setup integrates FinBERT into your pipeline, leveraging its financial domain knowledge while adapting it to your 4-class task. Let me know if you need help with tokenizer specifics or further tweaks!
