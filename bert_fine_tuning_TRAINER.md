Let’s create a well-organized code structure for fine-tuning a BERT model for your multi-class classification task, based on the **full fine-tuning approach with class weights** to handle potential imbalance, as outlined in your article. I’ll provide a file organization structure, skeleton code for each file, and detailed comments explaining what each part does, why it’s used, and the input/output flow. This will align with your practical roadmap while keeping the implementation modular, reusable, and easy to debug.

### File Organization
Here’s a suggested directory structure to keep your project clean and scalable:

```
bert_classification_project/
│
├── data/                   # Directory for raw and processed data
│   ├── raw_data.csv        # Input: Questions and labels (e.g., "question,label")
│   └── processed/          # Output: Tokenized datasets (optional caching)
│
├── models/                 # Directory to save trained model and tokenizer
│   └── bert_finetuned/     # Output: Saved fine-tuned BERT model
│
├── src/                    # Source code directory
│   ├── __init__.py         # Makes src a Python package
│   ├── data_processing.py  # Data loading, tokenization, and dataset creation
│   ├── model.py            # Model definition and initialization
│   ├── train.py            # Training logic with class weights
│   └── evaluate.py         # Evaluation and inference utilities
│
├── config.py               # Configuration settings (hyperparameters, paths)
├── main.py                 # Entry point to run the pipeline
└── requirements.txt        # Dependencies (e.g., transformers, torch, etc.)
```

### Skeleton Code with Explanations

#### 1. `config.py`
**Purpose**: Centralizes hyperparameters and file paths for easy tweaking and reproducibility.  
**Why**: Avoids hardcoding values in multiple files, making the codebase maintainable.

```python
# config.py
# Stores all configurable parameters and paths

# Paths
DATA_PATH = "data/raw_data.csv"
MODEL_DIR = "models/bert_finetuned"
PROCESSED_DIR = "data/processed"

# Model settings
MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model from Hugging Face
NUM_LABELS = 5                    # Number of classes in your task
MAX_LENGTH = 128                  # Max token length for questions

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0                  # For learning rate scheduler
```

**Input**: None (static config).  
**Output**: Variables used across the project.

---

#### 2. `src/data_processing.py`
**Purpose**: Loads data, tokenizes it, and creates a PyTorch Dataset.  
**Why**: Prepares your raw CSV data (questions + labels) for BERT training by converting text to model-readable tokens.

```python
# src/data_processing.py
# Handles data loading, tokenization, and dataset creation

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from config import DATA_PATH, MODEL_NAME, MAX_LENGTH

class QADataset(Dataset):
    """Custom Dataset for question-label pairs."""
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Tokenized input (input_ids, attention_mask)
        self.labels = labels        # Class labels (e.g., 0, 1, 2, ...)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def load_and_split_data():
    """Load CSV and split into train/val/test sets."""
    # Input: CSV with "question" and "label" columns
    df = pd.read_csv(DATA_PATH)
    questions = df['question'].tolist()
    labels = df['label'].tolist()

    # Stratified split to maintain class proportions
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        questions, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    # Output: Lists of texts and labels for each split
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def tokenize_data(texts, tokenizer):
    """Tokenize text using BERT tokenizer."""
    # Input: List of questions
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    # Output: Dictionary with input_ids, attention_mask, etc.
    return encodings

def prepare_datasets():
    """Main function to process and return datasets."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_split_data()

    # Tokenize each split
    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    # Create datasets
    train_dataset = QADataset(train_encodings, train_labels)
    val_dataset = QADataset(val_encodings, val_labels)
    test_dataset = QADataset(test_encodings, test_labels)

    # Output: Datasets ready for training and evaluation
    return train_dataset, val_dataset, test_dataset, train_labels  # train_labels for class weights
```

**Input**: CSV file (`data/raw_data.csv`) with columns `question` (text) and `label` (int).  
**Output**: PyTorch Datasets (`train_dataset`, `val_dataset`, `test_dataset`) and `train_labels` for class weight computation.

---

#### 3. `src/model.py`
**Purpose**: Loads and configures the BERT model for classification.  
**Why**: Initializes the pre-trained BERT model with a classification head tailored to your number of classes.

```python
# src/model.py
# Defines and initializes the BERT model

from transformers import BertForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def load_model():
    """Load pre-trained BERT model with classification head."""
    # Input: Pre-trained model name and number of labels
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    # Output: Configured BERT model ready for fine-tuning
    return model
```

**Input**: Model name and number of labels from `config.py`.  
**Output**: A `BertForSequenceClassification` instance.

---

#### 4. `src/train.py`
**Purpose**: Implements the training loop with class weights to handle imbalance.  
**Why**: Fine-tunes the model on your data, optimizing weights to classify questions accurately while addressing imbalance.

```python
# src/train.py
# Training logic with class weights for imbalance

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS
from torch.utils.data import DataLoader

def compute_class_weights(labels, num_classes):
    """Compute weights to balance classes."""
    # Input: Train labels and number of classes
    class_weights = compute_class_weight('balanced', classes=range(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(model, train_dataset, val_dataset, train_labels, device):
    """Train the BERT model with class-weighted loss."""
    model.to(device)

    # DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # Class-weighted loss
    class_weights = compute_class_weights(train_labels, num_classes=model.config.num_labels).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)  # Use weighted loss

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss_fn(outputs.logits, labels).item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    # Output: Trained model (still in memory, save in main.py)
    return model
```

**Input**: Model, datasets, and training labels.  
**Output**: Fine-tuned BERT model (updated weights).

---

#### 5. `src/evaluate.py`
**Purpose**: Evaluates the model and provides inference functionality.  
**Why**: Assesses performance on the test set and allows predictions on new data.

```python
# src/evaluate.py
# Evaluation and inference utilities

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from config import BATCH_SIZE

def evaluate_model(model, test_dataset, device):
    """Evaluate model performance on test set."""
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # Output: Metrics
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    return accuracy, f1

def predict(model, tokenizer, question, device):
    """Predict class for a single question."""
    model.to(device)
    model.eval()
    encodings = tokenizer([question], truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_class = outputs.logits.argmax(dim=-1).item()
    # Output: Predicted class ID
    return pred_class
```

**Input**: Model, test dataset or single question.  
**Output**: Metrics (accuracy, F1) or predicted class.

---

#### 6. `main.py`
**Purpose**: Orchestrates the entire pipeline.  
**Why**: Ties all modules together into a single executable script.

```python
# main.py
# Entry point to run the fine-tuning pipeline

import torch
from src.data_processing import prepare_datasets
from src.model import load_model
from src.train import train_model
from src.evaluate import evaluate_model
from config import MODEL_DIR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_dataset, val_dataset, test_dataset, train_labels = prepare_datasets()
    
    # Load model
    model = load_model()
    
    # Train
    trained_model = train_model(model, train_dataset, val_dataset, train_labels, device)
    
    # Evaluate
    evaluate_model(trained_model, test_dataset, device)
    
    # Save model
    trained_model.save_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
```

**Input**: None (uses config and data files).  
**Output**: Trained model saved to `MODEL_DIR`, plus printed metrics.

---

#### 7. `requirements.txt`
**Purpose**: Lists dependencies for easy setup.  
**Why**: Ensures reproducibility across environments.

```
transformers==4.36.2
torch==2.0.1
pandas==2.0.3
scikit-learn==1.3.0
imbalanced-learn==0.11.0
```

---

### How It Fits Your Approach
- **Full Fine-Tuning**: All BERT layers + classification head are trained (`src/train.py`).
- **Class Weights**: Implemented in `train.py` to handle imbalance, as per your preference.
- **Practical Tips**: Modular design allows starting small (e.g., subset data in `data_processing.py`), monitoring via printed losses, and experimenting with `config.py`.

### Running the Code
1. Create `data/raw_data.csv` with your ~1500 samples (e.g., `question,label` format).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run: `python main.py`.

This structure provides a robust, extensible foundation for your BERT fine-tuning task, aligning with your article’s roadmap! Let me know if you’d like adjustments or additional features (e.g., early stopping, data augmentation).