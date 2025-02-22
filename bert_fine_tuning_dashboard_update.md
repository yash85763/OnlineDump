You’re right to point out potential issues—indentation is critical in Python, especially in complex functions like the training loop in `train.py`. I’ll rewrite the entire codebase for clarity, ensuring proper indentation and consistency across all files. I’ll include the updates for the Dash dashboard, metrics saving, and label mappings as requested, while double-checking the structure of `train.py` (I assume "trainer.py" was a typo for `src/train.py`). Below is the corrected and complete set of files.

---

### File Structure Recap
```
bert_classification_project/
├── config.py
├── main.py
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── dashboard.py
│   └── __init__.py
├── requirements.txt
└── data/raw_data.csv  # Example: "question,label"
```

---

### 1. `config.py`
```python
# config.py
# Configuration settings

DATA_PATH = "data/raw_data.csv"
MODEL_DIR = "models/bert_finetuned"
PROCESSED_DIR = "data/processed"

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5
MAX_LENGTH = 128

BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
PATIENCE = 3
```

---

### 2. `main.py`
```python
# main.py
# Entry point with dashboard integration

import torch
import json
import subprocess
from src.data_processing import prepare_datasets
from src.model import load_model
from src.train import train_model
from src.evaluate import evaluate_model
from config import MODEL_DIR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_dataset, val_dataset, test_dataset, train_labels, label_to_id, id_to_label = prepare_datasets()
    assert len(label_to_id) == 5, f"Expected 5 classes, got {len(label_to_id)}"

    # Train and evaluate
    model = load_model()
    trained_model = train_model(model, train_dataset, val_dataset, train_labels, device)
    metrics = evaluate_model(trained_model, test_dataset, device, id_to_label)

    # Save model and mappings
    trained_model.save_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(MODEL_DIR)
    with open(f"{MODEL_DIR}/label_mappings.json", "w") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": {str(k): v for k, v in id_to_label.items()}}, f)
    print(f"Model and mappings saved to {MODEL_DIR}")

    # Launch dashboard
    print("Launching performance dashboard...")
    subprocess.Popen(["python", "src/dashboard.py"])

if __name__ == "__main__":
    main()
```

---

### 3. `src/data_processing.py`
```python
# src/data_processing.py
# Data loading, tokenization, and label encoding

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from config import DATA_PATH, MODEL_NAME, MAX_LENGTH

class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def check_label_type(labels):
    unique_labels = set(labels)
    return all(isinstance(label, int) for label in unique_labels)

def create_label_mappings(labels):
    unique_labels = sorted(set(labels))
    if check_label_type(labels):
        label_to_id = {label: label for label in unique_labels}
        id_to_label = {label: label for label in unique_labels}
    else:
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def encode_labels(labels, label_to_id):
    return [label_to_id[label] for label in labels]

def load_and_split_data():
    df = pd.read_csv(DATA_PATH)
    questions = df['question'].tolist()
    raw_labels = df['label'].tolist()

    label_to_id, id_to_label = create_label_mappings(raw_labels)
    encoded_labels = encode_labels(raw_labels, label_to_id)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        questions, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_to_id, id_to_label

def tokenize_data(texts, tokenizer):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    return encodings

def prepare_datasets():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_to_id, id_to_label = load_and_split_data()

    train_encodings = tokenize_data(train_texts, tokenizer)
    val_encodings = tokenize_data(val_texts, tokenizer)
    test_encodings = tokenize_data(test_texts, tokenizer)

    train_dataset = QADataset(train_encodings, train_labels)
    val_dataset = QADataset(val_encodings, val_labels)
    test_dataset = QADataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset, train_labels, label_to_id, id_to_label
```

---

### 4. `src/model.py`
```python
# src/model.py
# Model definition

from transformers import BertForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    return model
```

---

### 5. `src/train.py`
Here’s the corrected version with proper indentation:

```python
# src/train.py
# Training logic with metrics saving

import torch
import pandas as pd
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, MODEL_DIR, PATIENCE

def compute_class_weights(labels, num_classes):
    class_weights = compute_class_weight('balanced', classes=range(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(model, train_dataset, val_dataset, train_labels, device, patience=PATIENCE):
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    class_weights = compute_class_weights(train_labels, num_classes=model.config.num_labels).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_losses, val_losses = [], []
    train_accs, val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        train_preds, train_true = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_preds.extend(outputs.logits.argmax(-1).cpu().tolist())
            train_true.extend(labels.cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = accuracy_score(train_true, train_preds)
        train_accs.append(train_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss_fn(outputs.logits, labels).item()
                val_preds.extend(outputs.logits.argmax(-1).cpu().tolist())
                val_true.extend(labels.cpu().tolist())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(val_true, val_preds)
        val_accs.append(val_acc)
        print(f"Validation Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

    # Save learning curves
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Accuracy': train_accs,
        'Val Accuracy': val_accs
    })
    metrics_df.to_csv(f"{MODEL_DIR}/learning_curves.csv", index=False)
    print(f"Learning curves saved to {MODEL_DIR}/learning_curves.csv")

    return model
```

#### Indentation Fixes
- Ensured all blocks under `for epoch in range(NUM_EPOCHS)` are properly indented (4 spaces).
- Nested loops (`for batch in train_loader/val_loader`) and conditionals (`if avg_val_loss < best_val_loss`) are correctly aligned.
- No stray dedents or over-indents that could break the logic.

---

### 6. `src/evaluate.py`
```python
# src/evaluate.py
# Evaluation and inference with metric saving

import torch
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from config import BATCH_SIZE, MODEL_DIR

def evaluate_model(model, test_dataset, device, id_to_label=None):
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
            preds.extend(outputs.logits.argmax(-1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average=None)
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
    conf_matrix = confusion_matrix(true_labels, preds)

    class_names = [id_to_label[i] if id_to_label else str(i) for i in range(len(set(true_labels)))]

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    print("\nPer-Class Metrics:")
    print(classification_report(true_labels, preds, target_names=class_names))

    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    metrics_df.to_csv(f"{MODEL_DIR}/test_metrics.csv", index=False)
    conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    conf_df.to_csv(f"{MODEL_DIR}/confusion_matrix.csv")

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix
    }

def predict(model=None, tokenizer=None, question=None, device="cpu", use_saved_mappings=True):
    if model is None:
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    
    id_to_label = None
    if use_saved_mappings:
        try:
            with open(f"{MODEL_DIR}/label_mappings.json", "r") as f:
                label_mappings = json.load(f)
                id_to_label = {int(k): v for k, v in label_mappings["id_to_label"].items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"Failed to load label mappings: {e}. Returning ID only.")
    
    model.to(device)
    model.eval()
    encodings = tokenizer([question], truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_id = outputs.logits.argmax(dim=-1).item()
    
    pred_label = id_to_label[pred_id] if id_to_label is not None else None
    return pred_id, pred_label
```

---

### 7. `src/dashboard.py`
```python
# src/dashboard.py
# Dash dashboard for visualizing performance

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from config import MODEL_DIR

learning_curves = pd.read_csv(f"{MODEL_DIR}/learning_curves.csv")
test_metrics = pd.read_csv(f"{MODEL_DIR}/test_metrics.csv")
conf_matrix = pd.read_csv(f"{MODEL_DIR}/confusion_matrix.csv", index_col=0)

app = dash.Dash(__name__)

loss_fig = px.line(
    learning_curves, x='Epoch', y=['Train Loss', 'Val Loss'],
    title='Loss Over Epochs', labels={'value': 'Loss', 'variable': 'Dataset'}
)
acc_fig = px.line(
    learning_curves, x='Epoch', y=['Train Accuracy', 'Val Accuracy'],
    title='Accuracy Over Epochs', labels={'value': 'Accuracy', 'variable': 'Dataset'}
)
metrics_fig = px.bar(
    test_metrics, x='Class', y=['Precision', 'Recall', 'F1-Score'],
    barmode='group', title='Per-Class Precision, Recall, and F1-Score',
    labels={'value': 'Score', 'variable': 'Metric'}
)
conf_fig = px.imshow(
    conf_matrix, text_auto=True, aspect="auto",
    labels=dict(x="Predicted", y="True", color="Count"),
    title='Confusion Matrix', x=conf_matrix.columns, y=conf_matrix.index
)

app.layout = html.Div([
    html.H1("BERT Fine-Tuning Performance Dashboard"),
    html.H2("Learning Curves"),
    dcc.Graph(figure=loss_fig),
    dcc.Graph(figure=acc_fig),
    html.H2("Test Set Performance"),
    html.P(f"Overall Accuracy: {test_metrics['F1-Score'].mean():.4f}"),
    dcc.Graph(figure=metrics_fig),
    dcc.Graph(figure=conf_fig)
])

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

---

### 8. `requirements.txt`
```
transformers==4.36.2
torch==2.0.1
pandas==2.0.3
scikit-learn==1.3.0
imbalanced-learn==0.11.0
dash==2.9.3
plotly==5.14.1
```

---

### Verification
- **Indentation in `train.py`**:
  - The `for epoch` loop is at the top level after variable declarations.
  - `model.train()` and `model.eval()` sections are indented under the loop.
  - Batch loops (`for batch in ...`) are nested correctly.
  - Early stopping logic is properly aligned under the epoch loop.
  - File saving is outside the loop, at the function level.

- **Running**:
  1. Ensure `data/raw_data.csv` exists (e.g., `"question,label"` format with 5 classes).
  2. Run `python main.py`.
  3. Check `models/bert_finetuned/` for saved files (`learning_curves.csv`, `test_metrics.csv`, `confusion_matrix.csv`, `label_mappings.json`).
  4. Open `http://localhost:8050` to view the dashboard.

---

### Explanation of Fixes
- **Indentation**: I’ve ensured all blocks in `train.py` follow Python’s 4-space convention. The previous version was conceptually correct, but I’ve rewritten it to eliminate any ambiguity (e.g., ensuring `print` statements and metric appending align with their respective training/validation phases).
- **Consistency**: All files use consistent indentation and structure, avoiding potential syntax errors.

This should resolve any indentation issues and provide a fully functional pipeline with the Dash dashboard. Let me know if you spot anything else or need further adjustments!