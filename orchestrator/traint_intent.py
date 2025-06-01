# traint_intent.py – Fine-tunes a BERT‐tiny model on intent labels
import os
import random
import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

# -----------------------------------------------
# 1) Reproducibility: Set a seed for all libraries
# -----------------------------------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------
# 2) Paths and helper
# -----------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
def here(*paths):
    return ROOT.joinpath(*paths).resolve()

CSV_PATH = here("data", "week_label_intent.csv")
OUT_DIR = here("scripts", "models", "intent_bert")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------
# 3) Load DataFrame and map labels to IDs
# -----------------------------------------------
df = pd.read_csv(CSV_PATH)

# Build label2id / id2label
unique_labels = sorted(df.label.unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

df["label_id"] = df.label.map(label2id)

# -----------------------------------------------
# 4) Split into train / validation
#    (only stratify if every class has ≥ 2 samples)
# -----------------------------------------------
label_counts = df["label_id"].value_counts()

if all(label_counts[lbl] >= 2 for lbl in label_counts.index):
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df["label_id"],
        random_state=SEED
    )
else:
    print(
        "⚠️  Warning: Some labels have fewer than 2 samples. "
        "Proceeding with an un‐stratified split."
    )
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED
    )

# Keep only “text” and the integer label
train_df = train_df[["text", "label_id"]].rename(columns={"label_id": "label"})
val_df   = val_df[["text", "label_id"]].rename(columns={"label_id": "label"})

# -----------------------------------------------
# 5) Tokenizer + Dataset preparation
# -----------------------------------------------
MODEL_CHECKPOINT = "prajjwal1/bert-tiny"
tok = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_batch(batch):
    # Truncate/pad sequences to max_length=128 (adjust if you like)
    return tok(batch["text"], truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df).map(
    tokenize_batch, batched=True, remove_columns=["text"]
)
val_ds = Dataset.from_pandas(val_df).map(
    tokenize_batch, batched=True, remove_columns=["text"]
)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------------------------
# 6) Define compute_metrics
# -----------------------------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------------------------
# 7) Load model with the correct number of labels
# -----------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# -----------------------------------------------
# 8) Define minimal training arguments
#    (avoid any unsupported flags)
# -----------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    # Minimal required arguments; these should exist even in older versions
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir=str(OUT_DIR / "logs"),
    logging_steps=50,
    save_total_limit=2,
    seed=SEED
)

data_collator = DataCollatorWithPadding(tokenizer=tok)

# -----------------------------------------------
# 9) Initialize Trainer
# -----------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------------------------------
# 10) Train
# -----------------------------------------------
trainer.train()

# -----------------------------------------------
# 11) Evaluate on the validation set after training
# -----------------------------------------------
print("\n🧪 Running final evaluation on validation set:")
metrics = trainer.evaluate(eval_dataset=val_ds)
print(metrics)

# -----------------------------------------------
# 12) Save the fine‐tuned model + tokenizer
# -----------------------------------------------
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))

print(f"\n✅  Finished training & evaluation. Model + tokenizer saved to {OUT_DIR}")
