# train_emotion.py — Transformer-safe legacy version (old HuggingFace)
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# ---------------- PATH SETUP ----------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
def here(*p): return ROOT.joinpath(*p).resolve()

CSV = here("data", "week_label_emotion.csv")  # Make sure this is emoji-free and clean
OUT = here("scripts", "models", "emotion_bert")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD & CLEAN DATA ----------------
df = pd.read_csv(CSV).dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Label encoding
unique_labels = sorted(df["label"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["label"].map(label2id)

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# ---------------- TOKENIZATION ----------------
tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
def encode(batch): return tok(batch["text"], truncation=True, padding=True)

train_ds = Dataset.from_pandas(train_df).map(encode, batched=True, remove_columns=["text"])
val_ds   = Dataset.from_pandas(val_df).map(encode, batched=True, remove_columns=["text"])

# ---------------- MODEL ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ---------------- TRAINING ARGUMENTS (SAFE) ----------------
args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-5,
    logging_dir=str(OUT / "logs")
)

# ---------------- TRAINING ----------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok)
)

trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"✅ emotion_bert saved to {OUT}")
