"""Fine‑tune DialoGPT‑medium on MindMate data"""
import torch, pathlib
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, DataCollatorForLanguageModeling,
                          Trainer)

BASE = "microsoft/DialoGPT-medium"
TOK  = AutoTokenizer.from_pretrained(BASE)
TOK.pad_token = TOK.eos_token

# ----- dataset ------------------------------------------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/valid.jsonl",
    },
)

def tokenize(batch):
    out = TOK(batch["text"], truncation=True, padding="max_length", max_length=256)
    out["labels"] = out["input_ids"].copy()
    return out

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ----- model --------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(BASE)

args = TrainingArguments(
    output_dir="models/dialo_ft",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(TOK, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("models/dialo_ft")
    TOK.save_pretrained("models/dialo_ft")
    print("[✔] DialoGPT fine‑tune complete → models/dialo_ft/")