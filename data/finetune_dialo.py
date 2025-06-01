import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

print("📦 [Step 1] Loading pretrained DialoGPT model...")
try:
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ DialoGPT-small loaded successfully.")
except Exception as e:
    print("❌ Failed to load model/tokenizer.")
    raise e

print("📁 [Step 2] Loading fine-tuning dataset from empathetic_finetune_dataset.jsonl...")
try:
    dataset_path = "empathetic_finetune_dataset.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        for line in lines:
            if "prompt" in line and "response" in line:
                line["text"] = f"{line['prompt']}\n{line['response']}"
            else:
                raise KeyError("Missing 'prompt' or 'response' key in a data entry.")
        dataset = Dataset.from_list(lines)
    print(f"✅ Loaded {len(dataset)} samples from: {dataset_path}")
except Exception as e:
    print("❌ Failed to load or parse your JSON dataset.")
    raise e

print("🔪 [Step 3] Tokenizing samples...")
try:
    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized_dataset = dataset.map(tokenize, batched=True)
    print("✅ Tokenization completed.")
except Exception as e:
    print("❌ Error during tokenization.")
    raise e

print("⚙️ [Step 4] Setting up training configuration...")
output_dir = "./finetuned_mindmate"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print("🚀 [Step 5] Starting training...")
try:
    trainer.train()
    print("✅ Training completed.")
except Exception as e:
    print("❌ Training failed.")
    raise e

print(f"🗓️ [Step 6] Saving model and tokenizer to: {output_dir}")
try:
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model + Tokenizer saved successfully.")
except Exception as e:
    print("❌ Saving failed.")
    raise e
