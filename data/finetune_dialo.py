# finetune_dialo.py

import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

print("📦 [Step 1] Loading pretrained DialoGPT model...")
try:
    # === Load DialoGPT model + tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ✅ ADD THIS LINE
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ DialoGPT-small loaded successfully.")

except Exception as e:
    print("❌ Failed to load model/tokenizer.")
    raise e

print("📁 [Step 2] Loading your fine-tuning dataset...")
try:
    data_file = os.path.join(os.path.dirname(__file__), "dialo_finetune.json")
    dataset = load_dataset("json", data_files={"train": data_file}, split="train")
    print(f"✅ Loaded {len(dataset)} samples from: {data_file}")
except Exception as e:
    print("❌ Failed to load or parse your JSON dataset.")
    raise e

print("🧪 [Step 3] Tokenizing samples...")
try:
    def tokenize(batch):
        input_ids = tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=64)
        target_ids = tokenizer(batch["response"], truncation=True, padding="max_length", max_length=64)
        input_ids["labels"] = target_ids["input_ids"]
        return input_ids

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
    logging_steps=20,
    save_steps=200,
    #evaluation_strategy="no",
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

print(f"💾 [Step 6] Saving model and tokenizer to: {output_dir}")
try:
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model + Tokenizer saved successfully.")
except Exception as e:
    print("❌ Saving failed.")
    raise e
