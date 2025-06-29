#run this on colab
# 📌 2. Import everything
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import torch

# 📌 3. Load tokenizer and model
model_name = "facebook/blenderbot_small-90M"
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)

# 📌 4. Load your dataset from JSONL
import json

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

train_data = load_jsonl("/content/train_formatted.jsonl")  # <-- Upload via left panel or Drive

# 📌 5. Convert to HuggingFace Dataset
dataset = Dataset.from_list(train_data)
import os
os.environ["WANDB_DISABLED"] = "true"


# 📌 6. Tokenization function
def tokenize(example):
    model_inputs = tokenizer(example["instruction"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(text_target=example["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

training_args = TrainingArguments(
    output_dir="./blenderbot-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=20,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
)

# 📌 8. Train using Hugging Face Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 📌 9. Save model
model.save_pretrained("/content/mindmate_finetuned")
tokenizer.save_pretrained("/content/mindmate_finetuned")

# 📌 10. Sample Inference
def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

chat("You are MindMate. Respond to a [rant] message: 'I studied all night and still failed my exam 😭'")
