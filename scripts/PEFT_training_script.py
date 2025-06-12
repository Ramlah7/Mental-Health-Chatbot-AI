import re
import pathlib
import json
import time
import traceback
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# --- Configuration ---
MAX_LENGTH = 512                   # Increased to capture longer bot replies
BATCH_SIZE = 4                     # Adjust based on 6GB GPU; lower if OOM
ACCUMULATION_STEPS = 2             # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
NUM_EPOCHS = 5                     # Use early stopping
LEARNING_RATE = 5e-5               # Tuned for small dataset
BASE_MODEL = 'microsoft/DialoGPT-small'

# --- Paths ---
ROOT = pathlib.Path('/content')
DATA_DIR = ROOT / 'content'       # train.jsonl & valid.jsonl here
OUT_DIR = pathlib.Path('/content/drive/MyDrive/mindmate_dialo_model')
TRAIN_FILE = DATA_DIR / 'train.jsonl'
VALID_FILE = DATA_DIR / 'valid.jsonl'

# --- Logging helper ---
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- Read JSONL ---
def read_jsonl(path: pathlib.Path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# --- Prepare texts & mask labels ---
def prepare_texts(records, tokenizer, max_length=MAX_LENGTH):
    texts = []
    for r in records:
        ctx = str(r.get('context', '')).strip()
        resp = str(r.get('response', '')).strip()
        if not ctx or not resp:
            continue
        ctx = re.sub(r"\s+", " ", ctx)
        resp = re.sub(r"\s+", " ", resp)
        text = f"<|user|>{ctx}{tokenizer.eos_token}<|bot|>{resp}{tokenizer.eos_token}"
        texts.append(text)

    enc = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    # Mask labels: only compute loss on bot responses
    labels = enc.input_ids.clone()
    bot_id = tokenizer.convert_tokens_to_ids('<|bot|>')
    for i, seq in enumerate(enc.input_ids):
        positions = (seq == bot_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            start = int(positions[0])
            labels[i, :start] = -100
        else:
            labels[i, :] = -100
    enc['labels'] = labels
    return enc

# --- Find latest checkpoint ---
def get_latest_checkpoint(checkpoint_dir: pathlib.Path):
    pts = list(checkpoint_dir.glob('checkpoint-*'))
    if not pts:
        return None
    pts = sorted(pts, key=lambda x: int(x.name.split('-')[-1]))
    return str(pts[-1])

# --- Main training ---
def main():
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        raise FileNotFoundError(f"train.jsonl or valid.jsonl missing in {DATA_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenizer & model
    log('Loading tokenizer and base model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|bot|>']})
    tokenizer.pad_token = tokenizer.eos_token

    log('Initializing LoRA-PEFT configuration...')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    # Read data
    log('Reading train/valid files...')
    train_raw = read_jsonl(TRAIN_FILE)
    valid_raw = read_jsonl(VALID_FILE)
    log(f"Loaded {len(train_raw)} train and {len(valid_raw)} valid examples.")

    # Tokenize
    log('Tokenizing datasets...')
    train_enc = prepare_texts(train_raw, tokenizer, MAX_LENGTH)
    valid_enc = prepare_texts(valid_raw, tokenizer, MAX_LENGTH)
    train_dataset = Dataset.from_dict(train_enc).shuffle(seed=42)
    valid_dataset = Dataset.from_dict(valid_enc)
    log(f"Dataset sizes → train: {len(train_dataset)}, valid: {len(valid_dataset)}")

    # Training arguments with eval & early stopping
    log('Setting up training arguments...')
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        eval_strategy='steps',       # updated keyword
        eval_steps=500,
        logging_strategy='steps',
        logging_steps=100,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_total_limit=2,
        fp16=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    latest = get_latest_checkpoint(OUT_DIR)
    log(f"Starting training from: {latest or 'scratch'}")
    trainer.train(resume_from_checkpoint=latest)

    # Save final model
    log('Saving final model and tokenizer...')
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    log(f"Model saved at {OUT_DIR}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Training interrupted.')
    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc()
