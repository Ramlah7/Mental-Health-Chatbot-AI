# ✅ DialoGPT Training on EmpatheticDialogues with Google Drive + Checkpointing (Auto-Resume)

# Step 1: Mount Google Drive
from google.colab import drive
import pathlib, json, time, sys, traceback
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)

drive.mount('/content/drive')

# Step 2: Define paths
BASE_MODEL = "microsoft/DialoGPT-small"
ROOT = pathlib.Path("/content")
DATA_DIR = ROOT / "content"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"
OUT = pathlib.Path("/content/drive/MyDrive/mindmate_dialo_model")

# Step 3: Logging function
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Step 4: Read JSONL function
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Step 5: Auto-detect latest checkpoint
def get_latest_checkpoint(checkpoint_dir: pathlib.Path):
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[-1]))
    return str(checkpoints[-1]) if checkpoints else None

# Step 6: Main Training Logic
def main():
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        sys.exit("❌ train.jsonl or valid.jsonl not found in /content/content")

    OUT.mkdir(parents=True, exist_ok=True)

    log("🔁 Loading tokenizer and model …")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    log("📄 Reading and tokenizing dataset manually …")
    train_data_raw = read_jsonl(TRAIN_FILE)
    valid_data_raw = read_jsonl(VALID_FILE)

    train_enc = tok([x["text"] for x in train_data_raw], truncation=True, padding="max_length", max_length=256)
    valid_enc = tok([x["text"] for x in valid_data_raw], truncation=True, padding="max_length", max_length=256)

    train_enc["labels"] = train_enc["input_ids"]
    valid_enc["labels"] = valid_enc["input_ids"]

    train_dataset = Dataset.from_dict(train_enc)
    valid_dataset = Dataset.from_dict(valid_enc)

    log("🛠️ Setting up training arguments …")
    args = TrainingArguments(
        output_dir=str(OUT),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_total_limit=2,
        save_steps=500,
        logging_dir=str(OUT / "logs"),
        logging_steps=50,
        logging_first_step=True,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
    )

    latest_ckpt = get_latest_checkpoint(OUT)
    log(f"🚀 Starting training from checkpoint: {latest_ckpt if latest_ckpt else 'scratch'}")
    trainer.train(resume_from_checkpoint=latest_ckpt)

    log("✅ Training complete. Saving model …")
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    log(f"📦 Final model saved at: {OUT}")

try:
    main()
except KeyboardInterrupt:
    log("⏸️ Training interrupted — resume later.")
except Exception as e:
    log(f"❌ Error: {e}")
    traceback.print_exc()
