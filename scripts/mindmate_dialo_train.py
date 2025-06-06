# scripts/finetune_dialo.py
from __future__ import annotations
import pathlib, sys, time, traceback
from packaging import version
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
import transformers as _tf

# Paths
d = pathlib.Path(__file__).resolve().parents[1]
DATA = d / "data" / "processed"
OUT  = d / "models" / "mindmate_dialo"
TRAIN_FILE = DATA / "train.jsonl"
VALID_FILE = DATA / "valid.jsonl"
BASE_MODEL = "microsoft/DialoGPT-small"

def log(msg: str): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        sys.exit("Missing JSONL files.")
    OUT.mkdir(parents=True, exist_ok=True)

    log("Loading model …")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    log("Loading & tokenising dataset …")
    ds = load_dataset("json", data_files={"train":str(TRAIN_FILE),"validation":str(VALID_FILE)})
    ds = ds.map(lambda b: tok(b["text"],truncation=True,padding="max_length",max_length=256), batched=True)
    for sp in ["train","validation"]: ds[sp] = ds[sp].add_column("labels", ds[sp]["input_ids"])

    common = dict(output_dir=str(OUT), overwrite_output_dir=True,
                  num_train_epochs=8, per_device_train_batch_size=1,
                  gradient_accumulation_steps=4, learning_rate=2e-4,
                  save_total_limit=2, fp16=False, report_to="none",
                  logging_steps=50, logging_first_step=True)
    try:
        args = TrainingArguments(**common, evaluation_strategy="epoch", save_strategy="epoch")
    except TypeError:
        log("Fallback to step-based eval/save")
        args = TrainingArguments(**common, eval_steps=500, save_steps=500)

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"], eval_dataset=ds["validation"],
                      tokenizer=tok, data_collator=DataCollatorForLanguageModeling(tok,mlm=False))

    ckpts = sorted(OUT.glob("checkpoint-*"))
    resume = ckpts[-1] if ckpts else None
    if resume: log(f"Resuming {resume.name}")

    log("Starting training (Ctrl-C to pause)…")
    trainer.train(resume_from_checkpoint=resume)
    log("Training complete.")

    log("Saving model …")
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    log(f"Saved to {OUT.relative_to(d)}")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: log("Interrupted — resume later.")
    except Exception as e: log(f"Error: {e}"); traceback.print_exc()