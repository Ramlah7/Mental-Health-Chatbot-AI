from datasets import load_dataset
import json
from pathlib import Path
import random

# ✅ Allow remote code execution required by the dataset
dataset = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)

# Combine context + reply as (user, bot)
pairs = []
for example in dataset["train"]:
    user_utt = example["utterance"]
    bot_utt = example["context"]
    pairs.append((f"User: {bot_utt}", f"Bot: {user_utt}"))

# Shuffle and split
random.seed(42)
random.shuffle(pairs)
split_idx = int(0.9 * len(pairs))
train_pairs = pairs[:split_idx]
valid_pairs = pairs[split_idx:]

# Save to JSONL
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
    for u, b in train_pairs:
        f.write(json.dumps({"text": f"{u}\n{b}"}, ensure_ascii=False) + "\n")

with (output_dir / "valid.jsonl").open("w", encoding="utf-8") as f:
    for u, b in valid_pairs:
        f.write(json.dumps({"text": f"{u}\n{b}"}, ensure_ascii=False) + "\n")

print("✅ Dataset prepared and saved to data/processed/")
