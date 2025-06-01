import pandas as pd
import json
import os

# === Paths ===
csv_path = os.path.join(os.path.dirname(__file__), "empathetic_dataset_cleaned.csv")
json_path = os.path.join(os.path.dirname(__file__), "dialo_finetune.json")

# === Load and clean the CSV ===
df = pd.read_csv(csv_path)
df = df.dropna()
df = df[df['user_input'].str.len() > 1]
df = df[df['bot_reply'].str.len() > 1]

# === Format for DialoGPT fine-tuning ===
with open(json_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = row['user_input'].strip()
        response = row['bot_reply'].strip()
        combined = {
            "text": f"User: {prompt}\nAssistant: {response}"
        }
        f.write(json.dumps(combined) + "\n")

print(f"✅ DialoGPT-formatted dataset saved to: {json_path}")
