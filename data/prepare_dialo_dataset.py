# data/prepare_dialo_dataset.py

import pandas as pd
import json
import os

csv_path = os.path.join(os.path.dirname(__file__), "empathetic_dataset_cleaned.csv")
json_path = os.path.join(os.path.dirname(__file__), "dialo_finetune.json")

df = pd.read_csv(csv_path)
df = df.dropna()
df = df[df['user_input'].str.len() > 0]
df = df[df['bot_reply'].str.len() > 0]

with open(json_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = row['user_input'].strip()
        response = row['bot_reply'].strip()
        json_line = {"prompt": prompt, "response": response}
        f.write(json.dumps(json_line) + "\n")

print(f"✅ Dataset saved to: {json_path}")
