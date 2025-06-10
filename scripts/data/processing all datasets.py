import os
import pandas as pd
import json
import random

# === Configuration ===
# Determine the directory of this script and locate 'Final_Datasets' subfolder
def base_dir():
    return os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(base_dir(), "Final_Datasets")

# List of expected CSV filenames in 'Final_Datasets'
default_csv_files = [
    "Amod_mental_health_counseling_conversations.csv",
    "cleaned_nbert_counsel_chat.csv",
    "cleaned_shenLabMentalChat16K.csv",
    "heliosbrahma_mental_health_chatbot_dataset.csv",
    "mental_health_faq_cleaned.csv",
    "tolu07_Mental_Health_FAQ.csv",
    "ZahrizhalAli_mental_health_conversational_dataset.csv",
]

# Output JSONL files (written to script directory)
output_train = os.path.join(base_dir(), "train.jsonl")
output_valid = os.path.join(base_dir(), "valid.jsonl")

# Fraction of data reserved for validation
split_ratio = 0.1
# Random seed for reproducibility
seed = 42

# Possible column schema options (lowercase matching)
schema_options = [
    ("user", "bot"),
    ("context", "response"),
    ("question", "answer"),
    ("prompt", "completion"),
]

records = []

# 1) Load, detect schema, standardize each CSV
for fname in default_csv_files:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        continue

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        continue

    cols_lower = {col.lower(): col for col in df.columns}
    mapping = None
    for ctx_col, resp_col in schema_options:
        if ctx_col in cols_lower and resp_col in cols_lower:
            mapping = (cols_lower[ctx_col], cols_lower[resp_col])
            break
    if mapping is None:
        print(f"[Skipping] {fname}: no matching schema. Columns: {list(df.columns)}")
        continue

    real_ctx, real_resp = mapping
    df = df[[real_ctx, real_resp]].rename(columns={real_ctx: "context", real_resp: "response"})

    # 2) Clean: strip whitespace and drop empty entries
    df["context"] = df["context"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[(df["context"].str.len() > 0) & (df["response"].str.len() > 0)]

    # 3) Collect records
    records.extend(df.to_dict(orient="records"))

# 4) Deduplicate context–response pairs
unique = {(r["context"], r["response"]): r for r in records}
all_records = list(unique.values())

# 5) Shuffle & split into train/validation
random.seed(seed)
random.shuffle(all_records)

n_total = len(all_records)
n_valid = int(n_total * split_ratio)
valid_records = all_records[:n_valid]
train_records = all_records[n_valid:]

# 6) Write out JSONL files
with open(output_train, "w", encoding="utf-8") as ft:
    for rec in train_records:
        ft.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(output_valid, "w", encoding="utf-8") as fv:
    for rec in valid_records:
        fv.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 7) Summary log
print(f"Final datasets directory: {data_dir}")
print(f"Total dialog pairs: {n_total}")
print(f"Training examples: {len(train_records)} -> {output_train}")
print(f"Validation examples: {len(valid_records)} -> {output_valid}")