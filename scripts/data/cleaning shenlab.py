import pandas as pd
import json
import re
from pathlib import Path

# —— CONFIG ——
script_dir = Path(__file__).parent
INPUT_CSV = (script_dir
            .parent
            / 'data'# ...\data
            / 'dataset'          # ...\data\dataset
            / 'ShenLab_MentalChat16K.csv')  # file name

CLEAN_CSV   = Path('cleaned_shenLAbMentalChat16K.csv')
OUTPUT_JSON = Path('MentalChat16K_ft.jsonl')

# —— 1. LOAD & STRIP SYSTEM PREFIX ——
df = pd.read_csv(INPUT_CSV, dtype=str)

# Define the exact prompt prefix to remove
PREFIX = (
    r'^You are a helpful mental health counselling assistant, please answer the mental health questions '
    r'based on the patient\'s description\.\s*'
    r'The assistant gives helpful, comprehensive, and appropriate answers to the user\'s questions\.\s*'
)

# Remove it from the start of each user field
df['user'] = df['user'] \
    .astype(str) \
    .str.replace(PREFIX, '', regex=True) \
    .str.strip()  # drop leading/trailing spaces

# —— 2. NORMALIZE INSIDE FIELDS ——
for col in ['user','bot']:
    df[col] = (
        df[col]
        .str.replace(r'\s+', ' ', regex=True)   # collapse all whitespace
        .str.strip().str.strip('"')             # drop outer quotes
    )

# —— 3. DROP EXACT DUPLICATES ——
df = df.drop_duplicates(subset=['user','bot'])

# —— 4. COLLAPSE MULTIPLE ANSWERS PER USER ——
# Keep the longest bot reply for each user prompt
df = (
    df.assign(bot_len=df['bot'].str.len())
      .sort_values(['user','bot_len'], ascending=[True, False])
      .drop_duplicates(subset=['user'], keep='first')
      .drop(columns='bot_len')
)

# —— 5. SAVE CLEAN CSV ——
df.to_csv(CLEAN_CSV, index=False)
print(f"✔ Clean CSV written to {CLEAN_CSV}")

# —— 6. EXPORT JSONL FOR FINE-TUNING ——
with open(OUTPUT_JSON, 'w', encoding='utf-8') as out_f:
    for _, row in df.iterrows():
        record = {
            "prompt": row['user'] + "\n",       # trailing newline helps delimiting
            "completion": row['bot'] + "  "     # two spaces = end-of-completion token
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✔ JSONL ready for finetuning: {OUTPUT_JSON}")
