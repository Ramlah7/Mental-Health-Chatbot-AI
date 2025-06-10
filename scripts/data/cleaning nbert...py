import pandas as pd
import json
from pathlib import Path

# —— CONFIG ——
script_dir = Path(__file__).parent
INPUT_CSV = (script_dir
            .parent
            / 'data'# ...\data
            / 'dataset'          # ...\data\dataset
            / 'nbertagnolli_counsel_chat.csv')  # file name

CLEANED_CSV = Path('cleaned_nbert_counsel_chat.csv')
OUTPUT_JSONL = Path('training_data.jsonl')

# —— 1. LOAD & NORMALIZE WHITESPACE ——
df = pd.read_csv(INPUT_CSV, dtype=str)

# Strip outer quotes if present
df['user'] = df['user'].str.strip().str.strip('"').str.strip()
df['bot']  = df['bot'].str.strip().str.strip('"').str.strip()

# Collapse any weird internal newlines/spaces
df['user'] = df['user'].str.replace(r'\s+', ' ', regex=True)
df['bot']  = df['bot'].str.replace(r'\s+', ' ', regex=True)

# —— 2. REMOVE EXACT DUPLICATES ——
df = df.drop_duplicates(subset=['user', 'bot'])

# —— 3. COLLAPSE MULTIPLE ANSWERS PER QUESTION ——
#    Heuristic: keep the longest bot response for each user prompt.
df = (
    df
    .assign(bot_len = df['bot'].str.len())
    .sort_values(['user', 'bot_len'], ascending=[True, False])
    .drop_duplicates(subset=['user'], keep='first')
    .drop(columns='bot_len')
)

# —— 4. SAVE CLEAN CSV ——
df.to_csv(CLEANED_CSV, index=False)
print(f"✔ Clean CSV written to {CLEANED_CSV}")

# —— 5. EXPORT JSONL FOR FINE-TUNING ——
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out:
    for _, row in df.iterrows():
        # Format matches OpenAI fine-tuning: {"prompt": "...", "completion": "..."}
        record = {
            "prompt": row['user'] + "\n",       # you can tweak separators
            "completion": row['bot'] + "  "      # two spaces at end signals end-of-response
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✔ JSONL written to {OUTPUT_JSONL}")
