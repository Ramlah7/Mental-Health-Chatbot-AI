from pathlib import Path
import pandas as pd
import sys

# 1. Locate the CSV relative to this script
script_dir = Path(__file__).parent  # ...\data\data
# assume your CSV is at ...\data\dataset\mental_health_faq.csv
csv_path = (script_dir
            .parent
            / 'data'# ...\data
            / 'dataset'          # ...\data\dataset
            / 'mental_health_faq.csv')  # file name

# 2. If it doesn’t exist, let the user know exactly where Python looked
if not csv_path.exists():
    sys.exit(f"\nERROR: CSV not found at {csv_path!r}\n"
             "Please move your file there, or adjust `csv_path` accordingly.\n")

# 3. Read & clean
df = pd.read_csv(csv_path, dtype=str)

# Define and strip off the prefixes
df['user'] = df['user'].str.replace(r'^\s*<HUMAN>:\s*', '', regex=True).str.strip()
df['bot']  = df['bot'].str.replace(r'^\s*<ASSISTANT>:\s*', '', regex=True).str.strip()

# 4. Write out
out_path = script_dir / 'mental_health_faq_cleaned.csv'
df.to_csv(out_path, index=False)
print(f"✔ Cleaned data written to {out_path}")
