#!/usr/bin/env python3
"""
Clean + split conversational dataset for MindMate
-------------------------------------------------
Reads    : data/conversation_pairs.csv  (columns: user_input, bot_reply)
Writes   : data/processed/train.jsonl  (90 %)
           data/processed/valid.jsonl  (10 %)

Cleaning : • Unicode normalisation (NFKC)
           • collapse whitespace
           • strip leading/trailing spaces
"""

import pandas as pd
import unicodedata, re, pathlib, random, json, sys

# ---------------------------------------------------------------------------
# 1.  Resolve project-root, independent of where you launch the script
#     (…/Mental-Health-Chatbot-AI in your repo structure)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent              # …/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                                    # repo root

CSV_IN  = PROJECT_ROOT / "data" / "conversation_pairs.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# 2.  Helper: pretty-print a path *relative to* PROJECT_ROOT when possible
# ---------------------------------------------------------------------------
def nice(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

# ---------------------------------------------------------------------------
# 3.  Text cleaning routine
# ---------------------------------------------------------------------------
_ws_re = re.compile(r"\s+")

def _clean(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    return _ws_re.sub(" ", text).strip()

# ---------------------------------------------------------------------------
# 4.  Load, clean, shuffle, split, save
# ---------------------------------------------------------------------------
if not CSV_IN.exists():
    sys.exit(f"❌ CSV not found: {CSV_IN}")

print(f"[+] Reading  {nice(CSV_IN)} …")
df = pd.read_csv(CSV_IN, encoding="utf-8")

if not {"user_input", "bot_reply"}.issubset(df.columns):
    sys.exit("❌ CSV must contain 'user_input' and 'bot_reply' columns")

print("[+] Cleaning rows …")
df = df.dropna(subset=["user_input", "bot_reply"]).copy()
df["user_input"] = df["user_input"].apply(_clean)
df["bot_reply"]  = df["bot_reply"].apply(_clean)

pairs = list(df[["user_input", "bot_reply"]].itertuples(index=False, name=None))
random.Random(SEED).shuffle(pairs)

split_idx = int(len(pairs) * 0.9)
splits = {"train": pairs[:split_idx], "valid": pairs[split_idx:]}

for name, chunk in splits.items():
    out_path = OUT_DIR / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for u, b in chunk:
            fh.write(json.dumps({"text": f"{u}  {b}"}, ensure_ascii=False) + "\n")
    print(f"    ✓ {name:<5} → {nice(out_path)}  ({len(chunk)} rows)")

print("[✔] Pre-processing complete.")
