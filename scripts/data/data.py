# ================================================================
# Python script: load_and_parse_mental_health_datasets.py
# ================================================================
# Dependencies:
#   pip install datasets pandas kaggle html5lib beautifulsoup4
#
# For Kaggle datasets: set up ~/.kaggle/kaggle.json with your Kaggle credentials.
# ================================================================

import os
import csv
import re
import html
from datasets import load_dataset
import pandas as pd

# For HTML tag removal
from bs4 import BeautifulSoup

# For Kaggle API (optional)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_available = True
except ImportError:
    kaggle_available = False

# Create output directory
OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_strip(text):
    """
    Safely strip whitespace. If None or not string, return empty string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    return text.strip()

def clean_html(raw_text):
    """
    Remove HTML tags and unescape HTML entities.
    """
    if raw_text is None:
        return ""
    # Use BeautifulSoup to remove tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ")
    return html.unescape(text).strip()

def inspect_dataset(dataset, num_examples: int = 3):
    """
    Print schema/features and a few examples.
    """
    print("=== Dataset Info ===")
    # Print features/schema
    if hasattr(dataset, 'features'):
        print("Features:", dataset.features)
    else:
        print("No `.features` attribute; printing column names if pandas DataFrame-like:")
        try:
            print("Columns:", dataset.column_names)
        except:
            pass
    print("=== Examples ===")
    for i, ex in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
        print(f"Example {i}:")
        print(ex)
    print("====================\n")

def parse_amod(example):
    """
    Parse Amod/mental_health_counseling_conversations,
    fields: 'Context' (question), 'Response' (answer).
    """
    user = safe_strip(example.get("Context"))
    bot = safe_strip(example.get("Response"))
    return user, bot

def parse_shenlab(example):
    """
    Parse ShenLab/MentalChat16K,
    fields: 'instruction', 'input', 'output'.
    Combine instruction + input for user if input non-empty.
    """
    instr = safe_strip(example.get("instruction"))
    inp = safe_strip(example.get("input"))
    out = safe_strip(example.get("output"))
    if inp:
        user = f"{instr} {inp}"
    else:
        user = instr
    bot = out
    return user, bot

def parse_zahrizhalali(example):
    """
    Parse ZahrizhalAli/mental_health_conversational_dataset,
    field: 'text' in format "<HUMAN>: ... <ASSISTANT>: ...".
    We extract first HUMAN/ASSISTANT pair.
    """
    text = example.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    # Split by tags
    # e.g., "<HUMAN>: question ... <ASSISTANT>: answer ..."
    user, bot = "", ""
    # Find all segments
    # Strategy: find all occurrences of "<HUMAN>:" then corresponding "<ASSISTANT>:"
    segments = re.split(r"<HUMAN>:", text)
    for seg in segments[1:]:
        parts = re.split(r"<ASSISTANT>:", seg)
        if len(parts) >= 2:
            u = parts[0].strip()
            # answer may contain further HUMAN segments; we take until next tag
            a = parts[1].strip()
            # Choose first non-empty
            if not user and u:
                user = safe_strip(u)
            if not bot and a:
                bot = safe_strip(a.split("<HUMAN>:")[0].strip())
            if user and bot:
                break
    # Fallback: if still empty, return full text as user, empty bot
    if not user:
        user = safe_strip(text)
    return user, bot

def parse_heliosbrahma(example):
    """
    Parse heliosbrahma/mental_health_chatbot_dataset,
    same format as ZahrizhalAli: 'text' field.
    """
    return parse_zahrizhalali(example)

def parse_counsel_chat(example):
    """
    Parse nbertagnolli/counsel-chat.
    Fields: 'questionText', 'answerText'.
    """
    user = safe_strip(example.get("questionText"))
    bot = safe_strip(example.get("answerText"))
    return user, bot

def parse_faq_generic(example):
    """
    Generic FAQ parser: look for common fields 'question'/'answer' or 'Question'/'Answer'.
    If unknown, join all fields.
    """
    # Attempt common field names
    for q_field in ["question", "questionText", "Question", "QuestionText"]:
        for a_field in ["answer", "answerText", "Answer", "AnswerText"]:
            if q_field in example and a_field in example:
                return safe_strip(example.get(q_field)), safe_strip(example.get(a_field))
    # If only one field, e.g., 'text', return text->"" or skip
    if "text" in example:
        txt = safe_strip(example.get("text"))
        return txt, ""
    # Fallback: join fields
    all_vals = [safe_strip(v) for v in example.values() if v is not None]
    combined = " | ".join(all_vals)
    return combined, ""

def parse_esconv(example):
    """
    Parse thu-coai/esconv.
    Field: 'dialog' is a list of dicts with keys 'text' and 'speaker' ('usr' or 'sys').
    Extract all consecutive (usr, sys) pairs.
    Returns list of (user, bot) pairs.
    """
    pairs = []
    dialog = example.get("dialog", [])
    if isinstance(dialog, list):
        for i in range(len(dialog) - 1):
            curr = dialog[i]
            nxt = dialog[i+1]
            if curr.get("speaker") == "usr" and nxt.get("speaker") == "sys":
                user = safe_strip(curr.get("text"))
                bot = safe_strip(nxt.get("text"))
                pairs.append((user, bot))
    return pairs  # list

def parse_luang_empatic(example):
    """
    Parse LuangMV97/Empathetic_counseling_Dataset.
    Schema uncertain: inspect first, then attempt.
    We do: inspect keys; if fields like 'input'/'response', use those.
    Else, if single text, split on newline into pairs.
    """
    # If known fields:
    for q_field in ["input", "prompt", "question", "utterance", "text"]:
        for a_field in ["response", "reply", "answer", "output"]:
            if q_field in example and a_field in example:
                return safe_strip(example.get(q_field)), safe_strip(example.get(a_field))
    # Fallback: if a single 'text' containing alternating lines:
    # Split on newline and take consecutive pairs
    text = example.get("text") or example.get("dialog") or None
    if isinstance(text, str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            # Pair consecutive lines
            # Return first pair
            return safe_strip(lines[0]), safe_strip(lines[1])
    # Else return empty
    return "", ""

def save_pairs_to_csv(pairs, filename):
    """
    Save list of (user, bot) pairs to CSV with headers 'user','bot'.
    """
    df = pd.DataFrame(pairs, columns=["user", "bot"])
    # Drop empty user rows
    df = df[df['user'].astype(bool)]
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} pairs to {out_path}")

def process_hf_dataset(hf_name: str, parser_fn, multi_pair: bool=False, split: str="train"):
    """
    Load a Hugging Face dataset by name, inspect, parse, and save.
    - parser_fn: function(example) -> (user, bot) or -> list of (user, bot) if multi_pair=True
    - multi_pair: if True, parser_fn returns list of pairs; else single pair.
    """
    print(f"\nLoading dataset {hf_name} ...")
    try:
        ds = load_dataset(hf_name)
    except Exception as e:
        print(f"Failed to load {hf_name}: {e}")
        return

    # If dataset has splits, pick default split if present
    if isinstance(ds, dict):
        if split in ds:
            ds_part = ds[split]
        else:
            # pick first split
            ds_part = list(ds.values())[0]
    else:
        ds_part = ds

    inspect_dataset(ds_part, num_examples=2)

    all_pairs = []
    for example in ds_part:
        try:
            if multi_pair:
                pairs = parser_fn(example)
                for u, b in pairs:
                    u2, b2 = safe_strip(u), safe_strip(b)
                    if u2:
                        all_pairs.append((u2, b2))
            else:
                u, b = parser_fn(example)
                u2, b2 = safe_strip(u), safe_strip(b)
                if u2:
                    all_pairs.append((u2, b2))
        except Exception as e:
            # Skip problematic example
            continue

    # Clean text further if desired; e.g., remove HTML
    cleaned_pairs = []
    for u, b in all_pairs:
        cu = clean_html(u)
        cb = clean_html(b)
        cleaned_pairs.append((cu, cb))

    # Save
    dataset_name_sanitized = hf_name.replace("/", "_").replace("-", "_")
    filename = f"{dataset_name_sanitized}.csv"
    save_pairs_to_csv(cleaned_pairs, filename)

def process_kaggle_dataset(kaggle_dataset_identifier: str, local_file: str, parser_fn):
    """
    Template for Kaggle dataset:
    - kaggle_dataset_identifier: e.g., "username/dataset-name"
    - local_file: path to CSV after download
    - parser_fn: function(example_dict) -> (user, bot)
    """
    # 1) Download via Kaggle API if available
    if kaggle_available:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading Kaggle dataset {kaggle_dataset_identifier} ...")
        try:
            api.dataset_download_files(kaggle_dataset_identifier, path="kaggle_data", unzip=True, quiet=False)
            # Assume local_file path under kaggle_data
            local_path = os.path.join("kaggle_data", local_file)
        except Exception as e:
            print(f"Failed to download via Kaggle API: {e}")
            if not os.path.exists(local_file):
                print(f"Please manually download and place {local_file} locally.")
                return
            local_path = local_file
    else:
        if not os.path.exists(local_file):
            print(f"Kaggle API not available. Please place {local_file} locally.")
            return
        local_path = local_file

    # 2) Load CSV via pandas
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        print(f"Failed to read {local_path}: {e}")
        return
    print(f"Inspecting Kaggle DataFrame columns: {df.columns.tolist()}")
    # 3) Parse rows
    pairs = []
    for _, row in df.iterrows():
        example = row.to_dict()
        try:
            u, b = parser_fn(example)
            u2, b2 = safe_strip(u), safe_strip(b)
            if u2:
                pairs.append((clean_html(u2), clean_html(b2)))
        except:
            continue
    # Save
    dataset_name_sanitized = kaggle_dataset_identifier.replace("/", "_").replace("-", "_")
    filename = f"{dataset_name_sanitized}.csv"
    save_pairs_to_csv(pairs, filename)

def main():
    # ----------------------------------------
    # 1. Hugging Face datasets
    # ----------------------------------------
    hf_datasets = [
        # (hf_name, parser_fn, multi_pair)
        ("Amod/mental_health_counseling_conversations", parse_amod, False),
        ("ShenLab/MentalChat16K", parse_shenlab, False),
        ("ZahrizhalAli/mental_health_conversational_dataset", parse_zahrizhalali, False),
        ("heliosbrahma/mental_health_chatbot_dataset", parse_heliosbrahma, False),
        ("nbertagnolli/counsel-chat", parse_counsel_chat, False),
        ("tolu07/Mental_Health_FAQ", parse_faq_generic, False),
        ("thu-coai/esconv", parse_esconv, True),
        ("LuangMV97/Empathetic_counseling_Dataset", parse_luang_empatic, False),
        ("Felladrin/pretrain-mental-health-counseling-conversations", parse_amod, False),  # schema similar to Amod
        # Add other HF datasets as desired, e.g., augmentals or research datasets
    ]

    for hf_name, parser_fn, multi_pair in hf_datasets:
        process_hf_dataset(hf_name, parser_fn, multi_pair=multi_pair, split="train")

    # ----------------------------------------
    # 2. Kaggle datasets (template examples)
    # ----------------------------------------
    # Example: mental health conversations from Kaggle (replace with actual identifiers and files)
    # process_kaggle_dataset("someuser/mental-health-conversations", "mental_health_conversations.csv", parse_zahrizhalali_or_custom)
    # process_kaggle_dataset("someuser/mental-health-faq", "mental_health_faq.csv", parse_faq_generic)

    # If you have a specific Kaggle dataset:
    # def parse_my_kaggle_example(example):
    #     # Custom parsing based on columns in the CSV
    #     user = example.get("question") or example.get("input_text") or ""
    #     bot = example.get("answer") or example.get("response_text") or ""
    #     return user, bot
    #
    # process_kaggle_dataset("username/dataset-name", "downloaded_file.csv", parse_my_kaggle_example)

if __name__ == "__main__":
    main()
