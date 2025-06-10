import os
import json
import html
import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup

# Helper functions (safe_strip, clean_html, inspect_dataset) remain the same.

def safe_strip(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    return text.strip()

def clean_html(raw_text):
    if raw_text is None:
        return ""
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ")
    return html.unescape(text).strip()

def inspect_dataset(dataset, num_examples: int = 2):
    print("=== Dataset Info ===")
    if hasattr(dataset, 'features'):
        print("Features:", dataset.features)
    else:
        try:
            print("Columns:", dataset.column_names)
        except:
            pass
    print("=== Examples ===")
    for i, ex in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
        print(f"Example {i}: {ex}")
    print("====================\n")

def parse_amod(example):
    return safe_strip(example.get("Context")), safe_strip(example.get("Response"))

def parse_shenlab(example):
    instr = safe_strip(example.get("instruction"))
    inp = safe_strip(example.get("input"))
    out = safe_strip(example.get("output"))
    user = f"{instr} {inp}" if inp else instr
    return user, out

def parse_zahrizhalali(example):
    # unchanged from before
    text = example.get("text", "")
    # ... same logic ...
    # (Use the previous parse_zahrizhalali implementation)
    import re
    user, bot = "", ""
    segments = re.split(r"<HUMAN>:", text)
    for seg in segments[1:]:
        parts = re.split(r"<ASSISTANT>:", seg)
        if len(parts) >= 2:
            u = parts[0].strip()
            a = parts[1].strip()
            if not user and u:
                user = safe_strip(u)
            if not bot and a:
                bot = safe_strip(a.split("<HUMAN>:")[0].strip())
            if user and bot:
                break
    if not user:
        user = safe_strip(text)
    return user, bot

def parse_heliosbrahma(example):
    return parse_zahrizhalali(example)

def parse_counsel_chat(example):
    return safe_strip(example.get("questionText")), safe_strip(example.get("answerText"))

def parse_faq_generic(example):
    # unchanged generic FAQ parser
    for q_field in ["Questions", "question", "questionText", "Question"]:
        for a_field in ["Answers", "answer", "answerText", "Answer"]:
            if q_field in example and a_field in example:
                return safe_strip(example.get(q_field)), safe_strip(example.get(a_field))
    if "text" in example:
        return safe_strip(example.get("text")), ""
    all_vals = [safe_strip(v) for v in example.values() if v is not None]
    return " | ".join(all_vals), ""

def parse_esconv_fixed(example):
    text = example.get("text", "")
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    dialog = data.get("dialog", [])
    pairs = []
    if isinstance(dialog, list):
        for i in range(len(dialog) - 1):
            curr = dialog[i]
            nxt = dialog[i+1]
            if curr.get("speaker") == "usr" and nxt.get("speaker") == "sys":
                u = safe_strip(curr.get("text"))
                b = safe_strip(nxt.get("text"))
                if u:
                    pairs.append((u, b))
    return pairs

def parse_luang_fixed(example):
    return safe_strip(example.get("input")), safe_strip(example.get("label"))

# process_hf_dataset remains the same, except we feed the fixed parsers.

def process_hf_dataset(hf_name: str, parser_fn, multi_pair: bool=False, split: str="train"):
    print(f"\nLoading dataset {hf_name} ...")
    try:
        ds = load_dataset(hf_name)
    except Exception as e:
        print(f"[Error] Failed to load {hf_name}: {e}")
        return
    if isinstance(ds, dict):
        ds_part = ds.get(split) or next(iter(ds.values()))
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
                        all_pairs.append((clean_html(u2), clean_html(b2)))
            else:
                u, b = parser_fn(example)
                u2, b2 = safe_strip(u), safe_strip(b)
                if u2:
                    all_pairs.append((clean_html(u2), clean_html(b2)))
        except Exception:
            continue
    if not all_pairs:
        print(f"[Info] No pairs extracted for {hf_name} (check parser).")
    else:
        dataset_name_sanitized = hf_name.replace("/", "_").replace("-", "_")
        out_path = os.path.join("dataset", f"{dataset_name_sanitized}.csv")
        pd.DataFrame(all_pairs, columns=["user", "bot"]).to_csv(out_path, index=False)
        print(f"Saved {len(all_pairs)} pairs to {out_path}")

def main():
    os.makedirs("dataset", exist_ok=True)
    hf_datasets = [
        ("Amod/mental_health_counseling_conversations", parse_amod, False),
        ("ShenLab/MentalChat16K", parse_shenlab, False),
        ("ZahrizhalAli/mental_health_conversational_dataset", parse_zahrizhalali, False),
        ("heliosbrahma/mental_health_chatbot_dataset", parse_heliosbrahma, False),
        ("nbertagnolli/counsel-chat", parse_counsel_chat, False),
        ("tolu07/Mental_Health_FAQ", parse_faq_generic, False),
        ("thu-coai/esconv", parse_esconv_fixed, True),
        ("LuangMV97/Empathetic_counseling_Dataset", parse_luang_fixed, False),
        # Felladrin: skip pairing since no question field
        # ("Felladrin/pretrain-mental-health-counseling-conversations", ..., False),
    ]
    for name, parser_fn, multi in hf_datasets:
        process_hf_dataset(name, parser_fn, multi_pair=multi, split="train")

if __name__ == "__main__":
    main()
