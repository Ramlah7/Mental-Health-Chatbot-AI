# scripts/weak_label_intent.py
import pandas as pd, re, random, pathlib, json

RULES = {
    "greeting" : r"^(hi|hello|hey)\b",
    "definition_request": r"\bwhat\s+is\b|\bdefine\b",
    "self_harm": r"\bkill myself\b|\bsuicide\b",
    "coping_advice": r"\bhow (do|can) i\b|\bhelp me\b",
}

df = pd.read_csv("../data/conversation_pairs.csv")
labels = []
for txt in df["user_input"]:
    lab = "other"
    for name,pat in RULES.items():
        if re.search(pat, txt, flags=re.I):
            lab = name; break
    labels.append(lab)
df["label"] = labels
df.to_csv("data/intent_labeled.csv", index=False)
print(df["label"].value_counts())
