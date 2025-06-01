#!/usr/bin/env python3
"""
Build SBERT embeddings + FAISS index for MindMate
-------------------------------------------------
Reads   : data/conversation_pairs.csv         (user_input, bot_reply)
Creates : models/faiss_index/index.bin        (binary FAISS index)
          models/faiss_index/responses.csv    (bot replies in original order)
"""

import pathlib, sys, pandas as pd, numpy as np

# ---------------- project-root-agnostic paths ------------------------------
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent          # …/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                                # repo root

CSV   = PROJECT_ROOT / "data"   / "conversation_pairs.csv"
OUT   = PROJECT_ROOT / "models" / "faiss_index"
OUT.mkdir(parents=True, exist_ok=True)

def nice(p: pathlib.Path) -> str:
    try:  return str(p.relative_to(PROJECT_ROOT))
    except ValueError:  return str(p)

# ---------------- sanity checks --------------------------------------------
if not CSV.exists():
    sys.exit(f"❌ CSV not found: {CSV}")

# ---------------- load -----------------------------------------------------
print(f"[+] Loading pairs from {nice(CSV)} …")
df = pd.read_csv(CSV, encoding="utf-8").dropna(subset=["user_input", "bot_reply"])

if df.empty:
    sys.exit("❌ CSV is empty after dropping NaNs")

texts = df["user_input"].astype(str).tolist()
print(f"    → {len(texts)} user inputs to embed")

# ---------------- embed ----------------------------------------------------
from sentence_transformers import SentenceTransformer     # delayed import
print("[+] Encoding with Sentence-Transformers (all-mpnet-base-v2) …")
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")   # change to "cuda" if GPU ready
emb   = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True          # ✓ cosine-sim ready
).astype("float32")

# ---------------- build FAISS index ---------------------------------------
import faiss

dim   = emb.shape[1]
index = faiss.IndexFlatIP(dim)         # inner-product == cosine because vectors are normalised
index.add(emb)

idx_file = OUT / "index.bin"
faiss.write_index(index, str(idx_file))
print(f"[✓] FAISS index written → {nice(idx_file)}  ({index.ntotal} vectors)")

# ---------------- save bot replies ----------------------------------------
csv_out = OUT / "responses.csv"
df[["bot_reply"]].to_csv(csv_out, index=False)
print(f"[✓] Responses CSV written → {nice(csv_out)}")
