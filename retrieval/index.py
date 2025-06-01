"""
retrieval/index.py – FAQ response retriever using FAISS + Sentence-BERT
"""

import pathlib, faiss, pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# 🛠 Path helper – project root auto-detection
ROOT = pathlib.Path(__file__).resolve().parents[1]
def here(*parts) -> str:
    """Return absolute path (POSIX style) from project root."""
    return str(ROOT.joinpath(*parts).resolve().as_posix())

# 📦 Configuration
MODEL_NAME    = "all-mpnet-base-v2"
SIM_THRESHOLD = 0.78
INDEX_PATH    = here("scripts", "models", "faiss_index", "index.bin")
RESP_PATH     = here("scripts", "models", "faiss_index", "responses.csv")

# 🔄 Load components
print(f"[retrieval] Loading embedder → {MODEL_NAME}")
_model = SentenceTransformer(MODEL_NAME, device="cpu")

print(f"[retrieval] Reading FAISS index → {INDEX_PATH}")
_index = faiss.read_index(INDEX_PATH)

print(f"[retrieval] Reading responses CSV → {RESP_PATH}")
_responses = pd.read_csv(RESP_PATH)["bot_reply"].tolist()

assert _index.ntotal == len(_responses), (
    f"[retrieval] Index size ({_index.ntotal}) ≠ responses ({len(_responses)})"
)

# 🔍 API
def faq_query(text: str, k: int = 1) -> List[Tuple[str, float]]:
    """
    Search FAISS index and return up to `k` (reply, score) results.
    Only includes results with similarity ≥ SIM_THRESHOLD.
    """
    vec = _model.encode([text]).astype("float32")
    D, I = _index.search(vec, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= SIM_THRESHOLD:
            results.append((_responses[idx], float(score)))
    return results
