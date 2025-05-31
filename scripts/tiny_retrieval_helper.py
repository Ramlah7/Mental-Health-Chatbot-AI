from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pandas as pd
from pathlib import Path

EMB_MODEL = SentenceTransformer("all-mpnet-base-v2", device="cpu")
INDEX     = faiss.read_index("models/faiss_index/index.bin")
RESP      = pd.read_csv("models/faiss_index/responses.csv")["bot_reply"].tolist()

q   = "I am feeling happy"
vec = EMB_MODEL.encode([q]).astype("float32")
D,I = INDEX.search(vec, 1)
print("Query:", q)
print("Best match (score", D[0][0], ") →", RESP[I[0][0]])