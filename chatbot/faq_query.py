# chatbot/faq_query.py
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "model" / "faiss_index" / "index.bin"
RESPONSES_PATH = BASE_DIR / "model" / "faiss_index" / "responses.csv"

model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index(str(INDEX_PATH))
responses = pd.read_csv(RESPONSES_PATH)["bot_reply"].tolist()

def query_faiss(user_input: str, top_k: int = 1, threshold: float = 0.75) -> str:
    embedding = model.encode([user_input])
    distances, indices = index.search(np.array(embedding).astype("float32"), top_k)

    if distances[0][0] > threshold:
        return responses[indices[0][0]]
    else:
        return "I'm here for you — could you tell me a bit more so I can help better?"
