import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

from pathlib import Path
# Define absolute base path (e.g., project root)
BASE_DIR = Path(__file__).resolve().parents[1]  # Adjust parents[N] as needed

# Construct the full path to the CSV file
CSV_PATH = BASE_DIR / "data" / "mental_health_tagged.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Load SBERT model
model = SentenceTransformer("all-mpnet-base-v2")

# Get user inputs and compute embeddings
texts = df["user_input"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save index
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.bin")

# Save bot replies
df[["bot_reply"]].to_csv("faiss_index/responses.csv", index=False)

print("✅ FAISS index and responses.csv saved successfully.")
