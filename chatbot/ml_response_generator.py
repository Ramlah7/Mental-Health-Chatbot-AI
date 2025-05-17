# chatbot/ml_response_generator.py

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import traceback
import time

# === Load the Model ===
try:
    print("🧠 [Model] Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ [Model] Model loaded successfully.")
except Exception as e:
    print("❌ [Error] Failed to load the model.")
    traceback.print_exc()
    raise

# === Load the Dataset ===
data_path = os.path.join(os.path.dirname(__file__), "../data/empathetic_dataset_cleaned.csv")
try:
    print(f"📁 [Data] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if df.empty or 'user_input' not in df.columns or 'bot_reply' not in df.columns:
        raise ValueError("CSV is empty or missing required columns.")
except Exception as e:
    print("❌ [Error] Failed to load dataset.")
    traceback.print_exc()
    raise RuntimeError(f"Failed to load or process dataset: {e}")

# === Clean the Data + Filter Inappropriate Replies ===
try:
    df = df.dropna()
    df = df.query("user_input.str.len() > 0 and bot_reply.str.len() > 0", engine='python')

    # Remove unsafe replies
    unwanted_keywords = ['kill', 'sex', 'date', 'girlfriend', 'boyfriend', 'kiss',
                         'fashion', 'party', 'drunk', 'relationship', 'marriage']
    df = df[~df['bot_reply'].str.contains('|'.join(unwanted_keywords), case=False, na=False)]

    user_inputs = df['user_input'].tolist()
    bot_replies = df['bot_reply'].tolist()
    print(f"✅ [Data] Cleaned and ready with {len(user_inputs)} user inputs.")
except Exception as e:
    print("❌ [Error] While cleaning data.")
    traceback.print_exc()
    raise

# === Encode Embeddings and Save/Load ===
embeddings_file = os.path.join(os.path.dirname(__file__), "../data/empathetic_embeddings.pt")
try:
    if os.path.exists(embeddings_file):
        print("📦 [Embeddings] Loading precomputed embeddings...")
        user_embeddings = torch.load(embeddings_file)
    else:
        print("🔍 [Embeddings] Starting encoding...")
        start_time = time.time()
        user_embeddings = model.encode(user_inputs, convert_to_tensor=True, show_progress_bar=True)
        torch.save(user_embeddings, embeddings_file)
        print(f"✅ [Embeddings] Encoded {len(user_embeddings)} inputs in {time.time() - start_time:.2f} seconds and saved.")
except Exception as e:
    print("❌ [Error] Failed during embedding computation.")
    traceback.print_exc()
    raise

# === Semantic Matching Function ===
def find_best_match(user_query):
    """Find the best matching reply from the dataset using semantic similarity."""
    try:
        if not user_query.strip():
            return "I'm here to help whenever you're ready to talk."

        query_embedding = model.encode(user_query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, user_embeddings)[0]
        best_idx = int(torch.argmax(similarities))
        return bot_replies[best_idx]
    except Exception as e:
        print("❌ [Error] During response generation.")
        traceback.print_exc()
        return "[Bot error: Unable to process your message.]"

# === CLI Debug Mode ===
if __name__ == "__main__":
    print("🗣️ Type a message (or type 'exit'):")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            print("Bot:", find_best_match(user_input))
        except Exception as e:
            print("❌ Unexpected error in chat loop:", e)
