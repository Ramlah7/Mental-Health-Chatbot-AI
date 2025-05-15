# ml_response_generator.py

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import torch

# Load a fast, lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Good tradeoff between speed and accuracy

# Load and embed dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/mental_health_cleaned.csv")
df = pd.read_csv(data_path)

# Drop NA and clean short dialogues
df = df.dropna().query("user_input.str.len() > 0 and bot_reply.str.len() > 0", engine='python')

user_inputs = df['user_input'].tolist()
bot_replies = df['bot_reply'].tolist()

# Precompute embeddings only once
print("🔍 [Embeddings] Encoding dataset...")
user_embeddings = model.encode(user_inputs, convert_to_tensor=True)
print(f"✅ [Embeddings] {len(user_embeddings)} sentences embedded.")

def find_best_match(user_query):
    """Finds the best bot reply using semantic similarity."""
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, user_embeddings)[0]
    best_idx = int(torch.argmax(similarities))
    return bot_replies[best_idx]

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        print("Bot:", find_best_match(user_input))
