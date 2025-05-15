# dataset_loader.py

from datasets import load_dataset
import pandas as pd
import os

def download_and_prepare_empathetic_dataset():
    print("📥 [Dataset] Loading EmpatheticDialogues...")
    ed = load_dataset("empathetic_dialogues", trust_remote_code=True)

    user_inputs = []
    bot_replies = []

    # Extract dialogue turns (utterance = user, context = bot)
    for row in ed['train']:
        user_inputs.append(row['utterance'])
        bot_replies.append(row['context'])

    # Clean and filter
    df = pd.DataFrame({
        "user_input": user_inputs,
        "bot_reply": bot_replies
    })

    df = df.dropna()
    df = df[df['user_input'].str.len() > 0]
    df = df[df['bot_reply'].str.len() > 0]

    # Save to CSV
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'empathetic_dataset_cleaned.csv')
    df.to_csv(save_path, index=False)
    print(f"✅ [Dataset] Saved dataset to: {save_path}")

if __name__ == "__main__":
    download_and_prepare_empathetic_dataset()
