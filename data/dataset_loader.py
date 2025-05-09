from datasets import load_dataset
import pandas as pd
import os

def download_and_prepare_dailydialog():
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    dialogues = dataset['train']['dialog']

    user_inputs, bot_replies = [], []

    for dialogue in dialogues:
        for i in range(len(dialogue) - 1):
            user_sentence = dialogue[i].strip()
            bot_sentence = dialogue[i + 1].strip()
            user_inputs.append(user_sentence)
            bot_replies.append(bot_sentence)

    df = pd.DataFrame({
        "user_input": user_inputs,
        "bot_reply": bot_replies
    })

    # Ensure data folder exists
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, 'mental_health_cleaned.csv')
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved at: {save_path}")

if __name__ == "__main__":
    download_and_prepare_dailydialog()
