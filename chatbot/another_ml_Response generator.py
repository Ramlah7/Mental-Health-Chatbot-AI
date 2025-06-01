import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.sentiment_analysis import analyze_sentiment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Load the Fine-Tuned DialoGPT Model ===
MODEL_PATH = os.getenv("DIALOGPT_MODEL_PATH", "../data/finetuned_dialo")
print(f"📦 [Model] Loading DialoGPT from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

# === Prompt templates by sentiment ===
PROMPT_MAP = {
    "Positive": "You are a cheerful and supportive mental health assistant. Respond with kindness and encouragement.",
    "Negative": "You are an empathetic and calming mental health assistant. Comfort the user in a caring and non-judgmental way.",
    "Neutral":  "You are a balanced and helpful mental health assistant. Offer guidance and ask thoughtful questions."
}

def generate_dialogpt_reply(user_input):
    """Generate a sentiment-aware reply using fine-tuned DialoGPT."""
    try:
        if not user_input.strip():
            return "I'm here when you're ready to talk."

        sentiment = analyze_sentiment(user_input)
        print(f"[Sentiment] Detected: {sentiment}")

        prompt_instruction = PROMPT_MAP.get(sentiment, PROMPT_MAP["Neutral"])
        full_input = f"{prompt_instruction}\nUser: {user_input}\nAssistant:"

        input_ids = tokenizer.encode(full_input, return_tensors="pt")

        output_ids = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            num_return_sequences=1
        )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract assistant reply from generated text
        if "Assistant:" in decoded:
            reply = decoded.split("Assistant:")[-1].strip()
        else:
            reply = decoded.strip()

        if len(reply) < 3:
            return "I'm here to support you. Want to tell me more?"

        return reply

    except Exception as e:
        print(f"[DialoGPT Error] {e}")
        return "[Bot Error] Sorry, something went wrong."

# === Debugging CLI ===
if __name__ == "__main__":
    print("🧠 [MindMate - DialoGPT] Ready to chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        print("Bot:", generate_dialogpt_reply(user_input))
