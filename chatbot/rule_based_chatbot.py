# rule_based_chatbot.py

from utils.sentiment_analysis import analyze_sentiment
from chatbot.ml_response_generator import find_best_match
import traceback

def generate_bot_reply(user_message):
    try:
        sentiment = analyze_sentiment(user_message)
        print(f"🔎 Detected Sentiment: {sentiment}")
        return find_best_match(user_message)
    except Exception as e:
        print("❌ [Error] in generate_bot_reply:")
        traceback.print_exc()
        return f"[Bot error: {e}]"

# === CLI Debug Mode ===
if __name__ == "__main__":
    print("🗣️ Type a message (or type 'exit'):")
    while True:
        try:
            text = input("You: ")
            if text.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            print("Bot:", generate_bot_reply(text))
        except Exception as e:
            print("❌ Unexpected error in chat loop:", e)
