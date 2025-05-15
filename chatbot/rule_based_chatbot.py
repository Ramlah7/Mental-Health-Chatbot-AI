# chatbot/rule_based_chatbot.py

from utils.sentiment_analysis import analyze_sentiment
from chatbot.ml_response_generator import find_best_match
import traceback

# === Unsafe content checker ===
def is_safe_response(response):
    bad_words = ['kill', 'sex', 'girlfriend', 'boyfriend', 'date', 'fashion', 'kiss']
    return not any(bad_word in response.lower() for bad_word in bad_words)

# === Hybrid Reply Generator ===
def generate_bot_reply(user_message):
    try:
        sentiment = analyze_sentiment(user_message)
        print(f"🔎 Detected Sentiment: {sentiment}")

        reply = find_best_match(user_message)

        if not is_safe_response(reply):
            return "I'm really sorry you're going through this. Would you like a breathing exercise or a self-care tip?"

        return reply
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
