# chatbot/rule_based_chatbot.py

from utils.sentiment_analysis import analyze_sentiment
import random
from database.database_handler import create_session, log_message

# -------------------------------------------------------------------
# Reply templates
# -------------------------------------------------------------------
positive_replies = [
    "That's wonderful to hear! Keep smiling!",
    "I'm happy for you! Keep spreading positivity!",
    "Positive vibes are contagious. Stay awesome!"
]

negative_replies = [
    "I'm sorry to hear that. Would you like a relaxation tip?",
    "It’s okay to feel this way. I'm here to listen.",
    "Sometimes talking about it can help. I'm here for you."
]

neutral_replies = [
    "I'm here if you need to talk about anything.",
    "Feel free to share whatever is on your mind.",
    "I'm ready to listen whenever you are ready."
]

# -------------------------------------------------------------------
# Session initialization
# -------------------------------------------------------------------
def initialize_session():
    """
    Creates a new conversation session in the database.
    Returns the session_id for subsequent logging.
    """
    return create_session()

# -------------------------------------------------------------------
# Main reply function
# -------------------------------------------------------------------
def get_reply(session_id: int, user_message: str) -> str:
    """
    Logs the user's message, generates a sentiment-based reply,
    logs the bot's reply, and returns it.
    """
    # 1) Log user message
    log_message(session_id, 'user', user_message)

    # 2) Determine sentiment and choose a reply
    mood = analyze_sentiment(user_message)
    if mood == "Positive":
        reply = random.choice(positive_replies)
    elif mood == "Negative":
        reply = random.choice(negative_replies)
    else:
        reply = random.choice(neutral_replies)

    # 3) Log bot reply
    log_message(session_id, 'bot', reply)
    return reply

# -------------------------------------------------------------------
# CLI entry-point for quick testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    sess = initialize_session()
    print("MindMate (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Take care! Goodbye.")
            break
        bot_resp = get_reply(sess, user_input)
        print(f"Bot: {bot_resp}")
