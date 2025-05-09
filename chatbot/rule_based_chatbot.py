# rule_based_chatbot.py
from utils.sentiment_analysis import analyze_sentiment
import random

# Define different types of replies
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


def generate_bot_reply(user_message):
    """Generates a bot reply based on user's sentiment."""
    mood = analyze_sentiment(user_message)

    if mood == "Positive":
        return random.choice(positive_replies)
    elif mood == "Negative":
        return random.choice(negative_replies)
    else:
        return random.choice(neutral_replies)


# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Take care! Goodbye.")
            break
        bot_reply = generate_bot_reply(user_input)
        print(f"Bot: {bot_reply}")
