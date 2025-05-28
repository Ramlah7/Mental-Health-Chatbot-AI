import openai
import os
from dotenv import load_dotenv
from utils.sentiment_analysis import analyze_sentiment

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt templates by sentiment
PROMPT_MAP = {
    "Positive": "You are a cheerful and supportive mental health assistant. Respond with kindness and encouragement.",
    "Negative": "You are an empathetic and calming mental health assistant. Comfort the user in a caring and non-judgmental way.",
    "Neutral":  "You are a balanced and helpful mental health assistant. Offer guidance and ask thoughtful questions."
}

def generate_openai_reply(user_input):
    """Generate a sentiment-aware reply using OpenAI GPT."""
    try:
        if not user_input.strip():
            return "I'm here when you're ready to talk."

        # Step 1: Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        print(f"[Sentiment] Detected: {sentiment}")

        # Step 2: Compose prompt
        system_instruction = PROMPT_MAP.get(sentiment, PROMPT_MAP["Neutral"])
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ]

        # Step 3: Call OpenAI API (v1.x syntax)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=120,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.6
        )

        # Step 4: Return reply
        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return "I'm here to support you, even if I couldn't find the right words just now."

# Debugging CLI
if __name__ == "__main__":
    while True:
        text = input("You: ")
        if text.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        print("Bot:", generate_openai_reply(text))
