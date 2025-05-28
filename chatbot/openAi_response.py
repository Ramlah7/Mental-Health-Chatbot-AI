import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils.sentiment_analysis import analyze_sentiment

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if not GEMINI_API_KEY:
    raise ValueError("[Gemini API] API key not found in .env. Please set GEMINI_API_KEY.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Prompt templates by sentiment
PROMPT_MAP = {
    "Positive": "You are a cheerful and supportive mental health assistant. Respond with kindness and encouragement.",
    "Negative": "You are an empathetic and calming mental health assistant. Comfort the user in a caring and non-judgmental way.",
    "Neutral":  "You are a balanced and helpful mental health assistant. Offer guidance and ask thoughtful questions."
}

def generate_gemini_reply(user_input):
    """Generate a sentiment-aware reply using Google Gemini."""
    try:
        if not user_input.strip():
            return "I'm here when you're ready to talk."

        # Step 1: Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        print(f"[Sentiment] Detected: {sentiment}")

        # Step 2: Compose prompt
        system_instruction = PROMPT_MAP.get(sentiment, PROMPT_MAP["Neutral"])
        prompt = f"{system_instruction}\nUser: {user_input}\nAssistant:"

        # Step 3: Call Gemini API
        response = model.generate_content(prompt)

        # Step 4: Return reply
        reply = response.text.strip()
        return reply

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "I'm here to support you, even if I couldn't find the right words just now."

# Debugging CLI
if __name__ == "__main__":
    while True:
        text = input("You: ")
        if text.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        print("Bot:", generate_gemini_reply(text))
