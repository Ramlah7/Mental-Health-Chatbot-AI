import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# List all models
models = genai.list_models()

print("📋 Available Gemini Models:\n")
for model in models:
    print(f"- {model.name}  (Supports generate_content: )")
