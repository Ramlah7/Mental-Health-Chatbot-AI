# chatbot/rule_based_chatbot.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.sentiment_analysis import analyze_sentiment
import torch
import traceback
import time

print("🧠 [Init] Starting MindMate (GODEL + Sentiment Hybrid)...")

# === Load GODEL model and tokenizer ===
try:
    print("📦 [Step 1] Loading model and tokenizer from HuggingFace...")
    model_name = "microsoft/GODEL-v1_1-base-seq2seq"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ [Step 1 Complete] GODEL model loaded successfully.")
except Exception as e:
    print("❌ [Error] Failed to load model/tokenizer.")
    traceback.print_exc()
    raise

# === Main bot reply function ===
def generate_bot_reply(user_input):
    print("\n====================")
    print(f"💬 [User Input] {user_input}")
    print("====================")

    try:
        if not user_input.strip():
            print("⚠️ [Warning] Empty input received.")
            return "I'm here when you're ready to talk."

        # Step 2: Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        print(f"🔍 [Sentiment] Detected: {sentiment}")

        # Step 3: Select instruction based on sentiment
        if sentiment == "Positive":
            instruction = "Instruction: respond joyfully. "
        elif sentiment == "Negative":
            instruction = "Instruction: respond empathetically. "
        else:
            instruction = "Instruction: respond supportively. "

        # Step 4: Format prompt
        full_input = f"{instruction}Context: User: {user_input}"
        print(f"🧾 [Formatted Input] {full_input}")

        # Step 5: Tokenize
        inputs = tokenizer(full_input, return_tensors="pt", padding=True)
        print("🧪 [Tokenization] Done.")

        # Step 6: Generate response
        print("⚙️ [Generation] Generating reply...")
        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_length=60,
            num_beams=4,
            top_p=0.9,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start
        print(f"⏱️ [Time Taken] {elapsed:.2f} seconds")

        # Step 7: Decode + check reply
        reply = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"🤖 [Raw Bot Reply] {reply}")

        # Step 8: Fallback for echo or empty
        if (
            reply.lower().startswith("no answer") or
            len(reply.split()) < 2 or
            reply.strip().lower() == user_input.strip().lower()
        ):
            print("⚠️ [Fallback Triggered] Low-quality or echo reply detected.")
            if sentiment == "Positive":
                reply = "That's amazing! You should be proud — congratulations 🎉"
            elif sentiment == "Negative":
                reply = "I'm here for you. Want to talk more about what’s bothering you?"
            else:
                reply = "Tell me more — I’m listening."

        return reply

    except Exception as e:
        print("❌ [Generation Error] Failed during reply generation.")
        traceback.print_exc()
        return "[Bot error: Unable to process right now.]"

# === CLI Debug Mode ===
if __name__ == "__main__":
    print("\n🧠 [MindMate Ready] GODEL + VADER chatbot is live.")
    print("💬 Type your message (or 'exit' to quit):\n")

    while True:
        try:
            text = input("You: ")
            if text.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            response = generate_bot_reply(text)
            print(f"Bot: {response}")
        except Exception as e:
            print("❌ [Fatal CLI Error] Unexpected error in chat loop:")
            traceback.print_exc()
