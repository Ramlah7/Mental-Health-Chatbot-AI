from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import torch
import pathlib

# Auto-resolve absolute model path
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "mindmate_dialo"

tokenizer = BlenderbotSmallTokenizer.from_pretrained(MODEL_PATH)
model = BlenderbotSmallForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def build_prompt(user_text: str, intent: str) -> str:
    """Generate a contextually appropriate prompt for the model based on intent."""
    if intent == "trivial_crisis":
        return f"You're MindMate, a chill and funny friend. Respond to this small crisis: '{user_text}'"
    elif intent == "sadness":
        return f"You're MindMate, a warm companion. Comfort this person who said: '{user_text}'"
    elif intent == "overwhelmed":
        return f"You're MindMate, a calm and supportive friend. Help someone who said: '{user_text}'"
    elif intent == "achievement":
        return f"You're MindMate, a best friend who celebrates wins. Respond to: '{user_text}'"
    elif intent == "rant":
        return f"You're MindMate, a safe space to vent. Someone just said: '{user_text}'"
    elif intent == "social_awkward":
        return f"You're MindMate, a humorous friend who gets awkwardness. React to: '{user_text}'"
    elif intent == "introvert":
        return f"You're MindMate, someone who understands introverts. Support this: '{user_text}'"
    elif intent == "extrovert":
        return f"You're MindMate, someone who thrives on energy. Share excitement for: '{user_text}'"
    elif intent == "gratitude":
        return f"You're MindMate, a kind and grateful soul. Acknowledge this: '{user_text}'"
    elif intent == "frustration":
        return f"You're MindMate. Someone is really frustrated. Reply gently to: '{user_text}'"
    elif intent == "confusion":
        return f"You're MindMate, a helpful explainer. Clarify or reassure this: '{user_text}'"
    elif intent == "curiosity":
        return f"You're MindMate, a thoughtful guide. Explore this question: '{user_text}'"
    else:
        return f"You are MindMate. Respond with kindness to: '{user_text}'"

def generate_response(user_text: str, intent: str = "neutral") -> str:
    try:
        print(f"\n🟦 User input: {user_text}")
        print(f"📌 Detected intent: [{intent}]")
        prompt = build_prompt(user_text, intent)
        print(f"🧠 Formatted prompt → {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            temperature=0.85,
            num_return_sequences=1,
            repetition_penalty=1.2
        )
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"💬 Bot reply: {reply}")
        return reply
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "[MindMate encountered an issue generating a reply.]"
