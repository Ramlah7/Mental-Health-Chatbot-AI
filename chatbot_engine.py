from chatbot.generator import generate_response
from chatbot.intent import detect_intent
from chatbot.faq_query import query_faiss


class MindMateBot:
    def get_reply(self, user_text: str) -> str:
        try:
            print(f"🔍 Calling detect_intent() on: \"{user_text}\"")
            intent = detect_intent(user_text)
            print(f"✅ Intent detected: [{intent}]")

            print("🧠 Searching FAISS index...")
            reply = query_faiss(user_text)

            if "couldn’t find" in reply.lower() or "i'm here for you" in reply.lower():
                print("🔁 FAISS confidence too low. Using fallback intent-based response.")
                return generate_response(user_text, intent)

            print(f"✅ Bot reply received → {reply}")
            return reply

        except Exception as e:
            print(f"❌ Error in get_reply(): {e}")
            return "[MindMate error]"
