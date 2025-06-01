from orchestrator.router import route

class MindMateBot:

    def __init__(self):
        self.history = []  # list of {"user": str, "bot": str}
        print("[MindMateBot] initialised – history empty")

    def reset(self):
        self.history.clear()
        print("[MindMateBot] history reset")

    def get_reply(self, user_input: str) -> str:
        print(f"[MindMateBot] USER → {user_input}")
        self.history.append({"user": user_input, "bot": ""})

        bot_reply = route(user_input)
        self.history[-1]["bot"] = bot_reply
        print(f"[MindMateBot] BOT  → {bot_reply}")
        return bot_reply
