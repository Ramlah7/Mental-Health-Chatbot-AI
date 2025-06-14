# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


# chatbot_engine.py
from orchestrator.router import respond

class MindMateBot:
    def __init__(self):
        self.history: list[dict[str,str]] = []

    def reset(self) -> None:
        self.history.clear()

    def get_reply(self, user_input: str) -> str:
        self.history.append({"user": user_input, "bot": ""})
        bot_reply = respond(self.history)
        self.history[-1]["bot"] = bot_reply
        return bot_reply