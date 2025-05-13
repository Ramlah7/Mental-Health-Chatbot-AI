import sys
from PyQt5 import QtWidgets, QtCore
from gui.main_window_ui import Ui_MainWindow
from chatbot.rule_based_chatbot import generate_bot_reply  # using raw reply only, no DB

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("MindMate (Chatbot Test Only)")
        self.setFixedSize(820, 500)

        # LEFT panel (history)
        if self.ui.widget.layout() is None:
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.ui.historyLabel)
            layout.addWidget(self.ui.historyList)
            self.ui.widget.setLayout(layout)

        # RIGHT panel (scrollable chat area)
        self.chatContentWidget = QtWidgets.QWidget()
        self.messagesLayout = QtWidgets.QVBoxLayout(self.chatContentWidget)
        self.messagesLayout.setAlignment(QtCore.Qt.AlignTop)

        self.ui.scrollArea.takeWidget()
        self.ui.scrollArea.setWidget(self.chatContentWidget)
        self.ui.scrollArea.setWidgetResizable(True)

        # Connect input to chatbot
        self.ui.pushButton.clicked.connect(self.send_message)
        self.ui.lineEdit.returnPressed.connect(self.send_message)

    def send_message(self):
        text = self.ui.lineEdit.text().strip()
        if not text:
            return
        self.ui.lineEdit.clear()
        self.display_message(text, is_user=True)

        try:
            reply = generate_bot_reply(text)  # 🔁 Only sentiment logic, no DB
            self.display_message(reply, is_user=False)
        except Exception as e:
            self.display_message(f"[ERROR]: {e}", is_user=False)

    def display_message(self, text, is_user=True):
        bubble = QtWidgets.QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(400)
        bubble.setStyleSheet(f"""
            QLabel {{
                background-color: {'#cce5ff' if is_user else '#f0f0f0'};
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
                color: #2e2e2e;
            }}
        """)

        hbox = QtWidgets.QHBoxLayout()
        if is_user:
            hbox.addStretch()
            hbox.addWidget(bubble)
        else:
            hbox.addWidget(bubble)
            hbox.addStretch()

        self.messagesLayout.addLayout(hbox)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        self.ui.scrollArea.verticalScrollBar().setValue(
            self.ui.scrollArea.verticalScrollBar().maximum()
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
