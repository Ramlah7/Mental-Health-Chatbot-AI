import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from gui.main_window_ui import Ui_MainWindow

from chatbot.rule_based_chatbot import generate_bot_reply
from database.database_handler import (
    init_schema,
    create_session,
    log_message,
    fetch_sessions,
    fetch_messages,
    update_session_title
)


class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        print("🚀 Launching Application...")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("MindMate")
        self.setFixedSize(820, 500)

        QtCore.QTimer.singleShot(100, self.initialize_database_safely)

        if self.ui.widget.layout() is None:
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.ui.historyLabel)
            layout.addWidget(self.ui.historyList)
            self.ui.widget.setLayout(layout)

        self.chatContentWidget = QtWidgets.QWidget()
        self.messagesLayout = QtWidgets.QVBoxLayout(self.chatContentWidget)
        self.messagesLayout.setAlignment(QtCore.Qt.AlignTop)

        try:
            self.ui.scrollArea.takeWidget()
            self.ui.scrollArea.setWidget(self.chatContentWidget)
            self.ui.scrollArea.setWidgetResizable(True)
        except Exception as e:
            self.show_error("ScrollArea Error", str(e))

        self.ui.pushButton.clicked.connect(self.send_message)
        self.ui.lineEdit.returnPressed.connect(self.send_message)
        self.ui.historyList.itemClicked.connect(self.on_history_clicked)

    def initialize_database_safely(self):
        print("🛠 [DB] Initializing schema and session...")
        try:
            init_schema()
            self.current_session = create_session()
            print(f"✅ [DB] Session created: ID {self.current_session}")
            self.refresh_history_list()
        except Exception as e:
            self.show_error("Database Initialization Error", str(e))

    def send_message(self):
        text = self.ui.lineEdit.text().strip()
        if not text:
            return
        self.ui.lineEdit.clear()
        self.display_message(text, is_user=True)

        print(f"🧾 Logging to session {self.current_session}")
        try:
            log_message(self.current_session, 'user', text)

            # ✅ Update title if this session has no title yet
            sessions = fetch_sessions()
            for sess_id, title, _ in sessions:
                if sess_id == self.current_session and (title is None or title.strip() == ""):
                    update_session_title(self.current_session, text[:50])
                    print(f"✍️ [Title Set] Session {sess_id} → {text[:50]}")
                    self.refresh_history_list()
                    break

        except Exception as e:
            self.show_error("Logging User Message Failed", str(e))

        try:
            reply = generate_bot_reply(text)
        except Exception as e:
            reply = f"[Bot error: {e}]"

        self.display_message(reply, is_user=False)

        try:
            log_message(self.current_session, 'bot', reply)
        except Exception as e:
            self.show_error("Logging Bot Message Failed", str(e))

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

    def clear_chat_display(self):
        while self.messagesLayout.count():
            item = self.messagesLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                layout = item.layout()
                for i in reversed(range(layout.count())):
                    subitem = layout.takeAt(i)
                    if subitem.widget():
                        subitem.widget().deleteLater()
                layout.deleteLater()

    def refresh_history_list(self):
        print("📜 [History] Fetching sessions...")
        self.ui.historyList.clear()
        try:
            db_sessions = fetch_sessions()
        except Exception as e:
            self.show_error("Fetching Sessions Failed", str(e))
            return

        for sess_id, title, ts in db_sessions:
            label = title if title else ts.strftime("%Y-%m-%d %H:%M:%S")
            print(f"📌 Session {sess_id} → Label: {label}")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, sess_id)
            self.ui.historyList.addItem(item)
        print(f"✅ [History] {len(db_sessions)} session(s) loaded.")

    def on_history_clicked(self, item: QtWidgets.QListWidgetItem):
        sess_id = item.data(QtCore.Qt.UserRole)
        print(f"🖱️ [History Clicked] Selected session ID: {sess_id}")

        self.clear_chat_display()
        try:
            messages = fetch_messages(sess_id)
            print(f"📥 [DB] Retrieved {len(messages)} messages for session {sess_id}")
            for sender, content in messages:
                self.display_message(content, is_user=(sender == 'user'))
            self.current_session = sess_id
            print(f"✅ [GUI] Displayed all messages from session {sess_id}")
        except Exception as e:
            self.show_error("Loading Session Messages Failed", str(e))

    def show_error(self, title, message):
        print(f"❌ [{title}] {message}")
        QMessageBox.critical(self, title, message)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
