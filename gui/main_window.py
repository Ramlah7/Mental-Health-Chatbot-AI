# gui/main_window.py

import sys
from PyQt5 import QtWidgets, QtCore
from gui.main_window_ui import Ui_MainWindow

from database.database_handler import (
    init_schema,
    create_session,
    fetch_sessions,
    fetch_messages
)
from chatbot.rule_based_chatbot import get_reply

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 1) UI setup
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("MindMate")
        self.setFixedSize(820, 500)

        # 2) Database schema + new session
        init_schema()
        self.current_session = create_session()

        # 3) Reattach / style the History panel
        if self.ui.historyLabel and self.ui.historyList:
            history_layout = QtWidgets.QVBoxLayout()
            history_layout.setContentsMargins(10, 10, 10, 10)
            history_layout.setSpacing(8)
            history_layout.addWidget(self.ui.historyLabel)
            history_layout.addWidget(self.ui.historyList, stretch=1)
            self.ui.widget.setLayout(history_layout)

        # 4) Prepare chat scroll area
        self.chatContentWidget = QtWidgets.QWidget()
        self.messagesLayout = QtWidgets.QVBoxLayout(self.chatContentWidget)
        self.messagesLayout.setAlignment(QtCore.Qt.AlignTop)
        self.ui.scrollArea.setWidget(self.chatContentWidget)
        self.ui.scrollArea.setWidgetResizable(True)

        # 5) Populate history & connect signals
        self.refresh_history_list()
        self.ui.pushButton.clicked.connect(self.send_message)
        self.ui.lineEdit.returnPressed.connect(self.send_message)
        self.ui.historyList.itemClicked.connect(self.on_history_clicked)

    def send_message(self):
        text = self.ui.lineEdit.text().strip()
        if not text:
            return
        # Clear input, display user bubble
        self.ui.lineEdit.clear()
        self.display_message(text, is_user=True)

        # Get, log & display bot reply
        reply = get_reply(self.current_session, text)
        self.display_message(reply, is_user=False)

    def display_message(self, text: str, is_user: bool = True):
        """
        Add a single chat bubble to the scroll area.
        """
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
        bar = self.ui.scrollArea.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_chat_display(self):
        """
        Remove all existing bubbles/layouts from the chat area.
        """
        while self.messagesLayout.count():
            item = self.messagesLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # clear nested layout
                sublayout = item.layout()
                for i in reversed(range(sublayout.count())):
                    subitem = sublayout.takeAt(i)
                    if subitem.widget():
                        subitem.widget().deleteLater()
                sublayout.deleteLater()

    def refresh_history_list(self):
        """
        Load the list of past sessions into the left panel.
        """
        self.ui.historyList.clear()
        for sess_id, ts in fetch_sessions():
            label = ts.strftime("%Y-%m-%d %H:%M:%S")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, sess_id)
            self.ui.historyList.addItem(item)

    def on_history_clicked(self, item: QtWidgets.QListWidgetItem):
        """
        When a session is clicked, clear the chat pane and reload its transcript.
        """
        sess_id = item.data(QtCore.Qt.UserRole)
        self.clear_chat_display()
        for sender, content in fetch_messages(sess_id):
            self.display_message(content, is_user=(sender == 'user'))
        self.current_session = sess_id

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
