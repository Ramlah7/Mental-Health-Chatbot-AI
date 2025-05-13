import sys
from PyQt5 import QtWidgets, QtCore
from gui.main_window_ui import Ui_MainWindow

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Setup the scrollable chat content dynamically
        self.chatContentWidget = QtWidgets.QWidget()
        self.messagesLayout = QtWidgets.QVBoxLayout(self.chatContentWidget)
        self.messagesLayout.setAlignment(QtCore.Qt.AlignTop)

        self.ui.scrollArea.setWidget(self.chatContentWidget)
        self.ui.scrollArea.setWidgetResizable(True)

        # Connect buttons
        self.ui.pushButton.clicked.connect(self.send_message)
        self.ui.lineEdit.returnPressed.connect(self.send_message)

    def send_message(self):
        text = self.ui.lineEdit.text().strip()
        if not text:
            return
        self.ui.lineEdit.clear()

        # Show user message
        self.display_message(text, is_user=True)

        # Simulate bot reply after delay
        QtCore.QTimer.singleShot(600, lambda: self.display_message("I'm here for you. Tell me more?", is_user=False))

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
        bar = self.ui.scrollArea.verticalScrollBar()
        bar.setValue(bar.maximum())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.setWindowTitle("MindMate")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
