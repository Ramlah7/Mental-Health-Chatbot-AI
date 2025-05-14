import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from gui.loasing_screen_testui import Ui_Form
from main_window import ChatWindow  # your actual chatbot window


class LoadingScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("MindMate - Loading...")
        self.setFixedSize(820, 500)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.center_window()
        self.start_animations()
        self.show()

        # ⏳ After 5 seconds, open main window
        QtCore.QTimer.singleShot(5000, self.open_main_window)

    def center_window(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )

    def start_animations(self):
        try:
            self.movie1 = QtGui.QMovie("C:/Users/ramla/Downloads/Animation - 1746873587109.gif")
            self.ui.botIcon.setMovie(self.movie1)
            self.movie1.start()

            self.movie2 = QtGui.QMovie("C:/Users/ramla/Downloads/Animation - 1746882257108.gif")
            self.ui.animationLabel.setMovie(self.movie2)
            self.movie2.start()
        except Exception as e:
            print(f"⚠️ Failed to load GIFs: {e}")

    def open_main_window(self):
        self.main_window = ChatWindow()
        self.main_window.show()
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    splash = LoadingScreen()
    sys.exit(app.exec_())
