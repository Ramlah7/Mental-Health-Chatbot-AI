import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from gui.loading_window_ui1 import Ui_Form
from main_window import ChatWindow


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

        # Launch ChatWindow after 3 seconds
        QtCore.QTimer.singleShot(10000, self.open_main_window)

    def center_window(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def start_animations(self):
        try:
            gif1_path = "Animation - 1.gif"
            gif2_path = "Animation - 2.gif"

            self.movie1 = QtGui.QMovie(gif1_path)
            self.ui.botIcon.setMovie(self.movie1)
            self.movie1.start()

            self.movie2 = QtGui.QMovie(gif2_path)
            self.ui.animationLabel.setMovie(self.movie2)
            self.movie2.start()
        except Exception as e:
            print(f"⚠️ Failed to load GIFs: {e}")

    def open_main_window(self):
        self.main_window = ChatWindow()
        self.main_window.show()

        # 👇 These 3 lines ensure it appears on screen and is focused
        self.main_window.raise_()                 # Bring to front
        self.main_window.activateWindow()         # Request focus
        self.main_window.setWindowState(QtCore.Qt.WindowActive)

        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    splash = LoadingScreen()
    sys.exit(app.exec_())
