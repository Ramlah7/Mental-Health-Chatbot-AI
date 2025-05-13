import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QMovie
from gui.loasing_screen_testui import Ui_Form

class LoadingScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("MindMate")
        self.setFixedSize(797, 565)

        # ✅ Animate the chatbot icon
        self.bot_movie = QMovie("C:/Users/ramla/Downloads/Animation - 1746873587109.gif")
        self.ui.botIcon.setMovie(self.bot_movie)
        self.bot_movie.start()

        # ✅ Animate the small loading animation
        self.loading_movie = QMovie("C:/Users/ramla/Downloads/Animation - 1746882257108.gif")
        self.ui.animationLabel.setMovie(self.loading_movie)
        self.loading_movie.start()

        self.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LoadingScreen()
    sys.exit(app.exec_())
