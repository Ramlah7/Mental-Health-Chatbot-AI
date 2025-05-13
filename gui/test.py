import sys
from PyQt5 import QtWidgets
from gui.main_window_ui import Ui_MainWindow

class MinimalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("🚑 Minimal Crash Test")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MinimalWindow()
    window.show()
    sys.exit(app.exec_())
