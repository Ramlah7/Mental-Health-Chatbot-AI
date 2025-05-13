from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import QTimer, QPointF, Qt
from PyQt5.QtGui import QMovie, QColor, QPen
from gui.loading_screen_ui import Ui_loadingScreen
import sys
import random

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_loadingScreen()
        self.ui.setupUi(self)

        # ✅ Animated chatbot GIF
        self.chatbot_movie = QMovie("C:/Users/ramla/Downloads/Animation - 1746873587109.gif")
        self.ui.chatbotanimation.setMovie(self.chatbot_movie)
        self.chatbot_movie.start()

        # ✅ Animated loading GIF
        self.loading_movie = QMovie("C:/Users/ramla/Downloads/Animation - 1746874294705.gif")
        self.ui.loadinganimation.setMovie(self.loading_movie)
        self.loading_movie.start()

        # ✅ Loading text animation (if you use a label called loadingLabel)
        self.dot_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loading_text)
        self.timer.start(500)

        # ✅ Soft pastel floating sparkles
        self.setup_particles()

    def update_loading_text(self):
        if hasattr(self.ui, 'loadingLabel'):
            dots = '.' * ((self.dot_count % 3) + 1)
            self.ui.loadingLabel.setText(f"Loading{dots}")
            self.dot_count += 1

    def setup_particles(self):
        self.scene = QGraphicsScene()
        self.ui.particleview.setScene(self.scene)
        self.ui.particleview.setStyleSheet(
            "background: rgba(0, 255, 0, 30);")  # GREEN to check if it's rendering at all
        self.ui.particleview.setHorizontalScrollBarPolicy(1)  # ScrollBarAlwaysOff
        self.ui.particleview.setVerticalScrollBarPolicy(1)

        self.particles = []

        for _ in range(30):
            size = 10
            particle = QGraphicsEllipseItem(0, 0, size, size)
            particle.setBrush(QColor(255, 255, 255, 255))  # FULL white for visibility
            particle.setPen(QPen(Qt.NoPen))
            x = random.randint(0, self.width())
            y = random.randint(0, self.height())
            particle.setPos(x, y)
            self.scene.addItem(particle)
            self.particles.append(particle)

        self.particle_timer = QTimer()
        self.particle_timer.timeout.connect(self.animate_particles)
        self.particle_timer.start(50)

    def animate_particles(self):
        for p in self.particles:
            x, y = p.pos().x(), p.pos().y() - 1
            if y < -10:
                y = random.randint(int(self.height() * 0.6), self.height())
                x = random.randint(0, self.width())
            p.setPos(QPointF(x, y))

# 🚀 Run the window
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoadingScreen()
    window.show()
    sys.exit(app.exec_())


