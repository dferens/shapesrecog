import sys
import os.path
from PyQt4 import QtGui, uic


RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'res')


class MainWindow(QtGui.QMainWindow):

   def __init__(self):
       super(MainWindow, self).__init__()
       uic.loadUi(os.path.join(RESOURCES_DIR, 'app.ui'), self)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    app.setActiveWindow(window)
    window.show()
    sys.exit(app.exec_())
