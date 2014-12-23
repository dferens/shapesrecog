import sys
import os.path
from PyQt4 import QtGui, QtCore, uic

from . import core, settings


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


class MainWindow(QtGui.QMainWindow):

    NETWORK_FILE = os.path.join(settings.ROOT_DIR, 'state.net')

    def __init__(self):
        super(MainWindow, self).__init__()
        ui_file = os.path.join(settings.RESOURCES_DIR, 'app.ui')
        uic.loadUi(ui_file, self)
        self.__connect_signals()

    @property
    def hidden_neurons(self):
        return self.spHiddenNeurons.value()

    @property
    def trainer(self):
        if self.rbBackprop.isChecked():
            return 'backprop'
        else:
            return 'rprop'

    def __connect_signals(self):
        self.btSelectImage.clicked.connect(self.on_select_image_click)
        self.btCreate.clicked.connect(self.on_create_click)
        self.btRelearn.clicked.connect(self.on_relearn_click)
        self.btLoad.clicked.connect(self.on_load_click)
        self.btSave.clicked.connect(self.on_save_click)

    def on_select_image_click(self):
        file_path = QtGui.QFileDialog.getOpenFileName(self, 'Select image', filter='*.png')
        self.lbSelectedImage.setPixmap(QtGui.QPixmap(file_path))
        predicted_class = core.classify(self.network, file_path)
        self.lbClass.setText(predicted_class)

    def on_create_click(self):
        self.network = core.create_network(self.hidden_neurons)

    def on_relearn_click(self):
        core.learn_network(self.network, trainer=self.trainer)
        images_files = core.get_learn_dataset_images()
        predicted_count = 0

        for file_path in images_files:
            true_class = core.get_true_image_class(file_path)

            if true_class ==  core.classify(self.network, file_path):
                predicted_count += 1

        predicted_percents = float(predicted_count) / len(images_files) * 100
        self.lbTrainCA.setText('{0:.2f} % ({1}/{2})'.format(
            predicted_percents, predicted_count, len(images_files)
        ))

    def on_load_click(self):
        self.network = core.import_network(self.NETWORK_FILE)

    def on_save_click(self):
        core.export_network(self.network, self.NETWORK_FILE)


def run():
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    app.setActiveWindow(window)
    window.show()
    sys.exit(app.exec_())
