import sys
import threading
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

        self.btLoad.clicked.connect(self.on_load_click)
        self.btSave.clicked.connect(self.on_save_click)
        self.btSelectImage.clicked.connect(self.on_select_image_click)
        self.btRunCrossValidation.clicked.connect(self.on_run_click)

        self.cbxLayer.addItems(core.LAYERS.keys())

        self.network = None

    @property
    def layer_type(self):
        return str(self.cbxLayer.currentText())

    @layer_type.setter
    def layer_type(self, value):
        index = self.cbxLayer.findText(value)
        self.cbxLayer.setCurrentIndex(index)

    @property
    def bias(self):
        return self.cbBias.isChecked()

    @property
    def trainer(self):
        if self.rbBackprop.isChecked():
            return 'backprop'
        else:
            return 'rprop'

    @property
    def layer_neurons(self):
        text = str(self.tbHiddenLayerNeurons.text()).strip()
        layer_neurons = map(int, text.split())
        return layer_neurons

    @layer_neurons.setter
    def layer_neurons(self, value):
        text = ' '.join(value)
        self.tbHiddenLayerNeurons.setText(text)

    @property
    def folds_count(self):
        return self.sbFold.value()

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value
        network_avaliable =  value is not None
        self.btSave.setEnabled(network_avaliable)
        self.btSelectImage.setEnabled(network_avaliable)

    def on_run_click(self):
        self.pbCrossValidation.setValue(0)
        self.lbCrossValidationCA.setText('computing...')
        self.btRunCrossValidation.setEnabled(False)

        def task():
            accuracies = []
            accuracy_gen = core.validate_network(
                hidden_layer_neurons=self.layer_neurons,
                layer=self.layer_type,
                bias=self.bias,
                folds_count=self.folds_count,
                trainer_class=self.trainer
            )
            for step, (accuracy, network) in enumerate(accuracy_gen, start=1):
                accuracies.append(accuracy)
                self.pbCrossValidation.setValue(100.0 / self.folds_count * step)
                self.network = network

            average_accuracy = sum(accuracies) / len(accuracies) * 100
            self.lbCrossValidationCA.setText('{0:.2f} %'.format(average_accuracy))
            self.btRunCrossValidation.setEnabled(True)

        threading.Thread(target=task).start()

    def on_select_image_click(self):
        file_path = QtGui.QFileDialog.getOpenFileName(self, 'Select image', filter='*.png')
        pixmap = QtGui.QPixmap(file_path)
        preview_pixmap = pixmap.scaled(self.lbSelectedImage.size(), QtCore.Qt.KeepAspectRatio)
        self.lbSelectedImage.setPixmap(preview_pixmap)
        predicted_class = core.classify(self.network, file_path)
        self.lbClass.setText(predicted_class)
        pass

    def on_load_click(self):
        self.network = core.import_network(self.NETWORK_FILE)
        hidden_layers = self.network.modulesSorted[1:-1]

        if len(hidden_layers) > 0:
            self.hidden_neurons = hidden_layers[0].indim
            self.layer_type = core.get_layer_id(type(hidden_layers[0]))

    def on_save_click(self):
        core.export_network(self.network, self.NETWORK_FILE)


def run():
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    app.setActiveWindow(window)
    window.show()
    sys.exit(app.exec_())
