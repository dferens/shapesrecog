import os

from PyQt4.QtGui import QImage, qGray

from pybrain import LinearLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.xml import NetworkWriter, NetworkReader
from pybrain.supervised import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.shortcuts import buildNetwork

import settings

FIGURES = ('circle', 'triangle', 'rectangle')

IMG_SIZE = (64, 64)

TRAINERS = {
    'rprop': RPropMinusTrainer,
    'backprop': BackpropTrainer
}


def __get_input_vector(img_path):
    width, height = IMG_SIZE
    image = QImage(width, height, QImage.Format_RGB32)

    if image.load(img_path):
        vector = []
        get_value = lambda x, y: 1 - (qGray(image.pixel(x, y)) / 255.0)

        for y in range(height):
            for x in range(width):
                vector.append(get_value(x, y))

        return vector
    else:
        raise Exception('Can not load file, bad format: %s' % img_path)


def get_true_image_class(img_path):
    filename = os.path.basename(img_path)
    img_class =  filename.split('_')[0]
    assert img_class in FIGURES
    return img_class


def get_learn_dataset_images():
    """
    :rtype: [str]
    """
    dataset_dir = settings.DATASETS['learn']
    dataset_files = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)]
    image_files = [path for path in dataset_files if path.endswith('.png')]
    return image_files


def create_network(hidden_neurons=0):
    inputs_count = IMG_SIZE[0] * IMG_SIZE[1]
    outputs_count = len(FIGURES)

    if hidden_neurons == 0:
        args = (inputs_count, outputs_count)
    else:
        args = (inputs_count, hidden_neurons, outputs_count)

    kwargs = dict(hiddenclass=LinearLayer)
    network = buildNetwork(*args, **kwargs)
    return network


def learn_network(network, trainer='rprop'):
    assert trainer in TRAINERS

    inputs_count = IMG_SIZE[0] * IMG_SIZE[1]
    dataset = ClassificationDataSet(
        inputs_count,
        nb_classes=len(FIGURES),
        class_labels=FIGURES
    )
    for file_path in get_learn_dataset_images():
        input_vector = __get_input_vector(file_path)
        class_label = get_true_image_class(file_path)
        output_vector = [FIGURES.index(class_label)]
        dataset.appendLinked(input_vector, output_vector)

    dataset._convertToOneOfMany(bounds=[0, 1])
    TrainerClass = TRAINERS[trainer]
    trainer = TrainerClass(network, dataset=dataset, verbose=True)
    trainer.trainUntilConvergence(maxEpochs=200)


def classify(network, image_path):
    input_vector = __get_input_vector(image_path)
    output_vector = network.activate(input_vector)
    closest_vector = [abs(1 - abs(x)) for x in output_vector]
    closest_i = closest_vector.index(min(closest_vector))
    closest_class = FIGURES[closest_i]
    return closest_class


def export_network(network, file_path):
    NetworkWriter.writeToFile(network, file_path)


def import_network(file_path):
    return NetworkReader.readFrom(file_path)