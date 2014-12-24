import os
from itertools import chain, islice

from PyQt4.QtGui import QImage, qGray

from numpy import array_split, concatenate
from pybrain import structure
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

LAYERS = {
    'softmax': structure.SoftmaxLayer,
    'sigmoid': structure.SigmoidLayer,
    'linear': structure.LinearLayer,
    'tanh': structure.TanhLayer,
    'gaussian': structure.GaussianLayer
}

def get_layer_id(LayerClass):
    for key in LAYERS:
        if LayerClass == LAYERS[key]:
            return key

    raise KeyError('Layer class "%s" is not registered' % LayerClass.__name__)


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


def get_dataset_files():
    """
    :rtype: [str]
    """
    dataset_dir = settings.DATASET
    dataset_files = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)]
    image_files = [path for path in dataset_files if path.endswith('.png')]
    return image_files


def classify(network, image_path):
    input_vector = __get_input_vector(image_path)
    output_vector = network.activate(input_vector)
    closest_i = output_vector.argmax(axis=0)
    closest_class = FIGURES[closest_i]
    return closest_class


def validate_network(hidden_layer_neurons=(), layer='linear', bias=True,
                     folds_count=5, trainer_class='rprop'):
    assert layer in LAYERS
    assert trainer_class in TRAINERS

    inputs_count = IMG_SIZE[0] * IMG_SIZE[1]
    outputs_count = len(FIGURES)

    def network_builder():
        network_args = chain((inputs_count,), hidden_layer_neurons, (outputs_count,))
        return buildNetwork(
            *network_args,
            hiddenclass=LAYERS[layer],
            bias=bias,
            outclass=LAYERS['softmax']
        )

    def dataset_builder():
        return ClassificationDataSet(
            inputs_count,
            nb_classes=len(FIGURES),
            class_labels=FIGURES
        )

    data_vectors = []
    for file_path in get_dataset_files():
        input_vector = __get_input_vector(file_path)
        class_label = get_true_image_class(file_path)
        class_i = [FIGURES.index(class_label)]
        data_vectors.append((input_vector, class_i))


    chunk_size = len(data_vectors) / folds_count

    for i in range(folds_count):
        #
        # Step 1: prepare train & test datasets
        #
        train_vectors = []
        test_vectors = []
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size

        for vector_i, vectors in enumerate(data_vectors):
            if chunk_start <= vector_i <= chunk_end:
                test_vectors.append(vectors)
            else:
                train_vectors.append(vectors)

        train_dataset = dataset_builder()
        test_dataset = dataset_builder()

        for dataset, vectors in zip((train_dataset, test_dataset),
                                    (train_vectors, test_vectors)):
            for input_vector, class_i in vectors:
                dataset.appendLinked(input_vector, class_i)

            dataset._convertToOneOfMany(bounds=[0, 1])

        #
        # Step 2: train & test
        #
        network = network_builder()
        trainer = TRAINERS[trainer_class](network, dataset=train_dataset)
        trainer.train()

        predicted_count = 0
        for input_vector, class_i in test_vectors:
            predicted_vector = network.activate(input_vector)
            predicted_class_i = predicted_vector.argmax()

            if class_i == predicted_class_i:
                predicted_count += 1

        current_accuracy = float(predicted_count) / len(test_vectors)
        yield (current_accuracy, network)


def export_network(network, file_path):
    NetworkWriter.writeToFile(network, file_path)


def import_network(file_path):
    return NetworkReader.readFrom(file_path)