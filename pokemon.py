import os
import pickle
import numpy as np
from PIL import Image
from cnn.network import CnnNetwork
from cnn.steps import ConvolutionalStep
from cnn.steps import ReluActivation
from cnn.steps import MaxPoolingStep
from cnn.steps import OutputStep
from cnn.steps import Flatten
from cnn.network import CnnNetwork
from cnn.steps.activation import Sigmoid
from cnn.steps.activation import Relu
from cnn.steps.activation import Tanh
from cnn.steps.activation import Softmax
from cnn.cost_functions import MeanSquared
from cnn.cost_functions import CrossEntropyLogisticRegression


categories = {'trainer': 0, 'pokemon': 1, 'energy': 2}


def get_image_matrix(image_path, size, graysacle=False):
    image = Image.open(image_path)
    image.thumbnail((size, size), Image.ANTIALIAS)

    if graysacle:
        return np.asarray(image.convert("L"))
    else:
        result = np.zeros((3, size, size))
        matrix = np.asarray(image)
        red = matrix[:, :, 0]
        green = matrix[:, :, 1]
        blue = matrix[:, :, 2]

        # Image.fromarray(red).save('red.bmp')
        # Image.fromarray(green).save('green.bmp')
        # Image.fromarray(blue).save('blue.bmp')

        merged = np.stack([red, green, blue], axis=0)

        result[:merged.shape[0], :merged.shape[1], :merged.shape[2]] = merged
        return result.dot(1/256)

        # return merged.dot(1/256)


def get_cards(X, y):
    cards_dir = '/home/eric/dev/pokemon/cards/'

    for category, tag in categories.items():
        file_names = os.listdir(os.path.join(cards_dir, category))

        for index, image_path in enumerate(file_names[0:100]):
            image_path = os.path.join(cards_dir, category, image_path)

            image_matrix = get_image_matrix(image_path, 100)

            X.append(image_matrix)
            y.append(tag)

def validate():
    network = CnnNetwork()
    network.load('network.b')

    image_matrix = get_image_matrix('validation/charizard.png', 100)

def fit():
    X = []
    y = []
    get_cards(X, y)

    steps = [
        ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='random', activation=Relu),
        MaxPoolingStep(3),
        ConvolutionalStep(filter_size=(3, 3), num_of_kernels=5, x0='random', activation=Relu),
        MaxPoolingStep(3),
        ConvolutionalStep(filter_size=(3, 3), num_of_kernels=2, x0='random', activation=Relu),
        MaxPoolingStep(3),
        # ConvolutionalStep(filter_size=(3, 3), num_of_kernels=6, x0='random', activation=Relu),
        # MaxPoolingStep(3),
        # ConvolutionalStep(filter_size=(3, 3), num_of_kernels=5, x0='random', activation=Relu),
        # MaxPoolingStep(3),
        Flatten(),
        OutputStep(x0='random', activation=Softmax)
    ]

    network = CnnNetwork(steps)
    # network.fit(X, y, MeanSquared, iterations=1000, batch_size=32)

    # network.load('network.b')
    # network.fit([X[2]], [y[2]], MeanSquared, iterations=1000, batch_size=32, learning_rate=0.1)
    network.fit(X, y, MeanSquared, iterations=10, batch_size=len(X), learning_rate=0.001)
    # network.save('network.b')

fit()

