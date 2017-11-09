import os
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
from cnn.cost_functions import CrossEntropy

cards_dir = '.'
categories = {'trainer': 0, 'pokemon': 1, 'energy': 2}
image_size = (150, 110)


def get_image_matrix(image_path, size, graysacle=False):
    image = Image.open(image_path)

    revese_size = (size[1], size[0])
    image.thumbnail(revese_size, Image.ANTIALIAS)

    if graysacle:
        return np.asarray(image.convert("L"))
    else:
        result = np.zeros((3,) + size)
        matrix = np.asarray(image)
        red = matrix[:, :, 0]
        green = matrix[:, :, 1]
        blue = matrix[:, :, 2]

        merged = np.stack([red, green, blue], axis=0)

        result[:merged.shape[0], :merged.shape[1], :merged.shape[2]] = merged
        return result.dot(1 / 256)


def get_cards(X, y):
    for category, tag in categories.items():
        print(category)
        file_names = os.listdir(os.path.join(cards_dir, category))
        category_x = []
        category_y = []

        for index, image_path in enumerate(file_names):
            image_path = os.path.join(cards_dir, category, image_path)

            image_matrix = get_image_matrix(image_path, image_size)

            category_x.append(image_matrix)
            category_y.append(tag)

        X += category_x
        y += category_y


def fit():
    X = []
    y = []

    get_cards(X, y)

    filter_size = (6, 6)
    padding = 0
    stride = 1
    x0 = 'random'

    steps = [
        ConvolutionalStep(filter_size=filter_size, padding=padding, stride=stride, num_of_kernels=4, x0=x0,
                          activation=Relu),
        MaxPoolingStep(3),
        ConvolutionalStep(filter_size=filter_size, padding=padding, stride=stride, num_of_kernels=7, x0=x0,
                          activation=Relu),
        MaxPoolingStep(3),
        Flatten(),
        OutputStep(x0=x0, activation=Softmax)
    ]

    network = CnnNetwork(steps)

    network.fit(
        X, y,
        CrossEntropy,
        iterations=100,
        batch_size=len(X),
        learning_rate=1e-5,
        verbose=1,

    )


if __name__ == '__main__':
    fit()
