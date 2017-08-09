import numpy as np
from cnn.steps.basic import BasicStep
from cnn.common import get_array


class OutputStep(BasicStep):
    def __init__(self, activation, x0='random'):
        self.x0 = x0

        self.weights = None
        self.classes = None

    def prepare(self, X, y):
        self.classes = list(set(y))

    def forward_propagation(self, input):
        flatten_input = input.reshape(input.size)
        # bias
        flatten_input = np.append([1], flatten_input)

        if self.weights is None:
            self.weights = self.__initiliaze_weights(flatten_input)

        return np.array([flatten_input.dot(neuron_weights.T) for neuron_weights in self.weights])

    def __initiliaze_weights(self, flatten_input):
        # TODO: not use only ones - use what ever X0 is assign for
        return np.array([get_array(self.x0, flatten_input.size) for c in self.classes])
