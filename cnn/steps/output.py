import numpy as np
from cnn.steps.basic import BasicStep
from cnn.steps.activation import Linear
from cnn.common import get_array


class OutputStep(BasicStep):
    def __init__(self, activation=Linear, x0='random'):
        self.x0 = x0
        self.activation = activation

        self.weights = None
        self.classes = None

    def compile(self, X, y):
        self.classes = list(set(y))

    def forward_propagation(self, input):
        flatten_input = input.reshape(input.size)
        # bias
        flatten_input = np.append([1], flatten_input)

        if self.weights is None:
            self.weights = self.__initiliaze_weights(flatten_input)

        output = np.array([flatten_input.dot(neuron_weights.T) for neuron_weights in self.weights])
        return self.activation.forward_propagation(output)

    def update_weights(self, delta):
        raise Error("Implement")

    def __initiliaze_weights(self, flatten_input):
        # TODO: not use only ones - use what ever X0 is assign for
        return np.array([get_array(self.x0, flatten_input.size) for c in self.classes])
