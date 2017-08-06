import numpy as np
from cnn.steps.basic import BasicStep


class ClassificationOutputStep(BasicStep):
    def __init__(self, x0='random'):
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

        return [flatten_input.dot(neuron_weights.T) for neuron_weights in self.weights]

    def __initiliaze_weights(self, flatten_input):
        # TODO: not use only ones - use what ever X0 is assign for
        return np.array([np.ones(flatten_input.size) for c in self.classes])
