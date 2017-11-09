import numpy as np
from cnn.steps.basic import StepWithFilters
from cnn.steps.activation import Linear
from cnn.common import get_array


class OutputStep(StepWithFilters):
    def __init__(self, use_bias=True, activation=Linear, x0='random'):
        super().__init__(activation)

        self.x0 = x0
        self.activation = activation
        self.use_bias = use_bias

        self.classes = None

    def compile(self, X, y):
        self.classes = list(set(y))

    def back_prop(self, delta, leraning_rate):
        errors = np.zeros(len(self.inputs))

        for i, filter in enumerate(self.filters):
            error = delta[i]
            filter += np.dot(error, self.inputs.transpose()) * leraning_rate
            errors += error  # * filter

        return errors[1 if self.use_bias else 0:]

    def calc_neurons_values(self, input):
        inputs = np.copy(input)

        if self.use_bias:
            inputs = np.append([1], inputs)
        self.inputs = inputs

        if self.filters is None:
            self.filters = self.__initiliaze_weights(inputs)

        return np.array([inputs.dot(neuron_weights.T) for neuron_weights in self.filters])

    def __initiliaze_weights(self, flatten_input):
        return np.array([get_array(self.x0, flatten_input.size) for c in self.classes])
