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

    def back_prop(self, delta, leraning_rate=0.001):
        errors = []
        for i in range(self.filters.shape[1]):
            # filter += delta[i]
            neuron_weights = self.filters[:, i]
            neuron_deltas = (delta * self.activation.back_propagation(self.z)).dot(leraning_rate)
            self.filters[:, i] += neuron_deltas

            neuron_error = np.sum(neuron_deltas)

            errors.append(neuron_error)

        return errors[1 if self.use_bias else 0:]

    def calc_neurons_values(self, input):
        flatten_input = input.reshape(input.size)
        # bias
        flatten_input = np.append([1], flatten_input)

        if self.filters is None:
            self.filters = self.__initiliaze_weights(flatten_input)

        output = np.array([flatten_input.dot(neuron_weights.T) for neuron_weights in self.filters])
        return self.activation.forward_propagation(output)

    def __initiliaze_weights(self, flatten_input):
        # TODO: not use only ones - use what ever X0 is assign for
        return np.array([get_array(self.x0, flatten_input.size) for c in self.classes])
