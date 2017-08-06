import numpy as np
from cnn.steps.basic import BasicStep


class BasicActiviation(BasicStep):
    def forward_propagation(self, input):
        flatten = input.reshape(input.size)
        activated = [self.activation(value) for value in flatten]

        return np.array(activated).reshape(input.shape)

    def activation(self, value):
        raise

    def activation_derivative(self, input):
        raise


class ReluActivation(BasicActiviation):
    def activation(self, value):
        return max(0, value)


class SigmoidActivation(BasicActiviation):
    def activation(self, value):
        return 1 / (1 + math.exp(-value))
