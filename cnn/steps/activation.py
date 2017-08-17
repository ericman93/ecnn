import numpy as np
import math
from cnn.steps.basic import BasicStep


class BasicActiviation(BasicStep):
    def forward_propagation(self, input):
        flatten = input.reshape(input.size)
        activated = [self.activation(value) for value in flatten]

        return np.array(activated).reshape(input.shape)

    def activation(self, value):
        raise

    def derivative(self, input):
        raise


class LinearActivation(BasicActiviation):
    def activation(self, value):
        return value

    def derivative(self, value):
        return 1


class ReluActivation(BasicActiviation):
    def activation(self, value):
        return max(0, value)

    def derivative(self, value):
        return 0 if value < 0 else 1


class SigmoidActivation(BasicActiviation):
    def activation(self, value):
        return self.__sigmoid(value)

    def derivative(self, value):
        return self.__sigmoid(value) * (1 - self.__sigmoid(value))

    def __sigmoid(self, value):
        return 1.0 / (1 + math.exp(-value))


Sigmoid = SigmoidActivation()
Linear = LinearActivation()
Relu = ReluActivation()
