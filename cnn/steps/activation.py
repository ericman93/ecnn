import numpy as np
import math
from cnn.steps.basic import BasicStep


class BasicActiviation(BasicStep):
    def forward_propagation(self, inputs):
        flatten = inputs.reshape(inputs.size)
        activated = [self.activation(value) for value in flatten]

        return np.array(activated).reshape(inputs.shape)

    def back_propagation(self, inputs):
        flatten = inputs.reshape(inputs.size)
        deactivated = [self.derivative(value) for value in flatten]

        return np.array(deactivated).reshape(inputs.shape)

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

class TanhActivation(BasicActiviation):
    def forward_propagation(self, inputs):
        return np.tanh(inputs)

    def back_propagation(self, inputs):
        return 1.0 - np.tanh(inputs)**2

class SigmoidActivation(BasicActiviation):
    def activation(self, value):
        return self.__sigmoid(value)

    def derivative(self, value):
        return self.__sigmoid(value) * (1 - self.__sigmoid(value))

    def __sigmoid(self, value):
        return 1.0 / (1 + math.exp(-value))


class SoftmaxActivation(BasicActiviation):
    def forward_propagation(self, inputs):
        return np.exp(inputs) / np.sum(np.exp(inputs), axis=0)


Sigmoid = SigmoidActivation()
Linear = LinearActivation()
Relu = ReluActivation()
Tanh = TanhActivation()
Softmax = SoftmaxActivation()