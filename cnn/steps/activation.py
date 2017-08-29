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
    # def forward_propagation(self, inputs):
    #     inputs[inputs<0: 0]

    def activation(self, value):
        return max(0, value)

    def derivative(self, value):
        return 0 if value < 0 else 1


class TanhActivation(BasicActiviation):
    def forward_propagation(self, inputs):
        return np.tanh(inputs)

    def back_propagation(self, inputs):
        return 1.0 - np.tanh(inputs) ** 2


class SigmoidActivation(BasicActiviation):
    def forward_propagation(self, inputs):
        return self.__sigmoid(inputs)

    def derivative(self, inputs):
        return self.__sigmoid(inputs) * (1 - self.__sigmoid(inputs))

    def __sigmoid(self, inputs):
        return 1.0 / (1 + np.exp(-inputs))


class SoftmaxActivation(BasicActiviation):
    def forward_propagation(self, inputs):
        max_value = 500
        normalized = [min(i, max_value) if i > 0 else max(i, max_value * -1) for i in inputs]
        # print(f"normalized: {normalized}")

        self.inputs = inputs
        self.values = self.__signal(normalized)
        return self.values
        # max_value = 500
        #
        # normalized = [min(i, max_value) if i > 0 else max(i, max_value * -1) for i in inputs]
        # exps = np.exp(normalized)
        # # print(f"softmax {inputs}")
        # # exps = np.exp(inputs)
        #
        # self.inputs = inputs
        # self.values = exps / np.sum(exps, axis=0)
        #
        # return self.values

    def back_propagation(self, inputs):
        # SM = self.values.reshape((-1, 1))
        # jac = np.diag(self.values) - np.dot(SM, SM.T)
        # return np.sum(jac, axis=1)

        gradients = [0] * len(self.values)
        #
        for i, value in enumerate(self.values):
            for j, input in enumerate(self.inputs):
                if i == j:
                    gradients[i] += value * (1 - input)
                else:
                    gradients[i] += -value * input

        return np.array(gradients) #* -1

    def __signal(self, inputs):
        exps = np.exp(inputs)
        return exps / np.sum(exps, axis=0)


Sigmoid = SigmoidActivation()
Linear = LinearActivation()
Relu = ReluActivation()
Tanh = TanhActivation()
Softmax = SoftmaxActivation()
