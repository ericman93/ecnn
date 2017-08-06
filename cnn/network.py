import numpy as np


class CnnNetwork(object):
    def __init__(self, steps):
        self.steps = steps

    def predict(self, input):
        pass

    def forward_propagation(self, input):
        data = self.__to_3d_shape(np.array(input))
        for step in self.steps:
            data = step.forward_propagation(data)

        return data

    def fit(self, X, y):
        for step in self.steps:
            step.prepare(X, y)

    def __to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
