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
        self.compile(X, y)

    # TODO: maybe keras is right and I need to compile before fitting
    def compile(self, X, y):
        for step in self.steps:
            step.compile(X, y)

    def __back_propogation(self, X, y):
        for i, step in enumerate(reversed(step.steps)):
            if i == 0:
                pass
        # calc error with cost function

    def __to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
