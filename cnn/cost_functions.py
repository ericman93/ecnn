import numpy as np


class BasicErrorFunction(object):
    def cost(self, real, prediction):
        pass

    def derivative(self, real, prediction):
        pass


class MeanSquaredError(BasicErrorFunction):
    # TODO: Should i get the value of the prediction of the activation for the backprop
    def cost(self, real, prediction):
        return ((real - prediction) ** 2) / 2

    def derivative(self, real, prediction):
        return real - prediction


class CrossEntropyErrror(BasicErrorFunction):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-7

    def cost(self, real, prediction):
        t = np.array(real)
        p = np.clip(prediction, self.epsilon, 1 - self.epsilon)

        return -1 * np.sum(np.multiply(t, np.log(p)))

    def derivative(self, real, prediction):
        return real - prediction

class CrossEntropyLogisticRegressionError(BasicErrorFunction):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-7

    def cost(self, real, prediction):
        t = np.array(real)
        p = np.clip(prediction, self.epsilon, 1 - self.epsilon)

        return -1 * np.sum(np.multiply(t, np.log(p)) + np.multiply((1 - t), np.log(1 - p)))

    def derivative(self, real, prediction):
        return real - prediction


MeanSquared = MeanSquaredError()
CrossEntropyLogisticRegression = CrossEntropyLogisticRegressionError()
CrossEntropy = CrossEntropyErrror()
