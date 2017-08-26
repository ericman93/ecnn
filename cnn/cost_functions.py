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
        return prediction - real


class CrossEntropyLogisticRegressionError(BasicErrorFunction):
    def cost(self, real, prediction):
        return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
        # return np.sum(real * np.log(prediction)) * -1

    def derivative(self, real, prediction):
        raise


MeanSquared = MeanSquaredError()
