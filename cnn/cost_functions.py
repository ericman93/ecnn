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


class CrossEntropyLogisticRegressionError(BasicErrorFunction):
    def cost(self, real, prediction):
        sum = 0
        for i in range(len(real)):
            sum += (prediction[i] + np.log(real[i])) + (1 - prediction[i]) * np.log(1 - real[i])

        return sum
        # return -np.sum(np.multiply(prediction, np.log(real)) + np.multiply((1 - prediction), np.log(1 - real)))
        # return np.sum(real * np.log(prediction)) * -1

    def derivative(self, real, prediction):
        return prediction - real


MeanSquared = MeanSquaredError()
CrossEntropyLogisticRegression = CrossEntropyLogisticRegressionError()
