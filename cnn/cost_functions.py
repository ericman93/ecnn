class BasicErrorFunction(object):
    def cost(self, real, prediction):
        pass

    def derivative(self, real, prediction):
        pass


class MeanSquaredError(BasicErrorFunction):
    # TODO: Should i get the value of the prediction of the activation for the backprop
    def cost(self, real, prediction):
        return ((real-prediction) ** 2) / 2

    def derivative(self, real, prediction):
        return

MeanSquared = MeanSquaredError()