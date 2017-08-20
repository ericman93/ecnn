import numpy as np
from cnn.steps.basic import StepWithFilters
from cnn.exceptions import BadInputException


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

    def fit(self, X, y, cost_function, learning_rate=0.001, iterations=1000, batch_size=32, random=True, test_portion=0):
        self.__validate_input(X, y)

        self.compile(X, y)
        self.__back_propogation(X, y, cost_function, learning_rate, iterations)

    # TODO: maybe keras is right and I need to compile before fitting
    def compile(self, X, y):
        for step in self.steps:
            step.compile(X, y)

    def __to_categories(self, y):
        num_of_classes = len(set(y))
        for p in y:
            y_category = [0] * num_of_classes
            y_category[p] = 1
            yield y_category

    def __back_propogation(self, X, y, cost_function, learning_rate, iterations):
        # for iteration in range(iterations):
        inputs =  self.__to_3d_shape(np.array(X[0]))
        tag = list(self.__to_categories(y))[0]

        data = inputs
        for step in self.steps:
            data = step.forward_propagation(data)

        cost = cost_function.cost(tag, data)

        for step in reversed(self.steps):
            if not isinstance(step, StepWithFilters):
                continue

            delta = step.activation.back_propagation(step.z).dot(learning_rate) * cost
            step.update_weights(delta)






        # for step_index, step in enumerate(reversed(self.steps)):
        #     if step_index == 0:
        #         print(step)
        # calc error with cost function

    def __validate_input(self, X, y):
        num_of_samples = np.array(X).shape[0]
        num_of_tagging = np.array(y).shape[0]

        if num_of_samples != num_of_tagging:
            raise BadInputException(f"Wrong number of samples and tagged data (X={num_of_samples}, y={num_of_tagging})")

    def __to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
