import pickle
import json
import operator
import numpy as np
import os.path
from random import shuffle
from cnn.steps.basic import StepWithFilters
from cnn.exceptions import BadInputException
from cnn.fit_history import History


class CnnNetwork(object):
    def __init__(self, steps):
        self.steps = steps

    def predict(self, inputs):
        classes = self.forward_propagation(inputs)
        index, value = max(enumerate(classes), key=operator.itemgetter(1))

        return index, value

    def forward_propagation(self, input):
        data = self.to_3d_shape(np.array(input))
        for step in self.steps:
            data = step.forward_propagation(data)

        return data

    def fit(self, X, y, cost_function, learning_rate=0.001, iterations=1000, batch_size=32, random=True, verbose=False,
            test_portion=0):
        self.__validate_input(X, y)
        self.verbose = verbose

        self.compile(X, y)
        self.__back_propogation(X, y, cost_function, learning_rate, iterations, batch_size)

    def save(self, file_path):
        print("saving model")
        with open(file_path, 'wb') as file:
            pickle.dump(self.steps, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                while 1:
                    try:
                        self.steps = pickle.load(file)
                    except (EOFError):
                        break

    # TODO: maybe keras is right and I need to compile before fitting
    def compile(self, X, y):
        for step in self.steps:
            step.compile(X, y)

    def __to_categories(self, y):
        # num_of_classes = len(set(y))
        num_of_classes = 3

        for p in y:
            y_category = [0] * num_of_classes
            y_category[p] = 1
            yield y_category

    def __back_propogation(self, X, y, cost_function, learning_rate, iterations, batch_size):
        history = History()

        inputs = [self.to_3d_shape(np.array(x)) for x in X]
        tags = list(self.__to_categories(y))
        input_with_tag = list(zip(inputs, tags))

        for iteration in range(iterations):
            self.log("---------", 1)
            self.log(f"Iteration {iteration}", 1)
            self.log("---------", 1)

            shuffle(input_with_tag)
            batch = input_with_tag if batch_size == -1 else input_with_tag[0:batch_size]
            batch_cost = 0

            for batch_index, (batch_input, batch_tag) in enumerate(batch):
                data = batch_input

                for step in self.steps:
                    data = step.forward_propagation(data)
                    # print(f"data after {step.__class__.__name__}: {data.shape}")

                self.log(f"output : {data}", 1)
                self.log(f"y: {batch_tag}", 1)
                cost = cost_function.cost(batch_tag, data)
                history.costs.append(cost)
                self.log(f"cost: {cost}", 1)

                delta = cost_function.derivative(batch_tag, data)  # * step.activation.derivative(step.inputs)

                # print("Backprop")
                for i, step in enumerate(reversed(self.steps)):
                    delta = step.back_prop(delta, learning_rate)
                    self.log(f"{step.__class__.__name__} - MAX: {np.max(delta)} - MIN: {np.min(delta)}", 2)

                print(f'--- {batch_index}/{len(batch)} ({iteration})---')

            self.save(f"pokemon-cards.b")


        with open('costs.json', 'w') as costs_file:
            json.dump(history.costs, costs_file)

        self.save(f"pokemon-cards.b")
        self.log(f"cost: {cost}", 1)

    def __validate_input(self, X, y):
        num_of_samples = np.array(X).shape[0]
        num_of_tagging = np.array(y).shape[0]

        if num_of_samples != num_of_tagging:
            raise BadInputException(f"Wrong number of samples and tagged data (X={num_of_samples}, y={num_of_tagging})")

    def log(self, message, level):
        if self.verbose >= level:
            print(message)

    def to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
