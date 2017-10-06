import numpy as np
from cnn.steps.basic import StepWithFilters
from cnn.steps.activation import Linear
from cnn.common import get_array


class OutputStep(StepWithFilters):
    def __init__(self, use_bias=True, activation=Linear, x0='random'):
        super().__init__(activation)

        self.x0 = x0
        self.activation = activation
        self.use_bias = use_bias
        #
        self.classes = None

    def compile(self, X, y):
        self.classes = list([0,1,2])
        # self.classes = list(set(y))

    def back_prop(self, delta, leraning_rate):
        delta = delta * self.activation.back_propagation(self.z)
        # print(f"delta: {delta}")

        # errors = np.sum((np.array(self.filters).transpose() * delta), axis=1)
        errors = np.zeros(len(self.inputs))
        # print(f"error: {errors}")

        # self.filters += errors * self.inputs.transpose() * leraning_rate
        for i, filter in enumerate(self.filters):
            # error = filter * delta[i]
            error = delta[i] * self.inputs #* -1 # NO IDEA WHY IT IS WORKING WITH -1

            # SHOULD BE + SIGN
            # but it dont really working for softmax, so i changed it -

            filter += error * leraning_rate #* filter

            # filter += error * leraning_rate #* self.inputs
            # filter += np.sum(error * self.inputs) * leraning_rate

            errors += error #* filter
        #
        # print(f"inputs: {self.inputs.reshape(self.inputs.size)[: 10]}")
        # print(f"z: {self.z}")
        # print(f"a: {self.a}")
        # print(f"delta: {delta}")
        # after_z = np.array([self.inputs.dot(neuron_weights.T) for neuron_weights in self.filters])
        # print(f"z after: {after_z}")
        # print(f"a after: {self.activation.forward_propagation(after_z)}")

        # return errors[1 if self.use_bias else 0:]

        # errors = np.zeros(self.filters[0].size)
        # errors = []
        #
        # for i in range(self.inputs.shape[0]):
        #     error = self.filters[:, i] * delta
        #     self.filters[:, i] += error * self.inputs[:, i] * leraning_rate
        #
        #     errors.append(error)

        # z_prims = self.activation.back_propagation(self.z)

        # for i, filter in enumerate(self.filters):
        #     error = filter * delta[i] * z_prims[i]
        #     filter += error * leraning_rate
        #     errors += error
        # errors = np.dot(self.filters.tra)

        return errors[1 if self.use_bias else 0:]

    def calc_neurons_values(self, input):
        inputs = np.copy(input)

        if self.use_bias:
            inputs = np.append([1], inputs)
        self.inputs = inputs

        if self.filters is None:
            self.filters = self.__initiliaze_weights(inputs)

        # print(f"inputs: {self.inputs[0:10]}")

        z = np.array([inputs.dot(neuron_weights.T) for neuron_weights in self.filters])
        return z

    def __initiliaze_weights(self, flatten_input):
        # TODO: not use only ones - use what ever X0 is assign for
        return np.array([get_array(self.x0, flatten_input.size) for c in self.classes])
