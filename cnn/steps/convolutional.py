import numpy as np
from cnn.common import get_array
from cnn.steps.activation import Linear
from cnn.steps.basic import StepWithFilters


class ConvolutionalStep(StepWithFilters):
    def __init__(self, filter_size, num_of_kernels, activation=Linear, x0='random', padding=None,
                 stride=1):
        super().__init__(activation)

        self.num_of_filters = num_of_kernels
        self.filter_size = filter_size
        self.x0 = x0
        self.padding = padding
        self.activation = activation

        self.stride = stride

    # ~~~~~
    #TODO: All the convolution functions are doing the same thing.
    # use only one of them and delete the rest
    # ~~~~~

    def conv_backprop_features(self, delta, stride):
        filter_shape = self.filters[0].shape
        final = np.zeros((filter_shape[-2], filter_shape[-1]))

        padded_input = self.__add_padding(self.inputs, self.padding)
        # input_height, input_width = self.inputs.shape[-2], self.inputs.shape[-1]
        input_height, input_width = padded_input.shape[-2], padded_input.shape[-1]
        feature_height, feature_widht = delta.shape[-2], delta.shape[-1]

        for i in range(0, input_height, stride):
            row = []
            if i + feature_height > input_height:
                continue

            for j in range(0, input_width, stride):
                if j + feature_widht > input_width:
                    continue

                bulk = padded_input[:, i: i + feature_height, j: j + feature_widht]
                final[int(i / stride), int(j / stride)] = np.sum(bulk * delta)

        return np.array(final)

    def conv_backprop_delta(self, inputs, filter, stride):
        feature_height, feature_widht = filter.shape[-2], filter.shape[-1]

        input_height, input_width = inputs.shape[-2], inputs.shape[-1]
        final = np.zeros((self.inputs.shape[-2], self.inputs.shape[-1]))

        for i in range(0, input_height, stride):
            row = []
            if i + feature_height > input_height:
                continue

            for j in range(0, input_width, stride):
                if j + feature_widht > input_width:
                    continue

                bulk = inputs[i: i + feature_height, j: j + feature_widht]
                final[i, j] = np.sum(bulk * filter)

        return np.tile(final, (self.inputs.shape[0], 1, 1))

    def convolution(self, a, filter, stride):
        feature_height, feature_widht = filter.shape[-2], filter.shape[-1]

        if filter.shape[-1] >= a.shape[-1] or filter.shape[-2] >= a.shape[-2]:
            a = self.__add_padding(a, (feature_height - 1, feature_widht - 1))

        features_sum_of_products = []

        weights = self.__get_sub_arrays(a, stride, (feature_height, feature_widht))
        features_sum_of_products = [[self.get_bulk_sum_of_products(bulk, filter) for bulk in row] for row in weights]

        return np.array(features_sum_of_products)

    def back_prop(self, delta, leraning_rate):
        errors = []
        delta = delta * self.activation.back_propagation(self.z)

        for filter_index, filter in enumerate(self.filters):
            filter_delta = self.conv_backprop_delta(delta[filter_index], filter, self.stride)
            errors.append(filter_delta)

            error = self.conv_backprop_features(delta[filter_index], self.stride)
            filter += (error * leraning_rate)

        return np.stack(errors, axis=0)

    def calc_neurons_values(self, input):
        self.inputs = input
        input = self.__add_padding(input, self.padding)

        if self.filters is None:
            self.neuron_count = input.size
            self.filters = self.__initiliaze_features(input.shape)

        feature_height, feature_widht = self.filters[0].shape[1], self.filters[0].shape[2]
        bulks = self.__get_sub_arrays(input, self.stride, (feature_height, feature_widht))

        features_sum_of_products = []

        for feature in self.filters:
            features_sum_of_products.append(
                [[self.get_bulk_sum_of_products(bulk, feature) for bulk in row] for row in bulks])

        return np.stack(features_sum_of_products, axis=0)

    def __add_padding(self, input, padding):
        if padding is None:
            return input

        if not isinstance(padding, tuple):
            padding = (padding, padding)

        padding_size = [(0, 0)] + [padding] * 2

        return np.lib.pad(input, padding_size, 'constant')

    def get_bulk_sum_of_products(self, bulk, feature):
        multiplied = feature * bulk
        return np.sum(np.reshape(multiplied, multiplied.size))

    def __initiliaze_features(self, input_shape):
        feature_size = (input_shape[0],)

        if isinstance(self.filter_size, tuple):
            filter_height = self.filter_size[0]
            filter_width = self.filter_size[1]
        else:
            filter_height = filter_width = self.filter_size

        feature_size += (min(filter_height, input_shape[1]), min(filter_width, input_shape[2]))

        return [get_array(self.x0, feature_size) for i in range(self.num_of_filters)]

    def __get_sub_arrays_2d(self, input, stride, size):
        input_height, input_width = input.shape[0], input.shape[1]
        size_height = size[0]
        size_width = size[1]

        bulks = []

        for i in range(0, input_height, stride):
            row = []
            if i + size_height > input_height:
                continue

            for j in range(0, input_width, stride):
                if j + size_width > input_width:
                    continue

                bulk = input[i: i + size_height, j: j + size_width]
                row.append(bulk)

            bulks.append(row)

        return bulks

    def __get_sub_arrays(self, input, stride, size):
        input_height, input_width = input.shape[-2], input.shape[-1]
        size_height = size[0]
        size_width = size[1]

        bulks = []

        for i in range(0, input_height, stride):
            row = []
            if i + size_height > input_height:
                continue

            for j in range(0, input_width, stride):
                if j + size_width > input_width:
                    continue

                bulk = input[:, i: i + size_height, j: j + size_width]
                row.append(bulk)

            bulks.append(row)

        return bulks
