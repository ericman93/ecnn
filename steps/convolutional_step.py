from steps.basic import BasicStep
import numpy as np


class ConvolutionalStep(BasicStep):
    '''
    feature_size: tuple (height widht) or (width) for 1 dimentianl array
    x0: initiale features values (random, ones, zeros)
    padding: number for equels
    '''

    def __init__(self, filter_size, num_of_filters, x0='random', padding=None, stride=1):
        super().__init__()

        self.num_of_filters = num_of_filters
        self.filter_size = filter_size
        self.x0 = x0
        self.features = None
        self.padding = padding

        self.stride = stride

    def forward_propagation(self, input):
        # TODO: this is not the convolution job. its the CNN network it self
        input_3d_array = self.__to_3d_shape(self.__add_padding(input))

        if self.features is None:
            self.features = self.__initiliaze_features(input_3d_array.shape)

        feature_height, feature_widht = self.features[0].shape[1], self.features[0].shape[2]
        bulks = self.__get_sub_arrays(input_3d_array, self.stride, (feature_height, feature_widht))
        features_sum_of_products = []

        for feature in self.features:
            features_sum_of_products.append(
                [[self.__get_bulk_sum_of_products(bulk, feature) for bulk in row] for row in bulks])

        return np.stack(features_sum_of_products, axis=0)

    def __add_padding(self, input):
        if self.padding is None:
            return input

        padding_size = [(0,0)] + [(self.padding, self.padding)] * 2
        return np.lib.pad(input, padding_size, 'constant')

        # return input

    def __get_bulk_sum_of_products(self, bulk, feature):
        multiplied = feature * bulk
        return np.sum(np.reshape(multiplied, multiplied.size))

    def __initiliaze_features(self, input_shape):
        feature_size = (input_shape[0],)

        if isinstance(self.filter_size, tuple):
            filter_height = self.filter_size[0]
            filter_width = self.filter_size[1]
        else:
            filter_height = filter_width = self.filter_size

        feature_size += (min(filter_height, input_shape[1]), filter_width)

        # TODO: not use only ones - use what ever X0 is assign for
        return [np.ones(feature_size) for i in range(self.num_of_filters)]

    def __get_sub_arrays(self, input, stride, size):
        input_height, input_width = input.shape[1], input.shape[2]
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

                # bulk = input[:, i: input_width, j: j + feature_widht]
                # else:
                bulk = input[:, i: i + size_height, j: j + size_width]
                row.append(bulk)

            bulks.append(row)

        return bulks

    def __to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
