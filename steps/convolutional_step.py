from steps.basic import BasicStep
import numpy as np


class ConvolutionalStep(BasicStep):
    '''
    feature_size: tuple (height widht) or (width) for 1 dimentianl array
    x0: initiale features values (random, ones, zeros)
    '''

    def __init__(self, filter_size, num_of_filters, x0='random', padding=2, stride=1):
        self.num_of_filters = num_of_filters
        self.filter_size = filter_size
        self.x0 = x0
        self.features = None

        self.stride = stride

    def forward_propagation(self, input):
        input_3d_array = self.__to_3d_shape(input)

        if self.features is None:
            self.features = self.__initiliaze_features(input_3d_array.shape)

        # TODO: save output. understand how do I pile features as my depth of the matrix
        bulks = self.__get_sub_arrays(input_3d_array)
        features_sum_of_products = []

        for feature in self.features:
            features_sum_of_products.append(
                [[self.__get_bulk_sum_of_products(bulk, feature) for bulk in row] for row in bulks])

        return np.stack(features_sum_of_products, axis=0)

    def __get_bulk_sum_of_products(self, bulk, feature):
        multiplied = feature * bulk
        return np.sum(np.reshape(multiplied, multiplied.size))

    def __get_sub_arrays(self, input):
        input_height, input_width = input.shape[1], input.shape[2]
        feature_height, feature_widht = self.features[0].shape[1], self.features[0].shape[2]
        bulks = []

        for i in range(0, input_height, self.stride):
            row = []
            if i + feature_height > input_height:
                continue

            for j in range(0, input_width, self.stride):
                if j + feature_widht > input_width:
                    continue

                # if feature_height == 0:
                bulk = input[:, i: i + feature_height, j: j + feature_widht]
                # else:
                #     bulk = input[:, (i, i + feature_height, 1), (j, j + feature_widht, 1)]

                row.append(bulk)

            bulks.append(row)

        return bulks

    def __initiliaze_features(self, input_shape):
        feature_size = (input_shape[0],) + (
            self.filter_size if isinstance(self.filter_size, tuple) else (1, self.filter_size))
        # TODO: not use only ones - use what ever X0 is assign for
        return [np.ones(feature_size) for i in range(self.num_of_filters)]

    def __to_3d_shape(self, input):
        return np.reshape(input, (1,) * (3 - len(input.shape)) + input.shape)
