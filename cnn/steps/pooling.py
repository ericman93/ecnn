import numpy as np

from cnn.steps.basic import BasicStep


class PoolingStep(BasicStep):
    def __init__(self, size):
        self.size = size
        self.inputs = None

    def forward_propagation(self, input):
        self.inputs = input
        self.z = []
        return self.__get_pooled(input)

    def back_prop(self, delta, learnin_rate):
        delta_i = 0
        final_delta = []

        for p in self.z:
            if p == 1:
                final_delta.append(delta[delta_i])
                delta_i += 1
            else:
                final_delta.append(0)

        return np.array(final_delta)

    def __get_pooled(self, input):
        input_hight = input.shape[1]
        input_widht = input.shape[2]

        layers = []
        for layer_index in range(0, input.shape[0]):
            layer = []
            for i in range(0, input_hight, self.size):
                row = []
                for j in range(0, input_widht, self.size):
                    bulk = input[layer_index, i: i + self.size, j: j + self.size]
                    pooled = self.pool(bulk)
                    row.append(pooled)

                    self.z += list(self.__get_z_bulk(bulk, pooled))

                layer.append(row)

            layers.append(layer)

        return np.array(layers)

    def __get_z_bulk(self, bulk, pooled):
        found = False
        for value in bulk.reshape(bulk.size):
            if found:
                yield 0
            elif value == pooled:
                found = True
                yield 1
            else:
                yield 0


class MaxPoolingStep(PoolingStep):
    def __init__(self, size):
        super().__init__(size)

    def pool(self, input):
        return max(input.reshape(input.size))
