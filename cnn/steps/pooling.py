import numpy as np

from cnn.steps.basic import BasicStep


class PoolingStep(BasicStep):
    def __init__(self, size):
        self.size = size

    def forward_propagation(self, input):
        return self.__get_pooled(input)

    def __get_pooled(self, input):
        input_hight = input.shape[1]
        input_widht = input.shape[2]

        layers = []
        for layer_index in range(0, input.shape[0]):
            layer = []
            for i in range(0, input_hight, self.size):
                row = []
                for j in range(0, input_widht, self.size):
                    row.append(self.pool(input[layer_index, i: i + self.size, j: j + self.size]))
                layer.append(row)

            layers.append(layer)

        return np.array(layers)


class MaxPoolingStep(PoolingStep):
    def __init__(self, size):
        super().__init__(size)

    def pool(self, input):
        return max(input.reshape(input.size))
