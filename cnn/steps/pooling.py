import numpy as np

from cnn.steps.basic import BasicStep


class PoolingStep(BasicStep):
    def __init__(self, size):
        self.size = size
        self.inputs = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.z = []
        self.a = self.__get_pooled(inputs)
        return self.a

    def back_prop(self, delta, learnin_rate):
        flatten_delta = delta.reshape(delta.size)
        delta_i = 0
        final_delta = []

        bb = [1 for a in self.z.reshape(self.z.size) if a == 1]

        for p in self.z.reshape(self.z.size):
            if p == 1:
                final_delta.append(flatten_delta[delta_i])
                delta_i += 1
            else:
                final_delta.append(0)

        return np.array(final_delta).reshape(self.z.shape)

    def __get_pooled(self, inputs):
        input_hight = inputs.shape[-2]
        input_widht = inputs.shape[-1]

        layers = []
        z = []

        for layer_index in range(0, inputs.shape[0]):
            layer = []

            for i in range(0, input_hight, self.size):
                row = []

                for j in range(0, input_widht, self.size):
                    bulk = inputs[layer_index, i: i + self.size, j: j + self.size]
                    # z = np.copy(bulk)

                    pooled = self.pool(bulk)
                    row.append(pooled)

                    # If two values have the pooled value, two of them will mark as 1
                    # pooled_bulk = [1 if v == pooled else 0 for v in set(bulk.reshape(bulk.size))]
                    # pooled_bulk += [0] * (self.size * self.size - len(pooled_bulk)) # if I have multiple items with the same value

                    z += self.__get_z_bulk(bulk,pooled)


                layer.append(row)

            layers.append(layer)

        self.z = np.array(z).reshape(self.inputs.shape)
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
