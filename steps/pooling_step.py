from steps.basic import BasicStep


class PoolingStep(BasicStep):
    def __init__(self, size):
        self.size = size

    def forward_propagation(self, input):
        depth = input.shape[0]
        # bulks = self._get_sub_arrays(input, self.size, (1 if depth == 1 else self.size, self.size))

        # output = [d for d in bulks]
        # a = 2



class MaxPooling(PoolingStep):
    def __init__(self, size):
        super().__init__(size)

    def pool(self, input):
        return max(input.reshape(input.size))
