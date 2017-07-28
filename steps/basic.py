class BasicStep(object):
    def __init__(self):
        pass

    def forward_propagation(self, input):
        raise Error("Not implemented")

    def _get_sub_arrays(self, input, stride, size):
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
