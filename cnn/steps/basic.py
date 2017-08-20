class BasicStep(object):
    def __init__(self):
        pass

    def prepare(self, X, y):
        pass

    def forward_propagation(self, input):
        raise Error("Not implemented")

    def compile(self, X, y):
        pass


class StepWithFilters(BasicStep):
    def __init__(self, activation):
        super().__init__()

        self.filters = None

        self.activation = activation
        self.z = None
        self.a = None

    def update_weights(self, delta):
        for i in range(len(self.filters)):
            self.filters[i, :] += delta[i]


    def forward_propagation(self, inputs):
        self.z = self.calc_neurons_values(inputs)
        self.a = self.activation.forward_propagation(self.z)

        return self.a

    def calc_neurons_values(self, inputs):
        raise Error("Implement")



