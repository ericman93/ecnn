from cnn.steps.basic import BasicStep
import numpy as np

class Flatten(BasicStep):
    def __init__(self):
        super().__init__()

    def forward_propagation(self, inputs):
        self.inputs = inputs
        return inputs.reshape(inputs.size)

    def back_prop(self, delta, learning_rate):
        return delta.reshape(self.inputs.shape)