import unittest
import numpy as np
from cnn.steps import ConvolutionalStep
from cnn.steps import ReluActivation
from cnn.network import CnnNetwork


class TestNetwork(unittest.TestCase):
    def test_2_steps_flow_1d_input(self):
        # arrange
        steps = [
            ConvolutionalStep(filter_size=(3, 3), num_of_filters=1, x0='ones'),
            ReluActivation()
        ]
        network = CnnNetwork(steps)
        network.fit([], [])
        expected_output = np.array([[[6, 9, 5, 0]]])

        input = [1, 2, 3, 4, -2, -9]

        # act
        output = network.forward_propagation(input)

        # assert
        self.assertEqual(expected_output.shape, output.shape)
        self.assertTrue(
            all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))
