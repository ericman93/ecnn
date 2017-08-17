import unittest
import numpy as np
from cnn.steps import ConvolutionalStep
from cnn.steps import ReluActivation
from cnn.steps import MaxPoolingStep
from cnn.steps import OutputStep
from cnn.network import CnnNetwork
from cnn.steps.activation import Sigmoid
from cnn.steps.activation import Relu
from cnn.cost_functions import MeanSquared
from cnn.exceptions import BadInputException


class TestForworkPropogationNetwork(unittest.TestCase):
    def test_2_steps_flow_1d_input(self):
        # arrange
        steps = [
            ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='ones'),
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

    def test_3_steps_flow_1d_input(self):
        # arrange
        steps = [
            ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='ones'),
            ReluActivation(),
            MaxPoolingStep(2)
        ]
        network = CnnNetwork(steps)
        network.fit([], [])
        expected_output = np.array([[[9, 5]]])

        input = [1, 2, 3, 4, -2, -9]

        # act
        output = network.forward_propagation(input)

        # assert
        self.assertEqual(expected_output.shape, output.shape)
        self.assertTrue(
            all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_3_steps_with_linier_output_with_bias_2d_input(self):
        # arrange
        num_classes = 3
        steps = [
            ConvolutionalStep(filter_size=(3, 3), padding=1, num_of_kernels=1, x0='ones', stride=2),
            MaxPoolingStep(2),
            OutputStep(x0='ones')
        ]
        network = CnnNetwork(steps)
        network.fit([], list(range(num_classes)))
        expected_output = np.array([155, 155, 155])

        inputs = [
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 5],
            [9, 10, 11, 12, 5],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 5],
        ]

        # act
        output = network.forward_propagation(inputs)

        # assert
        self.assertEqual(expected_output.shape, output.shape)
        self.assertTrue(
            all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_3_steps_with_sigmoid_output_with_bias_2d_input(self):
        # arrange
        num_classes = 3
        steps = [
            ConvolutionalStep(filter_size=(3, 3), padding=1, num_of_kernels=1, x0='ones', stride=2),
            MaxPoolingStep(2),
            OutputStep(x0='ones', activation=Sigmoid)
        ]
        network = CnnNetwork(steps)
        network.fit([], list(range(num_classes)))
        expected_output = np.array([1, 1, 1])

        inputs = [
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 5],
            [9, 10, 11, 12, 5],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 5],
        ]

        # act
        output = network.forward_propagation(inputs)

        # assert
        self.assertEqual(expected_output.shape, output.shape)
        self.assertTrue(
            all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))


class TestBackPropogationNetwork(unittest.TestCase):
    def test_wrong_number_of_tagged_elements(self):
        # arrange
        steps = [
            ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='ones', activation=Relu),
            OutputStep(x0='ones', activation=Sigmoid)
        ]
        network = CnnNetwork(steps)
        X = [
            [1, 2, 3, 4],
            [4, 5, 6, 6],
            [2, 5, 5, 6]
        ]
        y = [0, 1]

        with self.assertRaises(BadInputException):
            network.fit(X, y, MeanSquared)

    def test_one_conv_layer(self):
        # arrange
        steps = [
            ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='ones', activation=Relu),
            OutputStep(x0='ones', activation=Sigmoid)
        ]
        network = CnnNetwork(steps)
        X = [
            [1, 2, 3, 4]
        ]
        y = [0]

        # act
        network.fit(X, y, MeanSquared, iterations=1, batch_size=1)
