import unittest
import numpy as np
from cnn.steps import ConvolutionalStep
from cnn.steps.activation import Relu


class ConvolutionalStepTests(unittest.TestCase):
    def test_one_feature_size_of_1d_input_without_padding_one_filter(self):
        # arrange
        input = np.array([[[1, 2, 3, 4]]])
        expected_output = np.array([[[3, 5, 7]]])
        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, x0='ones')

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.filters]))
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_1d_input_two_padding_one_filter(self):
        # arrange
        input = np.array([[[1, 2, 3, 4]]])
        expected_output = np.array([[[1, 3, 5, 7, 4]]])
        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, padding=1, x0='ones')

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.filters]))
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_1d_input_without_padding_one_filter_with_stide(self):
        # arrange
        input = np.array([[[1, 2, 3, 4]]])
        expected_output = np.array([[[3, 7]]])
        step = ConvolutionalStep(filter_size=2, stride=2, num_of_kernels=1, x0='ones')

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.filters]))
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_1d_input_without_padding_two_filters(self):
        # arrange
        input = np.array([[[1, 2, 3, 4]]])
        expected_output = np.array([[[3, 5, 7]], [[6, 10, 14]]])
        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, x0='ones')

        step.filters = [
            np.ones((1, 1, 2)),
            np.ones((1, 1, 2)) * 2
        ]

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(2, len(step.filters))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.filters]))
        self.assertEqual((2, 1, 3), output.shape)
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_2d_input_without_padding_one_filter_known_features(self):
        # arrange
        input = np.array([[
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
        ]])
        expected_output = np.array([
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [1, 3, 3, 1, 1],
            [3, 3, 1, 1, 0]
        ])
        step = ConvolutionalStep(filter_size=(3, 3), num_of_kernels=1, x0='ones')
        step.filters = [
            np.array([[
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ]])
        ]

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 3, 3) for f in step.filters]))
        self.assertEqual((1, 5, 5), output.shape)
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_two_features_size_of_2d_input_without_padding(self):
        raise ("Implement")

    def test_one_feature_size_of_2d_input_with_padding(self):
        raise ("Implement")

    def test_one_feature_2d_input_with_padding(self):
        # arrange
        input = np.array([[
            [2, 5, 7],
            [1, 4, 6],
        ]])
        expected = np.array([
            [2, 7, 12, 7],
            [3, 12, 22, 13],
            [1, 5, 10, 6]
        ])

        step = ConvolutionalStep(filter_size=(2, 2), num_of_kernels=1, x0='ones', padding=1)

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 2, 2) for f in step.filters]))
        self.assertTrue(expected.size, output.size)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))

    def test_one_feature_3d_input_no_padding(self):
        # arrange
        input = np.array([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19]
            ]
        ])
        expected = np.array([
            [64, 72],
            [88, 96]
        ])

        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, x0='ones')

        # arrange
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (2, 2, 2) for f in step.filters]))
        self.assertTrue(expected.size, output.size)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))

    def test_one_feature_3d_input_with_1_padding(self):
        # arrange
        input = np.array([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19]
            ]
        ])
        expected = np.array([
            [12, 26, 30, 16],
            [30, 64, 72, 38],
            [42, 88, 96, 50],
            [24, 50, 54, 28]
        ])

        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, x0='ones', padding=1)

        # arrange
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (2, 2, 2) for f in step.filters]))
        self.assertTrue(expected.size, output.size)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))

    def test_convolution_with_given_activation(self):
        # arrange
        input = np.array([[[1, -7, 3, 4]]])
        expected_output = np.array([[[0, 0, 7]]])
        step = ConvolutionalStep(filter_size=2, num_of_kernels=1, activation=Relu, x0='ones')

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.filters))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.filters]))
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))
