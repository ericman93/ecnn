import unittest
import numpy as np
from steps.convolutional_step import ConvolutionalStep


class ConvolutionalStepTests(unittest.TestCase):
    def test_one_feature_size_of_1d_input_without_padding_one_filter(self):
        # arrange
        input = np.array([1, 2, 3, 4])
        expected_output = np.array([[[3, 5, 7]]])
        step = ConvolutionalStep(filter_size=2, num_of_filters=1, x0='ones')

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.features))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.features]))
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_1d_input_without_padding_two_filters(self):
        # arrange
        input = np.array([1, 2, 3, 4])
        expected_output = np.array([[[3, 5, 7]], [[6, 10, 14]]])
        step = ConvolutionalStep(filter_size=2, num_of_filters=1, x0='ones')

        step.features = [
            np.ones((1, 1, 2)),
            np.ones((1, 1, 2)) * 2
        ]

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(2, len(step.features))
        self.assertTrue(all([f.shape == (1, 1, 2) for f in step.features]))
        self.assertEqual((2, 1, 3), output.shape)
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_one_feature_size_of_2d_input_without_padding_one_filter_known_features(self):
        # arrange
        input = np.array([
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
        ])
        expected_output = np.array([
            [1, 4, 3, 4, 1],
            [1, 2, 4, 3, 3],
            [1, 2, 3, 4, 1],
            [1, 3, 3, 1, 1],
            [3, 3, 1, 1, 0]
        ])
        step = ConvolutionalStep(filter_size=(3, 3), num_of_filters=1, x0='ones')
        step.features = [
            np.array([[
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ]])
        ]

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(1, len(step.features))
        self.assertTrue(all([f.shape == (1, 3, 3) for f in step.features]))
        self.assertEqual((1, 5, 5), output.shape)
        self.assertTrue(expected_output.size, output.size)
        self.assertTrue(all(np.equal(expected_output.reshape(expected_output.size), output.reshape(output.size))))

    def test_two_features_size_of_2d_input_without_padding(self):
        raise ("Implement")

    def test_one_feature_size_of_2d_input_with_padding(self):
        raise ("Implement")
