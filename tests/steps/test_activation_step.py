import unittest

import numpy as np

from cnn.steps import ReluActivation


class ReluActivationTests(unittest.TestCase):
    def test_relu_activate_1d_array(self):
        # arrange
        input = np.array([[[1, 2, -3, 4, 5]]])
        expected = np.array([[[1, 2, 0, 4, 5]]])
        relu = ReluActivation()

        # act
        output = relu.forward_propagation(input)

        # assert
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))

    def test_relu_activate_2d_array(self):
        # arrange
        input = np.array([[
            [1, 6, 7, -23, -6],
            [-5, 3, 6, -3, 5],
            [-6, 23, -23, 4, 9]
        ]])
        expected = np.array([[
            [1, 6, 7, 0, 0],
            [0, 3, 6, 0, 5],
            [0, 23, 0, 4, 9]
        ]])
        relu = ReluActivation()

        # act
        output = relu.forward_propagation(input)

        # assert
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))

    def test_relu_activate_3d_array(self):
        input = np.array([
            [
                [1, 2, -3],
                [4, -3, 2, ],
                [7, -4, 7],
            ],
            [
                [0, -23, 100],
                [2, 5, 6],
                [-23.9, 4, 20.7]
            ]
        ])
        expected = np.array([
            [
                [1, 2, 0],
                [4, 0, 2, ],
                [7, 0, 7],
            ],
            [
                [0, 0, 100],
                [2, 5, 6],
                [0, 4, 20.7]
            ]
        ])
        relu = ReluActivation()

        # act
        output = relu.forward_propagation(input)

        # assert
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))



