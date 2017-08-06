import unittest
import numpy as np
from cnn.steps.output_step import ClassificationOutputStep


class OutputStepTests(unittest.TestCase):
    def test_output_prepare(self):
        # arrange
        X = [1, 0, 1, 1, 2]
        y = [0, 1, 2, 1, 0]
        input = np.array([[[1, 2, 3, 4]]])
        expected_weights = np.ones((3, 4))

        step = ClassificationOutputStep(x0='ones')

        # act
        step.prepare(X, y)
        step.forward_propagation(input)

        # assert
        self.assertEqual(expected_weights.shape, step.weights.shape)
        self.assertTrue(
            all(np.equal(expected_weights.reshape(expected_weights.size), step.weights.reshape(step.weights.size))))

    def test_output_initalize_weights_1d_array_4_classes(self):
        # arrange
        input = np.array([[[1, 2, 3, 4]]])

        step = ClassificationOutputStep(x0='ones')
        step.prepare([], [0, 1, 2, 3])
        expected_weights = np.ones((4, 5))

        # act
        step.forward_propagation(input)

        # assert
        self.assertEqual(expected_weights.shape, step.weights.shape)
        self.assertTrue(
            all(np.equal(expected_weights.reshape(expected_weights.size), step.weights.reshape(step.weights.size))))

    def test_output_initalize_weights_2d_array_3_classes(self):
        input = np.array([[
            [2, 3, 4],
            [6, 4, 3]
        ]])
        step = ClassificationOutputStep(x0='ones')
        step.prepare([], [0, 1, 2])
        expected_weights = np.ones((3, 7))

        # act
        step.forward_propagation(input)

        # assert
        self.assertEqual(expected_weights.shape, step.weights.shape)
        self.assertTrue(
            all(np.equal(expected_weights.reshape(expected_weights.size), step.weights.reshape(step.weights.size))))

    def test_output_initalize_weights_3d_array_2_classes(self):
        input = np.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ]
        ])
        step = ClassificationOutputStep(x0='ones')
        step.prepare([], [0, 1])
        expected_weights = np.ones((2, 9))

        # act
        step.forward_propagation(input)

        # assert
        self.assertEqual(expected_weights.shape, step.weights.shape)
        self.assertTrue(
            all(np.equal(expected_weights.reshape(expected_weights.size), step.weights.reshape(step.weights.size))))

    def test_predictions_for_2_classes(self):
        input = np.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ]
        ])
        step = ClassificationOutputStep(x0='ones')
        step.weights = [
            np.array([1, 1, 2, 3, 4, 5, 6, 7, 8]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        ]
        exptected_output = np.array([205, 37])

        # act
        output = step.forward_propagation(input)

        # assert
        self.assertEqual(exptected_output.shape, output.shape)
        self.assertTrue(
            all(np.equal(exptected_output.reshape(output.size), output.reshape(output.size))))

