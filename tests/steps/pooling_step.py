import unittest
import numpy as np
from steps.pooling_step import MaxPooling

class PoolingActivationTests(unittest.TestCase):
    def test_pooling_1d_array(self):
        # arrange
        input = np.array([[[1, 2, -3, 4]]])
        expected = np.array([[[2, 4]]])
        relu = MaxPooling(2)

        # act
        output = relu.forward_propagation(input)

        # assert
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), output.reshape(output.size))))
