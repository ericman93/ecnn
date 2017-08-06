import unittest
import numpy as np
import cnn.common as common


class TestCommon(unittest.TestCase):
    def test_get_array_of_ones(self):
        # arrange
        expected = np.array([[
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]])

        # act
        array = common.get_array('ones', (1, 2, 4))

        # assert
        self.assertEqual(expected.shape, array.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), array.reshape(array.size))))

    def test_get_random_array(self):
        # act
        array1 = common.get_array('random', (2, 1, 5))
        array2 = common.get_array('random', (2, 1, 5))

        # assert
        self.assertEqual(array1.shape, array2.shape)

        array1 = array1.reshape(array1.size)
        array2 = array2.reshape(array2.size)
        for i in range(0, array1.size):
            self.assertNotEqual(array1[i], array2[i])

    def test_get_array_of_zeros(self):
        # arrange
        expected = np.array([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])

        # act
        array = common.get_array('zeros', (3, 2, 4))

        # assert
        self.assertEqual(expected.shape, array.shape)
        self.assertTrue(all(np.equal(expected.reshape(expected.size), array.reshape(array.size))))
