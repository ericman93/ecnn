import numpy as np

array_init_methods = {
    'random': lambda shape: np.random.uniform(-1, 1, shape),
    'ones': np.ones,
    'zeros': np.zeros,
    'test': lambda  shape: np.zeros(shape) * 0.0001
}


def get_array(x0, shape):
    return array_init_methods[x0](shape)
