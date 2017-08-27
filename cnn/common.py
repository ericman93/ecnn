import numpy as np

array_init_methods = {
    'random': lambda shape: np.random.uniform(-1, 1, shape),
    'ones': np.ones,
    'zeros': np.zeros
}


def get_array(x0, shape):
    return array_init_methods[x0](shape)
