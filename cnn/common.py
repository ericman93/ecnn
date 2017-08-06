import numpy as np

array_init_methods = {
    'random': np.random.random_sample,
    'ones': np.ones,
    'zeros': np.zeros
}


def get_array(x0, shape):
    return array_init_methods[x0](shape)
