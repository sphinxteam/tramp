import numpy as np
from .base_prior import Prior
import warnings


def group_soft_threshold(x, gamma, axis):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True) # broadcast against x
    return np.maximum(0, 1 - gamma / x_norm) * x

def v_group_soft_threshold(x, gamma, axis):
    d = x.shape[axis]
    x_norm = np.linalg.norm(x, axis=axis, keepdims=False)
    v = (x_norm > gamma) * (1 + (1 / d - 1) * gamma / x_norm)
    return np.mean(v)

class MAP_L21NormPrior(Prior):
    def __init__(self, size, scale, axis = 0):
        self.size = size
        self.scale = scale
        self.axis = axis
        self.repr_init()
        self.gamma = 1 / scale

    def sample(self):
        warnings.warn(
            "MAP_L21NormPrior.sample not implemented "
            "return zero array as a placeholder"
        )
        return np.zeros(self.size)

    def math(self):
        return r"$\Vert . \Vert_{2,1}$"

    def second_moment(self):
        warnings.warn(
            "MAP_L21NormPrior.second_moment not implemented "
            "return 1 as a placeholder"
        )
        return 1.

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * group_soft_threshold(bx, self.gamma, self.axis)
        vx = (1 / ax) * v_group_soft_threshold(bx, self.gamma, self.axis)
        return rx, vx

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError
