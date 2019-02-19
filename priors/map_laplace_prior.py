import numpy as np
from ..base import Prior


def soft_threshold(x, gamma):
    x_abs = np.abs(x)
    return np.maximum(0, 1 - gamma / x_abs) * x


def v_soft_threshold(x, gamma):
    return np.mean(np.abs(x) > gamma)


class MAP_LaplacePrior(Prior):
    def __init__(self, size, scale):
        self.size = size
        self.scale = scale
        self.repr_init()
        self.gamma = 1 / scale

    def sample(self):
        X = np.random.laplace(size=self.size, scale=self.scale)
        return X

    def math(self):
        return r"$\Vert . \Vert_1$"

    def second_moment(self):
        return 2 * self.scale**2

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        vx = (1 / ax) * v_soft_threshold(bx, self.gamma)
        return rx, vx

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError
