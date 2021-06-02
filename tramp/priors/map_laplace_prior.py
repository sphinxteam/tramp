import numpy as np
from .base_prior import Prior


def l1_norm(x):
    x_abs = np.abs(x)
    return x_abs.sum()


def soft_threshold(x, gamma):
    x_abs = np.abs(x)
    return np.maximum(0, 1 - gamma / x_abs) * x


def v_soft_threshold(x, gamma):
    return np.abs(x) > gamma


class MAP_LaplacePrior(Prior):
    def __init__(self, size, scale, isotropic=True):
        self.size = size
        self.scale = scale
        self.isotropic = isotropic
        self.repr_init()
        self.gamma = 1 / scale

    def sample(self):
        X = np.random.laplace(size=self.size, scale=self.scale)
        return X

    def math(self):
        return r"$\Vert . \Vert_1$"

    def second_moment(self):
        return 2 * self.scale**2

    def scalar_forward_mean(self, ax, bx):
        return (1 / ax) * soft_threshold(bx, self.gamma)

    def scalar_forward_variance(self, ax, bx):
        return (1 / ax) * v_soft_threshold(bx, self.gamma)

    def scalar_log_partition(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        A =  -self.gamma*np.abs(rx) -0.5*ax*(rx**2) + bx*rx
        return A

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        vx = (1 / ax) * v_soft_threshold(bx, self.gamma)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        A =  -self.gamma*np.abs(rx) + bx*rx - 0.5*ax*(rx**2)
        return A.mean()

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError
