import numpy as np
from .base_prior import Prior
from ..utils.integration import exponential_measure
from ..beliefs import exponential, positive


class ExponentialPrior(Prior):
    def __init__(self, size, mean=1, isotropic=True):
        assert mean > 0
        self.size = size
        self.mean = mean
        self.isotropic = isotropic
        self.repr_init()
        self.b = -1/mean

    def sample(self):
        X = np.random.exponential(scale=1/self.mean, size=self.size)
        return X

    def math(self):
        return r"\exp"

    def second_moment(self):
        return 2 * self.mean**2

    def second_moment_FG(self, tx_hat):
        a = tx_hat
        return positive.tau(a, self.b)

    def scalar_forward_mean(self, ax, bx):
        b = bx + self.b
        return positive.r(ax, b)

    def scalar_forward_variance(self, ax, bx):
        b = bx + self.b
        return positive.v(ax, b)

    def scalar_log_partition(self, ax, bx):
        b = bx + self.b
        A = positive.A(ax, b) - exponential.A(self.b)
        return A

    def compute_forward_posterior(self, ax, bx):
        b = bx + self.b
        rx = positive.r(ax, b)
        vx = positive.v(ax, b)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        b = bx + self.b
        A = positive.A(ax, b) - exponential.A(self.b)
        return A.mean()

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        return exponential_measure(self.mean, f)
