"""Implements the PositivePrior class."""
import numpy as np
from .base_prior import Prior
from ..beliefs import positive
from scipy.stats import halfnorm


class PositivePrior(Prior):
    r"""Positive prior :math:`p(x) = 2 * 1_+(x) \mathcal{N}(x|0,1)`

    Parameters
    ----------
    size : int or tuple of int
        Shape of x
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, isotropic=True):
        self.size = size
        self.isotropic = isotropic
        self.repr_init()
        # natural parameters
        self.a = 1.
        self.b = 0.

    def sample(self):
        X = halfnorm.rvs(size=self.size)
        return X

    def math(self):
        return r"\mathcal{N}_+"

    def second_moment(self):
        return 1.

    def forward_second_moment_FG(self, tx_hat):
        a = tx_hat + self.a
        return positive.tau(a, self.b)

    def scalar_forward_mean(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        return positive.r(a, b)

    def scalar_forward_variance(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        return positive.v(a, b)

    def scalar_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = positive.A(a, b) - positive.A(self.a, self.b)
        return A

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        rx = positive.r(a, b)
        vx = positive.v(a, b)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = positive.A(a, b) - positive.A(self.a, self.b)
        return A.mean()

    def b_measure(self, mx_hat, qx_hat, tx0_hat):
        raise NotImplementedError

    def bx_measure(self, mx_hat, qx_hat, tx0_hat):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError
