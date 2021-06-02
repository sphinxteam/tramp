import numpy as np
from .base_prior import Prior
from ..beliefs import positive
from scipy.stats import halfnorm


class PositivePrior(Prior):
    def __init__(self, size, isotropic=True):
        self.size = size
        self.isotropic = isotropic
        self.repr_init()
        self.a = 1.
        self.b = 0.

    def sample(self):
        X = halfnorm.rvs(size=self.size)
        return X

    def math(self):
        return r"\mathcal{N}_+"

    def second_moment(self):
        return 1.

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

    def measure(self, f):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        raise NotImplementedError
