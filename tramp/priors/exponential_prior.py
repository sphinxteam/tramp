import numpy as np
from .base_prior import Prior
from ..utils.integration import exponential_measure
from ..utils.truncated_normal import log_Phi
from ..utils.misc import phi_1


class ExponentialPrior(Prior):
    """
        - lambd * exp( -lambd * x) = (-b0) exp(bO * x)
        - mean = 1 / lambd and var = 1 / lambd**2
        - self.b = b0 = - lambd
    """

    def __init__(self, size, lambd=1):
        assert lambd > 0
        self.size = size
        self.mean = 1/lambd
        self.b = - lambd
        self.repr_init()

    def sample(self):
        X = np.random.exponential(scale=1/self.mean, size=self.size)
        return X

    def math(self):
        return r"\exp"

    def second_moment(self):
        return self.mean**2

    def measure(self, f):
        return exponential_measure(m=self.mean)

    def compute_log_partition(self, ax, bx):
        a = ax
        b = bx + self.b
        A = b**2/(2*a) + 1/2 * np.log(2*pi/a)
        z_pos = b / np.sqrt(a)
        A_pos = A + log_Phi(z_pos)
        A_z = -np.log(-self.b)
        logZ_i = A_pos - A_z
        logZ = np.sum(logZ_i)
        return logZ

    def compute_forward_posterior(self, ax, bx):
        a = ax
        b = bx + self.b
        z_pos = b / np.sqrt(a)
        # Use phi_1(x) = x + N(x) / Phi(x)
        pdf_cdf = phi_1(z_pos)-z_pos
        rx = 1/np.sqrt(a) * (z_pos + pdf_cdf)
        v = 1/a * (
            1 - z_pos * pdf_cdf - pdf_cdf**2)
        vx = np.mean(v)
        return rx, vx

    def beliefs_measure(self, ax, f):
        raise NotImplementedError
