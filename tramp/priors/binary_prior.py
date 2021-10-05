"""Implements the BinaryPrior class."""
import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure
from ..beliefs import binary


class BinaryPrior(Prior):
    r"""Binary prior :math:`p(x) = p_+ \delta_+(x) + p_- \delta_-(x)`

    Parameters
    ----------
    size : int or tuple of int
        Shape of x
    p_pos : float in (0,1)
        Parameter :math:`p_+` of the binary prior
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, p_pos=0.5, isotropic=True):
        self.size = size
        self.p_pos = p_pos
        self.isotropic = isotropic
        self.repr_init()
        self.p_neg = 1 - p_pos
        # natural parameters
        self.b = 0.5*np.log(self.p_pos / self.p_neg)

    def sample(self):
        p = [self.p_neg, self.p_pos]
        X = np.random.choice([-1, +1], size=self.size, replace=True, p=p)
        return X

    def math(self):
        return r"$p_\pm$"

    def second_moment(self):
        return 1.

    def forward_second_moment_FG(self, tx_hat):
        return binary.tau(self.b)

    def scalar_forward_mean(self, ax, bx):
        b = bx + self.b
        return binary.r(b)

    def scalar_forward_variance(self, ax, bx):
        b = bx + self.b
        return binary.v(b)

    def scalar_log_partition(self, ax, bx):
        b = bx + self.b
        A = binary.A(b) - binary.A(self.b) - 0.5*ax
        return A

    def compute_forward_posterior(self, ax, bx):
        b = bx + self.b
        rx = binary.r(b)
        vx = binary.v(b)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        b = bx + self.b
        A = binary.A(b) - binary.A(self.b) - 0.5*ax
        return A.mean()

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        mu_pos = gaussian_measure(+mx_hat, np.sqrt(qx_hat), f)
        mu_neg = gaussian_measure(-mx_hat, np.sqrt(qx_hat), f)
        mu = self.p_pos * mu_pos + self.p_neg * mu_neg
        return mu

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        mu_pos = +gaussian_measure(+mx_hat, np.sqrt(qx_hat), f)
        mu_neg = -gaussian_measure(-mx_hat, np.sqrt(qx_hat), f)
        mu = self.p_pos * mu_pos + self.p_neg * mu_neg
        return mu

    def beliefs_measure(self, ax, f):
        mu_pos = gaussian_measure(+ax, np.sqrt(ax), f)
        mu_neg = gaussian_measure(-ax, np.sqrt(ax), f)
        mu = self.p_pos * mu_pos + self.p_neg * mu_neg
        return mu

    def measure(self, f):
        return self.p_pos * f(+1) + self.p_neg * f(-1)
