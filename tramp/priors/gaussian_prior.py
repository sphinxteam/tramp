"""Implements the GaussianPrior class."""
import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure
from ..beliefs import normal


class GaussianPrior(Prior):
    r"""Gaussian prior $p(x)=\mathcal{N}(x|r, v)$

    Parameters
    ----------
    size : int or tuple of int
        Shape of x
    mean : float
        Mean parameter $r$ of the Gaussian prior
    var : float
        Variance parameter $v$ of the Gaussian prior
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, mean=0, var=1, isotropic=True):
        self.size = size
        self.mean = mean
        self.var = var
        self.isotropic = isotropic
        self.repr_init()
        self.sigma = np.sqrt(var)
        # natural parameters
        self.a = 1 / var
        self.b = mean / var

    def sample(self):
        X = self.mean + self.sigma * np.random.standard_normal(self.size)
        return X

    def math(self):
        return r"$\mathcal{N}$"

    def second_moment(self):
        return self.mean**2 + self.var

    def forward_second_moment_FG(self, tx_hat):
        a = tx_hat + self.a
        return normal.tau(a, self.b)

    def scalar_forward_mean(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        return b / a

    def scalar_forward_variance(self, ax, bx):
        a = ax + self.a
        return 1 / a

    def scalar_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = normal.A(a, b) - normal.A(self.a, self.b)
        return A

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        rx = b / a
        vx = 1 / a
        return rx, vx

    def compute_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = normal.A(a, b) - normal.A(self.a, self.b)
        return A.mean()

    def compute_forward_error(self, ax):
        a = ax + self.a
        vx = 1 / a
        return vx

    def compute_forward_v_BO(self, ax, tx0_hat):
        a = ax + self.a
        vx = 1 / a
        return vx

    def compute_forward_message(self, ax, bx):
        ax_new = self.a * np.ones_like(ax)
        bx_new = self.b * np.ones_like(bx)
        return ax_new, bx_new

    def compute_forward_state_evolution(self, ax):
        ax_new = self.a
        return ax_new

    def compute_forward_state_evolution_BO(self, ax, tx0_hat):
        ax_new = self.a
        return ax_new

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        a0 = self.a + tx0_hat
        b0 = self.b
        r0 = b0 / a0
        v0 = 1 / a0
        mu = gaussian_measure(
            mx_hat * r0, np.sqrt(qx_hat + (mx_hat**2) * v0), f
        )
        return mu

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        a0 = self.a + tx0_hat
        b0 = self.b
        r0 = b0 / a0
        v0 = 1 / a0
        ax_star = (mx_hat / qx_hat) * mx_hat

        def r_times_f(bx):
            bx_star = (mx_hat / qx_hat) * bx
            r = (b0 + bx_star) / (a0 + ax_star)
            return r * f(bx)
        mu = gaussian_measure(
            mx_hat * r0, np.sqrt(qx_hat + (mx_hat**2) * v0), r_times_f
        )
        return mu

    def beliefs_measure(ax, f):
        mu = gaussian_measure(
            ax * self.r0, np.sqrt(ax + (ax**2) * self.v0), f
        )
        return mu

    def measure(self, f):
        return gaussian_measure(self.mean, self.sigma, f)

    def compute_mutual_information(self, ax):
        a = ax + self.a
        I = 0.5*np.log(a*self.var)
        return I

    def compute_free_energy(self, ax):
        tau_x = self.second_moment()
        I = self.compute_mutual_information(ax)
        A = 0.5*ax*tau_x - I
        return A
