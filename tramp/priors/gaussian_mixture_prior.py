"""Implements the GaussianMixturePrior class."""
import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure
from ..beliefs import normal, mixture


class GaussianMixturePrior(Prior):
    r"""Gaussian mixture prior $p(x)=\sum_{k=1}^K p_k \mathcal{N}(x|r_k,v_k)$

    Parameters
    ----------
    size : int or tuple of int
        Shape of x
    probs : list of length K
        Probability parameters :math:`p_k` of the Gaussian mixture prior
    means : list of length K
        Mean parameters :math:`r_k` of the Gaussian mixture prior
    vars : list of length K
        Variance parameters :math:`r_k` of the Gaussian mixture prior
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, probs=[0.5, 0.5], means=[-1, 1], vars=[1, 1], isotropic=True):
        self.size = size
        assert len(probs) == len(means) == len(vars)
        self.K = len(probs)
        self.probs = np.array(probs)
        self.means = np.array(means)
        self.vars = np.array(vars)
        self.isotropic = isotropic
        self.repr_init()
        self.sigmas = np.sqrt(vars)
        # natural parameters
        self.a = 1 / self.vars
        self.b = self.means / self.vars
        self.eta = np.log(self.probs) - normal.A(self.a, self.b)

    def sample(self):
        shape = (self.size, self.K)
        X_gauss = self.means + self.sigmas * np.random.standard_normal(shape)
        X_cluster = np.random.multinomial(n=1, pvals=self.probs, size=self.size)
        X = (X_gauss*X_cluster).sum(axis=1)
        return X

    def math(self):
        return r"$\mathrm{GMM}$"

    def second_moment(self):
        tau = sum(
            prob * (mean**2 + var)
            for prob, mean, var in zip(self.probs, self.means, self.vars)
        )
        return tau

    def forward_second_moment_FG(self, tx_hat):
        a = tx_hat + self.a  # shape (K,)
        return mixture.tau(a, self.b, self.eta)

    def scalar_forward_mean(self, ax, bx):
        a = ax + self.a  # shape (K,)
        b = bx + self.b  # shape (K,)
        return mixture.r(a, b, self.eta)

    def scalar_forward_variance(self, ax, bx):
        a = ax + self.a  # shape (K,)
        b = bx + self.b  # shape (K,)
        return mixture.v(a, b, self.eta)

    def scalar_log_partition(self, ax, bx):
        a = ax + self.a  # shape (K,)
        b = bx + self.b  # shape (K,)
        A = mixture.A(a, b, self.eta) - mixture.A(self.a, self.b, self.eta)
        return A

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a[:, np.newaxis]  # shape (K, N)
        b = bx + self.b[:, np.newaxis]  # shape (K, N)
        eta = self.eta[:, np.newaxis]  # shape (K, 1)
        rx = mixture.r(a, b, eta)
        vx = mixture.v(a, b, eta)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        a = ax + self.a[:, np.newaxis]  # shape (K, N)
        b = bx + self.b[:, np.newaxis]  # shape (K, N)
        eta = self.eta[:, np.newaxis]  # shape (K, 1)
        A = mixture.A(a, b, eta) - mixture.A(self.a, self.b, self.eta)
        return A.mean()

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        # a0, b0, r0, v0, p0 : shape (K,)
        a0 = self.a + tx0_hat
        b0 = self.b
        r0 = b0 / a0
        v0 = 1 / a0
        p0 = mixture.p(a0, b0, self.eta)
        mu = 0
        for pk, rk, vk in zip(p0, r0, v0):
            mu += pk*gaussian_measure(
                mx_hat * rk, np.sqrt(qx_hat + (mx_hat**2) * vk), f
            )
        return mu

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        # a0, b0, r0, v0, p0 : shape (K,)
        a0 = self.a + tx0_hat
        b0 = self.b
        r0 = b0 / a0
        v0 = 1 / a0
        p0 = mixture.p(a0, b0, self.eta)
        ax_star = (mx_hat / qx_hat) * mx_hat
        mu = 0
        for ak, bk, pk, rk, vk in zip(a0, b0, p0, r0, v0):
            def r_times_f(bx):
                bx_star = (mx_hat / qx_hat) * bx
                r = (bk + bx_star) / (ak + ax_star)
                return r * f(bx)
            mu += pk*gaussian_measure(
                mx_hat * rk, np.sqrt(qx_hat + (mx_hat**2) * vk), r_times_f
            )
        return mu

    def beliefs_measure(self, ax, f):
        mu = sum(
            prob*gaussian_measure(ax * mean, np.sqrt(ax + (ax**2) * var), f)
            for prob, mean, var in zip(self.probs, self.means, self.vars)
        )
        return mu

    def measure(self, f):
        g = sum(
            prob*gaussian_measure(mean, sigma, f)
            for prob, mean, sigma in zip(self.probs, self.means, self.sigmas)
        )
        return g
