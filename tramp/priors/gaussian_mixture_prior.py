import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure
from ..beliefs import normal, mixture


class GaussianMixturePrior(Prior):
    def __init__(self, size, probs=[0.5, 0.5], means=[-1, 1], sigmas=[1, 1], isotropic=True):
        self.size = size
        assert len(probs)==len(means)==len(vars)
        self.K = len(probs)
        self.probs = np.array(probs)
        self.means = np.array(means)
        self.sigmas = np.array(sigmas)
        self.isotropic = isotropic
        self.repr_init()
        self.vars = self.sigmas**2
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

    def scalar_forward_mean(self, ax, bx):
        a = ax + self.a # shape (K,)
        b = bx + self.b # shape (K,)
        return mixture.r(a, b, self.eta)

    def scalar_forward_variance(self, ax, bx):
        a = ax + self.a # shape (K,)
        b = bx + self.b # shape (K,)
        return mixture.v(a, b, self.eta)

    def scalar_log_partition(self, ax, bx):
        a = ax + self.a # shape (K,)
        b = bx + self.b # shape (K,)
        A = mixture.A(a, b, self.eta) - mixture.A(self.a, self.b, self.eta)
        return A

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a[:, np.newaxis] # shape (K, N)
        b = bx + self.b[:, np.newaxis] # shape (K, N)
        rx = mixture.r(a, b, self.eta)
        vx = mixture.v(a, b, self.eta)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        a = ax + self.a[:, np.newaxis] # shape (K, N)
        b = bx + self.b[:, np.newaxis] # shape (K, N)
        A = mixture.A(a, b, self.eta) - mixture.A(self.a, self.b, self.eta)
        return A.mean()

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
