import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure
from ..beliefs import normal, sparse


class GaussBernoulliPrior(Prior):
    def __init__(self, size, rho=0.5, mean=0, var=1, isotropic=True):
        self.size = size
        self.rho = rho
        self.mean = mean
        self.var = var
        self.isotropic = isotropic
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = mean / var
        self.eta = normal.A(self.a, self.b) - np.log(rho / (1 - rho))

    def sample(self):
        X_gauss = self.mean + self.sigma * np.random.standard_normal(self.size)
        X_bernoulli = np.random.binomial(n=1, size=self.size, p=self.rho)
        X = X_gauss * X_bernoulli
        return X

    def math(self):
        return r"$\mathcal{N}_\rho$"

    def second_moment(self):
        return self.rho * (self.mean**2 + self.var)

    def second_moment_FG(self, tx_hat):
        a = tx_hat + self.a
        return sparse.tau(a, self.b, self.eta)

    def scalar_forward_mean(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        return sparse.r(a, b, self.eta)

    def scalar_forward_variance(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        return sparse.v(a, b, self.eta)

    def scalar_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = sparse.A(a, b, self.eta) - sparse.A(self.a, self.b, self.eta)
        return A

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        rx = sparse.r(a, b, self.eta)
        vx = sparse.v(a, b, self.eta)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = sparse.A(a, b, self.eta) - sparse.A(self.a, self.b, self.eta)
        return A.mean()

    def beliefs_measure(self, ax, f):
        mu_0 = gaussian_measure(0, np.sqrt(ax), f)
        mu_1 = gaussian_measure(
            ax * self.mean, np.sqrt(ax + (ax**2) * self.var), f
        )
        mu = (1 - self.rho) * mu_0 + self.rho * mu_1
        return mu

    def measure(self, f):
        g = gaussian_measure(self.mean, self.sigma, f)
        return (1 - self.rho) * f(0) + self.rho * g
