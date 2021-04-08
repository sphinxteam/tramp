import numpy as np
from .base_prior import Prior
from ..utils.integration import gaussian_measure


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GaussBernoulliPrior(Prior):
    def __init__(self, size, rho=0.5, mean=0, var=1):
        self.size = size
        self.rho = rho
        self.mean = mean
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = mean / var
        self.log_odds = np.log(rho / (1 - rho))

    def sample(self):
        X_gauss = self.mean + self.sigma * np.random.standard_normal(self.size)
        X_bernoulli = np.random.binomial(n=1, size=self.size, p=self.rho)
        X = X_gauss * X_bernoulli
        return X

    def math(self):
        return r"$\rho$"

    def second_moment(self):
        return self.rho * (self.mean**2 + self.var)

    def compute_forward_posterior(self, ax, bx):
        a = self.a + ax
        b = self.b + bx
        phi = 0.5 * (b**2 / a - self.b**2 / self.a + np.log(self.a / a))
        zeta = phi + self.log_odds
        s = sigmoid(zeta)
        s_prime = s * (1 - s)
        rx = (b / a) * s
        v = (1 / a) * s + (b / a)**2 * s_prime
        vx = np.mean(v)
        return rx, vx

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

    def compute_log_partition(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        A = 0.5 * (
            b**2 / a - self.b**2 / self.a + np.log(self.a/a)
        )
        logZ = np.sum(
            np.logaddexp(np.log(1 - self.rho), np.log(self.rho) + A)
        )
        return logZ
