import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from ..base import Prior


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GaussBernouilliPrior(Prior):
    def __init__(self, size, rho=0.5, mean=0, var=1):
        self.size = size
        self.rho = rho
        self.mean = mean
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = mean / var * np.ones(size)
        self.log_odds = np.log(rho / (1 - rho))

    def sample(self):
        X_gauss = self.mean + self.sigma * np.random.standard_normal(self.size)
        X_bernouilli = np.random.binomial(n=1, size=self.size, p=self.rho)
        X = X_gauss * X_bernouilli
        return X

    def math(self):
        return r"$\rho\mathcal{N}$"

    def second_moment(self):
        return self.rho * (self.mean**2 + self.var)

    def forward_posterior(self, message):
        ax, bx = self._parse_message_ab(message)
        a = self.a + ax
        b = self.b + bx
        phi = 0.5 * (b**2 / a - self.b**2 / self.a + np.log(self.a / a))
        zeta = phi + self.log_odds
        s = sigmoid(zeta)
        s_prime = s * (1 - s)
        r_hat = (b / a) * s
        v = (1 / a) * s + (b / a)**2 * s_prime
        v_hat = np.mean(v)
        return [(r_hat, v_hat)]

    def proba_beliefs(self, message):
        ax, bx = self._parse_message_ab(message)
        r1 = ax * self.mean
        s1 = np.sqrt(ax * (ax * self.var + 1))
        r2 = 0
        s2 = np.sqrt(ax)
        p1, p2 = self.rho, 1 - self.rho
        return (p1 * norm.pdf(bx, loc=r1, scale=s1) +
                p2 * norm.pdf(bx, loc=r2, scale=s2))
