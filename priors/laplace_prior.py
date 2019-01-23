import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from ..base import Prior

def soft_threshold(gamma, x):
    coeff = 1 - gamma / np.linalg.norm(x)
    return np.maximum(0, coeff) * x

def grad_soft_threshold(gamma, x):
    coeff = 1 - gamma / np.linalg.norm(x)
    return 1. * (coeff>0)

class LaplacePrior(Prior):
    def __init__(self, size, beta=1):
        self.size = size
        self.beta = beta
        self.repr_init()
        self.lambda = 1 / beta

    def sample(self):
        X = np.random.exponential(size=self.size, scale=self.beta)
        return X

    def math(self):
        return r"$\beta$"

    def second_moment(self):
        return 2 * self.beta**2

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * soft_threshold(self.lambda, bx)
        vx = (1 / ax) * grad_soft_threshold(self.lambda, bx)
        return rx, vx

    def beliefs_measure(self, ax, f):
        raise NotImplementedError
