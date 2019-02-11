import numpy as np
from ..base import Prior
from ..utils.integration import gaussian_measure


class BinaryPrior(Prior):
    def __init__(self, size, p_pos=0.5):
        self.size = size
        self.p_pos = p_pos
        self.repr_init()
        self.p_neg = 1 - p_pos
        self.log_odds = np.log(self.p_pos / self.p_neg)

    def sample(self):
        p = [self.p_neg, self.p_pos]
        X = np.random.choice([-1, +1], size=self.size, replace=True, p=p)
        return X

    def math(self):
        return r"$p_\pm$"

    def second_moment(self):
        return 1.

    def compute_forward_posterior(self, ax, bx):
        eta = bx + 0.5 * self.log_odds
        rx = np.tanh(eta)
        # 1 / cosh**2 leads to overflow
        v = 1 - np.tanh(eta)**2
        vx = np.mean(v)
        return rx, vx

    def beliefs_measure(self, ax, f):
        mu_pos = gaussian_measure(+ax, np.sqrt(ax), f)
        mu_neg = gaussian_measure(-ax, np.sqrt(ax), f)
        mu = self.p_pos * mu_pos + self.p_neg * mu_neg
        return mu

    def measure(self, f):
        return self.p_pos * f(+1) + self.p_neg * f(-1)
