import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from ..base import Prior


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

    def forward_posterior(self, message):
        ax, bx = self._parse_message_ab(message)
        eta = bx + 0.5 * self.log_odds
        r_hat = np.tanh(eta)
        v = 1 / (np.cosh(eta)**2)
        v_hat = np.mean(v)
        return [(r_hat, v_hat)]

    def proba_beliefs(self, message):
        ax, bx = self._parse_message_ab(message)
        r, s = ax, np.sqrt(ax)
        p1, p2 = self.p_pos, self.p_neg
        return (p1 * norm.pdf(bx, loc=+r, scale=s) +
                p2 * norm.pdf(bx, loc=-r, scale=s))
