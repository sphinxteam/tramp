import numpy as np
from scipy.stats import norm
from ..base import Likelihood


class AbsLikelihood(Likelihood):
    def __init__(self, y):
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.abs(X)

    def math(self):
        return r"$\mathrm{abs}$"

    def backward_posterior(self, message):
        az, bz = self._parse_message_ab(message)
        r_hat = bz * np.tanh(bz * self.y)
        v = (bz / np.cosh(bz * self.y))**2
        v_hat = np.mean(v)
        return [(r_hat, v_hat)]

    def proba_beliefs(self, message):
        az, bz, tau = self._parse_message_ab_tau(message)
        a_eff = az - 1 / tau
        # TODO y can be a vector !!
        m = self.y * a_eff
        s = np.sqrt(a_eff)
        proba =  0.5 * (norm.pdf(bz, loc=+m, scale=s) +
                        norm.pdf(bz, loc=-m, scale=s))
        return proba
