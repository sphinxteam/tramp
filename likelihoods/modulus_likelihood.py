import numpy as np
from scipy.stats import norm
from scipy.special import ive
from ..base import Likelihood


class ModulusLikelihood(Likelihood):
    def __init__(self, y):
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.absolute(X)

    def math(self):
        return r"$|.|$"

    def backward_posterior(self, message):
        az, bz = self._parse_message_ab(message)
        r = np.absolute(bz)
        z = r * self.y
        I = ive(1, z) / ive(0, z)
        r_hat = bz * (self.y / r) * I
        v = 0.5 * (self.y**2) * (1 - I**2)
        v_hat = np.mean(v)
        return [(r_hat, v_hat)]

    def proba_beliefs(self, message):
        raise NotImplementedError
