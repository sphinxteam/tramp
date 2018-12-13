import numpy as np
from scipy.special import erfcx
from scipy.stats import norm
from ..base import Likelihood


def phi_0(x):
    return 0.5 * erfcx(-x / np.sqrt(2))


def psi(x):
    phi = np.sqrt(2 * np.pi) * phi_0(x)
    return x + 1 / phi


def psi_prime(x):
    phi = np.sqrt(2 * np.pi) * phi_0(x)
    return 1 - (1 / phi) * (x + 1 / phi)


class SngLikelihood(Likelihood):
    def __init__(self, y):
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.sign(X)

    def math(self):
        return r"$\mathrm{sng}$"

    def backward_posterior(self, message):
        az, bz = self._parse_message_ab(message)
        x = self.y * bz / np.sqrt(az)
        r_hat = psi(x) * self.y / np.sqrt(az)
        v = psi_prime(x) / az
        v_hat = np.mean(v)
        return [(r_hat, v_hat)]

    def proba_beliefs(self, message):
        az, bz, tau = self._parse_message_ab_tau(message)
        x = self.y * bz / np.sqrt(az)
        s = np.sqrt(az * (az * tau - 1))
        return 2 * norm.cdf(x) * norm.pdf(bz, scale=s)
