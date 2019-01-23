import numpy as np
from scipy.special import erfcx
from scipy.stats import norm
from ..base import Likelihood
from ..utils.integration import gaussian_measure
from scipy.integrate import quad


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

    def compute_backward_posterior(self, az, bz, y):
        x = y * bz / np.sqrt(az)
        rz = psi(x) * y / np.sqrt(az)
        v = psi_prime(x) / az
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, tau, f):
        def f_pos(bz):
            return norm.cdf(+bz / np.sqrt(az)) * f(bz, +1)
        def f_neg(bz):
            return norm.cdf(-bz / np.sqrt(az)) * f(bz, -1)
        s = np.sqrt(az * (az * tau - 1))
        mu_pos = gaussian_measure(0, s, f_pos)
        mu_neg = gaussian_measure(0, s, f_neg)
        return mu_pos + mu_neg

    def measure(self, y, f, max = 10):
        if (y>0):
            return quad(f, 0, max)[0]
        else:
            return quad(f, -max, 0)[0]
