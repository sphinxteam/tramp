import numpy as np
from scipy.special import erfcx
from scipy.stats import norm
from ..base import Likelihood
from ..utils.integration import gaussian_measure
from scipy.integrate import quad
import logging
import warnings

def phi_0(x):
    "Computes N(x)/Phi(x)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.sqrt(2 * np.pi) * 0.5 * erfcx(-x / np.sqrt(2))
    return 1./d

def psi(x):
    phi = phi_0(x)
    return x + phi

def psi_prime(x):
    phi = phi_0(x)
    return 1 - phi * (x + phi)


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
        if (az <= 1 / tau):
            logging.info(f"az={az} <= 1/tau={1/tau} in {self}.beliefs_measure")
        a_eff = az * (az * tau - 1)
        s_eff = 0 if a_eff<=0 else np.sqrt(a_eff)
        def f_pos(bz):
            return norm.cdf(+bz / np.sqrt(az)) * f(bz, +1)
        def f_neg(bz):
            return norm.cdf(-bz / np.sqrt(az)) * f(bz, -1)
        mu_pos = gaussian_measure(0, s_eff, f_pos)
        mu_neg = gaussian_measure(0, s_eff, f_neg)
        return mu_pos + mu_neg

    def measure(self, y, f, max = 10):
        if (y>0):
            return quad(f, 0, max)[0]
        else:
            return quad(f, -max, 0)[0]
