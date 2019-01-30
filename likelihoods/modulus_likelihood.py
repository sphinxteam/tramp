import numpy as np
from scipy.stats import norm
from scipy.special import ive
from scipy.integrate import quad
from ..base import Likelihood
from ..utils.integration import gaussian_measure_2d
import logging


def _factor(x):
    return 2 * np.pi * x * (x>0)

class ModulusLikelihood(Likelihood):
    def __init__(self, y):
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.absolute(X)

    def math(self):
        return r"$|\cdot|$"

    def compute_backward_posterior(self, az, bz, y):
        b = np.absolute(bz)
        I = ive(1, b * y) / ive(0, b * y)
        rz = bz * (y / b) * I
        v = 0.5 * (y**2) * (1 - I**2)
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, tau, f):
        if (az <= 1 / tau):
            logging.info(f"az={az} <= 1/tau={1/tau} in {self}.beliefs_measure")
        a_eff = az * (az * tau - 1)
        s_eff = 0 if a_eff<=0 else np.sqrt(a_eff)
        def f_scaled(xi_b, xi_y):
            b = s_eff * xi_b
            y = b / az + xi_y / np.sqrt(az)
            return _factor(b) * _factor(y) * ive(0, b * y) * f(b, y)
        mu = gaussian_measure_2d(0, 1, 0, 1, f_scaled)
        return mu

    def measure(self, y, f):
        def polar_f(theta):
            return y * f(y * np.exp(theta * 1j))
        return quad(polar_f, 0, 2*np.pi)[0]
