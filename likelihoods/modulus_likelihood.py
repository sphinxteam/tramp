import numpy as np
from scipy.stats import norm
from scipy.special import ive
from scipy.integrate import quad
from ..base import Likelihood
from ..utils.integration import gaussian_measure_2d, gaussian_measure
from ..utils.misc import relu
import logging


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
        b_normed = 0 + 0j if b == 0 else bz / b
        rz = b_normed * y * I
        v = 0.5 * (y**2) * (1 - I**2)
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, tau, f):
        u_eff = np.maximum(0, az * tau - 1)
        # handling special case az * tau = 1 (no integration over b)
        if u_eff == 0:
            def f_scaled_y(xi_y):
                y = xi_y / np.sqrt(az)
                coef_y = np.sqrt(2 * np.pi * az)
                return coef_y * relu(y) * f(0, y)
            return gaussian_measure(0, 1, f_scaled_y)
        # typical case u_eff > 0
        s_eff = np.sqrt(az * u_eff)
        def f_scaled(xi_b, xi_y):
            b = s_eff * xi_b
            y = b / az + xi_y / np.sqrt(az)
            coef = 2 * np.pi / np.sqrt(u_eff)
            return coef * relu(b) * relu(y) * ive(0, b * y) * f(b, y)
        return gaussian_measure_2d(0, 1, 0, 1, f_scaled)

    def measure(self, y, f):
        def polar_f(theta):
            return y * f(y * np.exp(theta * 1j))
        return quad(polar_f, 0, 2 * np.pi)[0]
