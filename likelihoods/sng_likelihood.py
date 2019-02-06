import numpy as np
from ..base import Likelihood
from ..utils.integration import gaussian_measure
from ..utils.misc import phi_1, phi_2, norm_cdf
from scipy.integrate import quad
import logging


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
        rz = phi_1(x) * y / np.sqrt(az)
        v = phi_2(x) / az
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, tau, f):
        if (az <= 1 / tau):
            logging.info(f"az={az} <= 1/tau={1/tau} in {self}.beliefs_measure")
        a_eff = az * (az * tau - 1)
        s_eff = 0 if a_eff<=0 else np.sqrt(a_eff)
        def f_pos(bz):
            return norm_cdf(+bz / np.sqrt(az)) * f(bz, +1)
        def f_neg(bz):
            return norm_cdf(-bz / np.sqrt(az)) * f(bz, -1)
        mu_pos = gaussian_measure(0, s_eff, f_pos)
        mu_neg = gaussian_measure(0, s_eff, f_neg)
        return mu_pos + mu_neg

    def measure(self, y, f, max = 10):
        if (y>0):
            return quad(f, 0, max)[0]
        else:
            return quad(f, -max, 0)[0]
