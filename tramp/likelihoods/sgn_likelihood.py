import numpy as np
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure
from ..utils.misc import norm_cdf
from ..beliefs import positive
from scipy.integrate import quad


class SgnLikelihood(Likelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.isotropic = isotropic
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.sign(X)

    def math(self):
        return r"$\mathrm{sgn}$"

    def scalar_backward_mean(self, az, bz, y):
        return y * positive.r(az, bz * y)

    def scalar_backward_variance(self, az, bz, y):
        return positive.v(az, bz * y)

    def scalar_log_partition(self, az, bz, y):
        return positive.A(az, bz * y)

    def compute_backward_posterior(self, az, bz, y):
        rz = y * positive.r(az, bz * y)
        vz =  positive.v(az, bz * y)
        if self.isotropic:
            vz = vz.mean()
        return rz, vz

    def compute_log_partition(self, az, bz, y):
        A = positive.A(az, bz * y)
        return A.mean()

    def b_measure(self, mz_hat, qz_hat, tz0_hat, f):
        mz_star = mz_hat**2 / qz_hat
        az_star = mz_star + tz0_hat
        def f_pos(bz):
            bz_star = (mz_hat / qz_hat) * bz
            p = positive.p(az_star, +bz_star)
            return p * f(bz, +1)
        def f_neg(bz):
            bz_star = (mz_hat / qz_hat) * bz
            p = positive.p(az_star, -bz_star)
            return p * f(bz, -1)
        tz0 = 1 / tz0_hat
        sz_eff = np.sqrt(qz_hat + (mz_hat**2) * tz0)
        mu_pos = gaussian_measure(0, sz_eff, f_pos)
        mu_neg = gaussian_measure(0, sz_eff, f_neg)
        return mu_pos + mu_neg

    def bz_measure(self, mz_hat, qz_hat, tz0_hat, f):
        mz_star = mz_hat**2 / qz_hat
        az_star = mz_star + tz0_hat
        def f_pos(bz):
            bz_star = (mz_hat / qz_hat) * bz
            p = positive.p(az_star, +bz_star)
            r = positive.r(az_star, +bz_star)
            return p * r * f(bz, +1)
        def f_neg(bz):
            bz_star = (mz_hat / qz_hat) * bz
            p = positive.p(az_star, -bz_star)
            r = -positive.r(az_star, -bz_star)
            return p * r * f(bz, -1)
        tz0 = 1 / tz0_hat
        sz_eff = np.sqrt(qz_hat + (mz_hat**2) * tz0)
        mu_pos = gaussian_measure(0, sz_eff, f_pos)
        mu_neg = gaussian_measure(0, sz_eff, f_neg)
        return mu_pos + mu_neg

    def beliefs_measure(self, az, tau_z, f):
        mz_hat = az - 1 / tau_z
        assert mz_hat > 0 , "az must be greater than 1/ tau_z"

        def f_pos(bz):
            p = positive.p(az, +bz)
            return p * f(bz, +1)
        def f_neg(bz):
            p = positive.p(az, -bz)
            return p * f(bz, -1)
        sz_eff = np.sqrt(mz_hat + (mz_hat**2) * tau_z)
        mu_pos = gaussian_measure(0, sz_eff, f_pos)
        mu_neg = gaussian_measure(0, sz_eff, f_neg)
        return mu_pos + mu_neg

    def measure(self, y, f, max=10):
        if (y > 0):
            return quad(f, 0, max)[0]
        else:
            return quad(f, -max, 0)[0]
