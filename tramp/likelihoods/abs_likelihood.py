import numpy as np
from scipy.stats import norm
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure_2d
from ..beliefs import binary


class AbsLikelihood(Likelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.isotropic = isotropic
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.abs(X)

    def math(self):
        return r"$\mathrm{abs}$"

    def scalar_backward_mean(self, az, bz, y):
        return y * binary.r(bz * y)

    def scalar_backward_variance(self, az, bz, y):
        return (y**2) * binary.v(bz * y)

    def scalar_log_partition(self, az, bz, y):
        return -0.5*az*(y**2) + binary.A(bz * y)

    def compute_backward_posterior(self, az, bz, y):
        rz = y * binary.r(bz * y)
        vz = (y**2) * binary.v(bz * y)
        if self.isotropic:
            vz = vz.mean()
        return rz, vz

    def compute_log_partition(self, az, bz, y):
        A = -0.5*az*(y**2) + binary.A(bz * y)
        return A.mean()

    def b_measure(self, mz_hat, qz_hat, tz0_hat, f):
        def integrand(z, xi_b):
            bz = mz_hat * z + np.sqrt(qz_hat) * xi_b
            y = np.abs(z)
            return f(bz, y)
        tz0 = 1 / tz0_hat
        mu = gaussian_measure_2d(0, np.sqrt(tz0), 0, 1, integrand)
        return mu

    def bz_measure(self, mz_hat, qz_hat, tz0_hat, f):
        def integrand(z, xi_b):
            bz = mz_hat * z + np.sqrt(qz_hat) * xi_b
            y = np.abs(z)
            return z*f(bz, y)
        tz0 = 1 / tz0_hat
        mu = gaussian_measure_2d(0, np.sqrt(tz0), 0, 1, integrand)
        return mu

    def beliefs_measure(self, az, tau_z, f):
        mz_hat = az - 1 / tau_z
        assert mz_hat > 0 , "az must be greater than 1/tau_z"

        def integrand(z, xi_b):
            bz = mz_hat * z + np.sqrt(mz_hat) * xi_b
            y = np.abs(z)
            return f(bz, y)
        mu = gaussian_measure_2d(0, np.sqrt(tau_z), 0, 1, integrand)
        return mu

    def measure(self, y, f):
        return f(+y) + f(-y)
