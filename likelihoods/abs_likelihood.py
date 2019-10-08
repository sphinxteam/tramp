import numpy as np
from scipy.stats import norm
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure_2d


class AbsLikelihood(Likelihood):
    def __init__(self, y, y_name="y"):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.repr_init()
        self.y = y

    def sample(self, X):
        return np.abs(X)

    def math(self):
        return r"$\mathrm{abs}$"

    def compute_backward_posterior(self, az, bz, y):
        rz = y * np.tanh(bz * y)
        # 1 / cosh**2 leads to overflow
        v = (y**2) * (1 - np.tanh(bz * y)**2)
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, tau_z, f):
        "NB: Assumes that f(bz, y) pair in y."
        u_eff = np.maximum(0, az * tau_z - 1)
        sz_eff = np.sqrt(az * u_eff)

        def f_scaled(xi_b, xi_y):
            bz = sz_eff * xi_b
            y = bz / az + xi_y / np.sqrt(az)
            return f(bz, y)
        mu = gaussian_measure_2d(0, 1, 0, 1, f_scaled)
        return mu

    def measure(self, y, f):
        return f(+y) + f(-y)

    def compute_log_partition(self, az, bz, y):
        logZ = np.sum(
            -0.5*az*(y**2) + np.logaddexp(bz*y, -bz*y)
        )
        return logZ
