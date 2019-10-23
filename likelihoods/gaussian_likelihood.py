import numpy as np
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure


class GaussianLikelihood(Likelihood):
    def __init__(self, y, var=1, y_name="y"):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.var = var
        self.repr_init()
        self.y = y
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = None if y is None else y / var

    def sample(self, X):
        noise = self.sigma * np.random.standard_normal(X.shape)
        return X + noise

    def math(self):
        return r"$\mathcal{N}$"

    def compute_backward_posterior(self, az, bz, y):
        a = az + self.a
        b = bz + (y / self.var)
        rz = b / a
        vz = 1 / a
        return rz, vz

    def compute_backward_error(self, az, tau_z):
        a = az + self.a
        vz = 1 / a
        return vz

    def compute_backward_message(self, az, bz):
        az_new = self.a
        bz_new = self.b
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, tau_z):
        az_new = self.a
        return az_new

    def measure(self, y, f):
        return gaussian_measure(y, self.sigma, f)

    def compute_log_partition(self, az, bz, y):
        ay, by = self.a, self.a*y
        a = az + ay
        b = bz + by
        logZ = 0.5 * np.sum(
            b**2 / a - by**2 / ay + np.log(ay/a)
        )
        return logZ

    def compute_mutual_information(self, az, tau_z):
        a = az + self.a
        H = 0.5*np.log(2*np.pi*np.e*self.var)
        I = 0.5*np.log(a*tau_z) + H
        return I

    def compute_free_energy(self, az, tau_z):
        I = self.compute_mutual_information(az, tau_z)
        A = 0.5*az*tau_z - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A
