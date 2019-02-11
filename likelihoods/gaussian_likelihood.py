import numpy as np
from ..base import Likelihood
from ..utils.integration import gaussian_measure


class GaussianLikelihood(Likelihood):
    def __init__(self, y, var=1, y_name="y"):
        self.y_name = y_name
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.var = var
        self.repr_init()
        self.y = y
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = y / var

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

    def compute_backward_message(self, az, bz):
        az_new = self.a
        bz_new = self.b
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, tau):
        az_new = self.a
        return az_new

    def measure(self, y, f):
        return gaussian_measure(y, self.sigma, f)
