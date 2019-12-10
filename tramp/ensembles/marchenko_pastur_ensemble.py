import numpy as np
from .base_ensemble import Ensemble
from scipy.integrate import quad


class MarchenkoPasturEnsemble(Ensemble):
    def __init__(self, alpha):
        self.alpha = alpha
        self.repr_init()
        # Minimal and maximal eigenvalues (bulk)
        self.z_max = (1 + np.sqrt(alpha))**2
        self.z_min = (1 - np.sqrt(alpha))**2
        self.mean_spectrum = self.measure(lambda z: z)

    def generate(self, N=1000):
        """Generate gaussian iid matrix of size N"

        Returns
        -------
        - X : array of shape (M, N)
            X ~ iid N(var = 1/N) and M = alpha N
        """
        M = int(self.alpha * N)
        sigma_x = 1 / np.sqrt(N)
        X = sigma_x * np.random.randn(M, N)
        return X

    def bulk_density(self, z):
        return np.sqrt((z - self.z_min) * (self.z_max - z))/(2*np.pi*z)

    def measure(self, f):
        atomic = max(0, 1 - self.alpha) * f(0)

        def integrand(z):
            return f(z) * self.bulk_density(z)
        bulk = quad(integrand, self.z_min, self.z_max)[0]
        return atomic + bulk

    def compute_F(self, gamma):
        F = (np.sqrt(gamma*self.z_max + 1) - np.sqrt(gamma*self.z_min + 1))**2
        return F

    def eta_transform(self, gamma):
        F = self.compute_F(gamma)
        return 1 - F/(4*gamma)

    def shannon_transform(self, gamma):
        F = self.compute_F(gamma)
        S = (
            np.log(1 + self.alpha * gamma - F/4) +
            self.alpha * np.log(1 + gamma - F/4) - F / (4*gamma)
        )
        return S
