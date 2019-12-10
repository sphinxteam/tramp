import numpy as np
from .base_ensemble import Ensemble


class GaussianEnsemble(Ensemble):
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.repr_init()

    def generate(self):
        """Generate gaussian iid matrix.

        Returns
        -------
        - X : array of shape (M, N)
            X ~ iid N(var = 1/N)
        """
        sigma_x = 1 / np.sqrt(self.N)
        X = sigma_x * np.random.randn(self.M, self.N)
        return X
