import numpy as np
from .base_ensemble import Ensemble
from .gaussian_ensemble import GaussianEnsemble


class ComplexGaussianEnsemble(Ensemble):
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.repr_init()
        self.GE = GaussianEnsemble(M, N)

    def generate(self):
        """Generate complex gaussian iid matrix.

        Returns
        -------
        - X : complex array of shape (M, N)
            X.real and X.imag ~ iid N(var = 1/N)
        """
        X = self.GE.generate() + 1j*self.GE.generate()
        return X
