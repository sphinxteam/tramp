import numpy as np
from .base_ensemble import Ensemble


class ComplexUnitaryEnsemble(Ensemble):
    def __init__(self, M, N, scale=1):
        self.M = M
        self.N = N
        self.scale = scale
        self.repr_init()

    def generate(self):
        """Generate complex unitary ensemble matrix.

        Returns
        -------
        - X : array of shape (M, N)
            X ~ e^{i phi}
            phi ~ Unif(0;2*pi)
        """
        X = np.exp(2 * np.pi * 1j * np.random.uniform(size=(self.M, self.N)))
        return X
