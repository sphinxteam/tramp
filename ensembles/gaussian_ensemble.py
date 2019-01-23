import numpy as np
from ..base import Ensemble


class GaussianEnsemble(Ensemble):
    def __init__(self):
        self.repr_init()

    def generate(self, n_samples, n_features):
        """Generate gaussian iid matrix.

        Returns
        -------
        - X : array of shape (n_samples, n_features)
            X ~ iid N(0, 1/n_features)
        """
        sigma_x = 1 / np.sqrt(n_features)
        X = sigma_x * np.random.randn(n_samples, n_features)
        return X
