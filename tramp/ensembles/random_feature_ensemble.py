from .base_ensemble import Ensemble
import numpy as np


def relu(x):
    return np.where(x < 0, 0, x)

def relu_zero_mean(x):
    mean = 1 / np.sqrt(2*np.pi)
    return np.where(x < 0, 0, x) - mean

def abs_zero_mean(x):
    mean = np.sqrt(2 / np.pi)
    return np.abs(x) - mean


ACTIVATIONS = {
    "relu": relu,
    "relu_zero_mean": relu_zero_mean,
    "abs_zero_mean": abs_zero_mean,
    "abs": np.abs,
    "tanh": np.tanh,
    "sgn": np.sign
}


class RandomFeatureEnsemble(Ensemble):
    def __init__(self, M, N, f):
        self.M = M
        self.N = N
        self.f = ACTIVATIONS[f]
        self.repr_init()

    def generate(self):
        """Generate Random Feature matrix.

        Returns
        -------
        - Phi : array of shape (M, N)
            Z: (N, N) ~ iid N(0, 1) / sqrt(N)
            W: (M, N) ~ iid N(0, 1)
            X: (M, N) = f(WZ) / sqrt(N)
        """
        Z = np.random.randn(self.N, self.N) / np.sqrt(self.N)
        W = np.random.randn(self.M, self.N)
        X = self.f(W @ Z) / np.sqrt(self.N)
        return X
