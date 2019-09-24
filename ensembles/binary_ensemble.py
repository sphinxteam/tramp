import numpy as np
from .base_ensemble import Ensemble


class BinaryEnsemble(Ensemble):
    def __init__(self, M, N, p_pos=0.5):
        self.M = M
        self.N = N
        self.p_pos
        self.repr_init()
        self.p_neg = 1 - self.p_pos

    def generate(self):
        """Generate binary iid matrix.

        Returns
        -------
        - X : array of shape (M, N)
            Binary iid X where
            - X = + 1 / sqrt(N) with proba p_pos
            - X = - 1 / sqrt(N) with proba p_neg
        """
        p = [self.p_neg, self.p_pos]
        X = np.random.choice(
            [-1, +1], size=(self.M, self.N), replace=True, p=p
        )
        sigma_x = 1 / np.sqrt(self.N)
        X *= sigma_x
        return X
