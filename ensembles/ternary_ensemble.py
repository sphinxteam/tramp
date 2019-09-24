import numpy as np
from .base_ensemble import Ensemble


class TernaryEnsemble(Ensemble):
    def __init__(self, M, N, p_pos=0.33, p_neg=0.33):
        self.M = M
        self.N = N
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.repr_init()
        self.p_zero = 1 - (self.p_pos + self.p_neg)

    def generate(self):
        """Generate ternary iid matrix.

        Returns
        -------
        - X : array of shape (M, N)
            Ternary iid X where
            - X = + 1 / sqrt(N) with proba p_pos
            - X = 0 with proba p_zero
            - X = - 1 / sqrt(N) with proba p_neg
        """
        p = [self.p_neg, self.p_zero, self.p_pos]
        X = np.random.choice(
            [-1, 0, +1], size=(self.M, self.N), replace=True, p=p
        )
        sigma_x = 1 / np.sqrt(self.N)
        X *= sigma_x
        return X
