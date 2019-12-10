from .base_ensemble import Ensemble
from scipy.stats import unitary_group


class UnitaryEnsemble(Ensemble):
    def __init__(self, N):
        self.N = N
        self.repr_init()

    def generate(self):
        """Generate Haar U(N) matrix.

        Returns
        -------
        - U : array of shape (N, N)
            U ~ Haar U(N)
        """
        U = unitary_group.rvs(self.N)
        return U
