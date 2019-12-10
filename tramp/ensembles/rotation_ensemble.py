from .base_ensemble import Ensemble
from scipy.stats import special_ortho_group


class RotationEnsemble(Ensemble):
    def __init__(self, N):
        self.N = N
        self.repr_init()

    def generate(self):
        """Generate Haar SO(N) matrix.

        Returns
        -------
        - R : array of shape (N, N)
            R ~ Haar SO(N)
        """
        R = special_ortho_group.rvs(self.N)
        return R
