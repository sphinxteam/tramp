import numpy as np
from ..base_channel import Channel
from .AMP_matrix_factorization import VAMP_matrix_factorization
from .SE_matrix_factorization import SE_matrix_factorization


class LowRankGramChannel(Channel):
    """Low rank Gram matrix x = z z^T / sqrt(N).

    Notes
    -----
    - z : array of shape (N, K)
    - x : array of shape (N, N)
    In components x_ij = z_i.z_j / sqrt(N)
    """

    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.repr_init()

    def sample(self, Z):
        if Z.shape != (self.N, self.K):
            raise ValueError("Bad shape for Z")
        X = Z @ Z.T / np.sqrt(self.N)
        return X

    def math(self):
        return r"$zz^T$"

    def second_moment(self, tau_z):
        # we ignore O(1/N^2) terms
        tau_x = self.K * tau_z * tau_z / self.N
        return tau_x

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x; for x = zz^T / sqrt(N)"
        # FIXME : derive forward posterior for matrix factorization
        # Using placeholders
        rx, vx = np.ones_like(bx), 1.
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z; for x = zz^T / sqrt(N)"
        assert bz.shape == (self.N, self.K)
        assert bx.shape == (self.N, self.N)
        VAMP = VAMP_matrix_factorization(
            K=self.K, N=self.N, model='XX',
            au_av_bu_bv=[az, az, bz, bz], ax_bx=[ax, bx], verbose=False
        )
        # posterior on z
        rz, vz, _, _ = VAMP.VAMP_training()
        return rz, vz

    def compute_forward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def compute_backward_error(self, az, ax, tau_z):
        SE = SE_matrix_factorization(
            K=self.K, N=self.N, model='XX',
            au_av=[az, az], ax=ax, verbose=False
        )
        vz = SE.main()
        return vz
