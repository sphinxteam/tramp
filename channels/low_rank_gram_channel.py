import numpy as np
from ..base import Channel
from ..utils.AMP_matrix_factorization import VAMP_matrix_factorization
from ..utils.SE_matrix_factorization import SE_matrix_factorization


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

    def second_moment(self, tau):
        # we ignore O(1/N^2) terms
        tau = self.K * tau * tau / self.N
        return tau

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x; for x = zz^T / sqrt(N)"
        assert bz.shape == (self.N, self.K)
        assert bx.shape == (self.N, self.N)
        # intermediate matrices
        xz = bz / az
        G = xz @ xz.T / np.sqrt(self.N)
        H = (G**2 +  # (N,N)
             (xz**2).sum(axis=1).reshape(-1, 1) / (az * self.N) +
             self.K / (az * az * self.N))

        # posterior on x
        rx_new = G + bx * H
        vx_new = H.mean()
        #raise NotImplementedError
        return rx_new, vx_new

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z; for x = zz^T / sqrt(N)"
        assert bz.shape == (self.N, self.K)
        assert bx.shape == (self.N, self.N)

        VAMP = VAMP_matrix_factorization(
            K=self.K, N=self.N, model='XX', au_av_bu_bv=[az, az, bz, bz], ax_bx=[ax, bx], verbose=False)
        # posterior on z
        (rz_new, vz_new, _, _) = VAMP.VAMP_training()
        return rz_new, vz_new

    def compute_forward_state_evolution(self, az, ax, tau):

        raise NotImplementedError

    def compute_backward_state_evolution(self, az, ax, tau):
        SE = SE_matrix_factorization(
            K=self.K, N=self.N, model='XX', au_av=[az, az], ax=ax, verbose=False)
        mse_x = SE.main()
        return mse_x
