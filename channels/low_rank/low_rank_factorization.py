import numpy as np
from ..base_channel import MatrixFactorization
from .AMP_matrix_factorization import VAMP_matrix_factorization
from .SE_matrix_factorization import SE_matrix_factorization


class LowRankFactorization(MatrixFactorization):
    """Low rank factorization x = u v^T / sqrt(N).

    Notes
    -----
    - u : array of shape (M, K)
    - v : array of shape (N, K)
    - x : array of shape (M, N)
    In components x_ij = u_i.v_j / sqrt(N)
    """

    def __init__(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K
        self.repr_init()

    def sample(self, U, V):
        if U.shape != (self.M, self.K):
            raise ValueError("Bad shape for U")
        if V.shape != (self.N, self.K):
            raise ValueError("Bad shape for V")
        X = U @ V.T / np.sqrt(self.N)
        return X

    def math(self):
        return r"$uv^T$"

    def second_moment(self, tau_u, tau_v):
        tau_x = self.K * tau_u * tau_v / self.N
        return tau_x

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x; for x = uv^T / sqrt(N) and z = [u, v]"
        au, av = az
        bu, bv = bz
        # FIXME : derive forward posterior for matrix factorization
        # Using placeholders
        rx, vx = np.ones_like(bx), 1.
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z; for x = uv^T / sqrt(N) and z = [u, v]"
        au, av = az
        bu, bv = bz
        assert bu.shape == (self.M, self.K)
        assert bv.shape == (self.N, self.K)
        assert bx.shape == (self.M, self.N)
        VAMP = VAMP_matrix_factorization(
            K=self.K, N=self.N, M=self.M, model='UV',
            au_av_bu_bv=[au, av, bu, bv], ax_bx=[ax, bx], verbose=False
        )
        rz_u, vz_u, rz_v, vz_v = VAMP.VAMP_training()
        # posterior for z = [u, v]
        rz = [rz_u, rz_v]
        vz = [vz_u, vz_v]
        return rz, vz

    def compute_forward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def compute_backward_error(self, az, ax, tau_z):
        au, av = az
        SE = SE_matrix_factorization(
            K=self.K, N=self.N, M=self.M, model='UV',
            au_av=[au, av], ax=ax, verbose=False
        )
        vz_u,  vz_v = SE.main()
        # error for z = [u, v]
        vz = [vz_u, vz_v]
        return vz
