import numpy as np
from ..base import Factor, inv
from ..utils.AMP_matrix_factorization import VAMP_matrix_factorization
from ..utils.SE_matrix_factorization import SE_matrix_factorization


class LowRankFactorization(Factor):
    """Low rank factorization x = u v^T / sqrt(N).

    Notes
    -----
    - u : array of shape (M, K)
    - v : array of shape (N, K)
    - x : array of shape (M, N)
    In components x_ij = u_i.v_j / sqrt(N)
    """

    n_prev = 2
    n_next = 1

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
        tau = self.K * tau_u * tau_v / self.N
        return tau

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x; for x = uv^T / sqrt(N) and z = [u, v]"
        au, av = az
        bu, bv = bz
        assert bu.shape == (self.M, self.K)
        assert bv.shape == (self.N, self.K)
        assert bx.shape == (self.M, self.N)
        # intermediate matrices
        xu = bu / au
        xv = bv / av
        G = xu @ xv.T / np.sqrt(self.N)
        H = (
            G**2 +  # (N,M)
            (xu**2).sum(axis=1).reshape(-1, 1) / (av * self.N) +  # (N,1)
            (xv**2).sum(axis=1).reshape(1, -1) / (au * self.N) +  # (1,M)
            self.K / (au * av * self.N)
        )
        # posterior on x
        rx_new = G + bx * H
        vx_new = H.mean()
        return rx_new, vx_new

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z; for x = uv^T / sqrt(N) and z = [u, v]"
        au, av = az
        bu, bv = bz
        assert bu.shape == (self.M, self.K)
        assert bv.shape == (self.N, self.K)
        assert bx.shape == (self.M, self.N)

        VAMP = VAMP_matrix_factorization(
            K=self.K, N=self.N, M=self.M, model='UV', au_av_bu_bv=[au, av, bu, bv], ax_bx=[ax, bx], verbose=False)
        (rz_u, vz_u, rz_v, vz_v) = VAMP.VAMP_training()

        # posterior for z = [u, v]
        rz_new = [rz_u, rz_v]
        vz_new = [vz_u, vz_v]
        return rz_new, vz_new

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new = inv(vx) - ax
        bx_new = rx * inv(vx) - bx
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new = [inv(vk) - ak for ak, vk in zip(az, vz)]
        bz_new = [rk * inv(vk) - bk for bk, rk, vk in zip(bz, rz, vz)]
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        raise NotImplementedError

    def compute_backward_state_evolution(self, az, ax, tau):
        au, av = az
        SE = SE_matrix_factorization(
            K=self.K, N=self.N, M=self.M, model='UV', au_av=[au, av], ax=ax, verbose=False)
        mse_u, mse_v = SE.main()
        return mse_u, mse_v
