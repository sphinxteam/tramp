import numpy as np
from ..base import Channel


def check_unitary(U):
    if (U.shape[0] != U.shape[1]):
        raise ValueError(f"U.shape = {U.shape}")
    N = U.shape[0]
    if not np.allclose(U @ U.H, np.identity(N)):
        raise ValueError("U not unitary")


class UnitaryChannel(Channel):
    def __init__(self, U, U_name="U"):
        U = np.matrix(U)
        check_unitary(U)
        self.U_name = U_name
        self.N = U.shape[0]
        self.repr_init()
        self.U = U

    def sample(self, Z):
        X = self.U @ Z
        return X

    def math(self):
        return r"$"+self.U_name+"$"

    def second_moment(self, tau):
        return tau

    def compute_forward_message(self, az, bz, ax, bx):
        # x = U z
        ax_new = az
        bx_new = self.U @ bz
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        # z = U.H x
        az_new = ax
        bz_new = self.U.H @ bx
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        az_new = ax
        return az_new
