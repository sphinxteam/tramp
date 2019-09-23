import numpy as np
from ..base import Channel


def check_rotation(R):
    if (R.shape[0] != R.shape[1]):
        raise ValueError(f"R.shape = {R.shape}")
    N = R.shape[0]
    if not np.allclose(R @ R.T, np.identity(N)):
        raise ValueError("R not a rotation")


class RotationChannel(Channel):
    def __init__(self, R, R_name="R"):
        check_rotation(R)
        self.R_name = R_name
        self.N = R.shape[0]
        self.repr_init()
        self.R = R

    def sample(self, Z):
        X = self.R @ Z
        return X

    def math(self):
        return r"$"+self.R_name+"$"

    def second_moment(self, tau):
        return tau

    def compute_forward_message(self, az, bz, ax, bx):
        # x = R z
        ax_new = az
        bx_new = self.R @ bz
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        # z = R.T x
        az_new = ax
        bz_new = self.R.T @ bx
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        az_new = ax
        return az_new

    def log_partition(self, az, bz, ax, bx):
        b = bz + self.R.T @ bx
        a = az + ax
        logZ = 0.5 * np.sum(b**2 / a) + 0.5 * self.N * np.log(2 * np.pi / a)
        return logZ

    def free_energy(self, az, ax, tau):
        a = ax + az
        A = 0.5*(a*tau - 1 + np.log(2*np.pi / a))
        return A
