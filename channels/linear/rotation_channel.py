import numpy as np
from ..base_channel import Channel


def check_rotation(R):
    if (R.shape[0] != R.shape[1]):
        raise ValueError(f"R.shape = {R.shape}")
    N = R.shape[0]
    if not np.allclose(R @ R.T, np.identity(N)):
        raise ValueError("R not a rotation")


class RotationChannel(Channel):
    def __init__(self, R, name="R"):
        check_rotation(R)
        self.name = name
        self.N = R.shape[0]
        self.repr_init()
        self.R = R

    def sample(self, Z):
        X = self.R @ Z
        return X

    def math(self):
        return r"$"+self.name+"$"

    def second_moment(self, tau_z):
        return tau_z

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

    def compute_forward_state_evolution(self, az, ax, tau_z):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        az_new = ax
        return az_new

    def compute_log_partition(self, az, bz, ax, bx):
        b = bz + self.R.T @ bx
        a = az + ax
        logZ = 0.5 * np.sum(b**2 / a) + 0.5 * self.N * np.log(2 * np.pi / a)
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        a = ax + az
        I = 0.5*np.log(a*tau_z)
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A
