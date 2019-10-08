import numpy as np
from ..base_channel import Channel


class BiasChannel(Channel):
    def __init__(self, bias):
        self.repr_init()
        self.bias = bias

    def sample(self, Z):
        return Z + self.bias

    def math(self):
        return r"$+$"

    def second_moment(self, tau_z):
        tau_bias = (self.bias**2).mean()
        return tau_z + tau_bias

    def compute_forward_message(self, az, bz, ax, bx):
        ax_new = az
        bx_new = bz + az * self.bias
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        az_new = ax
        bz_new = bx - ax * self.bias
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        az_new = ax
        return az_new

    def compute_log_partition(self, az, bz, ax, bx):
        b = bx + bz - ax*self.bias
        a = ax + az
        logZ = 0.5 * np.sum(
            b**2 / a + np.log(2*np.pi / a) + 2*bx*self.bias - ax*(self.bias**2)
        )
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
