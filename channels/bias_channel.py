import numpy as np
from ..base import Channel


class BiasChannel(Channel):
    def __init__(self, bias):
        self.repr_init()
        self.bias = bias

    def sample(self, Z):
        return Z + self.bias

    def math(self):
        return r"$+$"

    def second_moment(self, tau):
        tau_bias = (self.bias**2).mean()
        return tau + tau_bias

    def compute_forward_message(self, az, bz, ax, bx):
        ax_new = az
        bx_new = bz + az * self.bias
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        az_new = ax
        bz_new = bx - ax * self.bias
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        az_new = ax
        return az_new
