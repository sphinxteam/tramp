import numpy as np
from ..base_channel import SOFactor


class SumChannel(SOFactor):

    def __init__(self, n_prev):
        self.n_prev = n_prev
        self.repr_init()

    def sample(self, *Zs):
        if len(Zs) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} arrays")
        X = sum(Zs)
        return X

    def math(self):
        return r"$\Sigma$"

    def second_moment(self, *tau_zs):
        if len(tau_zs) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} arrays")
        tau_z = sum(tau_zs)
        return tau_z

    def compute_forward_message(self, az, bz, ax, bx):
        "fwd message to x; for x = sum(z)"
        v_bar = sum(1 / a for a in az)
        r_bar = sum(b / a for a, b in zip(az, bz))
        ax_new = 1 / v_bar
        bx_new = r_bar / v_bar
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        "bwd message to z = {zk}; for x = sum(z)"
        v_bar = sum(1 / a for a in az)
        r_bar = sum(b / a for a, b in zip(az, bz))
        vx = 1 / ax
        rx = bx / ax
        vk = [vx + v_bar - 1 / a for a in az]
        rk = [rx - r_bar + b / a for a, b in zip(az, bz)]
        az_new = [1 / v for v in vk]
        bz_new = [r / v for v, r in zip(vk, rk)]
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        "fwd state evo to x; for x = sum(z)"
        v_bar = sum(1 / a for a in az)
        ax_new = 1 / v_bar
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        "bwd state evo to z = {zk}; for x = sum(z)"
        v_bar = sum(1 / a for a in az)
        vx = 1 / ax
        vk = [vx + v_bar - 1 / a for a in az]
        az_new = [1 / v for v in vk]
        return az_new
