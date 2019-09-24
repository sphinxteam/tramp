import numpy as np
from ..base_channel import Channel
from tramp.utils.misc import merge_estimates, hard_tanh
from tramp.utils.truncated_normal import TruncatedNormal
from tramp.utils.integration import gaussian_measure
from scipy.integrate import quad


class HardTanhChannel(Channel):
    def __init__(self):
        self.repr_init()
        self.pos = TruncatedNormal(zmin=-np.inf, zmax=-1, slope=0, z0=-1)
        self.mid = TruncatedNormal(zmin=-1, zmax=+1, slope=1, z0=0)
        self.neg = TruncatedNormal(zmin=1, zmax=np.inf, slope=0, z0=1)

    def sample(self, Z):
        X = hard_tanh(Z)
        return X

    def math(self):
        return r"$\textrm{h-tanh}$"

    def second_moment(self, tau_z):
        # TODO explicit formula
        tau_x = gaussian_measure(0, np.sqrt(tau_z), f = lambda z: hard_tanh(z)**2)
        return tau_x

    def compute_forward_posterior(self, az, bz, ax, bx):
        r_pos, v_pos, A_pos = self.pos.forward(az, bz, ax, bx)
        r_mid, v_mid, A_mid = self.mid.forward(az, bz, ax, bx)
        r_neg, v_neg, A_neg = self.neg.forward(az, bz, ax, bx)
        r_int, v_int, A_int = merge_estimates(
            r_pos, v_pos, A_pos, r_neg, v_neg, A_neg
        )
        r, v, A = merge_estimates(
            r_int, v_int, A_int, r_mid, v_mid, A_mid
        )
        v = np.mean(v)
        return r, v

    def compute_backward_posterior(self, az, bz, ax, bx):
        r_pos, v_pos, A_pos = self.pos.backward(az, bz, ax, bx)
        r_mid, v_mid, A_mid = self.mid.backward(az, bz, ax, bx)
        r_neg, v_neg, A_neg = self.neg.backward(az, bz, ax, bx)
        r_int, v_int, A_int = merge_estimates(
            r_pos, v_pos, A_pos, r_neg, v_neg, A_neg
        )
        r, v, A = merge_estimates(
            r_int, v_int, A_int, r_mid, v_mid, A_mid
        )
        v = np.mean(v)
        return r, v

    def beliefs_measure(self, az, ax, tau_z, f):
        raise NotImplementedError

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, hard_tanh(z))
        return quad(integrand, zmin, zmax)[0]
