import numpy as np
from ..base import Channel
from ..utils.integration import gaussian_measure_2d
from ..utils.misc import norm_cdf, phi_1, phi_2, sigmoid
from scipy.integrate import quad


class SngChannel(Channel):
    def __init__(self):
        self.repr_init()

    def sample(self, Z):
        X = np.sign(Z)
        return X

    def math(self):
        return r"$\mathrm{sng}$"

    def second_moment(self, tau):
        return 1.

    def compute_forward_posterior(self, az, bz, ax, bx):
        # estimate x from x = sng(z)
        x = bz / np.sqrt(az)
        p_pos = norm_cdf(+x)
        p_neg = norm_cdf(-x)
        eta = bx + 0.5 * np.log(p_pos / p_neg)
        rx = np.tanh(eta)
        v = 1 - rx**2
        vx = np.mean(v)
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        # estimate z from x = sng(z)
        xz = bz / np.sqrt(az)
        p_pos = norm_cdf(+xz)
        p_neg = norm_cdf(-xz)
        delta = 2 * bx + np.log(p_pos / p_neg)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        rz_pos = +phi_1(+xz) / np.sqrt(az)
        rz_neg = -phi_1(-xz) / np.sqrt(az)
        vz_pos = phi_2(+xz) / az
        vz_neg = phi_2(-xz) / az
        rz = sigma_pos * rz_pos + sigma_neg * rz_neg
        Dz = (rz_pos - rz_neg)**2
        v = sigma_pos * sigma_neg * Dz + sigma_pos * vz_pos + sigma_neg * vz_neg
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, ax, tau, f):
        def f_pos(bz, bx):
            return norm_cdf(+bz / np.sqrt(az)) * f(bz, bx)

        def f_neg(bz, bx):
            return norm_cdf(-bz / np.sqrt(az)) * f(bz, bx)
        a_eff = az * (az * tau - 1)
        s_eff = 0 if a_eff<=0 else np.sqrt(a_eff)
        mu_pos = gaussian_measure_2d(0, s_eff, +ax, np.sqrt(ax), f_pos)
        mu_neg = gaussian_measure_2d(0, s_eff, -ax, np.sqrt(ax), f_neg)
        return mu_pos + mu_neg

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, np.sign(z))
        return quad(integrand, zmin, zmax)[0]
