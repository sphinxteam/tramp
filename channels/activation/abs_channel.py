import numpy as np
from ..base_channel import Channel
from tramp.utils.integration import gaussian_measure_2d, gaussian_measure_2d_full
from tramp.utils.misc import norm_cdf, phi_0, phi_1, phi_2, sigmoid
from scipy.integrate import quad


class AbsChannel(Channel):
    def __init__(self):
        self.repr_init()

    def sample(self, Z):
        X = np.abs(Z)
        return X

    def math(self):
        return r"$\mathrm{abs}$"

    def second_moment(self, tau_z):
        return tau_z

    def compute_forward_posterior(self, az, bz, ax, bx):
        # estimate x from x = abs(z)
        a = ax + az
        x_pos = (bx + bz) / np.sqrt(a)
        x_neg = (bx - bz) / np.sqrt(a)
        delta = phi_0(x_pos) - phi_0(x_neg)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = phi_1(x_pos) / np.sqrt(a)
        r_neg = phi_1(x_neg) / np.sqrt(a)
        v_pos = phi_2(x_pos) / a
        v_neg = phi_2(x_neg) / a
        rx = sigma_pos * r_pos + sigma_neg * r_neg
        Dx = (r_pos - r_neg)**2
        v = sigma_pos * sigma_neg * Dx + sigma_pos * v_pos + sigma_neg * v_neg
        vx = np.mean(v)
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        # estimate z from x = abs(z)
        a = ax + az
        x_pos = (bx + bz) / np.sqrt(a)
        x_neg = (bx - bz) / np.sqrt(a)
        delta = phi_0(x_pos) - phi_0(x_neg)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = + phi_1(x_pos) / np.sqrt(a) # NB: + phi'(x_pos)
        r_neg = - phi_1(x_neg) / np.sqrt(a) # NB: - phi'(x_neg)
        v_pos = phi_2(x_pos) / a
        v_neg = phi_2(x_neg) / a
        rz = sigma_pos * r_pos + sigma_neg * r_neg
        Dz = (r_pos  - r_neg)**2
        v = sigma_pos * sigma_neg * Dz + sigma_pos * v_pos + sigma_neg * v_neg
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, ax, tau_z, f):
        u_eff = np.maximum(0, az * tau_z - 1)
        sz_eff = np.sqrt(az * u_eff)

        def f_pos(bz, bx):
            a = ax + az
            x_pos = (bx + bz) / np.sqrt(a)
            return norm_cdf(x_pos) * f(bz, bx)

        def f_neg(bz, bx):
            a = ax + az
            x_neg = (bx - bz) / np.sqrt(a)
            return norm_cdf(x_neg) * f(bz, bx)

        if ax==0 or u_eff==0:
            sx_eff = np.sqrt(ax * (ax * tau_z + 1))
            mu_pos = mu_neg = gaussian_measure_2d(0, sz_eff, 0, sx_eff, f_pos)
        else:
            cov_pos = np.array([
                [ax * (ax * tau_z + 1), +ax * u_eff],
                [+ax * u_eff, az * u_eff]
            ])
            cov_neg = np.array([
                [ax * (ax * tau_z + 1), -ax * u_eff],
                [-ax * u_eff, az * u_eff]
            ])
            mu_pos = gaussian_measure_2d_full(cov_pos, 0, f_pos)
            mu_neg = gaussian_measure_2d_full(cov_neg, 0, f_neg)
        return mu_pos + mu_neg

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, np.abs(z))
        return quad(integrand, zmin, zmax)[0]
