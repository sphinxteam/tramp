import numpy as np
from ..base_channel import Channel
from tramp.utils.integration import gaussian_measure_2d_full, gaussian_measure_2d
from tramp.utils.misc import norm_cdf, phi_0, phi_1, phi_2, sigmoid, leaky_relu
from scipy.integrate import quad


class LeakyReluChannel(Channel):
    def __init__(self, slope):
        self.slope = slope
        self.repr_init()

    def sample(self, Z):
        X = leaky_relu(Z, self.slope)
        return X

    def math(self):
        return r"$\textrm{l-relu}$"

    def second_moment(self, tau_z):
        return 0.5 * (1 + self.slope**2) * tau_z

    def compute_forward_posterior(self, az, bz, ax, bx):
        # estimate x from x = leaky_relu(z)
        a_pos = az + ax
        a_neg = az + (self.slope**2) * ax
        b_pos = bz + bx
        b_neg =  - bz - self.slope * bx
        x_pos = b_pos / np.sqrt(a_pos)
        x_neg = b_neg / np.sqrt(a_neg)
        delta = phi_0(x_pos) - phi_0(x_neg) + 0.5 * np.log(a_neg / a_pos)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = phi_1(x_pos) / np.sqrt(a_pos)
        r_neg = - self.slope * phi_1(x_neg) / np.sqrt(a_neg)
        v_pos = phi_2(x_pos) / a_pos
        v_neg = (self.slope**2) * phi_2(x_neg) / a_neg
        rx = sigma_pos * r_pos + sigma_neg * r_neg
        Dx = (r_pos - r_neg)**2
        v = sigma_pos * sigma_neg * Dx + sigma_pos * v_pos + sigma_neg * v_neg
        vx = np.mean(v)
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        # estimate z from x = leaky_relu(z)
        a_pos = az + ax
        a_neg = az + (self.slope**2) * ax
        b_pos = bz + bx
        b_neg =  - bz - self.slope * bx
        x_pos = b_pos / np.sqrt(a_pos)
        x_neg = b_neg / np.sqrt(a_neg)
        delta = phi_0(x_pos) - phi_0(x_neg) + 0.5 * np.log(a_neg / a_pos)
        sigma_pos = sigmoid(+delta)
        sigma_neg = sigmoid(-delta)
        r_pos = phi_1(x_pos) / np.sqrt(a_pos)
        r_neg = - phi_1(x_neg) / np.sqrt(a_neg)
        v_pos = phi_2(x_pos) / a_pos
        v_neg = phi_2(x_neg) / a_neg
        rz = sigma_pos * r_pos + sigma_neg * r_neg
        Dz = (r_pos - r_neg)**2
        v = sigma_pos * sigma_neg * Dz + sigma_pos * v_pos + sigma_neg * v_neg
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, ax, tau_z, f):
        u_eff = np.maximum(0, az * tau_z - 1)
        a_pos = az + ax
        a_neg = az + (self.slope**2) * ax

        def f_pos(bz, bx):
            b_pos = bz + bx
            x_pos = b_pos / np.sqrt(a_pos)
            return norm_cdf(x_pos) * f(bz, bx)

        def f_neg(bz, bx):
            b_neg =  - bz - self.slope * bx
            x_neg = b_neg / np.sqrt(a_neg)
            return norm_cdf(x_neg) * f(bz, bx)

        if ax==0 or u_eff==0:
            sx_eff_pos = np.sqrt(ax * (ax * tau_z + 1))
            sx_eff_neg = np.sqrt(ax * (self.slope**2 * ax * tau_z + 1))
            sz_eff = np.sqrt(az * u_eff)
            mu_pos = gaussian_measure_2d(0, sz_eff, 0, sx_eff_pos, f_pos)
            mu_neg = gaussian_measure_2d(0, sz_eff, 0, sx_eff_neg, f_pos)
        else:
            cov_pos = np.array([
                [ax * (ax * tau_z + 1), ax * u_eff],
                [ax * u_eff, az * u_eff]
            ])
            cov_neg = np.array([
                [ax * (self.slope**2 * ax * tau_z + 1), self.slope * ax * u_eff],
                [self.slope * ax * u_eff, az * u_eff]
            ])
            mu_pos = gaussian_measure_2d_full(cov_pos, 0, f_pos)
            mu_neg = gaussian_measure_2d_full(cov_neg, 0, f_neg)
        return mu_pos + mu_neg

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, leaky_relu(z, self.slope))
        return quad(integrand, zmin, zmax)[0]
