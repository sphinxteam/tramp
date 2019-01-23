import numpy as np
from scipy.stats import norm
from ..base import Channel


def compute_sng_proba_beliefs(az, bz, ax, bx, tau):
    # p(bz,bx|az,ax,tau) for x=sng(z)
    x = bz / np.sqrt(az)
    p_pos = norm.cdf(+x)
    p_neg = norm.cdf(-x)
    u0 = p_pos * np.exp(+bx) + p_neg * np.exp(-bx)
    sx = np.sqrt(ax)
    s_eff = np.sqrt(az * (az * tau - 1))
    return (u0 * np.exp(-0.5 * ax) *
            norm.pdf(bx, scale=sx) *
            norm.pdf(bz, scale=s_eff))

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
        p_pos = norm.cdf(+x)
        p_neg = norm.cdf(-x)
        u0 = p_pos * np.exp(+bx) + p_neg * np.exp(-bx)
        u1 = p_pos * np.exp(+bx) - p_neg * np.exp(-bx)
        phi = u1 / u0
        v = 1 - phi**2
        rx = phi
        vx = np.mean(v)
        return rx, vx

    def compute_backward_posterior(az, bz, ax, bx):
        # estimate z from x = sng(z)
        x = bz / np.sqrt(az)
        p_pos = norm.cdf(+x)
        p_neg = norm.cdf(-x)
        u0 = p_pos * np.exp(+bx) + p_neg * np.exp(-bx)
        u1 = 2 * norm.pdf(x) * np.sinh(bx) / np.sqrt(az)
        phi = u1 / u0
        v = 1 / az + phi * (- bz / az - phi) #check TODO
        rz = bz / az + phi
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, ax, tau, f):
        def f_pos(bz, bx):
            return norm.cdf(+bz / np.sqrt(az)) * f(bz, bx)
        def f_neg(bz, bx):
            return norm.cdf(-bz / np.sqrt(az)) * f(bz, bx)
        s_eff = np.sqrt(az * (az * tau - 1))
        mu_pos = gaussian_measure_2d(0, s_eff, +ax, np.sqrt(ax), f_pos)
        mu_neg = gaussian_measure_2d(0, s_eff, -ax, np.sqrt(ax), f_neg)
        return mu_pos + mu_neg
