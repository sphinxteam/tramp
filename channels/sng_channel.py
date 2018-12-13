import numpy as np
from scipy.stats import norm
from ..base import Channel


def compute_sng_forward_posterior(az, bz, ax, bx):
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

def compute_sng_backward_posterior(az, bz, ax, bx):
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

    def forward_posterior(self, message):
        az, bz, ax, bx = self._parse_message_ab(message)
        rx, vx = compute_sng_forward_posterior(az, bz, ax, bx)
        return [(rx, vx)]

    def backward_posterior(self, message):
        az, bz, ax, bx = self._parse_message_ab(message)
        rz, vz = compute_sng_backward_posterior(az, bz, ax, bx)
        return [(rz, vz)]

    def proba_beliefs(self, message):
        az, bz, ax, bx, tau = self._parse_message_ab_tau(message)
        proba  = compute_sng_proba_beliefs(az, bz, ax, bx, tau)
        return proba
