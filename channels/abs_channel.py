import numpy as np
from scipy.stats import norm
from scipy.special import erfcx
from ..base import Channel
import logging
from scipy.integrate import quad


def phi_0(x):
    return 0.5 * erfcx(-x / np.sqrt(2))


def phi_1(x):
    return x * phi_0(x) + 1 / np.sqrt(2 * np.pi)


def phi_2(x):
    return (1 + x**2) * phi_0(x) + x / np.sqrt(2 * np.pi)


class AbsChannel(Channel):
    def __init__(self):
        self.repr_init()

    def sample(self, Z):
        X = np.abs(Z)
        return X

    def math(self):
        return r"$\mathrm{abs}$"

    def second_moment(self, tau):
        return tau

    def check_params(self, ax, az, tau = None):
        if (ax <= 0):
            logging.warn(f"in AbsChannel negative ax={ax}")
            ax = 1e-11
        if (az <= 0):
            logging.warn(f"in AbsChannel negative az={az}")
            az = 1e-11
        if tau and (az < 1 / tau):
            logging.warn(f"in AbsChannel az={az}<1/tau={1/tau}")
            az = 1 / tau + 1e-11
        return ax, az

    def compute_forward_posterior(self, az, bz, ax, bx):
        ax, az = self.check_params(ax, az)
        # estimate x from x = abs(z)
        a = ax + az
        x_pos = (bx + bz) / np.sqrt(a)
        x_neg = (bx - bz) / np.sqrt(a)
        u0 = phi_0(x_pos) + phi_0(x_neg)
        u1 = (phi_1(x_pos) + phi_1(x_neg)) / np.sqrt(a)
        u2 = (phi_2(x_pos) + phi_2(x_neg)) / a
        v = u2 / u0 - (u1 / u0)**2
        rx = u1 / u0
        vx = np.mean(v)
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        ax, az = self.check_params(ax, az)
        # estimate z from x = abs(z)
        a = ax + az
        x_pos = (bx + bz) / np.sqrt(a)
        x_neg = (bx - bz) / np.sqrt(a)
        u0 = phi_0(x_pos) + phi_0(x_neg)
        u1 = (phi_1(x_pos) - phi_1(x_neg)) / np.sqrt(a)
        u2 = (phi_2(x_pos) + phi_2(x_neg)) / a
        v = u2 / u0 - (u1 / u0)**2
        rz = u1 / u0
        vz = np.mean(v)
        return rz, vz

    def beliefs_measure(self, az, ax, tau, f):
        ax, az = self.check_params(ax, az, tau)
        raise NotImplementedError

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, np.abs(z))
        return quad(integrand, zmin, zmax)[0]
