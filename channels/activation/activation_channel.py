import numpy as np
from ..base_channel import Channel
from tramp.utils.integration import gaussian_measure
from scipy.integrate import quad


class ActivationChannel(Channel):
    def __init__(self, func):
        self.name = func.__name__
        self.repr_init()
        self.func = func
        self.vector_forward_posterior = np.vectorize(
            self.scalar_forward_posterior
        )
        self.vector_backward_posterior = np.vectorize(
            self.scalar_backward_posterior
        )

    def sample(self, Z):
        X = self.func(Z)
        return X

    def math(self):
        return r"$\mathrm{" + self.name + r"}$"

    def second_moment(self, tau_z):
        tau_x = gaussian_measure(0, np.sqrt(tau_z), f = lambda z: self.func(z)**2)
        return tau_x

    def scalar_forward_posterior(self, az, bz, ax, bx):
        def belief(z, x):
            L = -0.5 * ax * (x**2) + bx * x - 0.5 * az * (z**2) + bz * z
            return np.exp(L)
        def x_belief(z, x):
            return x * belief(z, x)
        def x2_belief(z, x):
            return (x**2) * belief(z, x)
        zmin = bz / az - 10 / np.sqrt(az)
        zmax = bz / az + 10 / np.sqrt(az)
        Z = self.measure(belief, zmin, zmax)
        rx = self.measure(x_belief, zmin, zmax) / Z
        x2 = self.measure(x2_belief, zmin, zmax) / Z
        vx = x2 - rx**2
        return rx, vx

    def scalar_backward_posterior(self, az, bz, ax, bx):
        def belief(z, x):
            L = -0.5 * ax * (x**2) + bx * x - 0.5 * az * (z**2) + bz * z
            return np.exp(L)
        def z_belief(z, x):
            return z * belief(z, x)
        def z2_belief(z, x):
            return (z**2) * belief(z, x)
        zmin = bz / az - 10 / np.sqrt(az)
        zmax = bz / az + 10 / np.sqrt(az)
        Z = self.measure(belief, zmin, zmax)
        rz = self.measure(z_belief, zmin, zmax) / Z
        z2 = self.measure(z2_belief, zmin, zmax) / Z
        vz = z2 - rz**2
        return rz, vz

    def compute_forward_posterior(self, az, bz, ax, bx):
        rx, vx = self.vector_forward_posterior(az, bz, ax, bx)
        vx = vx.mean()
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        rz, vz = self.vector_backward_posterior(az, bz, ax, bx)
        vz = vz.mean()
        return rz, vz

    def beliefs_measure(self, az, ax, tau_z, f):
        raise NotImplementedError

    def measure(self, f, zmin, zmax):
        def integrand(z):
            return f(z, self.func(z))
        return quad(integrand, zmin, zmax)[0]
