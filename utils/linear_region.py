import numpy as np
from scipy.integrate import quad
from ..base import ReprMixin
from .truncated_normal import (
    truncated_normal_mean, truncated_normal_var, truncated_normal_logZ,
    truncated_normal_proba
)
from .integration import gaussian_measure_2d, gaussian_measure_2d_full


class LinearRegion(ReprMixin):
    "Inference knowing z in [zmin, zmax] and x = x0 + slope*z"

    def __init__(self, zmin, zmax, x0, slope):
        assert zmin < zmax
        self.zmin = zmin
        self.zmax = zmax
        self.x0 = x0
        self.slope = slope
        self.repr_init()

    def x(self, z):
        return self.x0 + self.slope*z

    def indicator(self, z):
        return (self.zmin <= z) * (z < self.zmax)

    def sample(self, Z):
        # zero outside of the region
        X = self.x(Z) * self.indicator(Z)
        return X

    def get_r0_v0(self, az, bz, ax, bx):
        a = az + self.slope**2 * ax
        b = bz + self.slope * (bx - ax * self.x0)
        r0 = b / a
        v0 = 1 / a
        return r0, v0

    def backward_mean(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        rz = truncated_normal_mean(r0, v0, self.zmin, self.zmax)
        return rz

    def backward_variance(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        vz = truncated_normal_var(r0, v0, self.zmin, self.zmax)
        return vz

    def forward_mean(self, az, bz, ax, bx):
        rz = self.backward_mean(az, bz, ax, bx)
        rx = self.slope * rz + self.x0
        return rx

    def forward_variance(self, az, bz, ax, bx):
        vz = self.backward_variance(az, bz, ax, bx)
        vx = self.slope**2 * vz
        return vx

    def log_partitions(self, az, bz, ax, bx):
        "Element-wise log_partition"
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        trunc_logZ = truncated_normal_logZ(r0, v0, self.zmin, self.zmax)
        logZ = trunc_logZ  - 0.5*ax*self.x0**2 + bx*self.x0
        return logZ

    def second_moment(self, tau_z):
        rz = truncated_normal_mean(0, tau_z, self.zmin, self.zmax)
        vz = truncated_normal_var(0, tau_z, self.zmin, self.zmax)
        rx = self.slope * rz + self.x0
        vx = self.slope**2 * vz
        tau_x = rx**2 + vx
        return tau_x

    def proba_tau(self, tau_z):
        p = truncated_normal_proba(0, tau_z, self.zmin, self.zmax)
        return p

    def proba_ab(self, az, bz, ax, bx):
        r0, v0 = self.get_r0_v0(az, bz, ax, bx)
        p = truncated_normal_proba(r0, v0, self.zmin, self.zmax)
        return p

    def beliefs_measure(self, az, ax, tau_z, f):
        u_eff = np.maximum(0, az * tau_z - 1)
        mean_x = ax*self.x0

        def integrand(bz, bx):
            return self.proba_ab(az, bz, ax, bx) * f(bz, bx)

        if ax == 0 or u_eff == 0 or self.slope == 0:
            sz_eff = np.sqrt(az * u_eff)
            sx_eff = np.sqrt(ax * (self.slope**2 * ax * tau_z + 1))
            mu = gaussian_measure_2d(0, sz_eff, mean_x, sx_eff, integrand)
        else:
            cov = np.array([
                [az * u_eff, self.slope * ax * u_eff],
                [self.slope * ax * u_eff, ax * (self.slope**2 * ax * tau_z + 1)]
            ])
            mean = np.array([0, mean_x])
            mu = gaussian_measure_2d_full(cov, mean, integrand)
        return mu

    def measure(self, f, zmin, zmax):
        assert zmin < zmax

        def integrand(z):
            return f(z, self.x(z))
        # integrate over intersection of [zmin,zmax] and [self.zmin,self.zmax]
        inter_min = max(zmin, self.zmin)
        inter_max = min(zmax, self.zmax)
        # void intersection
        if inter_min >= inter_max:
            return 0
        return quad(integrand, inter_min, inter_max)[0]
