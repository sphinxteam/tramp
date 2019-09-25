import numpy as np
from ..base_channel import Channel
from scipy.special import logsumexp, softmax
from tramp.utils.linear_region import LinearRegion


class PiecewiseLinearChannel(Channel):
    def __init__(self, name, regions):
        self.name = name
        self.repr_init()
        self.regions = [LinearRegion(**region) for region in regions]

    def sample(self, Z):
        X = sum(region.sample(Z) for region in self.regions)
        return X

    def math(self):
        return r"$\textrm{" + self.name + r"}$"

    def second_moment(self, tau_z):
        taus = [region.second_moment(tau_z) for region in self.regions]
        ps = [region.proba_tau(tau_z) for region in self.regions]
        tau_x = sum(p*tau for p, tau in zip(ps, taus))
        return tau_x

    def merge_estimates(self, rs, vs, As):
        ps = softmax(As)
        r = sum(p*r for p, r in zip(ps, rs))
        v = (
            sum(p*v for p, v in zip(ps, vs)) +
            sum(p*(r**2) for p, r in zip(ps, rs)) -
            sum(p*r for p, r in zip(ps, rs))**2
        )
        v = v.mean()
        return r, v

    def compute_forward_posterior(self, az, bz, ax, bx):
        rs = [region.forward_mean(az, bz, ax, bx) for region in self.regions]
        vs = [region.forward_variance(az, bz, ax, bx) for region in self.regions]
        As = [region.log_partition(az, bz, ax, bx) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def compute_backward_posterior(self, az, bz, ax, bx):
        rs = [region.backward_mean(az, bz, ax, bx) for region in self.regions]
        vs = [region.backward_variance(az, bz, ax, bx) for region in self.regions]
        As = [region.log_partition(az, bz, ax, bx) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def log_partition(self, az, bz, ax, bx):
        As = [region.log_partition(az, bz, ax, bx) for region in self.regions]
        A = logsumexp(As)
        return A

    def beliefs_measure(self, az, ax, tau_z, f):
        mu = sum(
            region.beliefs_measure(az, ax, tau_z, f) for region in self.regions
        )
        return mu

    def measure(self, f, zmin, zmax):
        assert zmin < zmax
        mu = sum(region.measure(f, zmin, zmax) for region in self.regions)
        return mu


class LeakyReluChannel(PiecewiseLinearChannel):
    def __init__(self, slope):
        self.slope = slope
        self.repr_init()
        pos = dict(zmin=-np.inf, zmax=0, slope=slope, x0=0)
        neg = dict(zmin=0, zmax=np.inf, slope=1, x0=0)
        super().__init__(name="l-relu", regions=[pos, neg])


class SgnChannel(PiecewiseLinearChannel):
    def __init__(self):
        self.repr_init()
        pos = dict(zmin=-np.inf, zmax=0, slope=0, x0=-1)
        neg = dict(zmin=0, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(name="sgn", regions=[pos, neg])


class AbsChannel(PiecewiseLinearChannel):
    def __init__(self):
        self.repr_init()
        pos = dict(zmin=-np.inf, zmax=0, slope=-1, x0=0)
        neg = dict(zmin=0, zmax=+np.inf, slope=+1, x0=0)
        super().__init__(name="abs", regions=[pos, neg])


class ReluChannel(PiecewiseLinearChannel):
    def __init__(self):
        self.repr_init()
        pos = dict(zmin=-np.inf, zmax=0, slope=0, x0=0)
        neg = dict(zmin=0, zmax=+np.inf, slope=1, x0=0)
        super().__init__(name="relu", regions=[pos, neg])


class HardTanhChannel(PiecewiseLinearChannel):
    def __init__(self):
        self.repr_init()
        pos = dict(zmin=-np.inf, zmax=-1, slope=0, x0=-1)
        mid = dict(zmin=-1, zmax=+1, slope=1, x0=0)
        neg = dict(zmin=1, zmax=np.inf, slope=0, x0=1)
        super().__init__(name="h-tanh", regions=[pos, mid, neg])
