import numpy as np
from ..base_channel import Channel
from scipy.special import logsumexp, softmax
from tramp.utils.linear_region import LinearRegion


class PiecewiseLinearChannel(Channel):
    def __init__(self, name, regions):
        self.repr_init()
        self.name = name
        self.regions = [LinearRegion(**region) for region in regions]
        self.n_regions = len(regions)

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
        ps = softmax(As, axis=0)
        r = sum(p*r for p, r in zip(ps, rs))
        Dr = sum(
            ps[i]*ps[j]*(rs[i] - rs[j])**2
            for i in range(self.n_regions)
            for j in range(i+1, self.n_regions)
        )
        v = sum(p*v for p, v in zip(ps, vs)) + Dr
        v = v.mean()
        return r, v

    def compute_forward_posterior(self, az, bz, ax, bx):
        rs = [region.forward_mean(az, bz, ax, bx) for region in self.regions]
        vs = [region.forward_variance(az, bz, ax, bx)
              for region in self.regions]
        As = [region.log_partitions(az, bz, ax, bx) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def compute_backward_posterior(self, az, bz, ax, bx):
        rs = [region.backward_mean(az, bz, ax, bx) for region in self.regions]
        vs = [region.backward_variance(az, bz, ax, bx)
              for region in self.regions]
        As = [region.log_partitions(az, bz, ax, bx) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def compute_log_partition(self, az, bz, ax, bx):
        As = [region.log_partitions(az, bz, ax, bx) for region in self.regions]
        A = logsumexp(As, axis=0)
        return A.sum()

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
        neg = dict(zmin=-np.inf, zmax=0, slope=slope, x0=0)
        pos = dict(zmin=0, zmax=np.inf, slope=1, x0=0)
        super().__init__(name="l-relu", regions=[pos, neg])


class SgnChannel(PiecewiseLinearChannel):
    def __init__(self):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=-1)
        pos = dict(zmin=0, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(name="sgn", regions=[pos, neg])


class AbsChannel(PiecewiseLinearChannel):
    def __init__(self):
        neg = dict(zmin=-np.inf, zmax=0, slope=-1, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=+1, x0=0)
        super().__init__(name="abs", regions=[pos, neg])


class AsymmetricAbsChannel(PiecewiseLinearChannel):
    def __init__(self, shift=1e-4):
        self.shift = shift
        neg = dict(zmin=-np.inf, zmax=shift, slope=-1, x0=0)
        pos = dict(zmin=shift, zmax=+np.inf, slope=+1, x0=0)
        super().__init__(name="a-abs", regions=[pos, neg])


class ReluChannel(PiecewiseLinearChannel):
    def __init__(self):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=1, x0=0)
        super().__init__(name="relu", regions=[pos, neg])


class HardTanhChannel(PiecewiseLinearChannel):
    def __init__(self):
        neg = dict(zmin=-np.inf, zmax=-1, slope=0, x0=-1)
        mid = dict(zmin=-1, zmax=+1, slope=1, x0=0)
        pos = dict(zmin=1, zmax=np.inf, slope=0, x0=1)
        super().__init__(name="h-tanh", regions=[pos, mid, neg])


class HardSigmoidChannel(PiecewiseLinearChannel):
    def __init__(self):
        L = 2.5
        neg = dict(zmin=-np.inf, zmax=-L, slope=0, x0=0)
        mid = dict(zmin=-L, zmax=+L, slope=1/(2*L), x0=0.5)
        pos = dict(zmin=L, zmax=np.inf, slope=0, x0=1)
        super().__init__(name="h-sigm", regions=[pos, mid, neg])


class SymmetricDoorChannel(PiecewiseLinearChannel):
    def __init__(self, width):
        self.width = width
        neg = dict(zmin=-np.inf, zmax=-width, slope=0, x0=+1)
        mid = dict(zmin=-width, zmax=+width, slope=0, x0=-1)
        pos = dict(zmin=+width, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(name="door", regions=[pos, mid, neg])
