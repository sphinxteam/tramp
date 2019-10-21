import numpy as np
from .base_likelihood import Likelihood
from scipy.special import logsumexp, softmax
from tramp.utils.linear_region import LinearRegionLikelihood


class PiecewiseLinearLikelihood(Likelihood):
    def __init__(self, name, regions, y, y_name="y"):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.repr_init()
        self.name = name
        self.y = y
        self.regions = [LinearRegionLikelihood(**region) for region in regions]
        self.n_regions = len(regions)

    def sample(self, Z):
        X = sum(region.sample(Z) for region in self.regions)
        return X

    def math(self):
        return r"$\textrm{" + self.name + r"}$"

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

    def compute_backward_posterior(self, az, bz, y):
        rs = [region.backward_mean(az, bz, y) for region in self.regions]
        vs = [region.backward_variance(az, bz, y) for region in self.regions]
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        r, v = self.merge_estimates(rs, vs, As)
        return r, v

    def compute_log_partition(self, az, bz, y):
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        A = logsumexp(As, axis=0)
        return A.sum()

    def beliefs_measure(self, az, tau_z, f):
        mu = sum(
            region.beliefs_measure(az, tau_z, f) for region in self.regions
        )
        return mu

    def measure(self, y, f):
        mu = sum(region.measure(y, f) for region in self.regions)
        return mu


class SgnLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y"):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=-1)
        pos = dict(zmin=0, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(name="sgn", regions=[pos, neg], y=y, y_name=y_name)


class AbsLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y"):
        neg = dict(zmin=-np.inf, zmax=0, slope=-1, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=+1, x0=0)
        super().__init__(name="abs", regions=[pos, neg], y=y, y_name=y_name)


class AsymmetricAbsLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", shift=1e-4):
        neg = dict(zmin=-np.inf, zmax=shift, slope=-1, x0=0)
        pos = dict(zmin=shift, zmax=+np.inf, slope=+1, x0=0)
        super().__init__(name="abs", regions=[pos, neg], y=y, y_name=y_name)


class ReluLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y"):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=1, x0=0)
        super().__init__(name="relu", regions=[pos, neg], y=y, y_name=y_name)


class LeakyReluLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, slope, y, y_name="y"):
        self.slope = slope
        neg = dict(zmin=-np.inf, zmax=0, slope=slope, x0=0)
        pos = dict(zmin=0, zmax=np.inf, slope=1, x0=0)
        super().__init__(name="l-relu", regions=[pos, neg], y=y, y_name=y_name)


class HardTanhLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y"):
        neg = dict(zmin=-np.inf, zmax=-1, slope=0, x0=-1)
        mid = dict(zmin=-1, zmax=+1, slope=1, x0=0)
        pos = dict(zmin=+1, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(
            name="h-tanh", regions=[pos, mid, neg], y=y, y_name=y_name
        )


class HardSigmoidLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y"):
        l = 2.5
        neg = dict(zmin=-np.inf, zmax=-l, slope=0, x0=0)
        mid = dict(zmin=-l, zmax=+l, slope=1/(2*l), x0=0.5)
        pos = dict(zmin=l, zmax=np.inf, slope=0, x0=1)
        super().__init__(
            name="h-sigm", regions=[pos, mid, neg], y=y, y_name=y_name
        )


class SymmetricDoorLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, width, y, y_name="y"):
        self.width = width
        neg = dict(zmin=-np.inf, zmax=-width, slope=0, x0=+1)
        mid = dict(zmin=-width, zmax=+width, slope=0, x0=-1)
        pos = dict(zmin=+width, zmax=+np.inf, slope=0, x0=+1)
        super().__init__(
            name="door", regions=[pos, mid, neg], y=y, y_name=y_name
        )
