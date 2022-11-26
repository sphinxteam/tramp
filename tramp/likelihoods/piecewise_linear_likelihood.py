import numpy as np
from scipy.special import logsumexp, softmax
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure, gaussian_measure_2d
from ..beliefs import truncated


# TODO: ConstantRegionLikelihood (slope=0)
class LinearRegionLikelihood(Likelihood):
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

    def strict_indicator(self, z):
        return (self.zmin < z) * (z < self.zmax)

    def sample(self, Z):
        # zero outside of the region
        X = self.x(Z) * (self.zmin <= Z) * (Z < self.zmax)
        return X

    def backward_mean(self, az, bz, y):
        if self.slope == 0:
            rz = truncated.r(az, bz, self.zmin, self.zmax)
        else:
            rz = (y - self.x0) / self.slope
        rz = np.where(self.contains(y), rz, 0)
        return rz

    def backward_variance(self, az, bz, y):
        if self.slope == 0:
            vz = truncated.v(az, bz, self.zmin, self.zmax)
        else:
            vz = 0
        vz = np.where(self.contains(y), vz, 0)
        return vz

    def contains(self, y):
        if self.slope == 0:
            y_in_region = (y == self.x0)
        else:
            z = (y - self.x0) / self.slope
            y_in_region = self.strict_indicator(z)
        return y_in_region

    def log_partitions(self, az, bz, y):
        "Element-wise log_partition"
        if self.slope == 0:
            logZ = truncated.A(az, bz, self.zmin, self.zmax)
        else:
            z = (y - self.x0) / self.slope
            logZ = -0.5*az*(z**2) + bz*z -np.log(self.slope)
        logZ = np.where(self.contains(y), logZ, -np.inf)
        return logZ

    def b_measure(self, mz_hat, qz_hat, tz0_hat, f):
        if self.slope == 0:
            mz_star = mz_hat**2 / qz_hat
            az_star = mz_star + tz0_hat

            def p_times_f(bz):
                bz_star = (mz_hat / qz_hat) * bz
                p = truncated.p(az_star, bz_star, self.zmin, self.zmax)
                return p * f(bz, self.x0)
            tz0 = 1 / tz0_hat
            sz_eff = np.sqrt(qz_hat + (mz_hat**2) * tz0)
            mu = gaussian_measure(0, sz_eff, p_times_f)
        else:
            def integrand(z, xi_b):
                """
                The integrand must return 1_R(z)*f(bz, y) but f(bz, y) is slow to
                compute. If z is outside R we directly return 0 to bypass the
                computation of f(bz, y).
                """
                if not self.strict_indicator(z):
                    return 0
                bz = mz_hat * z + np.sqrt(qz_hat) * xi_b
                y = self.x(z)
                return f(bz, y)
            tz0 = 1 / tz0_hat
            mu = gaussian_measure_2d(0, np.sqrt(tz0), 0, 1, integrand)
        return mu

    def bz_measure(self, mz_hat, qz_hat, tz0_hat, f):
        if self.slope == 0:
            mz_star = mz_hat**2 / qz_hat
            az_star = mz_star + tz0_hat

            def rp_times_f(bz):
                bz_star = (mz_hat / qz_hat) * bz
                r = truncated.r(az_star, bz_star, self.zmin, self.zmax)
                p = truncated.p(az_star, bz_star, self.zmin, self.zmax)
                return r * p * f(bz, self.x0)
            tz0 = 1 / tz0_hat
            sz_eff = np.sqrt(qz_hat + (mz_hat**2) * tz0)
            mu = gaussian_measure(0, sz_eff, rp_times_f)
        else:
            def integrand(z, xi_b):
                """
                The integrand must return 1_R(z)*z*f(bz, y) but f(bz, y) is slow
                to compute. If z is outside R we directly return 0 to bypass the
                computation of f(bz, y).
                """
                if not self.strict_indicator(z):
                    return 0
                bz = mz_hat * z + np.sqrt(qz_hat) * xi_b
                y = self.x(z)
                return z*f(bz, y)
            tz0 = 1 / tz0_hat
            mu = gaussian_measure_2d(0, np.sqrt(tz0), 0, 1, integrand)
        return mu

    def beliefs_measure(self, az, tau_z, f):
        mz_hat = az - 1 / tau_z
        assert mz_hat > 0 , "az must be greater than 1/ tau_z"

        if self.slope == 0:
            def integrand(bz):
                p = truncated.p(az, bz, self.zmin, self.zmax)
                return p * f(bz, self.x0)
            sz_eff = np.sqrt(mz_hat + (mz_hat**2) * tau_z)
            mu = gaussian_measure(0, sz_eff, integrand)
        else:
            def integrand(z, xi_b):
                """
                The integrand must return 1_R(z)*f(bz, y) but f(bz, y) is slow to
                compute. If z is outside R we directly return 0 to bypass the
                computation of f(bz, y).
                """
                if not self.strict_indicator(z):
                    return 0
                bz = mz_hat * z + np.sqrt(mz_hat) * xi_b
                y = self.x(z)
                return f(bz, y)
            mu = gaussian_measure_2d(0, np.sqrt(tau_z), 0, 1, integrand)
        return mu

    def measure(self, y, f):
        if not self.contains(y):
            return 0
        if self.slope == 0:
            return quad(f, self.zmin, self.zmax)[0]
        else:
            z = (y - self.x0) / self.slope
            return f(z)


class PiecewiseLinearLikelihood(Likelihood):
    def __init__(self, name, regions, y, y_name, isotropic):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.isotropic = isotropic
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

    def scalar_backward_mean(self, az, bz, y):
        rs = [region.backward_mean(az, bz, y) for region in self.regions]
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        ps = softmax(As, axis=0)
        rz = sum(p*r for p, r in zip(ps, rs))
        return rz

    def scalar_backward_variance(self, az, bz, y):
        rs = [region.backward_mean(az, bz, y) for region in self.regions]
        vs = [region.backward_variance(az, bz, y) for region in self.regions]
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        ps = softmax(As, axis=0)
        Dr = sum(
            ps[i]*ps[j]*(rs[i] - rs[j])**2
            for i in range(self.n_regions)
            for j in range(i+1, self.n_regions)
        )
        vz = sum(p*v for p, v in zip(ps, vs)) + Dr
        return vz

    def scalar_log_partition(self, az, bz, y):
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        A = logsumexp(As)
        return A

    def compute_backward_posterior(self, az, bz, y):
        rs = [region.backward_mean(az, bz, y) for region in self.regions]
        vs = [region.backward_variance(az, bz, y) for region in self.regions]
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        ps = softmax(As, axis=0)
        rz = sum(p*r for p, r in zip(ps, rs))
        Dr = sum(
            ps[i]*ps[j]*(rs[i] - rs[j])**2
            for i in range(self.n_regions)
            for j in range(i+1, self.n_regions)
        )
        vz = sum(p*v for p, v in zip(ps, vs)) + Dr
        if self.isotropic:
            vz = vz.mean()
        return rz, vz

    def compute_log_partition(self, az, bz, y):
        As = [region.log_partitions(az, bz, y) for region in self.regions]
        A = logsumexp(As, axis=0)
        return A.mean()

    def b_measure(self, mz_hat, qz_hat, tz0_hat, f):
        mu = sum(
            region.b_measure(mz_hat, qz_hat, tz0_hat, f) for region in self.regions
        )
        return mu

    def bz_measure(self, mz_hat, qz_hat, tz0_hat, f):
        mu = sum(
            region.bz_measure(mz_hat, qz_hat, tz0_hat, f) for region in self.regions
        )
        return mu

    def beliefs_measure(self, az, tau_z, f):
        mu = sum(
            region.beliefs_measure(az, tau_z, f) for region in self.regions
        )
        return mu

    def measure(self, y, f):
        mu = sum(region.measure(y, f) for region in self.regions)
        return mu


class SgnLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=-1)
        pos = dict(zmin=0, zmax=+np.inf, slope=0, x0=+1)
        regions = [pos, neg]
        super().__init__("sgn", regions, y, y_name, isotropic)


class AbsLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        neg = dict(zmin=-np.inf, zmax=0, slope=-1, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=+1, x0=0)
        regions = [pos, neg]
        super().__init__("abs", regions, y, y_name, isotropic)


class AsymmetricAbsLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True, shift=1e-4):
        neg = dict(zmin=-np.inf, zmax=shift, slope=-1, x0=0)
        pos = dict(zmin=shift, zmax=+np.inf, slope=+1, x0=0)
        regions = [pos, neg]
        super().__init__("a-abs", regions, y, y_name, isotropic)


class ReluLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        neg = dict(zmin=-np.inf, zmax=0, slope=0, x0=0)
        pos = dict(zmin=0, zmax=+np.inf, slope=1, x0=0)
        regions = [pos, neg]
        super().__init__("relu", regions, y, y_name, isotropic)


class LeakyReluLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, slope, y, y_name="y", isotropic=True):
        self.slope = slope
        neg = dict(zmin=-np.inf, zmax=0, slope=slope, x0=0)
        pos = dict(zmin=0, zmax=np.inf, slope=1, x0=0)
        regions = [pos, neg]
        super().__init__("l-relu", regions, y, y_name, isotropic)


class HardTanhLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        neg = dict(zmin=-np.inf, zmax=-1, slope=0, x0=-1)
        mid = dict(zmin=-1, zmax=+1, slope=1, x0=0)
        pos = dict(zmin=+1, zmax=+np.inf, slope=0, x0=+1)
        regions = [pos, mid, neg]
        super().__init__("h-tanh", regions, y, y_name, isotropic)


class HardSigmoidLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, y, y_name="y", isotropic=True):
        l = 3
        neg = dict(zmin=-np.inf, zmax=-l, slope=0, x0=0)
        mid = dict(zmin=-l, zmax=+l, slope=1/(2*l), x0=0.5)
        pos = dict(zmin=l, zmax=np.inf, slope=0, x0=1)
        regions = [pos, mid, neg]
        super().__init__("h-sigm", regions, y, y_name, isotropic)


class SymmetricDoorLikelihood(PiecewiseLinearLikelihood):
    def __init__(self, width, y, y_name="y", isotropic=True):
        self.width = width
        neg = dict(zmin=-np.inf, zmax=-width, slope=0, x0=+1)
        mid = dict(zmin=-width, zmax=+width, slope=0, x0=-1)
        pos = dict(zmin=+width, zmax=+np.inf, slope=0, x0=+1)
        regions = [pos, mid, neg]
        super().__init__("door", regions, y, y_name, isotropic)
