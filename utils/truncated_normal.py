"""
Computes the mean and variance of a trunctated normal distribution.

 Reference
 ---------
- https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf
"""

import numpy as np
from scipy.special import erf, erfc, erfcx

def switch(x, y):
    "Switch x and y values to have np.abs(x) <= np.abs(y)"
    x_new = np.where(np.abs(x) > np.abs(y), y, x)
    y_new = np.where(np.abs(x) > np.abs(y), x, y)
    return x_new, y_new


def F0_inf(x, y):
    "Return F0(x, y) where y is inf"
    return np.log(erfcx(np.sign(y)*x)) - x**2


def F0_close(x, y):
    e = y - x
    L = (
        - x*e
        + (1/6)*(x**2 - 2)*e**2
        - (1/180)*(x**4 + 2*x**2 - 8)
        + np.log(2*e/np.sqrt(np.pi))
    )
    return L - x**2


def F0_neg(x, y):
    D = np.exp(x**2 - y**2)
    L = np.log(np.abs(
        D * erfcx(-y) - erfcx(-x)
    ))
    return L - x**2


def F0_pos(x, y):
    D = np.exp(x**2 - y**2)
    L = np.log(np.abs(
        erfcx(x) - D * erfcx(y)
    ))
    return L - x**2


def F0_other(x, y):
    return np.log(np.abs(erf(y) - erf(x)))

def F0(x, y, thresh=1e-7):
    "Computes log|erf(y) - erf(x)|"
    x, y = switch(x, y)
    assert (np.abs(x) <= np.abs(y)).all()
    # conditions
    inf = np.isinf(y)
    close = ~inf & (np.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F0 values
    F = np.zeros_like(x, dtype=float)
    F[inf] = F0_inf(x[inf], y[inf])
    F[close] = F0_close(x[close], y[close])
    F[neg] = F0_neg(x[neg], y[neg])
    F[pos] = F0_pos(x[pos], y[pos])
    F[other] = F0_other(x[other], y[other])

    return F

def F1_inf(x, y):
    "Return F1(x, y) where y is inf"
    return np.sign(y)/erfcx(np.sign(y)*x)


def F1_close(x, y):
    e = y - x
    return np.sqrt(np.pi) * (
        x
        + (1/2)*e
        - (1/6)*e**2
        - (1/12)*e**3
        + (1/90)*x*(x**2+1.)*e**4
    )


def F1_neg(x, y):
    D = np.exp(x**2 - y**2)
    return (1 - D) / (D * erfcx(-y) - erfcx(-x))


def F1_pos(x, y):
    D = np.exp(x**2 - y**2)
    return (1 - D) / (erfcx(x) - D * erfcx(y))


def F1_other(x, y):
    D = np.exp(x**2 - y**2)
    return np.exp(-x**2) * (1 - D) / (erf(y) - erf(x))

def F1(x, y, thresh=1e-7):
    "Computes (exp(-x^2) - exp(-y^2)) / (erf(y) - erf(x))"
    x, y = switch(x, y)
    assert (np.abs(x) <= np.abs(y)).all()
    # conditions
    inf = np.isinf(y)
    close = ~inf & (np.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F1 values
    F = np.zeros_like(x, dtype=float)
    F[inf] = F1_inf(x[inf], y[inf])
    F[close] = F1_close(x[close], y[close])
    F[neg] = F1_neg(x[neg], y[neg])
    F[pos] = F1_pos(x[pos], y[pos])
    F[other] = F1_other(x[other], y[other])

    return F

def F2_inf(x, y):
    "Return F2(x, y) where y is inf"
    return np.sign(y)*x/erfcx(np.sign(y)*x)

def F2_close(x, y):
    "Taylor expansion of F2(x, x+e)"
    e = y - x
    return np.sqrt(np.pi) * (
        x**2 - 1/2
        + x*e
        - (1/3)*(x**2-1)*e**2
        - (1/3)*x*e**3
        + (1/90)*(2*x**4 + 3*x**2 - 8)*e**4
    )

def F2_neg(x, y):
    D = np.exp(x**2 - y**2)
    return (x - D * y) / (D * erfcx(-y) - erfcx(-x))

def F2_pos(x, y):
    D = np.exp(x**2 - y**2)
    return (x - D * y) / (erfcx(x) - D * erfcx(y))

def F2_other(x, y):
    D = np.exp(x**2 - y**2)
    return np.exp(-x**2) * (x - D * y) / (erf(y) - erf(x))

def F2(x, y, thresh=1e-7):
    "Computes (x*exp(-x^2) - y*exp(-y^2)) / (erf(y) - erf(x))"
    x, y = switch(x, y)
    assert (np.abs(x) <= np.abs(y)).all()
    # conditions
    inf = np.isinf(y)
    close = ~inf & (np.abs(x - y) <= thresh)
    neg = (x < 0) & (y < 0)
    pos = (x > 0) & (y > 0)
    other = ~(neg | pos)
    neg = ~inf & ~close & neg
    pos = ~inf & ~close & pos
    other = ~inf & ~close & other
    # F2 values
    F = np.zeros_like(x, dtype=float)
    F[inf] = F2_inf(x[inf], y[inf])
    F[close] = F2_close(x[close], y[close])
    F[neg] = F2_neg(x[neg], y[neg])
    F[pos] = F2_pos(x[pos], y[pos])
    F[other] = F2_other(x[other], y[other])

    return F

def G0(x, y):
    "Computes log|Phi(y) - Phi(x)|"
    return np.log(0.5) + F0(x/np.sqrt(2), y/np.sqrt(2))


def G1(x, y):
    "Computes [N(y) - N(x)] / [Phi(y) - Phi(x)]"
    return -np.sqrt(2/np.pi) * F1(x/np.sqrt(2), y/np.sqrt(2))


def G2(x, y):
    "Computes [-y*N(y) + x*N(x)] / [Phi(y) - Phi(x)]"
    return (2/np.sqrt(np.pi)) * F2(x/np.sqrt(2), y/np.sqrt(2))


def G0_inf(x, s):
    "Computes G0(x, +inf) or G0(x, -inf)"
    return np.log(0.5) + F0_inf(x/np.sqrt(2), s)


def G1_inf(x, s):
    "Computes G1(x, +inf) or G1(x, -inf)"
    return -np.sqrt(2/np.pi) * F1_inf(x/np.sqrt(2), s)


def G2_inf(x, s):
    "Computes G2(x, +inf) or G2(x, -inf)"
    return (2/np.sqrt(np.pi)) * F2_inf(x/np.sqrt(2), s)

def truncated_normal_mean(r0, v0, zmin, zmax):
    "Mean of N(z | r0 v0) delta_[zmin, zmin](z)"
    assert zmin < zmax
    s0 = np.sqrt(v0)
    ymin = (r0 - zmin) / s0
    ymax = (r0 - zmax) / s0
    if (zmax == +np.inf):
        g1 = G1_inf(ymin, -1)
    elif (zmin == -np.inf):
        g1 = G1_inf(ymax, +1)
    else:
        g1 = G1(ymin, ymax)
    r = r0 + s0 * g1
    return r


def truncated_normal_var(r0, v0, zmin, zmax):
    "Variance of N(z | r0 v0) delta_[zmin, zmin](z)"
    assert zmin < zmax
    s0 = np.sqrt(v0)
    ymin = (r0 - zmin) / s0
    ymax = (r0 - zmax) / s0
    if (zmax == +np.inf):
        g1 = G1_inf(ymin, -1)
        g2 = G2_inf(ymin, -1)
    elif (zmin == -np.inf):
        g1 = G1_inf(ymax, +1)
        g2 = G2_inf(ymax, +1)
    else:
        g1 = G1(ymin, ymax)
        g2 = G2(ymin, ymax)
    v = v0 * (1. + g2 - g1**2)
    return v


def truncated_normal_logZ(r0, v0, zmin, zmax):
    "Log Partition of N(z | r0 v0) delta_[zmin, zmin](z)"
    assert zmin < zmax
    s0 = np.sqrt(v0)
    ymin = (r0 - zmin) / s0
    ymax = (r0 - zmax) / s0
    if (zmax == +np.inf):
        g0 = G0_inf(ymin, -1)
    elif (zmin == -np.inf):
        g0 = G0_inf(ymax, +1)
    else:
        g0 = G0(ymin, ymax)
    logZ = 0.5*np.sqrt(2*np.pi*v0) + 0.5*r0**2/v0 + g0
    return logZ


class TruncatedNormal():
    "Inference knowing z in [zmin, zmax] and x = z0 + slope*z"
    def __init__(self, zmin, zmax, z0, slope):
        assert zmin < zmax
        self.zmin = zmin
        self.zmax = zmax
        self.z0 = z0
        self.slope = slope

    def backward(self, az, bz, ax, bx):
        a = az + self.slope**2 * ax
        b = bz + self.slope * (bx - ax * self.z0)
        r0 = b / a
        v0 = 1 / a
        rz = truncated_normal_mean(r0, v0, self.zmin, self.zmax)
        vz = truncated_normal_var(r0, v0, self.zmin, self.zmax)
        logZ = truncated_normal_logZ(r0, v0, self.zmin, self.zmax)
        A = logZ - 0.5*ax*self.z0**2 + bx*self.z0
        return rz, vz, A

    def forward(self, az, bz, ax, bx):
        rz, vz, A = self.backward(az, bz, ax, bx)
        rx = self.slope * rz + self.z0
        vx = self.slope**2 * vz
        return rx, vx, A
