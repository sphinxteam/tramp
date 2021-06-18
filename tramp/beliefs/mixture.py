from scipy.special import logsumexp, softmax
from . import normal


"""
Notes
-----
We assume a, b, eta are of of same shape and the first axis corresponds to the
K mixture components.

For instance:
- scalar case : a, b, eta are (K,) arrays
- vector case : a, b, eta are (K, N) arrays
"""


def A(a, b, eta):
    xi = eta + normal.A(a, b)
    return logsumexp(xi, axis=0)


def p(a, b, eta):
    xi = eta + normal.A(a, b)
    s = softmax(xi, axis=0)
    return s


def r(a, b, eta):
    s = p(a, b, eta)
    rs = s * normal.r(a, b)
    return rs.sum(axis=0)


def v(a, b, eta):
    s = p(a, b, eta)
    r_ = normal.r(a, b)
    vs = s * normal.v(a, b)
    Dr = 0.5 * sum(
        sk*sl*(rk-rl)**2 for sk, rk in zip(s, r_) for sl, rl in zip(s, r_)
    )
    return Dr + vs.sum(axis=0)


def tau(a, b, eta):
    s = p(a, b, eta)
    taus = s * normal.tau(a, b)
    return taus.sum(axis=0)
