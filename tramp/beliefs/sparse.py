import numpy as np
from scipy.special import expit
from . import normal

def A(a, b, eta):
    return np.logaddexp(eta, normal.A(a, b))


def p(a, b, eta):
    xi = normal.A(a, b) - eta
    s = expit(xi)
    return s


def r(a, b, eta):
    s = p(a, b, eta)
    return s * (b / a)


def v(a, b, eta):
    s = p(a, b, eta)
    return s / a + s * (1-s) * (b / a)**2


def tau(a, b, eta):
    s = p(a, b, eta)
    return s / a + s * (b / a)**2
