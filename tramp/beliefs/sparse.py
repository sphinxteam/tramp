import numpy as np
from scipy.special import expit
from . import normal

def A(a, b, eta):
    return np.logaddexp(eta, normal.A(a, b))


def r(a, b, eta):
    xi = normal.A(a, b) - eta
    s = expit(xi)
    return s * (b / a)


def v(a, b, eta):
    xi = normal.A(a, b) - eta
    s = expit(xi)
    return s / a + s * (1-s) * (b / a)**2
