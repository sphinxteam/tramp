import numpy as np

def A(a, b):
    return 0.5 * (b**2 / a + np.log(2*np.pi / a))


def r(a, b):
    return b / a


def v(a, b):
    return 1 / a


def tau(a, b):
    return 1 / a + (b / a)**2
