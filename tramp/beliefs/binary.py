import numpy as np


def A(b):
    return np.logaddexp(b, -b)  # ln 2 cosh(b) leads to overflow


def r(b):
    return np.tanh(b)


def v(b):
    return 1-np.tanh(b)**2  # 1 / cosh**2 leads to overflow


def tau(b):
    return 1.
